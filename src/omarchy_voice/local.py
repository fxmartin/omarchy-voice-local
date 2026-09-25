"""Local speech engine (`engine = "local"`).

Selection, configuration and microphone capture with endpointing so far: the
recognition, planning and speech pipeline is not wired in yet, so `run` refuses
to start rather than pretending.
"""

from __future__ import annotations

import asyncio
import collections
import ipaddress
import sys
from typing import Awaitable, Callable
from urllib.parse import urlsplit

from .config import Config
from .feedback import Feedback
from .realtime import _terminate, frame_level

# whisper.cpp consumes 16 kHz mono, so capture at that rate rather than resample.
CAPTURE_RATE = 16000
FRAME_MS = 30
_FRAME_BYTES = CAPTURE_RATE * FRAME_MS // 1000 * 2
# On the meter's 0..1 scale (room tone is 0, quiet speech ~0.36); a frame at or
# above this counts as speech. Energy-based on purpose: measure before adding a
# neural detector.
SPEECH_LEVEL = 0.2


def config_problems(config: Config) -> list[str]:
    problems: list[str] = []
    if not config.local_stt_url:
        problems.append("[local] stt_url is empty")
    if not config.local_planner_base_url:
        problems.append("[local] planner_base_url is empty")
    return problems


class Endpointer:
    """Cuts a PCM16 stream into utterances from frame energy alone.

    Pure and synchronous so it can be driven with synthetic audio. A frame is
    speech or not; an utterance opens on the first speech frame (with the recent
    pre-roll in front so the first syllable survives), closes after the
    configured silence or at the maximum length, and is dropped if it held less
    speech than the minimum.
    """

    def __init__(self, silence_ms: int, min_speech_ms: int, max_speech_ms: int,
                 preroll_ms: int):
        self._silence_frames = max(1, silence_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._max_frames = max(1, max_speech_ms // FRAME_MS)
        self._preroll: collections.deque[bytes] = collections.deque(
            maxlen=preroll_ms // FRAME_MS)
        self.reset()

    @classmethod
    def from_config(cls, config: Config) -> "Endpointer":
        return cls(config.local_endpoint_silence_ms, config.local_endpoint_min_speech_ms,
                   config.local_endpoint_max_speech_ms, config.local_endpoint_preroll_ms)

    def reset(self) -> None:
        """Forget everything, including a partial utterance."""
        self._preroll.clear()
        self._frames: list[bytes] = []
        self._speech = 0
        self._silence = 0

    def feed(self, frame: bytes) -> bytes | None:
        """Take one frame; return a finished utterance, or None."""
        is_speech = frame_level(frame) >= SPEECH_LEVEL
        if not self._frames:
            if not is_speech:
                self._preroll.append(frame)
                return None
            self._frames = [*self._preroll, frame]
            self._preroll.clear()
            self._speech, self._silence = 1, 0
        else:
            self._frames.append(frame)
            if is_speech:
                self._speech, self._silence = self._speech + 1, 0
            else:
                self._silence += 1
        if self._silence >= self._silence_frames or len(self._frames) >= self._max_frames:
            utterance = b"".join(self._frames) if self._speech >= self._min_speech_frames else None
            self._frames = []
            return utterance
        return None


class LocalCapture:
    """The microphone for the local engine.

    Same gate as the Realtime engine: listening is the lifetime of the
    `pw-record` process, so while it is off there is no recorder to leak from.
    Every exit path — toggle off, close, cancellation, an unexpected EOF, a
    failing consumer — ends with the recorder process gone.
    """

    def __init__(self, config: Config, feedback: Feedback,
                 on_utterance: Callable[[bytes], None],
                 spawn: Callable[[list[str]], Awaitable[asyncio.subprocess.Process]]
                 | None = None):
        self.config = config
        self.feedback = feedback
        self.on_utterance = on_utterance
        self._spawn = spawn or self._spawn_recorder
        self.endpointer = Endpointer.from_config(config)
        self.active = False
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None

    @staticmethod
    async def _spawn_recorder(cmd: list[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def set_active(self, active: bool) -> None:
        if active == self.active:
            return
        if active:
            await self._start()
        else:
            await self._stop()

    async def close(self) -> None:
        await self._stop()

    async def _start(self) -> None:
        cmd = ["pw-record", "--rate", str(CAPTURE_RATE), "--channels", "1",
               "--format", "s16", "--latency", "20ms"]
        if self.config.device:
            cmd += ["--target", self.config.device]
        cmd.append("-")
        try:
            self._proc = await self._spawn(cmd)
        except FileNotFoundError:
            self.feedback.log("error   pw-record is missing — install pipewire-audio")
            return
        self.endpointer.reset()
        self.active = True
        self.feedback.log("mic     capturing")
        self._task = asyncio.create_task(self._pump(self._proc))

    async def _stop(self) -> None:
        self.active = False
        task, self._task = self._task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._kill()
        self.endpointer.reset()

    async def _kill(self) -> None:
        # Also covers a task cancelled before its first step ever ran.
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            await _terminate(proc)

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        stdout = proc.stdout
        assert stdout is not None
        try:
            while True:
                try:
                    frame = await stdout.readexactly(_FRAME_BYTES)
                except asyncio.IncompleteReadError:
                    if self.active:
                        self.feedback.log("error   pw-record ended unexpectedly")
                    return
                self.feedback.level(frame_level(frame))
                utterance = self.endpointer.feed(frame)
                if utterance is not None:
                    self.on_utterance(utterance)
        except Exception as exc:
            self.feedback.log(f"error   capture failed: {type(exc).__name__}: {exc}")
        finally:
            self.active = False
            await self._kill()
            # Leave the meter at rest, or the orb keeps the last loud frame.
            self.feedback.level(0.0)
            self.feedback.log("mic     stopped")


def _is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def stt_warnings(config: Config) -> list[str]:
    """Privacy notices for doctor: a non-loopback server receives your voice."""
    if config.local_stt_url and not _is_loopback(config.local_stt_url):
        return [f"[local] stt_url {config.local_stt_url} is not a loopback address; "
                "audio leaves this machine"]
    return []


def run(config: Config) -> int:
    print("the local engine's audio pipeline is not implemented yet", file=sys.stderr)
    return 1
