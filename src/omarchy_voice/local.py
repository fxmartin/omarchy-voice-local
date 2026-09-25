"""Local speech engine (`engine = "local"`).

Configuration checks, microphone capture with endpointing, and the entry point
that starts the session loop in `local_session.py`.
"""

from __future__ import annotations

import asyncio
import collections
import ipaddress
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
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


def ready_problems(config: Config) -> list[str]:
    """What stands between here and a working local session.

    Unlike the Realtime engine this needs no websockets, and an API key only
    when the planner is configured to send one.
    """
    from .realtime import default_source
    problems = config_problems(config)
    key_env = config.local_planner_api_key_env
    if key_env and not os.environ.get(key_env):
        problems.append(f"{key_env} is not set (the local planner uses it)")
    for tool in ("pw-record", "pw-cat"):
        if not shutil.which(tool):
            problems.append(f"{tool} is missing (install pipewire-audio)")
    source = default_source()
    if not source:
        problems.append("PipeWire reports no audio input — is a microphone plugged in?")
    elif source.endswith(".monitor"):
        problems.append(f"the default input is {source}, which is not a microphone — "
                        "plug one in, or set `device` in config")
    return problems


def _probe_stt(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True  # it answered, just not with a page
    except (urllib.error.URLError, OSError):
        return False


def _service_model() -> str:
    """The model the whisper-server user service is configured to load, if any."""
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "whisper-server", "-p", "Environment", "--value"],
            capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for item in out.split():
        if item.startswith("WHISPER_MODEL="):
            return item.split("=", 1)[1]
    return ""


def _synthesize(model: str) -> int:
    """Bytes of audio Piper produces for a short phrase; 0 when it cannot speak."""
    try:
        return len(subprocess.run(["piper", "--model", model, "--output-raw"],
                                  input=b"Test.", capture_output=True, timeout=30).stdout)
    except (OSError, subprocess.TimeoutExpired):
        return 0


def setup_checks(config: Config, *, probe_stt=_probe_stt, service_model=_service_model,
                 which=shutil.which, synthesize=_synthesize) -> list[tuple[bool, str]]:
    """Doctor's checks of the local engine's two servers, each with its fix."""
    checks: list[tuple[bool, str]] = []
    if probe_stt(config.local_stt_url):
        checks.append((True, f"whisper.cpp server answering at {config.local_stt_url}"))
    else:
        checks.append((False, f"whisper.cpp server not answering at {config.local_stt_url} "
                              "— start it: systemctl --user start whisper-server "
                              "(setup in docs/local.md)"))
    model = service_model()
    if model:
        checks.append((Path(model).is_file(), f"recognition model {model}"))

    if not which("piper"):
        checks.append((False, "piper is not installed — uv tool install piper-tts "
                              "(the Arch `piper` package is an unrelated mouse tool)"))
    voice = config.local_piper_model
    if not voice:
        checks.append((False, "[local] piper_model is unset — see docs/local.md for a voice"))
        return checks
    if not Path(voice).is_file():
        checks.append((False, f"Piper voice not found: {voice}"))
        return checks
    if not Path(f"{voice}.json").is_file():
        checks.append((False, f"Piper voice config missing: {voice}.json"))
        return checks
    if which("piper"):
        spoke = synthesize(voice) > 0
        checks.append((spoke, f"Piper voice {Path(voice).stem}"
                              + ("" if spoke else " could not synthesize a test phrase")))
    return checks


def run(config: Config) -> int:
    problems = config_problems(config)
    if problems:
        for problem in problems:
            print(f"cannot start local engine: {problem}", file=sys.stderr)
        return 1
    from .local_session import LocalSession
    from .realtime import _run_until_done
    try:
        return _run_until_done(LocalSession(config))
    except KeyboardInterrupt:
        return 0
