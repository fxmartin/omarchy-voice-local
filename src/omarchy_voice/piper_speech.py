"""Streaming local speech: Piper synthesizes one sentence at a time and the
audio is played through PipeWire (`LiveSpeaker`, i.e. `pw-cat`).

The first sentence starts playing while later ones are still being synthesized.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from .playback import LiveSpeaker

DEFAULT_RATE = 22050
MAX_QUEUED_SECONDS = 4.0  # stay well under LiveSpeaker's 10 s buffer cap
_SENTENCE = re.compile(r".+?(?:[.!?…]+[\"')\]]*(?=\s|$)|$)", re.S)


def split_sentences(text: str) -> list[str]:
    return [s for part in _SENTENCE.findall(text) if (s := part.strip())]


def voice_rate(model: str) -> int:
    """Sample rate from the voice's `.onnx.json`, falling back to Piper's default."""
    try:
        return int(json.loads(Path(f"{model}.json").read_text())["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_RATE


class PiperSpeaker:
    def __init__(
        self,
        model: str = "",
        *,
        synthesize: Callable[[str], Awaitable[bytes]] | None = None,
        player=None,
        notify: Callable[[str, str], object] = lambda title, body: None,
        log: Callable[[str], object] = lambda line: None,
        which: Callable[[str], str | None] = shutil.which,
    ):
        self.model = model
        self._synthesize = synthesize or self._piper
        self._player = player or LiveSpeaker(voice_rate(model))
        self._notify = notify
        self._log = log
        self._which = which
        self._task: asyncio.Task | None = None
        self._failure_logged = False

    # -- state the engine polls ---------------------------------------------
    def is_speaking(self) -> bool:
        busy = self._task is not None and not self._task.done()
        return busy or self._player.is_playing()

    def level(self) -> float:
        return self._player.level_now()

    # -- control -------------------------------------------------------------
    def speak(self, text: str) -> None:
        """Replace whatever is being said with `text`. Must run in the event loop."""
        previous, self._task = self._task, None
        if previous and not previous.done():
            previous.cancel()
        self._task = asyncio.create_task(self._run(text, previous))

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._player.interrupt()

    async def wait(self) -> None:
        if self._task:
            await asyncio.shield(self._task)

    # -- internals -----------------------------------------------------------
    async def _run(self, text: str, previous: asyncio.Task | None) -> None:
        if previous:
            with contextlib.suppress(asyncio.CancelledError):
                await previous
            await self._player.interrupt()
        try:
            self._check_available()
            for sentence in split_sentences(text):
                pcm = await self._synthesize(sentence)
                while self._player.queued_seconds() > MAX_QUEUED_SECONDS:
                    await asyncio.sleep(.05)
                await self._player.write(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._notify("Oma", text)
            if not self._failure_logged:
                self._failure_logged = True
                self._log(f"local speech unavailable: {type(exc).__name__}: {exc}")

    def _check_available(self) -> None:
        if self._synthesize != self._piper:
            return
        if not self._which("piper"):
            raise FileNotFoundError("piper is not installed")
        if not self.model or not Path(self.model).is_file():
            raise FileNotFoundError(f"Piper voice model not found: {self.model or '(unset)'}")

    async def _piper(self, sentence: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "piper", "--model", self.model, "--output-raw",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await proc.communicate(sentence.encode())
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0 or not out:
            raise RuntimeError(f"piper exited with {proc.returncode}")
        return out[: len(out) // 2 * 2]
