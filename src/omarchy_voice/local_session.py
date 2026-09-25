"""The local engine's session: listen, recognise, plan, act and speak.

    pw-record ─▶ endpointing ─▶ whisper.cpp ─▶ planner + tools ─▶ Piper ─▶ pw-cat

It presents the same surface as the Realtime engine: the toggle is the only
thing that opens the microphone, the bar and orb see the same states, and the
control socket takes the same verbs. What differs is that nothing here streams
room audio anywhere; only the recognised text reaches the planner.

Every collaborator is injectable, so the whole loop is exercised offline with
fakes: no microphone, recognition server, model or speaker.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from .config import Config
from .session import ControlServer, _matches

# Utterances are 16 kHz mono PCM16, so this many bytes is one second.
_BYTES_PER_SECOND = 16000 * 2
# Her speech still rings in the room for a moment after playback ends.
_ECHO_TAIL_SECONDS = 0.3
_METER_INTERVAL = 0.05
_WATCH_POLL_SECONDS = 5.0
# How long the bar and orb show an error before returning to listening.
ERROR_HOLD_SECONDS = 2.0


class _CaptureMeter:
    """Feedback as the microphone sees it: one channel is yours, one is hers.

    Capture publishes the microphone level on every frame. Passing the real
    Feedback through would zero her channel, so the orb would go still exactly
    while she talks. This fills in her level, and holds yours at zero while she
    speaks, because what the microphone hears then is mostly her.
    """

    def __init__(self, feedback: Any, speaker: Any):
        self._feedback = feedback
        self._speaker = speaker

    def level(self, value: float, voice: float = 0.0) -> None:
        speaking = self._speaker.is_speaking()
        self._feedback.level(0.0 if speaking else value, self._speaker.level())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._feedback, name)


class LocalSession:
    def __init__(self, config: Config, *, feedback: Any = None, executor: Any = None,
                 planner: Any = None, capture: Any = None, speaker: Any = None,
                 transcribe: Callable[[bytes], str | None] | None = None):
        self.config = config
        if feedback is None:
            from .feedback import Feedback
            feedback = Feedback(config)
        self.feedback = feedback
        if executor is None:
            from .tools import Executor
            executor = Executor(config, on_action=self._on_action)
        self.executor = executor
        if planner is None:
            from .planner import Planner
            planner = Planner.for_local(config, executor)
        self.planner = planner
        if speaker is None:
            from .piper_speech import PiperSpeaker
            speaker = PiperSpeaker(config.local_piper_model,
                                   notify=feedback.notify, log=feedback.log)
        self.speaker = speaker
        if capture is None:
            from .local import LocalCapture
            capture = LocalCapture(config, _CaptureMeter(feedback, speaker),
                                   on_utterance=self._queue_utterance)
        self.capture = capture
        if transcribe is None:
            from .local_stt import transcribe as whisper
            transcribe = lambda pcm: whisper(config, pcm)  # noqa: E731
        # Raises on failure, so the session can say which stage failed.
        self.transcribe = transcribe
        self.error_hold = ERROR_HOLD_SECONDS

        # Always starts muted; only the toggle opens the microphone.
        self.active = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop = asyncio.Event()
        self._utterances: asyncio.Queue[bytes] = asyncio.Queue()
        # Serialises turns: a typed `say`, a spoken utterance and a local
        # confirm must never plan or act at the same time.
        self._turn = asyncio.Lock()
        self._spoke_until = 0.0
        self._held = 0

    # -- plumbing -----------------------------------------------------------
    def _on_action(self, name: str, description: str) -> None:
        self.feedback.state("acting", description)
        self.feedback.log(f"action  {description}")

    def _resting_state(self) -> None:
        if self.executor.pending:
            self.feedback.state("confirm", self.executor.describe(*self.executor.pending))
        else:
            self.feedback.state("listening" if self.active else "idle")

    def _say(self, text: str) -> None:
        self.feedback.log(f"reply   {text!r}")
        self.speaker.speak(text)
        self.note_speaking()

    def note_speaking(self) -> None:
        """Record that she is speaking now, for the half-duplex gate."""
        if self.speaker.is_speaking():
            self._spoke_until = time.monotonic() + _ECHO_TAIL_SECONDS

    def _queue_utterance(self, pcm: bytes) -> None:
        self._utterances.put_nowait(pcm)

    def _failed(self, stage: str, detail: str) -> None:
        """Say which stage failed, show it briefly, then go back to resting.

        Never raises and never stops listening: one bad turn is one bad turn.
        """
        self.feedback.log(f"error   {stage}: {detail}")
        self.feedback.state("error", f"{stage} failed")
        self.feedback.notify("Voice", f"{stage} failed: {detail}")
        if self.error_hold > 0 and self.loop is not None:
            self.loop.call_later(self.error_hold, self._resting_state)
        else:
            self._resting_state()

    # -- the microphone gate --------------------------------------------------
    async def set_active(self, active: bool) -> str:
        self.active = active
        await self.capture.set_active(active)
        if not active:
            await self.speaker.stop()
            vision = getattr(self.executor, "vision", None)
            if vision is not None:
                await asyncio.to_thread(vision.stop_owned)
        self._resting_state()
        self.feedback.notify("Listening" if active else "Sleeping")
        self.feedback.log(f"gate    {'listening' if active else 'muted'}")
        return "listening" if active else "idle"

    # -- turns ----------------------------------------------------------------
    async def handle_utterance(self, pcm: bytes) -> None:
        """One endpointed utterance from the microphone."""
        started = time.monotonic() - len(pcm) / _BYTES_PER_SECOND
        self.note_speaking()
        if started < self._spoke_until:
            if not self.config.barge_in:
                # Half duplex: what the microphone heard while she spoke is
                # mostly her, and must never be taken as an instruction.
                self._held += 1
                self.feedback.log(f"mic     held an utterance while speaking ({self._held})")
                return
            await self.speaker.stop()
        self.feedback.state("thinking")
        try:
            text = await asyncio.to_thread(self.transcribe, pcm)
        except Exception as exc:
            self._failed("speech recognition", str(exc) or type(exc).__name__)
            return
        text = (text or "").strip()
        if not text:
            self._resting_state()
            return
        self.feedback.log(f"heard   {text!r}")
        await self.handle_text(text)

    async def handle_text(self, text: str) -> None:
        """A recognised or typed instruction."""
        async with self._turn:
            if self.executor.pending:
                if _matches(text, self.config.confirm_words, allow_negation=False):
                    await self._release(f"confirm {text!r}")
                    return
                if _matches(text, self.config.cancel_words):
                    self._drop()
                    return
            self.feedback.state("thinking")
            turn = await asyncio.to_thread(self.planner.think, text)
            if self.executor.pending:
                held = self.executor.describe(*self.executor.pending)
                phrase = self.config.confirm_words[0] if self.config.confirm_words else "confirm"
                self._say(f"{turn.reply} Say {phrase} to run {held}, or cancel.")
            elif turn.reply:
                self._say(turn.reply)
            if turn.error:
                # A network drop lands here: only this turn's planning failed,
                # and recognition and speech keep working for the next one.
                self._failed("planning", turn.error)
                return
            self._resting_state()

    async def _release(self, why: str) -> str:
        held = self.executor.describe(*self.executor.pending)
        self.feedback.log(f"{why} released: {held}")
        result = await asyncio.to_thread(self.executor.run_pending)
        self._say(result.output or ("Done." if result.ok else "That failed."))
        self._resting_state()
        return result.output or ("done" if result.ok else "failed")

    def _drop(self) -> str:
        held = self.executor.drop_pending()
        if held is None:
            return "nothing to cancel"
        self.feedback.log(f"cancel  {held}")
        self._say(f"Cancelled {held}.")
        self._resting_state()
        return f"Cancelled: {held}. It was not run."

    # -- announcements ------------------------------------------------------------
    async def announce(self, job: dict) -> None:
        """Say a watched command finished, or notify when nobody is listening."""
        if job["vanished"]:
            headline = f"The pane running {job['label']} was closed."
        elif job["timed_out"]:
            headline = f"{job['label']} is still going after a long time."
        else:
            headline = f"{job['label']} finished in {job['seconds']:.0f} seconds."
        self.feedback.log(f"watch   {job['target']}: {headline}")
        if not self.active:
            self.feedback.notify("Oma", headline)
            return
        async with self._turn:
            self._say(headline)

    async def _poll_task_notices(self) -> None:
        if not self.config.tasks_enabled or not hasattr(self.executor, "task_manager"):
            return
        manager = await asyncio.to_thread(self.executor.task_manager)
        await asyncio.to_thread(manager.list)
        for notice in await asyncio.to_thread(manager.store.notices):
            if await asyncio.to_thread(self.feedback.notify, "OMA task", notice["text"]):
                await asyncio.to_thread(manager.store.acknowledge, notice["id"])

    async def _watch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(_WATCH_POLL_SECONDS)
                await self._poll_task_notices()
                for job in await asyncio.to_thread(self.executor.poll_watches):
                    await self.announce(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a watcher must never take the session down
                self.feedback.log(f"warn    watcher: {type(exc).__name__}: {exc}")

    async def _meter_loop(self) -> None:
        """Keep the half-duplex gate current, and her orb channel while muted."""
        while not self._stop.is_set():
            self.note_speaking()
            if not self.capture.active and self.speaker.is_speaking():
                self.feedback.level(0.0, self.speaker.level())
            await asyncio.sleep(_METER_INTERVAL)

    async def _turn_loop(self) -> None:
        while not self._stop.is_set():
            pcm = await self._utterances.get()
            if not self.active:
                continue
            try:
                await self.handle_utterance(pcm)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one broken turn must not end listening
                self._failed("that turn", f"{type(exc).__name__}: {exc}")

    # -- control socket -------------------------------------------------------------
    async def control_async(self, command: str) -> str:
        verb, _, rest = command.partition(" ")
        if verb == "toggle":
            return await self.set_active(not self.active)
        if verb in ("start", "stop"):
            return await self.set_active(verb == "start")
        if verb == "say":
            text = rest.strip()
            if not text:
                return "nothing to say"
            self.feedback.log(f"typed   {text!r}")
            await self.handle_text(text)
            return "sent"
        if verb == "confirm":
            if not self.executor.pending:
                return "nothing to confirm"
            async with self._turn:
                return await self._release("confirm local")
        if verb == "cancel":
            async with self._turn:
                return self._drop()
        if verb == "quit":
            self._stop.set()
            return "stopping"
        return f"unknown command {verb!r}"

    def _control(self, command: str) -> str:
        """Called on the ControlServer thread; hands work to the event loop."""
        if self.loop is None:
            return "not ready"
        if command.partition(" ")[0] == "quit":
            self.loop.call_soon_threadsafe(self._stop.set)
            return "stopping"
        future = asyncio.run_coroutine_threadsafe(self.control_async(command), self.loop)
        try:
            return future.result(timeout=60)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    # -- lifecycle -----------------------------------------------------------------
    async def run(self) -> int:
        return await self.serve()

    async def serve(self, *, start_control: bool = True) -> int:
        self.loop = asyncio.get_running_loop()
        control = ControlServer(self._control) if start_control else None
        if control is not None:
            control.start()
        tasks = [asyncio.create_task(self._turn_loop()),
                 asyncio.create_task(self._meter_loop()),
                 asyncio.create_task(self._watch_loop())]
        self._resting_state()
        self.feedback.log(f"start   engine=local stt={self.config.local_stt_url} "
                          f"planner={self.config.local_planner_model} "
                          f"dry_run={self.config.dry_run}")
        self.feedback.log("gate    muted — press SUPER + SHIFT + V to start listening")
        try:
            await self._stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # The recorder must be gone on every way out.
            await self.capture.close()
            await self.speaker.stop()
            if control is not None:
                control.stop()
            self.feedback.state("stopped")
            self.feedback.level(0.0, 0.0)
        return 0
