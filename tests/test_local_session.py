"""The local engine's session loop, driven entirely by fakes.

No microphone, recognition server, model or speaker: each collaborator the
session talks to is injected, so every behaviour here is checked offline.

Run with: python3 -m unittest discover -s tests
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice.config import Config
from omarchy_voice.local_session import LocalSession
from omarchy_voice.planner import Turn
from omarchy_voice.tools import Result

ONE_SECOND = b"\x00\x00" * 16000


class FakeFeedback:
    def __init__(self):
        self.states, self.lines, self.notes, self.levels = [], [], [], []

    def state(self, status, text=""):
        self.states.append((status, text))

    def log(self, line):
        self.lines.append(line)

    def notify(self, title, body="", urgency="low"):
        self.notes.append((title, body))
        return True

    def level(self, value, voice=0.0):
        self.levels.append((value, voice))


class FakeCapture:
    def __init__(self):
        self.active = False
        self.closed = False

    async def set_active(self, active):
        self.active = active

    async def close(self):
        self.active = False
        self.closed = True


class FakeSpeaker:
    def __init__(self):
        self.said, self.stopped, self.speaking = [], 0, False

    def speak(self, text):
        self.said.append(text)

    async def stop(self):
        self.stopped += 1
        self.speaking = False

    def is_speaking(self):
        return self.speaking

    def level(self):
        return 0.5 if self.speaking else 0.0


class FakeExecutor:
    def __init__(self):
        self.pending = None
        self.ran = []

    def describe(self, name, args):
        return f"{name} {args.get('command', '')}".strip()

    def run_pending(self):
        held, self.pending = self.pending, None
        self.ran.append(held)
        return Result(True, "ran it")

    def drop_pending(self):
        if not self.pending:
            return None
        held, self.pending = self.describe(*self.pending), None
        return held

    def poll_watches(self):
        return []


class FakePlanner:
    def __init__(self, executor, reply="Done.", hold=None, error=""):
        self.executor, self.reply, self.hold, self.error = executor, reply, hold, error
        self.heard = []

    def think(self, text):
        self.heard.append(text)
        if self.hold:
            self.executor.pending = self.hold
        return Turn(text=text, reply=self.reply, error=self.error)


def make(config=None, transcript="go to workspace two", **planner_kw):
    feedback, executor = FakeFeedback(), FakeExecutor()
    planner = FakePlanner(executor, **planner_kw)
    session = LocalSession(
        config or Config(), feedback=feedback, executor=executor, planner=planner,
        capture=FakeCapture(), speaker=FakeSpeaker(),
        transcribe=lambda pcm: transcript)
    return session, feedback, executor, planner


def run(coro):
    return asyncio.run(coro)


class UtteranceTests(unittest.TestCase):
    def test_utterance_is_heard_planned_and_spoken(self):
        session, feedback, _, planner = make(reply="Switched to workspace 2.")

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(planner.heard, ["go to workspace two"])
        self.assertEqual(session.speaker.said, ["Switched to workspace 2."])
        statuses = [s for s, _ in feedback.states]
        self.assertLess(statuses.index("thinking"), len(statuses) - 1)
        self.assertEqual(statuses[-1], "listening")
        self.assertTrue(any(line.startswith("reply") for line in feedback.lines))
        self.assertIn("heard   'go to workspace two'", feedback.lines)

    def test_nothing_recognised_goes_back_to_listening(self):
        session, feedback, _, planner = make(transcript=None)

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(planner.heard, [])
        self.assertEqual(session.speaker.said, [])
        self.assertEqual(feedback.states[-1][0], "listening")
        self.assertFalse(any(line.startswith("heard") for line in feedback.lines))

    def test_utterance_overlapping_her_speech_is_dropped(self):
        session, _, _, planner = make()

        async def go():
            await session.set_active(True)
            session.speaker.speaking = True
            session.note_speaking()
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(planner.heard, [])

    def test_barge_in_interrupts_her_instead(self):
        session, _, _, planner = make(Config(barge_in=True))

        async def go():
            await session.set_active(True)
            session.speaker.speaking = True
            session.note_speaking()
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(planner.heard, ["go to workspace two"])
        self.assertGreaterEqual(session.speaker.stopped, 1)


class ConfirmationTests(unittest.TestCase):
    def held(self):
        return make(reply="That needs confirmation.", hold=("run_shell", {"command": "rm x"}))

    def test_held_action_shows_confirm_and_asks(self):
        session, feedback, _, _ = self.held()

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(feedback.states[-1], ("confirm", "run_shell rm x"))
        self.assertIn("confirm", session.speaker.said[-1].lower())

    def test_spoken_confirm_word_releases_it_locally(self):
        session, feedback, executor, planner = self.held()

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
            session.transcribe = lambda pcm: "yes do it"
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(executor.ran, [("run_shell", {"command": "rm x"})])
        self.assertEqual(planner.heard, ["go to workspace two"])  # never asked the model
        self.assertEqual(feedback.states[-1][0], "listening")

    def test_negated_confirm_does_not_release(self):
        session, _, executor, _ = self.held()

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
            session.transcribe = lambda pcm: "don't confirm"
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(executor.ran, [])

    def test_spoken_cancel_drops_it(self):
        session, _, executor, _ = self.held()

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
            session.transcribe = lambda pcm: "never mind"
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertIsNone(executor.pending)
        self.assertEqual(executor.ran, [])
        self.assertIn("cancel", session.speaker.said[-1].lower())


class ControlTests(unittest.TestCase):
    def test_toggle_opens_and_closes_the_microphone(self):
        session, feedback, _, _ = make()

        async def go():
            self.assertEqual(await session.control_async("toggle"), "listening")
            self.assertTrue(session.capture.active)
            self.assertEqual(await session.control_async("toggle"), "idle")
            self.assertFalse(session.capture.active)
        run(go())
        self.assertGreaterEqual(session.speaker.stopped, 1)
        self.assertEqual(feedback.states[-1][0], "idle")

    def test_start_stop_say_confirm_cancel(self):
        session, _, executor, planner = make(
            reply="That needs confirmation.", hold=("run_shell", {"command": "rm x"}))

        async def go():
            self.assertEqual(await session.control_async("start"), "listening")
            self.assertEqual(await session.control_async("say close the window"), "sent")
            self.assertEqual(planner.heard, ["close the window"])
            self.assertEqual(await session.control_async("confirm"), "ran it")
            self.assertEqual(await session.control_async("cancel"), "nothing to cancel")
            self.assertEqual(await session.control_async("stop"), "idle")
        run(go())
        self.assertEqual(len(executor.ran), 1)

    def test_quit_ends_the_session(self):
        session, _, _, _ = make()

        async def go():
            runner = asyncio.create_task(session.serve(start_control=False))
            await asyncio.sleep(0)
            self.assertEqual(await session.control_async("quit"), "stopping")
            return await asyncio.wait_for(runner, 5)
        self.assertEqual(run(go()), 0)
        self.assertTrue(session.capture.closed)


class AnnouncementTests(unittest.TestCase):
    JOB = {"target": "%1", "label": "the build", "seconds": 42, "vanished": False,
           "timed_out": False}

    def test_finished_job_is_spoken_while_listening(self):
        session, feedback, _, _ = make()

        async def go():
            await session.set_active(True)
            await session.announce(self.JOB)
        run(go())
        self.assertIn("the build finished in 42 seconds.", session.speaker.said[-1])
        self.assertEqual(feedback.notes[-1][0], "Listening")

    def test_finished_job_is_a_notification_while_muted(self):
        session, feedback, _, _ = make()
        run(session.announce(self.JOB))
        self.assertEqual(session.speaker.said, [])
        self.assertIn("the build finished", feedback.notes[-1][1])


class FailureTests(unittest.TestCase):
    """Each stage can fail on its own; the session says which and keeps going."""

    def quick(self, session):
        session.error_hold = 0.0
        return session

    def test_recognition_failure_is_named_and_listening_resumes(self):
        session, feedback, _, planner = make()
        self.quick(session)

        def broken(pcm):
            raise RuntimeError("recognition server unreachable")
        session.transcribe = broken

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
            await asyncio.sleep(0.01)
        run(go())
        self.assertEqual(planner.heard, [])
        self.assertIn(("error", "speech recognition failed"), feedback.states)
        self.assertIn("speech recognition", feedback.notes[-1][1])
        self.assertEqual(feedback.states[-1][0], "listening")

    def test_planning_failure_is_named_and_spoken(self):
        session, feedback, _, _ = make(reply="Something went wrong with that.",
                                       error="planner HTTP 500: boom")
        self.quick(session)

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
            await asyncio.sleep(0.01)
        run(go())
        self.assertIn(("error", "planning failed"), feedback.states)
        self.assertEqual(session.speaker.said, ["Something went wrong with that."])
        self.assertEqual(feedback.states[-1][0], "listening")

    def test_a_crashing_turn_does_not_stop_the_next_one(self):
        session, feedback, _, planner = make()
        self.quick(session)
        calls = []

        def flaky(text):
            calls.append(text)
            if len(calls) == 1:
                raise ValueError("unexpected")
            return FakePlanner.think(planner, text)
        planner.think = flaky

        async def go():
            await session.set_active(True)
            runner = asyncio.create_task(session._turn_loop())
            session._queue_utterance(ONE_SECOND)
            session._queue_utterance(ONE_SECOND)
            for _ in range(200):
                if len(calls) == 2 and session.speaker.said:
                    break
                await asyncio.sleep(0.01)
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        run(go())
        self.assertEqual(len(calls), 2)
        self.assertEqual(session.speaker.said, ["Done."])
        self.assertTrue(any("ValueError" in line for line in feedback.lines))

    def test_network_drop_only_fails_the_planning_turn(self):
        session, feedback, _, planner = make()
        self.quick(session)
        outcomes = iter([("", "could not reach the planner: timed out"), ("Done.", "")])

        def think(text):
            reply, error = next(outcomes)
            planner.heard.append(text)
            return Turn(text=text, reply=reply or "Something went wrong with that.", error=error)
        planner.think = think

        async def go():
            await session.set_active(True)
            await session.handle_utterance(ONE_SECOND)
            await session.handle_utterance(ONE_SECOND)
        run(go())
        self.assertEqual(len(planner.heard), 2)  # recognition kept working
        self.assertEqual(session.speaker.said[-1], "Done.")

    def test_audio_is_never_logged(self):
        session, feedback, _, _ = make()

        async def go():
            await session.set_active(True)
            await session.handle_utterance(b"\x01\x02" * 16000)
        run(go())
        self.assertFalse(any("\\x01" in line or "\x01" in line for line in feedback.lines))


class FakeProc:
    """Stands in for pw-record: a stream nobody feeds, ended by terminate()."""

    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.returncode = None

    def terminate(self):
        self.returncode = -15
        self.stdout.feed_eof()

    def kill(self):
        self.returncode = -9
        self.stdout.feed_eof()

    async def wait(self):
        return self.returncode


class RecorderLifetimeTests(unittest.TestCase):
    """No recorder process outlives the session, however it ends."""

    def session_with_real_capture(self):
        from omarchy_voice.local import LocalCapture
        procs = []

        async def spawn(cmd):
            procs.append(FakeProc())
            return procs[-1]
        session, feedback, _, _ = make()
        session.capture = LocalCapture(Config(engine="local"), feedback,
                                       session._queue_utterance, spawn=spawn)
        return session, procs

    def test_quit_while_listening_stops_the_recorder(self):
        session, procs = self.session_with_real_capture()

        async def go():
            runner = asyncio.create_task(session.serve(start_control=False))
            await asyncio.sleep(0)
            await session.control_async("start")
            self.assertIsNone(procs[0].returncode)
            await session.control_async("quit")
            await asyncio.wait_for(runner, 5)
        run(go())
        self.assertEqual(len(procs), 1)
        self.assertIsNotNone(procs[0].returncode)

    def test_recorder_stops_even_after_a_turn_crashed(self):
        session, procs = self.session_with_real_capture()
        session.error_hold = 0.0

        def broken(pcm):
            raise RuntimeError("boom")
        session.transcribe = broken

        async def go():
            runner = asyncio.create_task(session.serve(start_control=False))
            await asyncio.sleep(0)
            await session.control_async("start")
            session._queue_utterance(ONE_SECOND)
            await asyncio.sleep(0.05)
            await session.control_async("quit")
            await asyncio.wait_for(runner, 5)
        run(go())
        self.assertIsNotNone(procs[0].returncode)


if __name__ == "__main__":
    unittest.main()
