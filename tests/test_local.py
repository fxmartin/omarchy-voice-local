"""Local engine selection: config problems and the run dispatch."""

from __future__ import annotations

import array
import asyncio
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import cli, config, local


class LocalEngineTests(unittest.TestCase):
    def test_defaults_have_no_problems(self):
        self.assertEqual(local.config_problems(config.Config(engine="local")), [])

    def test_empty_urls_are_reported(self):
        problems = local.config_problems(config.Config(
            engine="local", local_stt_url="", local_planner_base_url=""))
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("stt_url" in p for p in problems))
        self.assertTrue(any("planner_base_url" in p for p in problems))

    def test_run_refuses_until_pipeline_exists(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(local.run(config.Config(engine="local")), 1)
        self.assertIn("not implemented", err.getvalue())

    def test_cmd_run_dispatches_local(self):
        args = cli.build_parser().parse_args(["run", "--engine", "local"])
        with mock.patch.object(local, "run", return_value=0) as run:
            self.assertEqual(cli.cmd_run(args, config.Config(engine="local")), 0)
        run.assert_called_once()


RATE = local.CAPTURE_RATE
FRAME_MS = local.FRAME_MS
FRAME = RATE * FRAME_MS // 1000 * 2


def tone(ms: int, amplitude: int = 8000) -> bytes:
    """Synthetic PCM16: a square wave loud enough to count as speech."""
    n = RATE * ms // 1000
    return array.array("h", [amplitude if i % 20 < 10 else -amplitude for i in range(n)]).tobytes()


def quiet(ms: int) -> bytes:
    return bytes(RATE * ms // 1000 * 2)


def frames(pcm: bytes):
    for i in range(0, len(pcm), FRAME):
        yield pcm[i:i + FRAME]


def endpointer(**kw) -> local.Endpointer:
    return local.Endpointer(**{"silence_ms": 300, "min_speech_ms": 150,
                               "max_speech_ms": 5000, "preroll_ms": 90, **kw})


def run_all(ep: local.Endpointer, pcm: bytes) -> list[bytes]:
    return [u for f in frames(pcm) if (u := ep.feed(f)) is not None]


class EndpointerTests(unittest.TestCase):
    def test_speech_then_pause_yields_one_utterance_with_preroll(self):
        out = run_all(endpointer(), quiet(600) + tone(600) + quiet(900) + quiet(300))
        self.assertEqual(len(out), 1)
        # Speech is kept whole, with the 90 ms pre-roll before its first syllable.
        self.assertGreaterEqual(len(out[0]), len(tone(600)) + len(quiet(90)))
        self.assertTrue(out[0].startswith(quiet(90)))

    def test_no_utterance_until_silence_elapses(self):
        self.assertEqual(run_all(endpointer(), tone(600) + quiet(240)), [])

    def test_room_tone_sends_nothing(self):
        low = tone(2000, amplitude=20)
        self.assertEqual(run_all(endpointer(), quiet(1000) + low + quiet(500)), [])

    def test_brief_noise_below_minimum_is_discarded(self):
        self.assertEqual(run_all(endpointer(), quiet(300) + tone(60) + quiet(900)), [])

    def test_short_pause_inside_speech_does_not_split(self):
        pcm = tone(400) + quiet(150) + tone(400) + quiet(600)
        self.assertEqual(len(run_all(endpointer(), pcm)), 1)

    def test_overlong_utterance_is_cut_and_sent(self):
        out = run_all(endpointer(max_speech_ms=1000), tone(2500))
        self.assertEqual(len(out), 2)
        for utterance in out:
            self.assertLessEqual(len(utterance), len(tone(1000)) + len(quiet(90)))

    def test_reset_drops_a_partial_utterance(self):
        ep = endpointer()
        run_all(ep, tone(400))
        ep.reset()
        self.assertEqual(run_all(ep, quiet(900)), [])

    def test_from_config_uses_endpoint_settings(self):
        ep = local.Endpointer.from_config(config.Config(
            engine="local", local_endpoint_silence_ms=200, local_endpoint_min_speech_ms=100,
            local_endpoint_max_speech_ms=900, local_endpoint_preroll_ms=60))
        self.assertEqual(len(run_all(ep, tone(400) + quiet(300))), 1)
        self.assertEqual(len(run_all(ep, tone(2000))), 2)


class FakeProc:
    """Stands in for pw-record: a stream we feed, and terminate() that ends it."""

    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.returncode = None
        self.terminated = 0

    def terminate(self):
        self.terminated += 1
        self.returncode = -15
        self.stdout.feed_eof()

    def kill(self):
        self.returncode = -9
        self.stdout.feed_eof()

    async def wait(self):
        return self.returncode


class FakeFeedback:
    def __init__(self):
        self.levels: list[float] = []
        self.logs: list[str] = []

    def level(self, value, voice=0.0):
        self.levels.append(value)

    def log(self, line):
        self.logs.append(line)


class LocalCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.procs: list[FakeProc] = []
        self.feedback = FakeFeedback()
        self.utterances: list[bytes] = []
        self.commands: list[list[str]] = []

    async def spawn(self, cmd):
        self.commands.append(cmd)
        proc = FakeProc()
        self.procs.append(proc)
        return proc

    def capture(self, **cfg) -> local.LocalCapture:
        return local.LocalCapture(
            config.Config(engine="local", local_endpoint_silence_ms=300,
                          local_endpoint_min_speech_ms=150, **cfg),
            self.feedback, self.utterances.append, spawn=self.spawn)

    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0)

    async def test_no_recorder_until_listening(self):
        cap = self.capture()
        self.assertEqual(self.procs, [])
        await cap.close()
        self.assertEqual(self.procs, [])

    async def test_recorder_command_uses_device_and_rate(self):
        cap = self.capture(device="mic-x")
        await cap.set_active(True)
        cmd = self.commands[0]
        self.assertEqual(cmd[0], "pw-record")
        self.assertEqual(cmd[cmd.index("--rate") + 1], str(RATE))
        self.assertEqual(cmd[cmd.index("--target") + 1], "mic-x")
        await cap.close()

    async def test_speech_is_handed_over_and_level_published(self):
        cap = self.capture()
        await cap.set_active(True)
        self.procs[0].stdout.feed_data(quiet(300) + tone(500) + quiet(600))
        await self.settle()
        self.assertEqual(len(self.utterances), 1)
        self.assertTrue(any(v > 0.3 for v in self.feedback.levels))
        await cap.close()

    async def test_partial_frames_are_reassembled(self):
        cap = self.capture()
        await cap.set_active(True)
        pcm = tone(500) + quiet(600)
        for i in range(0, len(pcm), 1000):
            self.procs[0].stdout.feed_data(pcm[i:i + 1000])
            await self.settle()
        self.assertEqual(len(self.utterances), 1)
        await cap.close()

    async def test_toggle_off_exits_recorder_and_drops_partial(self):
        cap = self.capture()
        await cap.set_active(True)
        self.procs[0].stdout.feed_data(tone(400))
        await self.settle()
        await cap.set_active(False)
        self.assertIsNotNone(self.procs[0].returncode)
        self.assertEqual(self.utterances, [])
        self.assertEqual(self.feedback.levels[-1], 0.0)
        # Listening again starts a fresh recorder and a fresh endpointer.
        await cap.set_active(True)
        self.procs[1].stdout.feed_data(quiet(900))
        await self.settle()
        self.assertEqual(self.utterances, [])
        await cap.close()
        self.assertIsNotNone(self.procs[1].returncode)

    async def test_close_exits_recorder(self):
        cap = self.capture()
        await cap.set_active(True)
        await cap.close()
        self.assertIsNotNone(self.procs[0].returncode)

    async def test_cancellation_exits_recorder(self):
        cap = self.capture()
        await cap.set_active(True)
        await self.settle()
        cap._task.cancel()
        await asyncio.gather(cap._task, return_exceptions=True)
        self.assertIsNotNone(self.procs[0].returncode)
        await cap.close()

    async def test_unexpected_eof_exits_and_reports(self):
        cap = self.capture()
        await cap.set_active(True)
        self.procs[0].stdout.feed_eof()
        await self.settle()
        self.assertIsNotNone(self.procs[0].returncode)
        self.assertTrue(any("unexpectedly" in line for line in self.feedback.logs))
        self.assertFalse(cap.active)
        await cap.close()

    async def test_consumer_error_still_exits_recorder(self):
        def boom(_utterance):
            raise RuntimeError("consumer failed")
        cap = local.LocalCapture(config.Config(
            engine="local", local_endpoint_silence_ms=300, local_endpoint_min_speech_ms=150),
            self.feedback, boom, spawn=self.spawn)
        await cap.set_active(True)
        self.procs[0].stdout.feed_data(tone(500) + quiet(600))
        await self.settle()
        self.assertIsNotNone(self.procs[0].returncode)
        await cap.close()

    async def test_missing_recorder_is_reported(self):
        async def missing(cmd):
            raise FileNotFoundError(cmd[0])
        cap = local.LocalCapture(config.Config(engine="local"), self.feedback,
                                 self.utterances.append, spawn=missing)
        await cap.set_active(True)
        self.assertFalse(cap.active)
        self.assertTrue(any("pw-record is missing" in line for line in self.feedback.logs))


if __name__ == "__main__":
    unittest.main()
