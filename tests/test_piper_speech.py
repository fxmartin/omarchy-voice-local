"""Streaming Piper speech: sentence pipelining, state, stopping, missing Piper."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice.piper_speech import DEFAULT_RATE, PiperSpeaker, split_sentences, voice_rate


class FakePlayer:
    def __init__(self):
        self.written: list[bytes] = []
        self.interrupts = 0
        self.playing = False
        self.level = 0.0
        self.queued = 0.0

    async def write(self, pcm):
        self.written.append(pcm)
        self.playing = True

    def queued_seconds(self):
        return self.queued

    def is_playing(self, tail=0):
        return self.playing

    def level_now(self):
        return self.level

    async def interrupt(self):
        self.interrupts += 1
        self.playing = False


class SplitTests(unittest.TestCase):
    def test_splits_on_sentence_ends(self):
        self.assertEqual(split_sentences("Hello there. How are you? Fine!"),
                         ["Hello there.", "How are you?", "Fine!"])

    def test_keeps_trailing_fragment_and_drops_blanks(self):
        self.assertEqual(split_sentences("  One.\n\nTwo without stop  "),
                         ["One.", "Two without stop"])
        self.assertEqual(split_sentences("   "), [])


def make(synth, player=None, **kw):
    notices, logs = [], []
    speaker = PiperSpeaker(
        synthesize=synth, player=player or FakePlayer(),
        notify=lambda title, body: notices.append((title, body)),
        log=logs.append, **kw)
    return speaker, notices, logs


class SpeakTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_sentence_plays_before_rest_is_synthesized(self):
        gate = asyncio.Event()
        seen = []

        async def synth(sentence):
            seen.append(sentence)
            if sentence == "Second.":
                await gate.wait()
            return sentence.encode().ljust(4, b"_")[:4]

        speaker, notices, _ = make(synth)
        speaker.speak("First. Second.")
        for _ in range(100):
            if speaker._player.written:
                break
            await asyncio.sleep(.01)
        self.assertEqual(len(speaker._player.written), 1)
        self.assertTrue(speaker.is_speaking())
        gate.set()
        await speaker.wait()
        self.assertEqual(len(speaker._player.written), 2)
        self.assertEqual(notices, [])

    async def test_state_and_level_come_from_the_player(self):
        async def synth(sentence):
            return b"\0\0"

        speaker, _, _ = make(synth)
        self.assertFalse(speaker.is_speaking())
        speaker.speak("Hi.")
        await speaker.wait()
        speaker._player.level = .4
        self.assertTrue(speaker.is_speaking())
        self.assertEqual(speaker.level(), .4)
        speaker._player.playing = False
        self.assertFalse(speaker.is_speaking())

    async def test_stop_drops_queued_sentences_quickly(self):
        started = asyncio.Event()

        async def synth(sentence):
            if sentence == "Two.":
                started.set()
                await asyncio.sleep(30)
            return b"\0\0"

        speaker, _, _ = make(synth)
        speaker.speak("One. Two. Three.")
        await started.wait()
        t0 = time.monotonic()
        await speaker.stop()
        self.assertLess(time.monotonic() - t0, .25)
        self.assertEqual(speaker._player.interrupts, 1)
        self.assertEqual(len(speaker._player.written), 1)
        self.assertFalse(speaker.is_speaking())

    async def test_new_reply_replaces_the_current_one(self):
        block = asyncio.Event()

        async def synth(sentence):
            if sentence == "Old two.":
                await block.wait()
            return sentence.encode().ljust(2, b"_")

        speaker, _, _ = make(synth)
        speaker.speak("Old one. Old two. Old three.")
        await asyncio.sleep(.05)
        speaker.speak("New.")
        await speaker.wait()
        self.assertEqual(speaker._player.written[-1], b"New.")
        self.assertNotIn(b"Old three.", speaker._player.written)

    async def test_backpressure_waits_for_the_player_queue(self):
        async def synth(sentence):
            return b"\0\0"

        speaker, _, _ = make(synth)
        speaker._player.queued = 99
        speaker.speak("One. Two.")
        await asyncio.sleep(.1)
        self.assertEqual(speaker._player.written, [])
        speaker._player.queued = 0
        await speaker.wait()
        self.assertEqual(len(speaker._player.written), 2)

    async def test_synth_failure_notifies_once_and_logs_once(self):
        async def synth(sentence):
            raise FileNotFoundError("piper")

        speaker, notices, logs = make(synth)
        for _ in range(2):
            speaker.speak("Reply text.")
            await speaker.wait()
        self.assertEqual(len(notices), 2)  # each reply is still shown
        self.assertEqual(notices[0][1], "Reply text.")
        self.assertEqual(len(logs), 1)

    async def test_missing_piper_or_model_is_reported_before_synthesis(self):
        notices, logs = [], []
        speaker = PiperSpeaker(model="/nonexistent/voice.onnx", player=FakePlayer(),
                               notify=lambda t, b: notices.append((t, b)),
                               log=logs.append, which=lambda name: None)
        speaker.speak("Hello.")
        await speaker.wait()
        self.assertEqual(notices, [("Oma", "Hello.")])
        self.assertEqual(len(logs), 1)
        self.assertEqual(speaker._player.written, [])

    async def test_missing_model_file_is_reported_when_piper_exists(self):
        notices, logs = [], []
        speaker = PiperSpeaker(model="/nonexistent/voice.onnx", player=FakePlayer(),
                               notify=lambda t, b: notices.append((t, b)),
                               log=logs.append, which=lambda name: "/usr/bin/piper")
        speaker.speak("Hello.")
        await speaker.wait()
        self.assertEqual(notices, [("Oma", "Hello.")])
        self.assertIn("model", logs[0])


class VoiceRateTests(unittest.TestCase):
    def test_reads_sample_rate_from_voice_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp, "v.onnx")
            Path(f"{model}.json").write_text('{"audio": {"sample_rate": 16000}}')
            self.assertEqual(voice_rate(str(model)), 16000)

    def test_falls_back_when_config_is_missing_or_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp, "v.onnx")
            self.assertEqual(voice_rate(str(model)), DEFAULT_RATE)
            Path(f"{model}.json").write_text("{not json")
            self.assertEqual(voice_rate(str(model)), DEFAULT_RATE)
            Path(f"{model}.json").write_text("{}")
            self.assertEqual(voice_rate(str(model)), DEFAULT_RATE)


class PiperProcessTests(unittest.IsolatedAsyncioTestCase):
    """Drive the real `_piper` subprocess path with a fake `piper` executable."""

    def _install(self, script: str) -> tuple[str, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        exe = Path(tmp.name, "piper")
        exe.write_text("#!/bin/sh\n" + script)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        model = Path(tmp.name, "voice.onnx")
        model.write_bytes(b"")
        patcher = mock.patch.dict(os.environ, {"PATH": f"{tmp.name}:{os.environ['PATH']}"})
        patcher.start()
        self.addCleanup(patcher.stop)
        return str(exe), str(model)

    async def test_speaks_through_piper_and_trims_odd_byte(self):
        _, model = self._install("cat >/dev/null; printf 'abcde'")
        player = FakePlayer()
        speaker = PiperSpeaker(model, player=player, which=lambda n: "/x/piper")
        speaker.speak("Hello.")
        await speaker.wait()
        self.assertEqual(player.written, [b"abcd"])

    async def test_nonzero_exit_is_reported(self):
        _, model = self._install("cat >/dev/null; exit 3\n")
        notices, logs = [], []
        speaker = PiperSpeaker(model, player=FakePlayer(), which=lambda n: "/x/piper",
                               notify=lambda t, b: notices.append((t, b)), log=logs.append)
        speaker.speak("Hello.")
        await speaker.wait()
        self.assertEqual(notices, [("Oma", "Hello.")])
        self.assertIn("exited with 3", logs[0])

    async def test_stop_kills_a_running_piper(self):
        _, model = self._install("cat >/dev/null; sleep 30\n")
        speaker = PiperSpeaker(model, player=FakePlayer(), which=lambda n: "/x/piper")
        speaker.speak("Hello.")
        await asyncio.sleep(.3)
        await asyncio.wait_for(speaker.stop(), 5)
        self.assertFalse(speaker.is_speaking())

    async def test_stop_and_wait_are_safe_when_idle(self):
        speaker, _, _ = make(lambda s: None)
        await speaker.stop()
        await speaker.wait()
        self.assertEqual(speaker._player.interrupts, 1)

    def test_default_player_is_a_live_speaker_at_the_voice_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp, "v.onnx")
            Path(f"{model}.json").write_text('{"audio": {"sample_rate": 16000}}')
            speaker = PiperSpeaker(str(model))
            self.assertEqual(speaker._player.rate, 16000)
            self.assertEqual(speaker._player.queued_seconds(), 0)


if __name__ == "__main__":
    unittest.main()
