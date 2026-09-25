"""One end-to-end pass over the realtime wire protocol, against a fake server.

This is the only test that exercises `RealtimeSession.run` itself — the connect,
the opening `session.update`, the function-call round trip, and a clean shutdown
through the control socket. It needs the `websockets` package; without it the
whole module skips, so the suite still runs green on a machine that only uses
a machine without python-websockets.

No audio is involved: the session starts muted (mode is not "always"), so
pw-record is never spawned.

Run with: python3 -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from websockets.asyncio.server import serve
except ImportError:  # pragma: no cover - depends on the machine
    serve = None

from omarchy_voice import feedback, realtime, session as session_mod
from omarchy_voice.config import Config


class FakeRealtimeServer:
    """The smallest server that looks like the Realtime API from the client's side."""

    def __init__(self):
        self.received: list[dict] = []
        self.session_update = asyncio.Event()
        self.tool_answered = asyncio.Event()
        self.port = 0
        self._server = None

    async def start(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        await ws.send(json.dumps({"type": "session.created",
                                  "session": {"id": "sess_test"}}))
        async for raw in ws:
            event = json.loads(raw)
            self.received.append(event)
            kind = event.get("type")
            if kind == "session.update":
                self.session_update.set()
                # A turn the model decided needs a tool call.
                await ws.send(json.dumps({
                    "type": "response.done",
                    "response": {"status": "completed", "output": [{
                        "type": "function_call",
                        "name": "hypr_dispatch",
                        "call_id": "call_wire",
                        "arguments": json.dumps(
                            {"lua": 'hl.dsp.focus({ workspace = "3" })'}),
                    }]},
                }))
            elif kind == "response.create":
                await ws.send(json.dumps({
                    "type": "response.output_audio_transcript.done",
                    "transcript": "Switched to workspace 3.",
                }))
                self.tool_answered.set()

    def of_type(self, kind):
        return [e for e in self.received if e.get("type") == kind]


@unittest.skipIf(serve is None, "websockets is not installed")
class WireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patcher = mock.patch('omarchy_voice.network.STATE_DIR', root)
        patcher.start()
        self.addCleanup(patcher.stop)
        for module, names in ((feedback, ("LOG_FILE", "STATE_FILE", "STATE_DIR", "RUNTIME_DIR")),
                              (session_mod, ("SOCKET_PATH", "RUNTIME_DIR")),
                              (realtime, ("SAFETY_ID_FILE", "CONFIG_DIR"))):
            for name in names:
                value = root / ("session.log" if name == "LOG_FILE" else
                                "state.json" if name == "STATE_FILE" else
                                "control.sock" if name == "SOCKET_PATH" else
                                "safety-id" if name == "SAFETY_ID_FILE" else "")
                patcher = mock.patch.object(module, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)

        self.server = FakeRealtimeServer()
        await self.server.start()
        patcher = mock.patch.object(
            realtime, "REALTIME_URL", f"ws://127.0.0.1:{self.server.port}/v1/realtime")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_one_full_round_trip(self):
        config = Config(dry_run=True, notify=False)
        session = realtime.RealtimeSession(config)
        runner = asyncio.create_task(session.run())

        await asyncio.wait_for(self.server.tool_answered.wait(), timeout=30)
        session._user_quit = True
        session._stop.set()
        self.assertEqual(await asyncio.wait_for(runner, timeout=10), 0)

        # 1. The session was configured before anything else was sent.
        update = self.server.of_type("session.update")[0]["session"]
        self.assertEqual(update["type"], "realtime")
        self.assertEqual(update["model"], config.realtime_model)
        self.assertEqual(update["output_modalities"], ["audio"])
        self.assertEqual(update["audio"]["input"]["format"],
                         {"type": "audio/pcm", "rate": config.realtime_sample_rate})
        self.assertEqual(update["audio"]["input"]["turn_detection"],
                         {"type": "semantic_vad"})
        self.assertEqual(update["audio"]["output"]["voice"], config.realtime_voice)
        self.assertIn("confirm_last", {t["name"] for t in update["tools"]})
        self.assertIn("You are the voice control layer", update["instructions"])

        # 2. The tool call was answered, and a spoken reply asked for.
        answer = self.server.of_type("conversation.item.create")[0]["item"]
        self.assertEqual(answer["call_id"], "call_wire")
        self.assertIn("dry-run", answer["output"])
        self.assertTrue(self.server.of_type("response.create"))

        # 3. Muted throughout: no audio was ever appended.
        self.assertEqual(self.server.of_type("input_audio_buffer.append"), [])

    async def test_control_socket_toggles_and_quits(self):
        config = Config(dry_run=True, notify=False)
        session = realtime.RealtimeSession(config)
        runner = asyncio.create_task(session.run())
        await asyncio.wait_for(self.server.session_update.wait(), timeout=30)

        self.assertTrue(session_mod.daemon_running())

        # A real toggle would spawn pw-record, so the capture step is stubbed —
        # what is under test is that a socket command reaches the event loop.
        async def fake_set_active(active):
            return f"toggled {active}"

        with mock.patch.object(session, "_set_active", fake_set_active):
            self.assertEqual(await asyncio.to_thread(session_mod.send_control, "toggle"),
                             "toggled True")
        self.assertEqual(await asyncio.to_thread(session_mod.send_control, "quit"),
                         "stopping")
        self.assertEqual(await asyncio.wait_for(runner, timeout=10), 0)
        self.assertFalse(session_mod.daemon_running())


class FakeRecorder:
    """Stands in for pw-record: a stdout that hits EOF when the process dies."""

    def __init__(self):
        self.returncode = None
        self.stdout = asyncio.StreamReader()

    def terminate(self):
        self.kill()

    def kill(self):
        if self.returncode is None:
            self.returncode = -15
            self.stdout.feed_eof()

    async def wait(self):
        return self.returncode


class ScriptedServer:
    """Per-connection scripts: each connection runs the next callable in `plans`."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.connections = 0
        self.connected = [asyncio.Event() for _ in plans]
        self.port = 0
        self._server = None

    async def start(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, ws):
        index = min(self.connections, len(self.plans) - 1)
        self.connections += 1
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "s"}}))
        async for raw in ws:
            if json.loads(raw).get("type") == "session.update":
                self.connected[index].set()
                await self.plans[index](ws)
                break
        async for _ in ws:  # keep reading until the client goes away
            pass


@unittest.skipIf(serve is None, "websockets is not installed")
class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module, name, value in (
                (feedback, "LOG_FILE", self.root / "session.log"),
                (feedback, "STATE_FILE", self.root / "state.json"),
                (feedback, "STATE_DIR", self.root),
                (feedback, "RUNTIME_DIR", self.root),
                (session_mod, "SOCKET_PATH", self.root / "control.sock"),
                (session_mod, "RUNTIME_DIR", self.root),
                (realtime, "SAFETY_ID_FILE", self.root / "safety-id"),
                (realtime, "CONFIG_DIR", self.root),
                (realtime, "RECONNECT_BASE_DELAY", 0.05),
                (realtime, "RESPONSE_TIMEOUT_SECONDS", 0.3)):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch("omarchy_voice.network.STATE_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

        self.recorders: list[FakeRecorder] = []

        async def spawn(*_cmd, **_kw):
            rec = FakeRecorder()
            self.recorders.append(rec)
            return rec

        patcher = mock.patch.object(realtime.asyncio, "create_subprocess_exec", spawn)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _serve(self, plans):
        server = ScriptedServer(plans)
        await server.start()
        self.addCleanup(server.stop)
        patcher = mock.patch.object(
            realtime, "REALTIME_URL", f"ws://127.0.0.1:{server.port}/v1/realtime")
        patcher.start()
        self.addCleanup(patcher.stop)
        return server

    def _status(self):
        return json.loads((self.root / "state.json").read_text())["status"]

    async def _finish(self, session, runner):
        await asyncio.to_thread(session_mod.send_control, "quit")
        self.assertEqual(await asyncio.wait_for(runner, timeout=10), 0)

    async def test_drop_while_listening_leaves_no_recorder_behind(self):
        async def drop(ws):
            await asyncio.sleep(0.2)
            await ws.close()

        async def stay(ws):
            pass

        server = await self._serve([drop, stay])
        session = realtime.RealtimeSession(Config(dry_run=True, notify=False))
        runner = asyncio.create_task(session.run())
        await asyncio.wait_for(server.connected[0].wait(), timeout=30)
        await asyncio.to_thread(session_mod.send_control, "start")
        await asyncio.wait_for(server.connected[1].wait(), timeout=30)

        for _ in range(100):  # the mic loop starts just after the handshake
            if len(self.recorders) >= 2:
                break
            await asyncio.sleep(0.05)
        # The recorder from the dropped session is gone by the time the new
        # session is up; only the reopened one may run.
        self.assertEqual(len(self.recorders), 2)
        self.assertIsNotNone(self.recorders[0].returncode)
        self.assertIsNone(self.recorders[1].returncode)

        await asyncio.to_thread(session_mod.send_control, "stop")
        self.assertTrue(all(r.returncode is not None for r in self.recorders))
        self.assertEqual(self._status(), "idle")
        await self._finish(session, runner)

    async def test_silent_session_is_rebuilt_and_answers_next_utterance(self):
        async def silent(ws):
            await ws.send(json.dumps({"type": "input_audio_buffer.speech_stopped"}))

        async def answers(ws):
            await ws.send(json.dumps({"type": "input_audio_buffer.speech_stopped"}))
            await ws.send(json.dumps({"type": "response.created"}))
            await ws.send(json.dumps({"type": "response.done",
                                      "response": {"status": "completed", "output": []}}))

        server = await self._serve([silent, answers])
        session = realtime.RealtimeSession(Config(dry_run=True, notify=False))
        runner = asyncio.create_task(session.run())
        await asyncio.wait_for(server.connected[1].wait(), timeout=30)
        await asyncio.sleep(0.3)  # let the answer land

        self.assertEqual(server.connections, 2)
        self.assertIsNotNone(session.ws)
        self.assertEqual(self._status(), "idle")
        await self._finish(session, runner)

    async def test_health_check_gives_up_loudly_when_never_answered(self):
        async def silent(ws):
            await ws.send(json.dumps({"type": "input_audio_buffer.speech_stopped"}))

        server = await self._serve([silent])
        with mock.patch.object(realtime, "RECONNECT_ATTEMPTS", 2):
            session = realtime.RealtimeSession(Config(dry_run=True, notify=False))
            code = await asyncio.wait_for(session.run(), timeout=30)
        self.assertEqual(code, 1)
        self.assertGreaterEqual(server.connections, 2)
        self.assertEqual(self._status(), "error")


if __name__ == "__main__":
    unittest.main()
