"""Planner: configurable endpoint, key requirement, bounded in-memory history."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import config, planner


class FakeEndpoint:
    """An OpenAI-compatible chat server on a local socket; replies are queued."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(length)),
                })
                payload = json.dumps(outer.replies.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def say(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]}}]}


class FakeExecutor:
    pending = None

    def call(self, name, args):
        return mock.Mock(as_tool_result=lambda: "ok: closed")

    def describe(self, name, args):
        return f"{name}({args})"


def make(endpoint: FakeEndpoint, **kwargs) -> planner.Planner:
    cfg = config.Config()
    return planner.Planner(
        cfg, FakeExecutor(), base_url=endpoint.base_url, **kwargs)


class EndpointTests(unittest.TestCase):
    def test_posts_to_configured_base_url_with_key(self):
        with FakeEndpoint([say("hi")]) as ep, \
                mock.patch.dict("os.environ", {"FAKE_KEY": "sekrit"}):
            turn = make(ep, api_key_env="FAKE_KEY", key_required=True).think("hello")
        self.assertEqual(turn.reply, "hi")
        self.assertEqual(ep.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(ep.requests[0]["auth"], "Bearer sekrit")

    def test_missing_key_is_error_only_when_required(self):
        with FakeEndpoint([say("hi")]) as ep, \
                mock.patch.dict("os.environ", {}, clear=True):
            turn = make(ep, api_key_env="FAKE_KEY", key_required=True).think("hello")
            self.assertIn("FAKE_KEY is not set", turn.error)
            self.assertEqual(ep.requests, [])
            turn = make(ep, api_key_env="FAKE_KEY", key_required=False).think("hello")
        self.assertEqual(turn.error, "")
        self.assertIsNone(ep.requests[0]["auth"])

    def test_default_planner_keeps_openai_url_and_requires_key(self):
        p = planner.Planner(config.Config(), FakeExecutor())
        self.assertEqual(p.base_url, "https://api.openai.com/v1")
        self.assertTrue(p.key_required)
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIn("OPENAI_API_KEY", p.think("x").error)

    def test_for_local_uses_local_settings(self):
        cfg = config.Config(
            local_planner_base_url="http://127.0.0.1:1/v1",
            local_planner_model="m", local_planner_api_key_env="")
        p = planner.Planner.for_local(cfg, FakeExecutor())
        self.assertEqual(p.base_url, "http://127.0.0.1:1/v1")
        self.assertEqual(p.model, "m")
        self.assertFalse(p.key_required)
        self.assertGreater(p.history_turns, 0)


class MemoryTests(unittest.TestCase):
    def test_second_utterance_sees_earlier_exchange_with_tools(self):
        replies = [
            tool_call("close", {"target": "firefox"}),
            say("Closed it."),
            say("Done."),
        ]
        with FakeEndpoint(replies) as ep:
            p = make(ep, key_required=False, history_turns=5)
            p.think("open firefox")
            p.think("and close it")
        second = ep.requests[2]["body"]["messages"]
        roles = [m["role"] for m in second]
        self.assertEqual(roles, ["system", "user", "assistant", "tool",
                                 "assistant", "user"])
        self.assertEqual(second[1]["content"], "open firefox")
        self.assertEqual(second[2]["tool_calls"][0]["function"]["name"], "close")
        self.assertEqual(second[3]["content"], "ok: closed")
        self.assertEqual(second[-1]["content"], "and close it")

    def test_history_is_bounded_oldest_dropped_first(self):
        with FakeEndpoint([say(str(i)) for i in range(4)]) as ep:
            p = make(ep, key_required=False, history_turns=2)
            for i in range(4):
                p.think(f"u{i}")
        last = ep.requests[3]["body"]["messages"]
        users = [m["content"] for m in last if m["role"] == "user"]
        self.assertEqual(users, ["u1", "u2", "u3"])

    def test_default_planner_has_no_memory(self):
        with FakeEndpoint([say("a"), say("b")]) as ep, \
                mock.patch.dict("os.environ", {"K": "x"}):
            p = make(ep, api_key_env="K", key_required=True)
            p.think("one")
            p.think("two")
        users = [m["content"] for m in ep.requests[1]["body"]["messages"]
                 if m["role"] == "user"]
        self.assertEqual(users, ["two"])

    def test_failed_turn_is_not_remembered(self):
        with FakeEndpoint([say("ok")]) as ep:
            p = make(ep, key_required=False, history_turns=3)
            with mock.patch.object(planner, "_chat", side_effect=RuntimeError("x")):
                p.think("broken")
            p.think("fine")
        users = [m["content"] for m in ep.requests[0]["body"]["messages"]
                 if m["role"] == "user"]
        self.assertEqual(users, ["fine"])

    def test_history_never_touches_disk(self):
        with FakeEndpoint([say("a")]) as ep, \
                mock.patch("builtins.open") as opened, \
                mock.patch("pathlib.Path.write_text") as write:
            make(ep, key_required=False, history_turns=3).think("one")
        opened.assert_not_called()
        write.assert_not_called()


class LoopEdgeTests(unittest.TestCase):
    def test_usage_is_reported_as_tokens(self):
        reply = say("hi")
        reply["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
        with FakeEndpoint([reply]) as ep:
            turn = make(ep, key_required=False).think("x")
        self.assertEqual(turn.tokens, {"in": 7, "out": 3})

    def test_unparseable_tool_arguments_are_reported_to_the_model(self):
        bad = {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "close", "arguments": "{not json"}}]}}]}
        with FakeEndpoint([bad, say("sorry")]) as ep:
            turn = make(ep, key_required=False).think("x")
        tool_msg = ep.requests[1]["body"]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("could not parse arguments", tool_msg["content"])
        self.assertEqual(turn.actions, [])

    def test_pending_confirmation_returns_and_is_remembered(self):
        executor = FakeExecutor()
        executor.pending = object()
        with FakeEndpoint([tool_call("close", {"target": "x"})]) as ep:
            p = planner.Planner(config.Config(), executor, base_url=ep.base_url,
                                key_required=False, history_turns=2)
            turn = p.think("close x")
        self.assertEqual(turn.reply, "That needs confirmation.")
        self.assertEqual(len(ep.requests), 1)
        self.assertEqual(p._history[0][-1],
                         {"role": "assistant", "content": "That needs confirmation."})

    def test_step_budget_exhaustion_replies_and_remembers(self):
        cfg = config.Config(max_turns=2)
        calls = [tool_call("close", {"target": "x"}) for _ in range(2)]
        with FakeEndpoint(calls) as ep:
            p = planner.Planner(cfg, FakeExecutor(), base_url=ep.base_url,
                                key_required=False, history_turns=2)
            turn = p.think("loop")
        self.assertEqual(turn.reply, "Ran out of steps on that one.")
        self.assertEqual(len(p._history), 1)


class HttpErrorTests(unittest.TestCase):
    def test_http_error_becomes_unavailable(self):
        import io
        import urllib.error
        err = urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"bad gateway"))
        with mock.patch.object(planner.urllib.request, "urlopen", side_effect=err):
            with self.assertRaises(planner.PlannerUnavailable) as ctx:
                planner._chat([], [], config.Config(), "")
        self.assertIn("planner HTTP 500: bad gateway", str(ctx.exception))

    def test_unreachable_endpoint_becomes_unavailable(self):
        import urllib.error
        err = urllib.error.URLError("refused")
        with mock.patch.object(planner.urllib.request, "urlopen", side_effect=err):
            turn = planner.Planner(config.Config(), FakeExecutor(),
                                   key_required=False).think("x")
        self.assertIn("could not reach the planner: refused", turn.error)
        self.assertEqual(turn.reply, "My planner isn't configured yet.")


if __name__ == "__main__":
    unittest.main()
