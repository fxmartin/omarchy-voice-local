"""whisper.cpp recognition client, against a fake server on a local socket."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import config, local, local_stt

PCM = b"\x01\x00" * 1600


class FakeWhisper(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, status=200, body=None, delay=0.0):
        self.status, self.delay = status, delay
        self.body = json.dumps({"text": " open the browser. "}).encode() if body is None else body
        self.requests: list[tuple[str, dict, bytes]] = []
        handler = self._handler()
        super().__init__(("127.0.0.1", 0), handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                server.requests.append((self.path, dict(self.headers), self.rfile.read(length)))
                if server.delay:
                    time.sleep(server.delay)
                try:
                    self.send_response(server.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(server.body)))
                    self.end_headers()
                    self.wfile.write(server.body)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        return Handler

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class FakeFeedback:
    def __init__(self):
        self.logs, self.notices = [], []

    def log(self, line):
        self.logs.append(line)

    def notify(self, title, body="", urgency="low"):
        self.notices.append((title, body))
        return True


class WavTests(unittest.TestCase):
    def test_pcm_becomes_a_valid_wav_in_memory(self):
        with wave.open(io.BytesIO(local_stt.pcm_to_wav(PCM)), "rb") as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
            self.assertEqual(w.readframes(w.getnframes()), PCM)


class TranscribeTests(unittest.TestCase):
    def serve(self, **kw):
        server = FakeWhisper(**kw)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def cfg(self, server, **kw):
        return config.Config(engine="local", local_stt_url=server.url, **kw)

    def test_posts_wav_to_inference_and_returns_stripped_text(self):
        server = self.serve()
        text = local_stt.transcribe(self.cfg(server, local_language="fr"), PCM)
        self.assertEqual(text, "open the browser.")
        path, headers, body = server.requests[0]
        self.assertEqual(path, "/inference")
        self.assertIn("multipart/form-data; boundary=", headers["Content-Type"])
        self.assertIn(b'name="file"; filename="utterance.wav"', body)
        self.assertIn(b"RIFF", body)
        self.assertIn(b'name="language"\r\n\r\nfr', body)
        self.assertIn(b'name="response_format"\r\n\r\njson', body)

    def test_base_url_trailing_slash_is_tolerated(self):
        server = self.serve()
        cfg = config.Config(engine="local", local_stt_url=server.url + "/")
        self.assertEqual(local_stt.transcribe(cfg, PCM), "open the browser.")
        self.assertEqual(server.requests[0][0], "/inference")

    def test_http_error_raises_short_message(self):
        server = self.serve(status=500, body=b"boom")
        with self.assertRaises(local_stt.SttError) as ctx:
            local_stt.transcribe(self.cfg(server), PCM)
        self.assertIn("500", str(ctx.exception))

    def test_server_error_field_raises(self):
        server = self.serve(body=b'{"error": "no model"}')
        with self.assertRaises(local_stt.SttError):
            local_stt.transcribe(self.cfg(server), PCM)

    def test_malformed_body_raises(self):
        server = self.serve(body=b"not json")
        with self.assertRaises(local_stt.SttError):
            local_stt.transcribe(self.cfg(server), PCM)

    def test_slow_server_times_out(self):
        server = self.serve(delay=1.5)
        with self.assertRaises(local_stt.SttError) as ctx:
            local_stt.transcribe(self.cfg(server, local_stt_timeout_seconds=0.2), PCM)
        self.assertIn("timed out", str(ctx.exception))

    def test_server_down_raises(self):
        server = self.serve()
        cfg = self.cfg(server)
        server.shutdown()
        server.server_close()
        with self.assertRaises(local_stt.SttError) as ctx:
            local_stt.transcribe(cfg, PCM)
        self.assertIn("unreachable", str(ctx.exception))

    def test_non_http_scheme_is_refused(self):
        cfg = config.Config(engine="local", local_stt_url="file:///etc/passwd")
        with self.assertRaises(local_stt.SttError):
            local_stt.transcribe(cfg, PCM)

    def test_audio_is_never_written_to_disk(self):
        server = self.serve()
        with mock.patch("builtins.open", side_effect=AssertionError("disk write")), \
             mock.patch("tempfile.NamedTemporaryFile", side_effect=AssertionError("tmp")), \
             mock.patch("tempfile.TemporaryFile", side_effect=AssertionError("tmp")):
            self.assertEqual(local_stt.transcribe(self.cfg(server), PCM), "open the browser.")


class HeardTests(unittest.TestCase):
    def test_success_logs_heard(self):
        server = FakeWhisper()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        fb = FakeFeedback()
        text = local_stt.hear(config.Config(local_stt_url=server.url), fb, PCM)
        self.assertEqual(text, "open the browser.")
        self.assertEqual(fb.logs, ["heard   'open the browser.'"])

    def test_failure_is_reported_and_returns_none(self):
        server = FakeWhisper(status=503, body=b"busy")
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        fb = FakeFeedback()
        self.assertIsNone(local_stt.hear(config.Config(local_stt_url=server.url), fb, PCM))
        self.assertEqual(len(fb.notices), 1)
        self.assertIn("503", fb.logs[0])

    def test_empty_transcript_is_not_logged_as_heard(self):
        server = FakeWhisper(body=b'{"text": "  "}')
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        fb = FakeFeedback()
        self.assertIsNone(local_stt.hear(config.Config(local_stt_url=server.url), fb, PCM))
        self.assertEqual(fb.logs, [])
        self.assertEqual(fb.notices, [])


class PrivacyWarningTests(unittest.TestCase):
    def warnings(self, url):
        return local.stt_warnings(config.Config(engine="local", local_stt_url=url))

    def test_loopback_urls_do_not_warn(self):
        for url in ("http://127.0.0.1:9000", "http://localhost:9000",
                    "http://[::1]:9000", "http://127.1.2.3/"):
            self.assertEqual(self.warnings(url), [], url)

    def test_remote_urls_warn_audio_leaves_machine(self):
        for url in ("http://192.168.1.5:9000", "https://stt.example.com", "http://0.0.0.0:9000"):
            found = self.warnings(url)
            self.assertEqual(len(found), 1, url)
            self.assertIn("leaves this machine", found[0])

    def test_empty_url_gives_no_warning(self):
        self.assertEqual(self.warnings(""), [])


if __name__ == "__main__":
    unittest.main()
