"""Doctor's view of the local engine: readiness, setup checks and privacy.

Every probe is injected, so nothing here needs a recognition server, Piper,
a microphone or a network.

Run with: python3 -m unittest discover -s tests
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import cli, config, local

ROOT = Path(__file__).resolve().parent.parent


class ReadyTests(unittest.TestCase):
    def ready(self, cfg, env=None, source="alsa_input.mic"):
        with mock.patch.dict(os.environ, env or {}, clear=True), \
             mock.patch.object(local.shutil, "which", return_value="/usr/bin/x"), \
             mock.patch("omarchy_voice.realtime.default_source", return_value=source):
            return local.ready_problems(cfg)

    def test_needs_no_websockets_and_no_key_without_one_configured(self):
        cfg = config.Config(engine="local", local_planner_api_key_env="")
        with mock.patch.dict(sys.modules, {"websockets": None}):
            self.assertEqual(self.ready(cfg), [])

    def test_configured_planner_key_must_be_set(self):
        cfg = config.Config(engine="local", local_planner_api_key_env="OPENAI_API_KEY")
        self.assertIn("OPENAI_API_KEY is not set (the local planner uses it)", self.ready(cfg))
        self.assertEqual(self.ready(cfg, {"OPENAI_API_KEY": "k"}), [])

    def test_monitor_source_is_not_a_microphone(self):
        cfg = config.Config(engine="local", local_planner_api_key_env="")
        problems = self.ready(cfg, source="alsa_output.speaker.monitor")
        self.assertTrue(any("not a microphone" in p for p in problems))


class SetupCheckTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.voice = Path(tmp.name) / "en_US-lessac-medium.onnx"
        self.voice.write_bytes(b"onnx")
        Path(f"{self.voice}.json").write_text("{}")
        self.model = Path(tmp.name) / "ggml-base.en.bin"
        self.model.write_bytes(b"ggml")

    def checks(self, **overrides):
        cfg = config.Config(engine="local", local_piper_model=str(self.voice))
        probes = dict(probe_stt=lambda url: True, service_model=lambda: str(self.model),
                      which=lambda name: f"/usr/bin/{name}", synthesize=lambda model: 4410)
        probes.update(overrides)
        return local.setup_checks(cfg, **probes)

    def test_everything_present(self):
        checks = self.checks()
        self.assertTrue(all(ok for ok, _ in checks), checks)
        text = " ".join(line for _, line in checks)
        self.assertIn("whisper.cpp server answering", text)
        self.assertIn("ggml-base.en.bin", text)
        self.assertIn("en_US-lessac-medium", text)

    def test_configured_model_missing_on_disk_is_reported(self):
        self.model.unlink()
        failed = [line for ok, line in self.checks() if not ok]
        self.assertEqual(failed, [f"recognition model {self.model}"])

    def test_silent_server_says_how_to_start_it(self):
        failed = [line for ok, line in self.checks(probe_stt=lambda url: False) if not ok]
        self.assertEqual(len(failed), 1)
        self.assertIn("systemctl --user start whisper-server", failed[0])

    def test_missing_piper_points_at_the_right_package(self):
        failed = [line for ok, line in self.checks(
            which=lambda name: None if name == "piper" else "/usr/bin/x") if not ok]
        self.assertTrue(any("uv tool install piper-tts" in line for line in failed), failed)

    def test_missing_voice_config_is_reported(self):
        Path(f"{self.voice}.json").unlink()
        failed = [line for ok, line in self.checks() if not ok]
        self.assertTrue(any(".onnx.json" in line for line in failed), failed)

    def test_voice_that_cannot_speak_is_reported(self):
        failed = [line for ok, line in self.checks(synthesize=lambda model: 0) if not ok]
        self.assertTrue(any("could not synthesize" in line for line in failed), failed)

    def test_unset_voice_is_reported_without_synthesizing(self):
        calls = []
        cfg = config.Config(engine="local", local_piper_model="")
        checks = local.setup_checks(cfg, probe_stt=lambda url: True, service_model=lambda: "",
                                    which=lambda name: "/usr/bin/x",
                                    synthesize=lambda model: calls.append(model) or 1)
        self.assertEqual(calls, [])
        self.assertTrue(any(not ok and "piper_model" in line for ok, line in checks))


class DoctorPrivacyTests(unittest.TestCase):
    def test_local_doctor_does_not_claim_audio_streams_to_openai(self):
        args = cli.build_parser().parse_args(["doctor"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
             mock.patch.object(local, "setup_checks", return_value=[(True, "fine")]), \
             mock.patch.object(local, "ready_problems", return_value=[]):
            cli.cmd_doctor(args, config.Config(engine="local"))
        text = out.getvalue()
        self.assertNotIn("streams continuously to OpenAI", text)
        self.assertIn("audio stays on this machine", text)


class ServiceUnitTests(unittest.TestCase):
    def test_whisper_server_unit_binds_loopback_only(self):
        unit = (ROOT / "share/whisper-server.service").read_text()
        exec_line = next(l for l in unit.splitlines() if l.startswith("ExecStart="))
        self.assertIn("/usr/bin/whisper-server", exec_line)
        self.assertIn("--host 127.0.0.1", exec_line)
        self.assertIn("--port 9000", exec_line)
        self.assertIn("WHISPER_MODEL", unit)
        self.assertIn("NoNewPrivileges=true", unit)


if __name__ == "__main__":
    unittest.main()
