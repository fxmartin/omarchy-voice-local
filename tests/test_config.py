"""Config loading: unknown keys, additive policy lists."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import config as cfg
from omarchy_voice.config import DEFAULT_CONFIRM, DEFAULT_DENY


class ConfigLoadTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "config.toml"
        path.write_text(text)
        return path

    def test_unknown_keys_are_kept_for_doctor(self):
        path = self.write('[ears]\nenginee = "realtime"\n')
        loaded = cfg.load(path)
        self.assertIn("enginee", loaded.unknown_keys)

    def test_a_retired_key_is_not_reported_as_a_typo(self):
        # Listening is toggle-only now. Every config written before that says
        # `mode = "push"`, and none of them should make doctor shout about it.
        path = self.write('[ears]\nmode = "push"\n')
        loaded = cfg.load(path)
        self.assertEqual(loaded.unknown_keys, [])
        self.assertIn("mode", loaded.retired_keys)

    def test_parallel_limit_and_personal_news_sources_load(self):
        path = self.write('[live]\nmax_parallel_tools = 4\n[navigation]\nnews_sources = ["https://apnews.com/"]\n')
        loaded = cfg.load(path)
        self.assertEqual(loaded.live_max_parallel_tools, 4)
        self.assertEqual(loaded.news_sources, ["https://apnews.com/"])
        self.assertEqual(loaded.unknown_keys, [])

    def test_a_retired_key_still_gets_explained(self):
        self.assertIn("toggle", cfg.RETIRED_KEYS["mode"])

    def test_there_is_no_always_on_setting(self):
        path = self.write('[ears]\nmode = "always"\n')
        loaded = cfg.load(path)
        self.assertFalse(hasattr(loaded, "mode"))

    def test_confirm_patterns_union_with_defaults(self):
        path = self.write('[hands]\nconfirm_patterns = ["\\\\bformat\\\\b"]\n')
        loaded = cfg.load(path)
        self.assertIn(r"\bformat\b", loaded.confirm_patterns)
        for builtin in DEFAULT_CONFIRM:
            self.assertIn(builtin, loaded.confirm_patterns)

    def test_confirm_patterns_replace_drops_defaults(self):
        path = self.write(
            '[hands]\n'
            'confirm_patterns = ["\\\\bformat\\\\b"]\n'
            'confirm_patterns_replace = true\n'
        )
        loaded = cfg.load(path)
        self.assertEqual(loaded.confirm_patterns, [r"\bformat\b"])
        self.assertNotIn(r"\breboot\b", loaded.confirm_patterns)

    def test_deny_patterns_union_with_defaults(self):
        path = self.write('[hands]\ndeny_patterns = ["\\\\bwipe\\\\b"]\n')
        loaded = cfg.load(path)
        self.assertIn(r"\bwipe\b", loaded.deny_patterns)
        self.assertIn(DEFAULT_DENY[0], loaded.deny_patterns)


if __name__ == "__main__":
    unittest.main()


class LocalEngineConfigTests(unittest.TestCase):
    def load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(text)
            return cfg.load(path)

    def test_local_keys_are_prefixed_and_have_defaults(self):
        default = cfg.Config()
        self.assertEqual(default.engine, "realtime")
        loaded = self.load('[local]\nstt_url = "http://x:1"\nplanner_model = "m"\n')
        self.assertEqual(loaded.local_stt_url, "http://x:1")
        self.assertEqual(loaded.local_planner_model, "m")
        self.assertEqual(loaded.planner_model, default.planner_model)
        self.assertEqual(loaded.unknown_keys, [])

    def test_local_planner_defaults_to_openai_chat(self):
        # Speech stays local; the brain starts on OpenAI chat with the existing
        # key and moves to a local server by changing these keys only.
        default = cfg.Config()
        self.assertEqual(default.local_planner_base_url, "https://api.openai.com/v1")
        self.assertEqual(default.local_planner_model, default.planner_model)
        self.assertEqual(default.local_planner_api_key_env, default.api_key_env)

    def test_engine_local_is_selectable_and_flag_overrides(self):
        self.assertEqual(self.load('[openai]\nengine = "local"\n').engine, "local")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[openai]\nengine = "live"\n')
            self.assertEqual(cfg.load(path, engine="local").engine, "local")

    def test_misspelled_local_key_is_reported(self):
        loaded = self.load('[local]\nstt_ulr = "x"\n')
        self.assertEqual(loaded.unknown_keys, ["local_stt_ulr"])

    def test_cli_accepts_engine_local(self):
        from omarchy_voice import cli
        self.assertEqual(
            cli.build_parser().parse_args(["run", "--engine", "local"]).engine, "local")

    def test_example_config_keys_are_known(self):
        text = (Path(__file__).resolve().parent.parent / "share/config.example.toml").read_text()
        uncommented = text.replace("# stt_url", "stt_url").replace("# [local]", "[local]")
        for key in ("language", "piper_model", "planner_base_url", "planner_model",
                    "planner_api_key_env", "endpoint_silence_ms", "endpoint_min_speech_ms",
                    "endpoint_max_speech_ms", "endpoint_preroll_ms"):
            uncommented = uncommented.replace(f"# {key} =", f"{key} =")
        self.assertEqual(self.load(uncommented).unknown_keys, [])
