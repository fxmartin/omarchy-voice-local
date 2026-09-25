"""Local engine selection: config problems and the run dispatch."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
