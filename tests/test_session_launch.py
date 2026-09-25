"""Apps the assistant opens run in the desktop session, not in its sandbox.

The daemon is a systemd service with PrivateTmp and a read-only filesystem.
Anything it started directly, or through uwsm-app (which only creates a scope,
and a scope keeps the caller's namespace), inherited both. A Chromium started
that way could not see the running browser's singleton socket in /tmp, opened a
second instance on the same profile, and crashed on the shared GPU cache.

Run with: python3 -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice.config import Config
from omarchy_voice.tools import USER_MANAGER, Executor, Result


class SessionLaunchTests(unittest.TestCase):
    def setUp(self):
        self.executor = Executor(Config(dry_run=False))
        shell = mock.patch.object(Executor, "_shell", return_value=Result(True, "started"))
        self.shell = shell.start()
        self.addCleanup(shell.stop)

    def command(self):
        return self.shell.call_args[0][0]

    def assertInSession(self, tail):
        self.assertEqual(self.command(), [*USER_MANAGER, *tail])

    def test_user_manager_keeps_the_launchers_output_and_exit_status(self):
        # Launch failures (unknown route, missing binary) are how the model
        # learns not to repeat itself, so the launcher must stay attached.
        self.assertEqual(USER_MANAGER[:2], ["systemd-run", "--user"])
        self.assertIn("--pipe", USER_MANAGER)
        self.assertIn("--wait", USER_MANAGER)
        self.assertEqual(USER_MANAGER[-1], "--")

    def test_desktop_app(self):
        with mock.patch("omarchy_voice.tools._desktop_entry_exists", return_value=True), \
             mock.patch("omarchy_voice.tools.shutil.which", return_value="/usr/bin/uwsm-app"):
            self.executor._tool_launch_app("sdlc")
        self.assertInSession(["/usr/bin/uwsm-app", "sdlc.desktop"])

    def test_desktop_app_action(self):
        with mock.patch("omarchy_voice.tools._desktop_entry_exists", return_value=True), \
             mock.patch("omarchy_voice.tools.desktop_actions", return_value=["new-window"]), \
             mock.patch("omarchy_voice.tools.shutil.which", return_value="/usr/bin/uwsm-app"):
            self.executor._tool_launch_app("chromium:new-window")
        self.assertInSession(["/usr/bin/uwsm-app", "chromium.desktop:new-window"])

    def test_url(self):
        with mock.patch.object(self.executor, "_validate_launch_app", return_value=None):
            self.executor._tool_launch_app("", url="https://example.com/")
        self.assertInSession(["xdg-open", "https://example.com/"])

    def test_omarchy_launch(self):
        self.executor._tool_omarchy_cli("launch webapp https://example.com/")
        self.assertInSession(["omarchy", "launch", "webapp", "https://example.com/"])

    def test_compose_panes(self):
        with mock.patch.object(self.executor, "_query_json", return_value=[]), \
             mock.patch.object(self.executor, "_dispatch_lua", return_value=Result(True, "ok")), \
             mock.patch.object(self.executor, "_await_new_window", return_value=None):
            self.executor.call("compose_windows", {
                "panes": [{"kind": "web", "target": "https://a.test", "name": "A"},
                          {"kind": "terminal", "target": "", "name": "shell"}],
                "workspace": "current"})
        launched = [c[0][0] for c in self.shell.call_args_list]
        self.assertIn([*USER_MANAGER, "omarchy", "launch", "webapp", "https://a.test"], launched)
        self.assertIn([*USER_MANAGER, "omarchy", "launch", "terminal"], launched)

    def test_other_omarchy_commands_stay_direct(self):
        # Queries and settings return output the model reads; they start nothing.
        self.executor._tool_omarchy_cli("theme list")
        self.assertEqual(self.command()[:2], ["omarchy", "theme"])


if __name__ == "__main__":
    unittest.main()
