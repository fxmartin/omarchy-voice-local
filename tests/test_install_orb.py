"""install.sh / uninstall.sh must ship the orb overlay without resetting the shell."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_PYTHON = shutil.which("python3")


# install.sh refuses to run as root by design, so these end-to-end runs cannot
# execute in root CI containers; they still run for every non-root developer.
@unittest.skipIf(os.geteuid() == 0, "install.sh refuses root; run as a desktop user")
class InstallOrbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        (self.home / ".config/omarchy").mkdir(parents=True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.log = self.tmp / "calls.log"
        self.fake("hyprctl", "exit 0")
        self.fake("omarchy", 'echo "omarchy $*" >> "$CALLS"')
        self.fake("omarchy-shell", 'echo "omarchy-shell $*" >> "$CALLS"')
        self.fake("omarchy-restart-shell", 'echo "restart-shell" >> "$CALLS"')
        self.fake("omarchy-refresh-shell", 'echo "refresh-shell" >> "$CALLS"; exit 1')
        self.fake("python3", f'[ "$1 $2" = "-c import websockets" ] && exit 0; exec {REAL_PYTHON} "$@"')
        self.env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home),
                    "CALLS": str(self.log)}

    def fake(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)

    def run_script(self, script, answers=""):
        return subprocess.run(["bash", str(ROOT / script)], input=answers, text=True,
                              capture_output=True, env=self.env, timeout=60)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def install(self):
        # plugins: y, omarchy voice commands: n, systemd unit: n
        result = self.run_script("install.sh", "y\nn\nn\n")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_install_copies_orb_beside_widget_and_enables_it(self):
        self.install()
        plugins = self.home / ".config/omarchy/plugins"
        self.assertTrue((plugins / "voice.indicator/manifest.json").is_file())
        self.assertTrue((plugins / "voice.orb/manifest.json").is_file())
        self.assertIn("omarchy plugin enable voice.orb", self.calls())

    def test_install_rescans_plugins_before_enabling_without_resetting_the_shell(self):
        # A freshly copied plugin is "not known" to the running shell until it
        # rescans, so enabling first always failed on a real desktop.
        self.install()
        calls = self.calls()
        rescan = calls.index("omarchy-shell shell rescanPlugins")
        self.assertLess(rescan, calls.index("omarchy plugin enable voice.orb"))
        self.assertNotIn("restart-shell", calls)
        self.assertNotIn("refresh-shell", calls)

    def test_failed_enable_tells_user_the_commands(self):
        self.fake("omarchy", 'echo "omarchy $*" >> "$CALLS"; case "$*" in *enable*) exit 1;; esac')
        result = self.run_script("install.sh", "y\nn\nn\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout + result.stderr
        self.assertIn("omarchy-shell shell rescanPlugins", output)
        self.assertIn("omarchy plugin enable voice.orb", output)
        self.assertNotIn("refresh-shell", self.calls())

    def test_failed_rescan_does_not_abort_the_install(self):
        self.fake("omarchy-shell", "exit 1")
        result = self.run_script("install.sh", "y\nn\nn\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("omarchy plugin enable voice.orb", self.calls())

    def test_uninstall_without_omarchy_cli_still_removes_orb(self):
        self.install()
        (self.bin / "omarchy").unlink()
        result = self.run_script("uninstall.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.home / ".config/omarchy/plugins/voice.orb").exists())

    def test_declining_desktop_integration_leaves_orb_out(self):
        result = self.run_script("install.sh", "n\nn\nn\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.home / ".config/omarchy/plugins/voice.orb").exists())
        self.assertNotIn("omarchy plugin enable voice.orb", self.calls())

    def test_uninstall_disables_and_removes_orb(self):
        self.install()
        self.log.write_text("")
        result = self.run_script("uninstall.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.home / ".config/omarchy/plugins/voice.orb").exists())
        self.assertFalse((self.home / ".config/omarchy/plugins/voice.indicator").exists())
        self.assertIn("omarchy plugin disable voice.orb", self.calls())
        self.assertNotIn("refresh-shell", self.calls())


if __name__ == "__main__":
    unittest.main()
