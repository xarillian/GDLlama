"""Regression checks for how the Godot compatibility checker judges editor logs."""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from tools import check_godot_compatibility as checker

PROGRESS_DIALOG = "ERROR: Do not use progress dialog (task) while flushing the message queue or using call_deferred()!"
SETTINGS_RACE = 'ERROR: EditorSettings not instantiated yet when getting setting "export/android/android_sdk_path".'


class EditorLogTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.engine = self.root / "godot"
        self.engine.write_text(f"#!{sys.executable}\nimport os, sys\nsys.stdout.write(open(os.environ['FAKE_LOG']).read())\n")
        self.engine.chmod(self.engine.stat().st_mode | stat.S_IEXEC)
        (self.root / "project").mkdir()

    def judge(self, log: str, version: str) -> bool:
        (self.root / "log.txt").write_text(log)
        environment = dict(os.environ, FAKE_LOG=str(self.root / "log.txt"))
        step = checker.run(str(self.engine), self.root / "project", "import", ["--editor"], environment, clean=True,
                           allowed_errors=checker.known_editor_errors(version))
        return step["passed"]

    def test_a_known_error_passes_even_when_godot_colours_it(self):
        self.assertTrue(self.judge(f"\x1b[1;31m{PROGRESS_DIALOG[:6]}\x1b[0;91m{PROGRESS_DIALOG[6:]}\x1b[0m\n",
                                   "4.4.stable.official.4c311cbee"))

    def test_an_unknown_error_fails(self):
        self.assertFalse(self.judge("ERROR: Chorus failed to register a class.\n", "4.4.stable.official.4c311cbee"))

    def test_the_settings_race_passes_on_any_dotnet_editor_but_not_a_standard_one(self):
        log = f".NET: hostfxr initialized\n{SETTINGS_RACE}\n"
        self.assertTrue(self.judge(log, "4.6.3.stable.mono.official.7d41c59c4"))
        self.assertFalse(self.judge(log, "4.6.3.stable.official.7d41c59c4"))


if __name__ == "__main__":
    unittest.main()
