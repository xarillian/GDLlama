"""Distribution checks for the staged Godot addon's license bundle."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools import stage_godot as stage


class StageGodotTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.plugin = self.root / "plugin"
        self.plugin.mkdir()
        for name in ("chorus.gdextension", "plugin.cfg", "plugin.gd", "icon.png"):
            shutil.copy2(stage.PLUGIN_SOURCE / name, self.plugin / name)
        (self.plugin / "doc_classes").mkdir()
        (self.plugin / "doc_classes" / "Example.xml").write_text("<class name='Example' />")
        self.notices = {
            name: (stage.ROOT / name).read_bytes()
            for name in (
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                *(str(path.relative_to(stage.ROOT)) for path in (stage.ROOT / "licenses").glob("*.txt")),
            )
        }
        for name, content in self.notices.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (self.root / "bin").mkdir()
        for suffix in ("so", "dll", "dylib"):
            (self.root / "bin" / f"libgodot_chorus.{suffix}").write_bytes(b"library fixture")
        self.destination = self.plugin / "addons" / "chorus"
        self.test_destination = self.root / "tests" / "godot" / "addons" / "chorus"
        for name, value in (
            ("ROOT", self.root),
            ("PLUGIN_SOURCE", self.plugin),
            ("PLUGIN_DESTINATION", self.destination),
            ("TEST_DESTINATION", self.test_destination),
        ):
            override = patch.object(stage, name, value)
            override.start()
            self.addCleanup(override.stop)

    def test_staged_addons_carry_the_complete_notice_bundle(self):
        platforms = {stage.sys.platform: stage.library_name(), "win32": "libgodot_chorus.dll"}
        for platform, library in platforms.items():
            with self.subTest(platform=platform), patch.object(stage.sys, "platform", platform):
                stage.stage_plugin()
                for destination in (self.destination, self.test_destination):
                    self.assertEqual(b"library fixture", (destination / "bin" / library).read_bytes())
                    for name, content in self.notices.items():
                        self.assertEqual(content, (destination / name).read_bytes(), name)
                self.assertEqual(platform != "win32", self.test_destination.is_symlink())

    def test_missing_notice_leaves_the_existing_addons_untouched(self):
        for destination in (self.destination, self.test_destination):
            destination.mkdir(parents=True)
            (destination / "existing").write_text("previous distribution")
        for name, content in self.notices.items():
            with self.subTest(notice=name):
                source = self.root / name
                source.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Required license notice not found"):
                    stage.stage_plugin()
                for destination in (self.destination, self.test_destination):
                    self.assertEqual(["existing"], [path.name for path in destination.iterdir()])
                    self.assertEqual("previous distribution", (destination / "existing").read_text())
                source.write_bytes(content)

    def test_restaging_refreshes_notices_and_removes_obsolete_files(self):
        stage.stage_plugin()
        obsolete = self.destination / "licenses" / "obsolete-commercial-terms.txt"
        obsolete.write_text("obsolete fixture")
        updated = self.notices["LICENSE"].replace(b"2025-2026", b"2025-2027")
        (self.root / "LICENSE").write_bytes(updated)
        stage.stage_plugin()
        self.assertFalse(obsolete.exists())
        for destination in (self.destination, self.test_destination):
            self.assertEqual(updated, (destination / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
