"""Regression checks for the tracked llama.cpp patch build boundary."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import materialize_llama as llama


class MaterializeLlamaTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=llama.ROOT / "bin")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.vendor = self.root / "vendor"
        self.vendor.mkdir()
        subprocess.run(["git", "init", "-q", str(self.vendor)], check=True)
        for name in ("src/llama-model-loader.cpp", "src/llama-batch.cpp", "ggml/CMakeLists.txt"):
            path = self.vendor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=llama.VENDOR))
        self.commit()
        self.addCleanup(patch.stopall)
        patch.object(llama, "VENDOR", self.vendor).start()
        patch.object(llama, "OUTPUT", self.root / "output").start()

    def commit(self):
        subprocess.run(["git", "add", "."], cwd=self.vendor, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "fixture"], cwd=self.vendor, check=True)

    def test_pristine_repeat_helper_change_and_tampered_cache(self):
        tree, revision, identity = llama.materialize()
        self.assertIn("struct upload_resources", (tree / "src/llama-model-loader.cpp").read_text())
        self.assertIn("std::make_unique<llama_batch_ext>", (tree / "src/llama-batch.cpp").read_text())
        self.assertEqual((tree, revision, identity), llama.materialize())
        helper = self.root / "changed-helper.py"
        helper.write_bytes(Path(llama.__file__).read_bytes() + b"\n")
        with patch.object(llama, "__file__", str(helper)):
            changed, _, changed_identity = llama.materialize()
        self.assertNotEqual(identity, changed_identity)
        self.assertNotEqual(tree, changed)
        altered_patch = self.root / "changed.patch"
        altered_patch.write_bytes(llama.PATCH.read_bytes() + b"\n")
        with patch.object(llama, "PATCH", altered_patch):
            patch_changed, _, patch_identity = llama.materialize()
        self.assertNotEqual(identity, patch_identity)
        self.assertNotEqual(tree, patch_changed)
        (tree / "src/llama-model-loader.cpp").write_text("damaged")
        with self.assertRaisesRegex(RuntimeError, "incompatible generated"):
            llama.materialize()

    def test_real_crlf_checkout_preserves_patch_application(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=checkout, check=True)
        attrs = checkout / ".gitattributes"
        attrs.write_text("* text=auto\n")
        checked_patch = checkout / "patches" / llama.PATCH.name
        checked_patch.parent.mkdir()
        checked_patch.write_bytes(llama.PATCH.read_bytes())
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "CRLF baseline"], cwd=checkout, check=True)
        checked_patch.unlink()
        subprocess.run(["git", "checkout", "HEAD", "--", str(checked_patch.relative_to(checkout))], cwd=checkout, check=True)
        self.assertIn(b"\r\n", checked_patch.read_bytes())
        with patch.object(llama, "PATCH", checked_patch):
            with self.assertRaises(subprocess.CalledProcessError):
                llama.materialize()

        attrs.write_bytes((llama.ROOT / ".gitattributes").read_bytes())
        subprocess.run(["git", "add", ".gitattributes"], cwd=checkout, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "Patch EOL policy"], cwd=checkout, check=True)
        attrs.unlink()
        subprocess.run(["git", "checkout", "HEAD", "--", ".gitattributes"], cwd=checkout, check=True)
        checked_patch.unlink()
        subprocess.run(["git", "checkout", "HEAD", "--", str(checked_patch.relative_to(checkout))], cwd=checkout, check=True)
        self.assertNotIn(b"\r\n", checked_patch.read_bytes())
        with patch.object(llama, "PATCH", checked_patch):
            tree, _, _ = llama.materialize()
        self.assertIn("std::make_unique<llama_batch_ext>", (tree / "src/llama-batch.cpp").read_text())

    def test_conflicting_and_partially_applied_input_fail(self):
        loader = self.vendor / "src/llama-model-loader.cpp"
        source = loader.read_text()
        loader.write_text(source.replace("std::vector<ggml_backend_buffer_t> host_buffers;",
                                         "std::vector<ggml_backend_buffer_t> damaged;", 1))
        self.commit()
        with self.assertRaises(subprocess.CalledProcessError):
            llama.materialize()
        loader.write_text(source)
        self.commit()
        tree, _, _ = llama.materialize()
        patched = (tree / "src/llama-model-loader.cpp").read_text()
        boundary = "    size_t buffer_idx"
        loader.write_text(patched.split(boundary, 1)[0] + boundary + source.split(boundary, 1)[1])
        self.commit()
        with self.assertRaises(subprocess.CalledProcessError):
            llama.materialize()
        loader.write_text(source)
        with self.assertRaisesRegex(RuntimeError, "tracked edits"):
            llama.materialize()


if __name__ == "__main__":
    unittest.main()
