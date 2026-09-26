"""Regression checks for the tracked llama.cpp patch build boundary."""

from concurrent.futures import ProcessPoolExecutor
import errno
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import materialize_llama as llama


def synchronize_publication(vendor, output, barrier):
    llama.VENDOR = vendor
    llama.OUTPUT = output
    rename = llama.os.rename

    def publish(source, destination):
        barrier.wait(timeout=10)
        return rename(source, destination)

    llama.os.rename = publish


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

    def test_concurrent_first_materializations_reuse_one_verified_tree(self):
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=context,
                                 initializer=synchronize_publication,
                                 initargs=(self.vendor, llama.OUTPUT, context.Barrier(2))) as builders:
            attempts = [builders.submit(llama.materialize) for _ in range(2)]
            first, second = [attempt.result(timeout=30) for attempt in attempts]
        self.assertEqual(first, second)
        self.assertEqual(first, llama.materialize())
        self.assertEqual([first[0]], list(llama.OUTPUT.iterdir()))

    def test_publication_collision_rejects_incomplete_or_changed_tree(self):
        rename = llama.os.rename
        for damage in ("missing-marker", "changed-source"):
            with self.subTest(damage=damage), patch.object(llama, "OUTPUT", self.root / damage):
                def publish(source, destination):
                    shutil.copytree(source, destination)
                    if damage == "missing-marker":
                        (destination / llama.MARKER).unlink()
                    else:
                        (destination / "src/llama-model-loader.cpp").write_text("damaged")
                    return rename(source, destination)

                with patch.object(llama.os, "rename", side_effect=publish):
                    with self.assertRaisesRegex(RuntimeError, "incompatible generated"):
                        llama.materialize()
                remaining = list(llama.OUTPUT.iterdir())
                self.assertEqual(1, len(remaining))
                if damage == "missing-marker":
                    self.assertFalse((remaining[0] / llama.MARKER).exists())
                else:
                    self.assertEqual("damaged", (remaining[0] / "src/llama-model-loader.cpp").read_text())

    def test_unrelated_publication_error_propagates_and_removes_staging_tree(self):
        failure = PermissionError(errno.EACCES, "publication denied")
        with patch.object(llama.os, "rename", side_effect=failure):
            with self.assertRaises(PermissionError) as raised:
                llama.materialize()
        self.assertIs(failure, raised.exception)
        self.assertEqual([], list(llama.OUTPUT.iterdir()))

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
