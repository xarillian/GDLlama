"""Build a verified, patched llama.cpp source tree without modifying its submodule."""

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "third-party/llama.cpp"
PATCH = ROOT / "patches/llama-resource-cleanup.patch"
OUTPUT = ROOT / "bin/vendor/llama.cpp"
MARKER = ".chorus-source.json"


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(tree):
    return {
        str(path.relative_to(tree)): digest(path)
        for path in sorted(tree.rglob("*"))
        if path.is_file() and path.name != MARKER
    }


def verify_source_tree(tree, revision, identity):
    marker = tree / MARKER
    if not marker.is_file() or json.loads(marker.read_text()) != {
        "revision": revision, "identity": identity, "files": inventory(tree)
    }:
        raise RuntimeError(f"incompatible generated llama.cpp tree: {tree}; remove this generated directory")


def materialize():
    revision = run("git", "rev-parse", "HEAD", cwd=VENDOR)
    if run("git", "status", "--porcelain", "--untracked-files=no", cwd=VENDOR):
        raise RuntimeError("llama.cpp submodule has tracked edits; restore them before building")
    paths = run("git", "apply", "--numstat", str(PATCH), cwd=VENDOR).splitlines()
    if {line.split("\t")[-1] for line in paths} != {
        "src/llama-model-loader.cpp", "src/llama-batch.cpp", "ggml/CMakeLists.txt", "common/jinja/runtime.cpp",
        "common/chat.cpp", "common/jinja/value.cpp"
    } or len(paths) != 6:
        raise RuntimeError(
            "llama patch must touch only upload cleanup, batch conversion, GGML revision guard, Jinja loop scopes "
            "and reentrant local time"
        )
    # Check against HEAD even when a previously materialized tree exists.
    run("git", "apply", "--check", str(PATCH), cwd=VENDOR)
    identity = hashlib.sha256(PATCH.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:20]
    destination = OUTPUT / (revision + "-" + identity)
    if destination.exists():
        verify_source_tree(destination, revision, identity)
        return destination, revision, identity

    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".materializing-", dir=OUTPUT))
    archive = temporary.with_suffix(".tar")
    try:
        run("git", "archive", f"--output={archive}", revision, cwd=VENDOR)
        shutil.unpack_archive(str(archive), str(temporary), format="tar")
        archive.unlink()
        run("git", "init", "-q", cwd=temporary)
        run("git", "apply", "--check", str(PATCH), cwd=temporary)
        run("git", "apply", str(PATCH), cwd=temporary)
        run("git", "apply", "--reverse", "--check", str(PATCH), cwd=temporary)
        shutil.rmtree(temporary / ".git")
        manifest = {"revision": revision, "identity": identity, "files": inventory(temporary)}
        (temporary / MARKER).write_text(json.dumps(manifest, sort_keys=True))
        try:
            os.rename(temporary, destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            verify_source_tree(destination, revision, identity)
    finally:
        if archive.exists():
            archive.unlink()
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination, revision, identity


if __name__ == "__main__":
    try:
        path, revision, identity = materialize()
        print(f"{path} revision={revision} patch-and-helper={identity}")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        raise SystemExit(f"llama.cpp materialization failed: {error}")
