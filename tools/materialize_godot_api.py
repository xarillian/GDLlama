"""Use the official Godot 4.4 API while retaining the pinned godot-cpp fixes."""

import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GODOT_CPP = ROOT / "third-party/godot-cpp"
SOURCE = "gdextension/extension_api.json"
PATCH = ROOT / "patches/godot-4.4-api.patch"
OUTPUT = ROOT / "bin/gen/godot-api/extension_api.json"
API_SHA256 = "8a8386e3597083cf4357b3dbf501ede3d38a4e3f7ff75da86dfef0d1d9c3e3a8"


def materialize():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent) as directory:
        temporary = Path(directory)
        api = temporary / "extension_api.json"
        # Read from git's object store and apply without conversion: a Windows checkout with
        # core.autocrlf rewrites line endings, and the API must hash byte for byte.
        api.write_bytes(subprocess.run(
            ["git", "show", f"HEAD:{SOURCE}"], cwd=GODOT_CPP, check=True, capture_output=True
        ).stdout)
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply",
             f"--directory={temporary.relative_to(ROOT).as_posix()}", str(PATCH)],
            cwd=ROOT, check=True,
        )
        content = api.read_bytes()
        if hashlib.sha256(content).hexdigest() != API_SHA256:
            raise RuntimeError("Generated API does not match the official Godot 4.4 API")
        if not OUTPUT.exists() or OUTPUT.read_bytes() != content:
            api.replace(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(materialize())
