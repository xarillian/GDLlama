"""Download pinned test models, or verify local fixtures without network access."""

import argparse
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/model-fixtures.json"
DESTINATION = ROOT / "tests/models"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all fixtures without downloading")
    args = parser.parse_args()
    try:
        models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
        for model in models:
            path = DESTINATION / model["filename"]
            if args.check or path.exists():
                verify_fixture(path, model)
            else:
                download_fixture(path, model)
            print(f"Verified {model['filename']}", flush=True)
    except (OSError, ValueError, HTTPException) as error:
        print(f"Model fixtures: {error}", file=sys.stderr)
        return 1
    return 0


def verify_fixture(path, model):
    if not path.is_file():
        raise ValueError(f"Missing {model['filename']}; run 'just download-fixtures'.")
    if path.stat().st_size != model["size_bytes"]:
        raise ValueError(f"Size mismatch for {model['filename']}; expected {model['size_bytes']} bytes.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != model["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {model['filename']}.")


def download_fixture(path, model):
    print(f"Downloading {model['filename']} ({model['size_bytes']} bytes)", flush=True)
    print(f"Terms: {model['license_url']}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(model["url"], timeout=60) as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        verify_fixture(temporary, model)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
