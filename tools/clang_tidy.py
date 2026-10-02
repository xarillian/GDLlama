"""Run the pinned clang-tidy over every first-party translation unit; fail on any diagnostic."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATABASE = ROOT / "compile_commands.json"
TIDY_DATABASE = ROOT / "bin/clang-tidy/compile_commands.json"
FIRST_PARTY = ("src/", "tests/")
# GCC accepts these flags; clang rejects them before any check runs.
GCC_ONLY_FLAGS = {"-fno-gnu-unique"}


def relative(entry):
    path = Path(entry["file"])
    if not path.is_absolute():
        path = Path(entry["directory"]) / path
    return path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else None


def first_party_database():
    entries, seen = [], set()
    for entry in json.loads(SOURCE_DATABASE.read_text()):
        name = relative(entry)
        if name is None or not name.startswith(FIRST_PARTY) or name in seen:
            continue
        seen.add(name)
        if "arguments" in entry:
            entry["arguments"] = [a for a in entry["arguments"] if a not in GCC_ONLY_FLAGS]
        else:
            entry["command"] = " ".join(a for a in entry["command"].split(" ") if a not in GCC_ONLY_FLAGS)
        entries.append(entry)
    TIDY_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    TIDY_DATABASE.write_text(json.dumps(entries, indent=1))
    return sorted(seen)


def lint(name):
    result = subprocess.run([str(ROOT / "tools/clang-tidy"), "-p", str(TIDY_DATABASE.parent), "--quiet", name],
                            cwd=ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main():
    if not SOURCE_DATABASE.is_file():
        raise SystemExit("compile_commands.json is missing; run `just compiledb` first")
    files = sys.argv[1:] or first_party_database()
    if sys.argv[1:]:
        first_party_database()
    failed = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for name, (code, output) in zip(files, pool.map(lint, files)):
            if code != 0:
                failed += 1
                print(output.strip(), flush=True)
    print(f"clang-tidy: {len(files) - failed}/{len(files)} files clean")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
