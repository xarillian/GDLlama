"""Launch Godot editors to check imports, settings persistence and optional runtime bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "plugin/addons/chorus"

# Stock 4.4.0 emits these even when importing an empty project without Chorus.
GODOT_44_HEADLESS_ERRORS = {
    "ERROR: Do not use progress dialog (task) while flushing the message queue or using call_deferred()!",
    'ERROR: Condition "!tasks.has(p_task)" is true. Returning: canceled',
    'ERROR: Condition "!tasks.has(p_task)" is true.',
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engines", nargs="*", help="Godot editor executables or commands")
    parser.add_argument("--versions", nargs="+", help="Download standard and .NET editors for these stable versions")
    parser.add_argument("--output", type=Path, required=True, help="New directory for disposable projects and logs")
    parser.add_argument("--repetitions", type=int, default=3, help="Fresh projects per plugin state (default: 3)")
    parser.add_argument("--editor-only", action="store_true", help="Skip the model-dependent runtime suite")
    args = parser.parse_args()
    if bool(args.engines) == bool(args.versions):
        parser.error("Supply editor executables or --versions, not both")
    if args.versions and any(not re.fullmatch(r"4\.\d+(?:\.\d+)?", version) for version in args.versions):
        parser.error("Versions must be stable Godot 4.x release numbers, such as 4.4 or 4.7.2")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not (ADDON / "chorus.gdextension").is_file():
        parser.error("Build and stage the addon with 'just godot' first")
    engines = []
    for name in args.engines:
        executable = shutil.which(name)
        if executable is None:
            parser.error(f"Godot executable not found: {name}")
        engines.append(str(Path(executable).resolve()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.versions:
        engines = download_editors(args.versions)
    results = []
    for index, engine in enumerate(engines):
        directory = output / str(index)
        directory.mkdir()
        result = check_engine(engine, directory, args.repetitions, editor_only=args.editor_only)
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{result['version']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    return 0 if all(result["passed"] for result in results) else 1


def download_editors(versions: list[str]):
    directory = ROOT / "bin/godot-engines"
    directory.mkdir(parents=True, exist_ok=True)
    for version in versions:
        release = f"https://github.com/godotengine/godot-builds/releases/download/{version}-stable"
        with urllib.request.urlopen(release + "/SHA512-SUMS.txt", timeout=90) as response:
            checksums = {line.split()[-1].lstrip("*"): line.split()[0] for line in response.read().decode().splitlines() if line.strip()}
        for dotnet in (False, True):
            flavor = "dotnet" if dotnet else "standard"
            if sys.platform == "win32":
                suffix = "mono_win64" if dotnet else "win64.exe"
            elif sys.platform == "darwin":
                suffix = ("mono_" if dotnet else "") + "macos.universal"
            else:
                suffix = "mono_linux_x86_64" if dotnet else "linux.x86_64"
            archive = directory / f"Godot_v{version}-stable_{suffix}.zip"
            if not archive.exists():
                temporary = archive.with_suffix(".zip.partial")
                with urllib.request.urlopen(f"{release}/{archive.name}", timeout=90) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(archive)
            with archive.open("rb") as source:
                if hashlib.file_digest(source, "sha512").hexdigest() != checksums[archive.name]:
                    raise RuntimeError(f"Checksum mismatch: {archive.name}")
            destination = directory / f"{version}-{flavor}"
            destination.mkdir(exist_ok=True)
            if sys.platform == "darwin":
                subprocess.run(["ditto", "-xk", str(archive), str(destination)], check=True)
                engine, = destination.glob("**/Contents/MacOS/Godot")
            else:
                with zipfile.ZipFile(archive) as package:
                    package.extractall(destination)
                pattern = "*_console.exe" if sys.platform == "win32" else "*.x86_64"
                engine, = destination.rglob(pattern)
            engine.chmod(0o755)
            yield str(engine)


def check_engine(engine: str, directory: Path, repetitions: int, *, editor_only: bool = False) -> dict:
    environment = dict(os.environ)
    for variable, name in (("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data")):
        environment[variable] = str(directory / name)
    version = subprocess.check_output([engine, "--version"], text=True, timeout=15).strip()
    steps = []
    editor_errors = GODOT_44_HEADLESS_ERRORS if version.startswith("4.4.stable.") else set()
    warmup = directory / "cache-warmup"
    make_project(warmup, addon=False)
    steps.append(run(engine, warmup, "import", ["--editor", "--import"], environment, clean=True, allowed_errors=editor_errors))
    for enabled in (False, True):
        for repetition in range(repetitions):
            project = directory / f"import-plugin-{int(enabled)}-{repetition}"
            make_project(project, plugin=enabled)
            step = run(engine, project, "import", ["--editor", "--import"], environment, clean=True, allowed_errors=editor_errors)
            settings = project / "chorus/settings.json"
            step["passed"] = step["passed"] and settings.is_file() and json.loads(settings.read_text()) == {"version": 1, "generation": {}}
            steps.append(step)
    steps.extend(check_settings(engine, directory, environment, editor_errors))
    if editor_only:
        return {"engine": engine, "version": version, "passed": all(step["passed"] for step in steps), "steps": steps}
    suite = directory / "suite"
    shutil.copytree(ROOT / "tests/godot", suite, ignore=shutil.ignore_patterns(".godot", "addons", "chorus"))
    shutil.copytree(ADDON, suite / "addons/chorus")
    (directory / "models").symlink_to(ROOT / "tests/models", target_is_directory=True)
    imported = run(engine, suite, "import", ["--editor", "--import"], environment, clean=True, allowed_errors=editor_errors)
    steps.append(imported)
    if imported["passed"]:
        steps.append(run(engine, suite, "runner", ["--scene", "res://tests_gdscript/runner.tscn"], environment, marker="FINAL SUMMARY: ALL TESTS PASSED (", timeout=360))
    return {"engine": engine, "version": version, "passed": all(step["passed"] for step in steps), "steps": steps}


def check_settings(engine: str, directory: Path, environment: dict, editor_errors: set[str]) -> list[dict]:
    project = directory / "settings"
    make_project(project, plugin=True)
    imported = run(engine, project, "import", ["--editor", "--import"], environment, clean=True, allowed_errors=editor_errors)
    steps = [imported]
    if not imported["passed"]:
        return steps
    smoke = project / "addons/editor_smoke"
    smoke.mkdir()
    shutil.copy2(ROOT / "tools/godot_editor_smoke.gd", smoke / "plugin.gd")
    (smoke / "plugin.cfg").write_text('[plugin]\nname="Chorus compatibility"\ndescription=""\nauthor="Chorus"\nversion="1"\nscript="plugin.gd"\n')
    configuration = project / "project.godot"
    configuration.write_text(configuration.read_text().replace('"res://addons/chorus/plugin.cfg")', '"res://addons/chorus/plugin.cfg", "res://addons/editor_smoke/plugin.cfg")'))
    steps.append(run(engine, project, "edit", ["--editor"], environment, clean=True, marker="CHORUS EDITOR SETTINGS SAVED", allowed_errors=editor_errors))
    steps.append(run(engine, project, "reopen", ["--editor", "--", "--chorus-read-settings"], environment, clean=True, marker="CHORUS EDITOR SETTINGS RESTORED", allowed_errors=editor_errors))
    return steps


def make_project(project: Path, *, addon: bool = True, plugin: bool = False) -> None:
    project.mkdir()
    configuration = 'config_version=5\n[application]\nconfig/name="Chorus compatibility"\n'
    if plugin:
        configuration += '\n[editor_plugins]\nenabled=PackedStringArray("res://addons/chorus/plugin.cfg")\n'
    (project / "project.godot").write_text(configuration)
    if addon:
        shutil.copytree(ADDON, project / "addons/chorus")


def run(engine: str, project: Path, name: str, arguments: list[str], environment: dict, *, clean: bool = False, marker: str = "", timeout: int = 90, allowed_errors: set[str] | None = None) -> dict:
    command = [engine, "--headless", "--verbose", "--path", str(project), *arguments]
    log = project.parent / f"{project.name}-{name}.log"
    with log.open("w") as output:
        try:
            code = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    text = log.read_text(errors="replace")
    passed = code == 0 and marker in text and "SCRIPT ERROR:" not in text and "Parse Error:" not in text and "leaked" not in text
    errors = [line.strip() for line in text.splitlines() if "ERROR:" in line]
    if clean:
        passed = passed and not any(line not in (allowed_errors or set()) for line in errors)
    if ".stable.mono." in text:
        passed = passed and ".NET: hostfxr initialized" in text
    return {"command": command, "exit": code, "passed": passed, "engine_diagnostics": errors, "log": str(log)}


if __name__ == "__main__":
    raise SystemExit(main())
