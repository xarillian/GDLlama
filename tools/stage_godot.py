from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SOURCE = ROOT / "plugin"
PLUGIN_DESTINATION = PLUGIN_SOURCE / "addons" / "chorus"
TEST_DESTINATION = ROOT / "tests" / "godot" / "addons" / "chorus"
NOTICE_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/Apache-2.0.txt",
    "licenses/CC0-1.0.txt",
    "licenses/GCC-Runtime-Library-Exception-3.1.txt",
    "licenses/GPL-3.0.txt",
)


def library_name() -> str:
    if sys.platform.startswith("linux"):
        return "libgodot_chorus.so"
    if sys.platform == "win32":
        return "libgodot_chorus.dll"
    if sys.platform == "darwin":
        return "libgodot_chorus.dylib"
    raise RuntimeError(f"Unsupported Godot staging platform: {sys.platform}")


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def stage_plugin() -> None:
    library = ROOT / "bin" / library_name()
    if not library.is_file():
        raise FileNotFoundError(f"Built Chorus library not found: {library}")
    for name in NOTICE_FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Required license notice not found: {ROOT / name}")

    remove_path(PLUGIN_DESTINATION)
    (PLUGIN_DESTINATION / "bin").mkdir(parents=True)
    for name in ("chorus.gdextension", "plugin.cfg", "plugin.gd", "icon.png"):
        shutil.copy2(PLUGIN_SOURCE / name, PLUGIN_DESTINATION / name)
    shutil.copytree(PLUGIN_SOURCE / "doc_classes", PLUGIN_DESTINATION / "doc_classes")
    shutil.copy2(library, PLUGIN_DESTINATION / "bin" / library.name)
    for name in NOTICE_FILES:
        destination = PLUGIN_DESTINATION / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    remove_path(TEST_DESTINATION)
    TEST_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        shutil.copytree(PLUGIN_DESTINATION, TEST_DESTINATION)
    else:
        TEST_DESTINATION.symlink_to(PLUGIN_DESTINATION, target_is_directory=True)


if __name__ == "__main__":
    stage_plugin()
