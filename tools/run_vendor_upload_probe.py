"""Exercise the maintained upload fix against the built Linux Vulkan vendor archive."""

from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from materialize_llama import ROOT, build_directory, materialize


def main():
    if not sys.platform.startswith("linux"):
        raise SystemExit("the GNU linker wrap probe is supported on Linux only")
    source, revision, identity = materialize()
    build = build_directory(revision, identity, "linux-x86_64", "vulkan")
    if not (build / "src/libllama.a").is_file():
        raise SystemExit("build the patched Vulkan test variant first: scons test use_vulkan=yes -j2")
    binary = ROOT / "bin/vendor_upload_probe"
    names = (
        "ggml_backend_buft_alloc_buffer", "ggml_backend_buffer_free", "ggml_backend_event_new",
        "ggml_backend_event_synchronize", "ggml_backend_event_free", "ggml_backend_event_record",
        "ggml_backend_dev_init", "ggml_backend_free", "ggml_backend_synchronize",
    )
    command = ["c++", "-std=c++20", "-O1", "-fopenmp", f"-I{source / 'include'}",
               f"-I{source / 'src'}", f"-I{source / 'ggml/include'}", str(ROOT / "tests/native/probes/vendor_upload_probe.cpp"),
               "-o", str(binary)]
    command.extend(f"-Wl,--wrap={name}" for name in names)
    command.extend(f"-L{build / path}" for path in ("src", "ggml/src", "ggml/src/ggml-vulkan"))
    command.extend(("-lllama", "-lggml", "-lggml-cpu", "-lggml-base", "-lggml-vulkan",
                    "-lvulkan", "-ldl", "-pthread"))
    subprocess.run(command, cwd=ROOT, check=True)
    model = ROOT / "tests/models/gemma-3-270m-it-F16.gguf"
    if not model.is_file():
        raise SystemExit(f"required existing model fixture missing: {model}")
    for mode in ("mixed", "buffer-fail", "event-fail", "backend-fail", "allocation-throw", "batch-fault"):
        subprocess.run([binary, model, mode], cwd=ROOT, check=True, timeout=180)


if __name__ == "__main__":
    main()
