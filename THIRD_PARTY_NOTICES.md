# Third-party notices

Chorus is licensed under the [MIT License](LICENSE). Third-party components retain their own licenses; the Chorus license does not replace them.

Keep `LICENSE`, this file, and the `licenses/` directory with redistributed Chorus libraries or addons. The Godot staging command includes these files. Components vary by build configuration; a notice here does not mean that every component is present in every binary.

## Inherited attribution

Chorus descends from GDLlama and [godot-llm](https://github.com/Adriankhl/godot-llm). The original MIT copyright notice, `Copyright (c) 2024 k.h.lai`, is retained in `LICENSE`.

## MIT components

The following copyright notices accompany the MIT terms reproduced below.

### godot-cpp and the Godot extension interface

Source: [godot-cpp](https://github.com/godotengine/godot-cpp), vendored at `third-party/godot-cpp`. Used by the Godot adapter.

```text
Copyright (c) 2017-present Godot Engine contributors.
Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
```

### llama.cpp and ggml

Source: [llama.cpp](https://github.com/ggml-org/llama.cpp), vendored at `third-party/llama.cpp`. Used by the Llama provider. Chorus applies the source patch in `patches/llama-resource-cleanup.patch` when building this dependency.

```text
Copyright (c) 2023-2026 The ggml authors
```

The CPU matrix-multiplication code in `ggml/src/ggml-cpu/llamafile/sgemm.cpp` also carries:

```text
Copyright 2024 Mozilla Foundation
```

The YaRN implementations in `ggml/src/ggml-cpu/ops.cpp` and `ggml/src/ggml-metal/kernels/rope.metal` also carry:

```text
Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
```

### JSON for Modern C++

Source: [nlohmann/json](https://github.com/nlohmann/json), version 3.12.0, vendored at `third-party/nlohmann-json` and in `third-party/llama.cpp/vendor/nlohmann`. Used by host settings and the Llama provider.

```text
Copyright (c) 2013-2025 Niels Lohmann
```

The bundled UTF-8 decoder and Grisu2 number-formatting implementation carry these additional MIT notices:

```text
Copyright (c) 2008-2009 Björn Hoehrmann <bjoern@hoehrmann.de>
Copyright (c) 2009 Florian Loitsch
```

Hedley and Abseil portions have additional terms listed below.

### cpp-httplib

Source: [cpp-httplib](https://github.com/yhirose/cpp-httplib), vendored at `third-party/llama.cpp/vendor/cpp-httplib`. Used by llama.cpp common utilities.

```text
Copyright (c) 2017 yhirose
```

### Vulkan headers and C++ bindings

Sources: [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) and [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp), supplied by the Vulkan SDK or system development packages. The Vulkan backend uses the MIT option of their dual Apache-2.0/MIT licensing.

```text
Copyright (c) 2015-2023 The Khronos Group Inc.
Copyright 2015-2026 The Khronos Group Inc.
```

### MIT terms

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Additional JSON for Modern C++ components

### Hedley

Source: [Hedley](https://github.com/nemequ/hedley), bundled in the JSON headers.

```text
Copyright (c) 2016-2021 Evan Nemerson <evan@nemerson.com>
```

The upstream JSON README identifies Hedley as CC0-1.0. The full terms are in [licenses/CC0-1.0.txt](licenses/CC0-1.0.txt).

### Abseil

Source: [Abseil](https://github.com/abseil/abseil-cpp), portions bundled in the JSON headers' C++11 compatibility code.

```text
Copyright 2018 The Abseil Authors
```

Licensed under Apache-2.0. The full terms are in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

## Public-domain components in llama.cpp

`common/base64.hpp` and `vendor/sheredom/subprocess.h` use the Unlicense. The subprocess library is from [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h).

```text
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
```

## GCC runtime libraries

GCC-built binaries may include runtime code from libgcc and libstdc++, or use libgomp for OpenMP. These libraries are copyright Free Software Foundation, Inc., and use GPL-3.0-or-later with the GCC Runtime Library Exception 3.1.

The exception permits eligible compiled combinations with independent modules to be conveyed under the terms chosen for those modules. It does not change Chorus's MIT license. The full texts are in [licenses/GPL-3.0.txt](licenses/GPL-3.0.txt) and [licenses/GCC-Runtime-Library-Exception-3.1.txt](licenses/GCC-Runtime-Library-Exception-3.1.txt).

## GoogleTest

Source: [GoogleTest](https://github.com/google/googletest), vendored at `third-party/googletest`. Used only by native tests, not the shipped Chorus libraries or addon.

```text
Copyright 2008, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Models and other redistributed components

Chorus does not include model weights. Models have their own licenses and usage terms, which are not replaced by the Chorus license.

The staging command copies the Chorus extension, not the Godot executable, GPU drivers or system shared libraries. If you distribute those components, additional backends or other dependencies, include their applicable licenses and notices. Recheck this inventory when changing dependency revisions, toolchains or build options.
