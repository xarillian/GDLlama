# Version

## [2.0.0] - Oct 2026
### Breaking Changes
- Renamed GDLlama to Chorus and replaced the legacy Godot API with `GodotChorus` and typed request resources. Existing scenes and scripts must migrate; no compatibility layer is provided.
- Moved shared generation defaults to native Godot Project Settings with portable JSON persistence. Request choices override host choices, while absent choices remain provider defaults.

### Added
- Host-independent C++ runtime and C ABI, with separate host adapters and inference providers
- Concurrent inference through shared batches, priority scheduling, and chunked prompt processing
- Asynchronous model loading with progress reporting and cancellation
- Explicit request admission, request identities, cancellation, and exactly one terminal result per accepted request
- Conversation import, export, editing, regeneration, temporary context, and prompt fitting with history rollback on failure or cancellation
- Separate reasoning and visible-text streams for supported models
- Embedded Godot editor documentation and an updated [Godot guide](GODOT.md)

### Changed
- Provider capability reporting and explicit rejection of unsupported options

## [1.0.0] - Oct 2025
### Added
- Initial stable release
- Multi-platform support (Windows, Linux, macOS)
- GPU acceleration (Vulkan, Metal)
- Conversational AI with context management
- Embedding generation and similarity search
- GBNF grammar and JSON schema support
- Comprehensive documentation
