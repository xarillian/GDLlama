class_name ModelPaths
extends RefCounted

const VALID_GGUF := "models/gemma-3-270m-it-F16.gguf"
const MISSING_GGUF := "models/does-not-exist.gguf"
const REASONING_GGUF := "models/Qwen3-0.6B-Q8_0.gguf"


static func tests_root() -> String:
	return ProjectSettings.globalize_path("res://").path_join("..").simplify_path()


static func valid_gguf() -> String:
	return tests_root().path_join(VALID_GGUF)


static func missing_gguf() -> String:
	return tests_root().path_join(MISSING_GGUF)


static func reasoning_gguf() -> String:
	return tests_root().path_join(REASONING_GGUF)
