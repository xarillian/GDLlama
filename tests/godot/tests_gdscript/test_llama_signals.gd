class_name TestLlamaSignals
extends RefCounted

## Narrowly scoped to the two signals Echo can't reach (reasoning_token_generated,
## history_truncated -- both gated on a real context budget). Generation quality /
## sampling correctness stays the native suite's job; this only checks the signal
## reaches Godot with the right shape.

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.backend = GodotChorus.BACKEND_LLAMA
	chorus.model_path = ModelPaths.reasoning_gguf()
	parent.add_child(chorus)

	await TestReport.run("a reasoning-capable model loads", func():
		TestReport.check(chorus.load_model(), "expected load_model() to succeed on the reasoning-capable model")
	)

	# TODO: reasoning_token_generated fires at least once with thinking left at its default
	# TODO: thinking: false yields no reasoning_token_generated events
	# TODO: history_truncated fires once a session's history exceeds the configured context_size

	chorus.stop_all()
	chorus.queue_free()
