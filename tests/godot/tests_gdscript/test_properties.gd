class_name TestProperties
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	parent.add_child(chorus)

	await TestReport.run("numeric and boolean properties round-trip", func():
		chorus.context_size = 4096
		TestReport.check(chorus.context_size == 4096, "expected context_size to round-trip")

		chorus.thread_count = 8
		TestReport.check(chorus.thread_count == 8, "expected thread_count to round-trip")

		chorus.use_gpu = false
		TestReport.check(chorus.use_gpu == false, "expected use_gpu to round-trip")

		chorus.backend = GodotChorus.BACKEND_ECHO
		TestReport.check(chorus.backend == GodotChorus.BACKEND_ECHO, "expected backend to round-trip")
	)

	# TODO: gpu_layers, num_slots, tokens_per_tick, n_batch, n_ubatch, main_gpu round-trip
	# TODO: chat_template round-trips and defaults to ""
	# TODO: generation_defaults defaults to null until a ChorusGenerationDefaults resource is assigned
	# TODO: regenerate(session) with no overrides argument (DEFVAL) doesn't error
	# TODO: render_chat_prompt(session) with defaulted template_override/inject doesn't error

	chorus.queue_free()
