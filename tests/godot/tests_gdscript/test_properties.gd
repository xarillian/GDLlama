class_name TestProperties
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	parent.add_child(chorus)

	# The llama backend declares these; the node renders them from that
	# declaration rather than binding them one by one, so the whole set is worth
	# exercising: a name, type, or default lost in translation shows up here.
	const LLAMA_OPTIONS := {
		"context_size": 2048,
		"thread_count": 4,
		"use_gpu": true,
		"gpu_layers": -1,
		"num_slots": 1,
		"tokens_per_tick": 512,
		"n_batch": 2048,
		"n_ubatch": 512,
		"main_gpu": 0,
	}

	await TestReport.run("declared backend options expose their defaults", func():
		chorus.backend = GodotChorus.BACKEND_LLAMA
		for name in LLAMA_OPTIONS:
			TestReport.check(
				chorus.get(name) == LLAMA_OPTIONS[name],
				"expected %s to default to %s, got %s" % [name, LLAMA_OPTIONS[name], chorus.get(name)]
			)
	)

	await TestReport.run("declared backend options round-trip", func():
		chorus.backend = GodotChorus.BACKEND_LLAMA
		chorus.context_size = 4096
		TestReport.check(chorus.context_size == 4096, "expected context_size to round-trip")

		chorus.thread_count = 8
		TestReport.check(chorus.thread_count == 8, "expected thread_count to round-trip")

		chorus.gpu_layers = 12
		TestReport.check(chorus.gpu_layers == 12, "expected gpu_layers to round-trip")

		chorus.num_slots = 4
		TestReport.check(chorus.num_slots == 4, "expected num_slots to round-trip")

		chorus.tokens_per_tick = 256
		TestReport.check(chorus.tokens_per_tick == 256, "expected tokens_per_tick to round-trip")

		chorus.n_batch = 4096
		TestReport.check(chorus.n_batch == 4096, "expected n_batch to round-trip")

		chorus.n_ubatch = 1024
		TestReport.check(chorus.n_ubatch == 1024, "expected n_ubatch to round-trip")

		chorus.main_gpu = 1
		TestReport.check(chorus.main_gpu == 1, "expected main_gpu to round-trip")

		chorus.use_gpu = false
		TestReport.check(chorus.use_gpu == false, "expected use_gpu to round-trip")
	)

	await TestReport.run("declared backend options appear in the property list", func():
		chorus.backend = GodotChorus.BACKEND_LLAMA
		var listed := {}
		for entry in chorus.get_property_list():
			listed[entry["name"]] = entry
		for name in LLAMA_OPTIONS:
			TestReport.check(listed.has(name), "expected %s in the property list" % name)
	)

	await TestReport.run("backend options revert to the declared default", func():
		chorus.backend = GodotChorus.BACKEND_LLAMA
		chorus.context_size = 8192
		TestReport.check(
			chorus.property_get_revert("context_size") == 2048,
			"expected context_size to revert to the backend's declared default"
		)
	)

	await TestReport.run("a backend declaring no options exposes none", func():
		chorus.backend = GodotChorus.BACKEND_ECHO
		TestReport.check(chorus.backend == GodotChorus.BACKEND_ECHO, "expected backend to round-trip")
		var listed := {}
		for entry in chorus.get_property_list():
			listed[entry["name"]] = entry
		TestReport.check(
			not listed.has("context_size"),
			"expected llama's options to disappear when Echo is selected"
		)
	)

	await TestReport.run("switching backends preserves a configured option", func():
		chorus.backend = GodotChorus.BACKEND_LLAMA
		chorus.num_slots = 7
		chorus.backend = GodotChorus.BACKEND_ECHO
		chorus.backend = GodotChorus.BACKEND_LLAMA
		TestReport.check(chorus.num_slots == 7, "expected num_slots to survive a backend round-trip")
	)

	# TODO: chat_template round-trips and defaults to ""
	# TODO: generation_defaults defaults to null until a ChorusGenerationDefaults resource is assigned
	# TODO: regenerate(session) with no overrides argument (DEFVAL) doesn't error
	# TODO: render_chat_prompt(session) with defaulted template_override/inject doesn't error

	chorus.queue_free()
