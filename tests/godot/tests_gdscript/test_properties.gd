class_name TestProperties
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	parent.add_child(chorus)

	# The llama provider declares these; the node renders them from that
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

	await TestReport.run("declared provider options expose their defaults", func():
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		for name in LLAMA_OPTIONS:
			TestReport.check(
				chorus.get(name) == LLAMA_OPTIONS[name],
				"expected %s to default to %s, got %s" % [name, LLAMA_OPTIONS[name], chorus.get(name)]
			)
	)

	await TestReport.run("declared provider options round-trip", func():
		chorus.provider = GodotChorus.PROVIDER_LLAMA
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

	await TestReport.run("declared provider options appear in the property list", func():
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		var listed := {}
		for entry in chorus.get_property_list():
			listed[entry["name"]] = entry
		for name in LLAMA_OPTIONS:
			TestReport.check(listed.has(name), "expected %s in the property list" % name)
	)

	await TestReport.run("provider options revert to the declared default", func():
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		chorus.context_size = 8192
		TestReport.check(
			chorus.property_get_revert("context_size") == 2048,
			"expected context_size to revert to the provider's declared default"
		)
	)

	await TestReport.run("a provider declaring no options exposes none", func():
		chorus.provider = GodotChorus.PROVIDER_ECHO
		TestReport.check(chorus.provider == GodotChorus.PROVIDER_ECHO, "expected provider to round-trip")
		var listed := {}
		for entry in chorus.get_property_list():
			listed[entry["name"]] = entry
		TestReport.check(
			not listed.has("context_size"),
			"expected llama's options to disappear when Echo is selected"
		)
	)

	await TestReport.run("switching providers preserves a configured option", func():
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		chorus.num_slots = 7
		chorus.provider = GodotChorus.PROVIDER_ECHO
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		TestReport.check(chorus.num_slots == 7, "expected num_slots to survive a provider round-trip")
	)

	chorus.queue_free()
