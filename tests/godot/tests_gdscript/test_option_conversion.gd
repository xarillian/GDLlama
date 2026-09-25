class_name TestOptionConversion
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func check_rejection(result: Variant, scope: String, reason: String) -> void:
	TestReport.check(result.error == GodotChorus.ERR_INVALID_REQUEST, "expected invalid request for " + scope)
	TestReport.check(result.message.contains(scope) and result.message.contains(reason), "expected scoped conversion diagnostic: " + result.message)
	if result is ChorusSubmitResult:
		TestReport.check(not result.accepted and result.request_id == -1, "expected rejection without a request identity")


static func reject_then_reuse(chorus: GodotChorus, options: Dictionary, reason: String) -> void:
	for in_defaults in [false, true]:
		var request := ChorusRequest.chat(&"conversion-recovery", "usable after rejection")
		var defaults := ChorusGenerationDefaults.new()
		chorus.generation_defaults = defaults
		var scope := "generation_defaults.provider_options" if in_defaults else "provider_options"
		if in_defaults:
			defaults.provider_options = options
		else:
			request.provider_options = options

		check_rejection(chorus.generate(request), scope, reason)
		check_rejection(chorus.regenerate(request), scope, reason)
		check_rejection(chorus.render_prompt(request), scope, reason)
		var batch := chorus.generate_batch([request, request])
		TestReport.check(batch.size() == 2, "expected positional batch rejections")
		for result in batch:
			check_rejection(result, scope, reason)
		TestReport.check(chorus.is_loaded(), "expected conversion rejection to leave the engine loaded")
		TestReport.check(chorus.active_request_for_session("conversion-recovery") == -1, "expected no session reservation on rejection")
		TestReport.check(chorus.export_conversation_history(&"conversion-recovery").is_empty(), "expected no history mutation on rejection")

		request.provider_options = {}
		defaults.provider_options = {}
		var recovered := chorus.generate(request)
		TestReport.check(recovered.accepted, "expected the same request and engine to remain usable")
		if recovered.accepted:
			var event: Array = await chorus.generation_complete
			TestReport.check(event[0] == recovered.request_id, "expected the recovered request to complete")
			TestReport.check(event[3].contains("usable after rejection"), "expected recovered generation output")
		TestReport.check(chorus.clear_conversation_history(&"conversion-recovery").ok, "expected recovered session to be idle")
	chorus.generation_defaults = null


static func nested_options(depth: int, mixed: bool) -> Dictionary:
	var value: Variant = true
	for level in range(depth - 1):
		value = [value] if mixed and level % 2 == 0 else {"child": value}
	return {"foreign": value}


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	await LoadWaiter.load(chorus)
	var terminal_ids: Array[int] = []
	chorus.generation_complete.connect(func(id, _session, _message_id, _text, _reasoning): terminal_ids.append(id))
	chorus.generation_error.connect(func(id, _session, _error, _message): terminal_ids.append(id))

	await TestReport.run("option conversion rejects dictionary self-cycles in requests and defaults", func():
		var options := {}
		options["foreign"] = options
		await reject_then_reuse(chorus, options, "cycle")
		options.clear()
	)

	await TestReport.run("option conversion rejects array self-cycles in requests and defaults", func():
		var array := []
		array.append(array)
		await reject_then_reuse(chorus, {"foreign": array}, "cycle")
		array.clear()
	)

	await TestReport.run("option conversion rejects mixed container cycles in requests and defaults", func():
		var options := {}
		var array := [options]
		options["foreign"] = array
		await reject_then_reuse(chorus, options, "cycle")
		array.clear()
		options.clear()
	)

	await TestReport.run("option conversion rejects excessive dictionary and mixed nesting", func():
		await reject_then_reuse(chorus, nested_options(65, false), "nesting depth of 64")
		await reject_then_reuse(chorus, nested_options(65, true), "nesting depth of 64")
	)

	await TestReport.run("option conversion accepts the depth boundary and acyclic sharing", func():
		var shared_map := {"value": 7}
		var shared_array := [shared_map, shared_map]
		var shared := {"foreign": {"first": shared_array, "second": shared_array, "equal": [{"value": 7}, {"value": 7}]}}
		for options in [nested_options(64, false), nested_options(64, true), shared]:
			var request := ChorusRequest.stateless("shared options remain usable")
			request.provider_options = options
			var defaults := ChorusGenerationDefaults.new()
			defaults.provider_options = options
			chorus.generation_defaults = defaults
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected valid depth and sharing in requests and defaults")
			if result.accepted:
				var event: Array = await chorus.generation_complete
				TestReport.check(event[0] == result.request_id and event[3].contains("shared options remain usable"), "expected shared acyclic options to complete")
		chorus.generation_defaults = null
	)

	await parent.get_tree().process_frame
	if TestReport.filter.is_empty():
		TestReport.check(terminal_ids.size() == 13, "expected terminals only for ten recovered and three valid requests")
	chorus.stop_all()
	chorus.queue_free()
	await parent.get_tree().process_frame
