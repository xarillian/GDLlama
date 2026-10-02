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
		var scope := "chorus/generation/provider_options" if in_defaults else "provider_options"
		if in_defaults:
			ProjectSettings.set_setting(scope, {"foreign": {"option": [options["foreign"]]}})
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
		ProjectSettings.set_setting("chorus/generation/provider_options", {})
		var recovered := chorus.generate(request)
		TestReport.check(recovered.accepted, "expected the same request and engine to remain usable")
		if recovered.accepted:
			var event: Array = await chorus.generation_complete
			TestReport.check(event[0] == recovered.request_id, "expected the recovered request to complete")
			TestReport.check(event[3].contains("usable after rejection"), "expected recovered generation output")
		TestReport.check(chorus.clear_conversation_history(&"conversion-recovery").ok, "expected recovered session to be idle")


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
		for options in [nested_options(63, false), nested_options(63, true), shared]:
			var request := ChorusRequest.stateless("shared options remain usable")
			request.provider_options = options
			var host_options: Dictionary = options
			if options.has("foreign") and options["foreign"] is not Dictionary:
				host_options = {"foreign": {"option": options["foreign"]}}
			ProjectSettings.set_setting("chorus/generation/provider_options", host_options)
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected valid depth and sharing in requests and defaults")
			if result.accepted:
				var event: Array = await chorus.generation_complete
				TestReport.check(event[0] == result.request_id and event[3].contains("shared options remain usable"), "expected shared acyclic options to complete")
		ProjectSettings.set_setting("chorus/generation/provider_options", {})
	)

	await TestReport.run("project conversion accepts codec depth and rejects float underflow", func():
		var value: Variant = false
		for _i in range(64):
			value = [value]
		ProjectSettings.set_setting("chorus/generation/provider_options", {"foreign": {"deep": value}})
		var accepted := chorus.generate(ChorusRequest.stateless("deep project value"))
		TestReport.check(accepted.accepted, "64 nested containers under one option must be accepted")
		if accepted.accepted:
			await chorus.generation_complete
		ProjectSettings.set_setting("chorus/generation/provider_options", {})
		value = [value]
		ProjectSettings.set_setting("chorus/generation/provider_options", {"foreign": {"deep": value}})
		TestReport.check(not chorus.generate(ChorusRequest.stateless("too deep")).accepted, "65 nested containers must reject")
		ProjectSettings.set_setting("chorus/generation/provider_options", {})
		ProjectSettings.set_setting("chorus/generation/temperature", {"value": 1e-100})
		TestReport.check(not chorus.generate(ChorusRequest.stateless("underflow")).accepted, "nonzero float32 underflow must reject")
		ProjectSettings.set_setting("chorus/generation/temperature", {"value": -0.0})
		var zero := chorus.generate(ChorusRequest.stateless("zero"))
		TestReport.check(zero.accepted, "explicit negative zero is representable")
		ProjectSettings.set_setting("chorus/generation/temperature", {})
		if zero.accepted:
			await chorus.generation_error
	)

	await TestReport.run("invalid native choice dictionaries reject transactionally", func():
		var request := ChorusRequest.stateless("native settings validation")
		for invalid in [{"value": 0, "present": true}, {"value": "zero"}, {"wrong": 0}]:
			ProjectSettings.set_setting("chorus/generation/max_tokens", invalid)
			var result := chorus.generate(request)
			TestReport.check(not result.accepted and result.error == GodotChorus.ERR_INVALID_REQUEST, "invalid choice wrapper must reject")
		ProjectSettings.set_setting("chorus/generation/max_tokens", {})
		for invalid in [{"value": {"kind": "unconstrained", "source": ""}}, {"value": {"kind": "gbnf"}}]:
			ProjectSettings.set_setting("chorus/generation/constraint", invalid)
			TestReport.check(not chorus.generate(request).accepted, "partial or extra constraint field rejects")
		ProjectSettings.set_setting("chorus/generation/constraint", {})
		for invalid in ["-1", "18446744073709551616", 0]:
			ProjectSettings.set_setting("chorus/generation/seed", {"value": invalid})
			TestReport.check(not chorus.generate(request).accepted, "seed must be unsigned uint64 decimal text")
		ProjectSettings.set_setting("chorus/generation/seed", {})
		ProjectSettings.set_setting("chorus/generation/provider_options", {1: {"key": true}})
		TestReport.check(not chorus.generate(request).accepted, "namespace keys require strings")
		ProjectSettings.set_setting("chorus/generation/provider_options", {})
	)

	await TestReport.run("request choices survive duplicate and resource round trips without selecting absent display values", func():
		var request := ChorusRequest.chat(&"request-round-trip", "round trip")
		request.max_tokens = 1 << 40
		request.clear_max_tokens()
		request.temperature = INF
		request.clear_temperature()
		request.seed = 0
		request.stop = PackedStringArray()
		request.show_thinking = false
		request.set_constraint(99, "invalid")
		request.clear_constraint()
		request.set_unconstrained()
		request.chat_template = "chosen template"
		for copy in [request.duplicate(true), request]:
			TestReport.check(not copy.has_max_tokens() and not copy.has_temperature(), "cleared invalid values must stay absent")
			TestReport.check(copy.has_seed() and copy.seed == 0 and copy.has_stop() and copy.stop.is_empty(), "zero and empty stop must stay present")
			TestReport.check(copy.has_show_thinking() and not copy.show_thinking and copy.has_constraint() and copy.is_unconstrained(), "false and unconstrained must stay present")
			TestReport.check(copy.has_chat_template() and copy.chat_template == "chosen template", "selected template must stay present")
		var path := "user://chorus_request_intent_test.tres"
		TestReport.check(ResourceSaver.save(request, path) == OK, "request resource should save")
		var loaded: ChorusRequest = ResourceLoader.load(path, "", ResourceLoader.CACHE_MODE_IGNORE)
		TestReport.check(loaded != null, "request resource should load")
		if loaded != null:
			TestReport.check(not loaded.has_max_tokens() and not loaded.has_temperature(), "load must not replay absent display values")
			TestReport.check(loaded.has_seed() and loaded.seed == 0 and loaded.has_stop() and loaded.stop.is_empty(), "load must retain zero and empty stop")
			TestReport.check(loaded.has_show_thinking() and not loaded.show_thinking and loaded.is_unconstrained(), "load must retain false and unconstrained")
			TestReport.check(loaded.has_chat_template() and loaded.chat_template == "chosen template", "load must retain selected template")
		TestReport.check(DirAccess.remove_absolute(ProjectSettings.globalize_path(path)) == OK, "temporary resource should be removed")
	)

	await TestReport.run("request compound constraints and selected empty template reject without stale choices", func():
		var request := ChorusRequest.chat(&"request-constraint", "hello")
		request.set_constraint(99, "invalid format")
		TestReport.check(not chorus.generate(request).accepted, "invalid selected format must reject")
		request.clear_constraint()
		TestReport.check(not request.has_constraint(), "clear must remove invalid format")
		request.set_unconstrained()
		TestReport.check(request.has_constraint() and request.is_unconstrained(), "explicit unconstrained must be a value")
		request.set_constraint(ChorusConstraintFormat.GBNF, "root ::= \"hello\"")
		TestReport.check(request.has_constraint() and not request.is_unconstrained(), "complete grammar must replace unconstrained")
		request.set_constraint(ChorusConstraintFormat.GBNF, "")
		TestReport.check(request.has_constraint(), "empty selected grammar source must remain selected")
		request.clear_constraint()
		request.chat_template = ""
		TestReport.check(request.has_chat_template() and not chorus.generate(request).accepted, "selected empty template must reject")
		request.clear_chat_template()
		TestReport.check(not request.has_chat_template(), "template clear must remove selection")
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "cleared invalid choices must allow submission")
		if result.accepted:
			await chorus.generation_complete
	)

	await TestReport.run("request provider dictionary conversion copies nested option values", func():
		var nested := {"value": [1]}
		var request := ChorusRequest.stateless("copied options")
		request.provider_options = {"foreign": {"option": nested, "empty": {}}}
		var copied := request.duplicate(true)
		nested["value"].append(2)
		TestReport.check(copied.provider_options["foreign"]["option"]["value"] == [1], "deep duplicate must not share nested dictionary arrays")
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "foreign nested options should be accepted")
		nested["value"].append(3)
		request.clear_provider_option("foreign", "option")
		TestReport.check(not request.provider_options["foreign"].has("option") and request.provider_options["foreign"].has("empty"), "key clear removes only local key")
		request.clear_provider_options()
		TestReport.check(request.provider_options.is_empty(), "dictionary clear removes only local choices")
		if result.accepted:
			await chorus.generation_complete
	)

	await TestReport.run("request storage excludes absent display values and obsolete controls", func():
		var resource := ChorusRequest.new()
		var properties: Array[Dictionary] = resource.get_property_list()
		var storage: Array[Dictionary] = properties.filter(func(item): return item.name == &"_choices")
		TestReport.check(storage.size() == 1 and (storage[0].usage & PROPERTY_USAGE_STORAGE), "request choice snapshot is stored")
		for name in ["max_tokens", "temperature", "top_k", "top_p", "seed", "frequency_penalty", "presence_penalty", "stop", "show_thinking"]:
			var found: Array[Dictionary] = properties.filter(func(item): return item.name == name)
			TestReport.check(found.size() == 1 and not (found[0].usage & PROPERTY_USAGE_STORAGE), "display choice must not replay absent " + name)
		TestReport.check(not ClassDB.class_exists("ChorusGenerationDefaults"), "obsolete defaults resource is removed")
	)

	await parent.get_tree().process_frame
	if TestReport.filter.is_empty():
		TestReport.check(terminal_ids.size() == 17, "expected terminals for recovered, valid and selected-choice requests")
	chorus.stop_all()
	chorus.queue_free()
	await parent.get_tree().process_frame
