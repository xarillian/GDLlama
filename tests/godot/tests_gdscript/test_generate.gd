class_name TestGenerate
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("typed factories and result source identity", func():
		var request := ChorusRequest.stateless("hello world")
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "expected typed stateless request to be accepted")
		TestReport.check(result.request == request, "expected submit result to retain request identity")
		request.content = "mutated after admission"
		var event: Array = await chorus.generation_complete
		TestReport.check(event[0] == result.request_id, "expected generation completion correlation")
		TestReport.check(event[2] == -1, "expected stateless completion message id to be absent")
	)

	await TestReport.run("empty chat factory session rejects instead of becoming stateless", func():
		var result := chorus.generate(ChorusRequest.chat(StringName(), "hello"))
		TestReport.check(not result.accepted and result.error == GodotChorus.ERR_INVALID_REQUEST, "expected empty chat session rejection")
	)

	await TestReport.run("staged embedding factory metadata documents the empty session", func():
		var documentation := FileAccess.get_file_as_string("res://addons/chorus/doc_classes/ChorusEmbeddingRequest.xml")
		TestReport.check(documentation.contains('<param index="1" name="session" type="StringName" default="&quot;&quot;"/></method>'), "expected an empty StringName default in staged metadata")
	)

	await TestReport.run("invalid active defaults constraint rejects admission", func():
		var defaults := ChorusGenerationDefaults.new()
		defaults.override_constraint = true
		defaults.constraint_format = 99
		chorus.generation_defaults = defaults
		TestReport.check(not chorus.generate(ChorusRequest.stateless("invalid defaults constraint")).accepted, "expected invalid active defaults constraint rejection")
		defaults.override_seed = true
		defaults.seed = -1
		TestReport.check(not chorus.generate(ChorusRequest.stateless("invalid defaults seed")).accepted, "expected invalid active defaults seed rejection")
		chorus.generation_defaults = null
	)

	await TestReport.run("typed collection metadata names every resource element", func():
		var methods := ClassDB.class_get_method_list("GodotChorus")
		var generate_batch: Dictionary = methods.filter(func(method): return method.name == &"generate_batch")[0]
		var import_history: Dictionary = methods.filter(func(method): return method.name == &"import_conversation_history")[0]
		var export_history: Dictionary = methods.filter(func(method): return method.name == &"export_conversation_history")[0]
		var inject: Dictionary = ClassDB.class_get_property_list("ChorusRequest").filter(func(property): return property.name == &"inject")[0]
		var completion: Dictionary = ClassDB.class_get_signal_list("GodotChorus").filter(func(item): return item.name == &"generation_complete")[0]
		TestReport.check(generate_batch.args[0].hint_string == "ChorusRequest" and generate_batch["return"].hint_string == "ChorusSubmitResult", "expected typed generation batch metadata")
		TestReport.check(import_history.args[1].hint_string == "ChorusMessage" and export_history["return"].hint_string == "ChorusMessage" and inject.hint_string == "ChorusInjectedMessage", "expected typed history and injection metadata")
		TestReport.check(completion.args[1].type == TYPE_STRING_NAME, "expected StringName generation completion session metadata")
	)

	await TestReport.run("typed session and embedding signals carry identity", func():
		var chat := ChorusRequest.chat(&"test-1", "hello 1")
		var chat_result := chorus.generate(chat)
		var embedding := ChorusEmbeddingRequest.create("cat sleeps on mat", &"memory")
		var embedding_result := chorus.embed(embedding)
		TestReport.check(chat_result.accepted and chat_result.request_message_id >= 0 and chat_result.response_message_id >= 0, "expected sessioned request identities")
		TestReport.check(embedding_result.accepted, "expected typed embedding request to be accepted")
		var completion: Array = await chorus.generation_complete
		TestReport.check(completion[0] == chat_result.request_id and completion[1] == "test-1" and completion[2] == chat_result.response_message_id, "expected generation signal correlation")
		var embedded: Array = await chorus.embedding_complete
		TestReport.check(embedded[0] == embedding_result.request_id and embedded[1] == &"memory" and embedded[2].size() == 128, "expected embedding signal correlation")
	)

	await TestReport.run("override state distinguishes empty set and clear", func():
		var request := ChorusRequest.stateless("state")
		request.stop = PackedStringArray(["x"])
		TestReport.check(request.stop_state == ChorusOverrideState.INHERIT, "value property must not activate override")
		request.set_stop(PackedStringArray())
		TestReport.check(request.stop_state == ChorusOverrideState.SET and request.stop.is_empty(), "expected empty SET")
		request.clear_stop()
		TestReport.check(request.stop_state == ChorusOverrideState.CLEAR, "expected CLEAR to remain distinct")
		request.seed_state = ChorusOverrideState.SET
		request.seed = -1
		TestReport.check(not chorus.generate(request).accepted, "expected negative active seed rejection")
		request.execution = 99
		TestReport.check(not chorus.generate(request).accepted, "expected invalid execution rejection")
	)

	await TestReport.run("batches isolate entries and preserve positions", func():
		var first := ChorusRequest.chat(&"batch-a", "first")
		var invalid := ChorusRequest.stateless("bad")
		invalid.execution = 99
		var third := ChorusRequest.stateless("third")
		var results := chorus.generate_batch([first, null, invalid, third])
		TestReport.check(results.size() == 4, "expected one result per input")
		TestReport.check(results[0].accepted and not results[1].accepted and not results[2].accepted and results[3].accepted, "expected independent admission")
		for _i in range(2):
			await chorus.generation_complete
	)

	await TestReport.run("embedding batches preserve positions and accept null entries independently", func():
		var requests: Array[ChorusEmbeddingRequest] = [
			ChorusEmbeddingRequest.create("first", &"embedding-a"),
			null,
			ChorusEmbeddingRequest.create("third", &"embedding-c")
		]
		var results := chorus.embed_batch(requests)
		TestReport.check(results.size() == 3, "expected one embedding result per input")
		TestReport.check(results[0].accepted and not results[1].accepted and results[2].accepted, "expected null embedding rejection to stay positional")
		for _i in range(2):
			await chorus.embedding_complete
	)

	await TestReport.run("invalid shared defaults reject every generation batch element", func():
		var defaults := ChorusGenerationDefaults.new()
		defaults.provider_options = {"llama": {"repeat_penalty": Vector2.ONE}}
		chorus.generation_defaults = defaults
		var first := ChorusRequest.stateless("first")
		var second := ChorusRequest.stateless("second")
		var results := chorus.generate_batch([first, second])
		TestReport.check(results.size() == 2 and not results[0].accepted and not results[1].accepted, "expected no batch dispatch when shared defaults fail")
		TestReport.check(results[0].request == first and results[1].request == second, "expected rejected batch source identity")
		chorus.generation_defaults = null
	)

	await TestReport.run("typed numeric overrides reject non-finite and out-of-range floats", func():
		var invalid_values := [NAN, INF, -INF, 1.0e40]
		for value in invalid_values:
			var temperature := ChorusRequest.stateless("temperature")
			temperature.set_temperature(value)
			TestReport.check(not chorus.generate(temperature).accepted, "expected invalid temperature to be rejected")
			var top_p := ChorusRequest.stateless("top-p")
			top_p.set_top_p(value)
			TestReport.check(not chorus.generate(top_p).accepted, "expected invalid top-p to be rejected")
			var frequency := ChorusRequest.stateless("frequency")
			frequency.set_frequency_penalty(value)
			TestReport.check(not chorus.generate(frequency).accepted, "expected invalid frequency penalty to be rejected")
			var presence := ChorusRequest.stateless("presence")
			presence.set_presence_penalty(value)
			TestReport.check(not chorus.generate(presence).accepted, "expected invalid presence penalty to be rejected")
	)

	await TestReport.run("render prompt returns typed diagnostics", func():
		var preview := chorus.render_prompt(ChorusRequest.stateless("preview"))
		TestReport.check(preview.ok and preview.text == "preview", "expected stateless typed prompt preview")
	)

	chorus.stop_all()
	chorus.queue_free()
	await parent.get_tree().process_frame
	await parent.get_tree().process_frame
