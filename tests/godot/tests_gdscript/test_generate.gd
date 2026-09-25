class_name TestGenerate
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func wait_for_event(chorus: GodotChorus, events: Dictionary, id: int) -> Array:
	var deadline := Time.get_ticks_msec() + 10000
	while not events.has(id) and Time.get_ticks_msec() < deadline:
		await chorus.get_tree().process_frame
	TestReport.check(events.has(id), "expected terminal for request " + str(id))
	return events.get(id, [])


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	await LoadWaiter.load(chorus)
	var terminals := {}
	chorus.generation_complete.connect(func(id, session, message_id, text, reasoning): terminals[id] = [id, session, message_id, text, reasoning])
	chorus.embedding_complete.connect(func(id, session, values): terminals[id] = [id, session, values])
	chorus.prompt_rendered.connect(func(id, session, text, omitted): terminals[id] = [id, session, text, omitted])
	chorus.message_token_counted.connect(func(id, count): terminals[id] = [id, count])
	chorus.generation_error.connect(func(id, session, code, message): terminals[id] = [id, session, code, message])

	await TestReport.run("typed factories and result source identity", func():
		var request := ChorusRequest.stateless("hello world")
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "expected typed stateless request to be accepted")
		TestReport.check(result.request == request, "expected submit result to retain request identity")
		request.content = "mutated after admission"
		var event: Array = await wait_for_event(chorus, terminals, result.request_id)
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
		var completion: Array = await wait_for_event(chorus, terminals, chat_result.request_id)
		TestReport.check(completion[0] == chat_result.request_id and completion[1] == "test-1" and completion[2] == chat_result.response_message_id, "expected generation signal correlation")
		var embedded: Array = await wait_for_event(chorus, terminals, embedding_result.request_id)
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

	await TestReport.run("inactive integer overrides ignore stored out-of-range values", func():
		for state in [ChorusOverrideState.INHERIT, ChorusOverrideState.CLEAR]:
			var request := ChorusRequest.stateless("inactive integers")
			request.max_tokens = 1 << 40
			request.max_tokens_state = state
			request.top_k = 1 << 40
			request.top_k_state = state
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected inactive out-of-range integers not to affect admission")
			if result.accepted:
				await wait_for_event(chorus, terminals, result.request_id)

		var max_tokens := ChorusRequest.stateless("active max tokens")
		max_tokens.set_max_tokens(1 << 40)
		TestReport.check(not chorus.generate(max_tokens).accepted, "expected an active out-of-range max_tokens value to reject admission")
		var top_k := ChorusRequest.stateless("active top k")
		top_k.set_top_k(1 << 40)
		TestReport.check(not chorus.generate(top_k).accepted, "expected an active out-of-range top_k value to reject admission")
	)

	await TestReport.run("batches isolate entries and preserve positions", func():
		var first := ChorusRequest.chat(&"batch-a", "first")
		var invalid := ChorusRequest.stateless("bad")
		invalid.execution = 99
		var third := ChorusRequest.stateless("third")
		var results := chorus.generate_batch([first, null, invalid, third])
		TestReport.check(results.size() == 4, "expected one result per input")
		TestReport.check(results[0].accepted and not results[1].accepted and not results[2].accepted and results[3].accepted, "expected independent admission")
		for result in results:
			if result.accepted:
				await wait_for_event(chorus, terminals, result.request_id)
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
		for result in results:
			if result.accepted:
				await wait_for_event(chorus, terminals, result.request_id)
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

	await TestReport.run("provider option dictionaries require string keys recursively", func():
		var top_level := ChorusRequest.stateless("top-level provider key")
		top_level.provider_options = {1: {"x": true}}
		TestReport.check(not chorus.generate(top_level).accepted, "expected a non-string provider namespace to reject admission")

		var nested := ChorusRequest.stateless("nested provider key")
		nested.provider_options = {"foreign": {1: true}}
		TestReport.check(not chorus.generate(nested).accepted, "expected a nested non-string provider option key to reject admission")

		var valid := ChorusRequest.stateless("string provider keys")
		valid.provider_options = {"foreign": {"x": true}}
		var result := chorus.generate(valid)
		TestReport.check(result.accepted, "expected recursively string-keyed provider options to remain valid")
		if result.accepted:
			await wait_for_event(chorus, terminals, result.request_id)
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
		var request := ChorusRequest.stateless("preview")
		var preview := chorus.render_prompt(request)
		TestReport.check(preview.accepted and preview.request == request, "expected preview admission and source identity")
		TestReport.check(not terminals.has(preview.request_id), "preview must not signal inline")
		request.content = "changed after preview admission"
		var event := await wait_for_event(chorus, terminals, preview.request_id)
		TestReport.check(event[2] == "preview", "expected frozen preview text")
		var counted := chorus.count_message_tokens("hello")
		TestReport.check(not counted.accepted and counted.error == GodotChorus.ERR_UNSUPPORTED_FEATURE, "Echo must not invent a tokenizer")
		TestReport.check(counted.request == null and not chorus.supports_message_token_counting(), "count source is null and capability is honest")
		var methods := ClassDB.class_get_method_list("GodotChorus")
		var render: Dictionary = methods.filter(func(method): return method.name == &"render_prompt")[0]
		var signal_info: Dictionary = ClassDB.class_get_signal_list("GodotChorus").filter(func(item): return item.name == &"prompt_rendered")[0]
		TestReport.check(render["return"].class_name == &"ChorusSubmitResult", "preview returns admission metadata")
		TestReport.check(signal_info.args[1].type == TYPE_STRING_NAME, "preview session metadata is StringName")
		TestReport.check(not ClassDB.class_exists("ChorusRenderResult"), "obsolete synchronous result is removed")
	)

	await TestReport.run("real model async counts preview frame progress and cancellation", func():
		chorus.stop_all()
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		chorus.model_path = ModelPaths.valid_gguf()
		chorus.set("use_gpu", false)
		chorus.set("context_size", 512)
		await LoadWaiter.load(chorus)
		TestReport.check(chorus.supports_message_token_counting(), "expected loaded tokenizer capability")
		var count_method: Dictionary = ClassDB.class_get_method_list("GodotChorus").filter(func(item): return item.name == &"count_message_tokens")[0]
		var count_signal: Dictionary = ClassDB.class_get_signal_list("GodotChorus").filter(func(item): return item.name == &"message_token_counted")[0]
		TestReport.check(count_method["return"].class_name == &"ChorusSubmitResult" and count_signal.args[1].type == TYPE_INT, "expected typed count method and signal metadata")
		var counted := chorus.count_message_tokens("日本語 😀 <bos>")
		TestReport.check(counted.accepted and counted.request == null, "expected sessionless count admission")
		TestReport.check(not terminals.has(counted.request_id), "count success must not signal inline")
		var count_event := await wait_for_event(chorus, terminals, counted.request_id)
		TestReport.check(count_event.size() == 2 and count_event[1] > 0, "expected a positive literal Unicode token count")
		var empty := chorus.count_message_tokens("")
		var empty_event := await wait_for_event(chorus, terminals, empty.request_id)
		TestReport.check(empty_event.size() == 2 and empty_event[1] == 0, "empty content must count as zero")

		var history: Array[ChorusMessage] = [ChorusMessage.create(0, ChorusRole.SYSTEM, "You are the village blacksmith.")]
		for i in range(256):
			history.append(ChorusMessage.create(2 * i + 1, ChorusRole.USER, "Tell me about the weather in the village today."))
			history.append(ChorusMessage.create(2 * i + 2, ChorusRole.ASSISTANT, "The sun is shining and the village is peaceful."))
		TestReport.check(chorus.import_conversation_history(&"frame-progress", history).ok, "expected long history import outside timers")
		var request := ChorusRequest.chat(&"frame-progress", "What do you remember?")
		request.set_max_tokens(64)
		var begin := Time.get_ticks_usec()
		var preview := chorus.render_prompt(request)
		var admission_us := Time.get_ticks_usec() - begin
		TestReport.check(preview.accepted and not terminals.has(preview.request_id), "expected asynchronous long-history preview")
		TestReport.check(chorus.active_request_for_session("frame-progress") == -1, "preview must not occupy the session")
		var frames := 0
		while not terminals.has(preview.request_id) and Time.get_ticks_usec() - begin < 10000000:
			await parent.get_tree().process_frame
			frames += 1
		TestReport.check(terminals.has(preview.request_id), "expected preview completion while frames advance")
		var event: Array = terminals.get(preview.request_id, [])
		TestReport.check(event.size() == 4 and event[2] is String and event[3].size() > 0, "expected rendered text and omitted identities")
		print("F3_GODOT turns=256 admission_ms=", admission_us / 1000.0, " completion_ms=", (Time.get_ticks_usec() - begin) / 1000.0, " frames=", frames)

		var large_content := "literal draft content ".repeat(200000)
		var cancelled := chorus.count_message_tokens(large_content)
		TestReport.check(cancelled.accepted, "expected cancellable count admission")
		TestReport.check(chorus.cancel_request(cancelled.request_id), "expected cancellation to return while preparing")
		var cancelled_event := await wait_for_event(chorus, terminals, cancelled.request_id)
		TestReport.check(cancelled_event.size() == 4 and cancelled_event[2] == GodotChorus.ERR_CANCELLED, "expected one cancelled auxiliary terminal")
		TestReport.check(chorus.export_conversation_history(&"frame-progress").size() == history.size(), "auxiliary work must not change history")
	)

	chorus.stop_all()
	chorus.queue_free()
	await parent.get_tree().process_frame
	await parent.get_tree().process_frame
