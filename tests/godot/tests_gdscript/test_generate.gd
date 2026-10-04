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
	chorus.generation_complete.connect(func(id, session, message_id, text, reasoning, usage): terminals[id] = [id, session, message_id, text, reasoning, usage])
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

	await TestReport.run("completion carries the provider's token usage", func():
		var request := ChorusRequest.stateless("one two three")
		request.max_tokens = 1
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "expected capped request to be accepted")
		var event: Array = await wait_for_event(chorus, terminals, result.request_id)
		var usage: ChorusGenerationUsage = event[5]
		TestReport.check(usage.prompt_tokens == 3, "expected Echo to count three prompt chunks")
		TestReport.check(usage.cached_prompt_tokens == 0, "expected Echo to reuse nothing")
		TestReport.check(usage.generated_tokens == 1, "expected the cap to bound generated chunks")
	)

	await TestReport.run("empty chat factory session rejects instead of becoming stateless", func():
		var result := chorus.generate(ChorusRequest.chat(StringName(), "hello"))
		TestReport.check(not result.accepted and result.error == GodotChorus.ERR_INVALID_REQUEST, "expected empty chat session rejection")
	)

	await TestReport.run("generation settings selector rejects rooted paths without changing project choices", func():
		var selector := "chorus/generation/settings_path"
		var selected_path: String = ProjectSettings.get_setting(selector)
		var selected_choice: Dictionary = ProjectSettings.get_setting("chorus/generation/max_tokens").duplicate(true)
		var bridge := ChorusProjectSettings.new()
		for rooted in ["res:///outside.json", "res://C:/outside.json", "res://\\\\server\\share\\settings.json"]:
			ProjectSettings.set_setting(selector, rooted)
			var result: Dictionary = bridge.reload_generation_defaults()
			TestReport.check(result.status == ChorusProjectSettings.SOURCE_FAILED, "rooted settings selector must fail: " + rooted)
			TestReport.check(ProjectSettings.get_setting(selector) == selected_path, "failed selection retains the prior path")
			TestReport.check(ProjectSettings.get_setting("chorus/generation/max_tokens") == selected_choice, "failed selection retains prior choices")
	)

	await TestReport.run("shared project defaults freeze and request overrides clear locally", func():
		var settings := ProjectSettings
		settings.set_setting("chorus/generation/max_tokens", {})
		var baseline := chorus.generate(ChorusRequest.stateless("provider fallback"))
		TestReport.check(baseline.accepted, "absence reaches Echo")
		if baseline.accepted:
			var event := await wait_for_event(chorus, terminals, baseline.request_id)
			TestReport.check(event[3].contains("provider fallback"), "provider supplies absent max_tokens")
		settings.set_setting("chorus/generation/max_tokens", {"value": 0})
		var frozen := chorus.generate(ChorusRequest.stateless("frozen"))
		settings.set_setting("chorus/generation/max_tokens", {})
		TestReport.check(frozen.accepted, "snapshot admitted")
		if frozen.accepted:
			var event := await wait_for_event(chorus, terminals, frozen.request_id)
			TestReport.check(event[3] == "", "accepted work retains zero")
		settings.set_setting("chorus/generation/max_tokens", {"value": 0})
		var request := ChorusRequest.stateless("request wins")
		request.max_tokens = 20
		var override := chorus.generate(request)
		TestReport.check(override.accepted, "request override admitted")
		if override.accepted:
			var event := await wait_for_event(chorus, terminals, override.request_id)
			TestReport.check(event[3].contains("request wins"), "request wins")
		request.clear_max_tokens()
		var inherited := chorus.generate(request)
		TestReport.check(inherited.accepted, "clear resumes project choice")
		if inherited.accepted:
			var event := await wait_for_event(chorus, terminals, inherited.request_id)
			TestReport.check(event[3] == "", "project zero restored")
		settings.set_setting("chorus/generation/max_tokens", {})
	)

	await TestReport.run("two nodes share one project choice without scene defaults", func():
		var other: GodotChorus = ChorusNodeScene.instantiate()
		other.provider = GodotChorus.PROVIDER_ECHO
		parent.add_child(other)
		await LoadWaiter.load(other)
		var other_events := {}
		other.generation_complete.connect(func(id, _session, _message_id, text, _reasoning, _usage): other_events[id] = text)
		ProjectSettings.set_setting("chorus/generation/max_tokens", {"value": 0})
		var first := chorus.generate(ChorusRequest.stateless("first node"))
		var second := other.generate(ChorusRequest.stateless("second node"))
		TestReport.check(first.accepted and second.accepted, "both nodes admit with project zero")
		if first.accepted and second.accepted:
			var first_event := await wait_for_event(chorus, terminals, first.request_id)
			var deadline := Time.get_ticks_msec() + 10000
			while not other_events.has(second.request_id) and Time.get_ticks_msec() < deadline:
				await parent.get_tree().process_frame
			TestReport.check(first_event[3] == "" and other_events.get(second.request_id, "missing") == "", "both nodes receive zero")
		ProjectSettings.set_setting("chorus/generation/max_tokens", {})
		other.stop_all()
		other.queue_free()
		await parent.get_tree().process_frame
	)

	await TestReport.run("project constraint and template respect provider boundary", func():
		ProjectSettings.set_setting("chorus/generation/constraint", {"value": {"kind": "gbnf", "source": "root ::= \"ok\""}})
		var grammar := chorus.generate(ChorusRequest.stateless("grammar"))
		TestReport.check(grammar.accepted, "grammar reaches provider")
		if grammar.accepted:
			var event := await wait_for_event(chorus, terminals, grammar.request_id)
			TestReport.check(event[2] == GodotChorus.ERR_UNSUPPORTED_FEATURE, "Echo rejects grammar")
		ProjectSettings.set_setting("chorus/generation/constraint", {"value": {"kind": "unconstrained"}})
		var unconstrained := chorus.generate(ChorusRequest.stateless("unconstrained"))
		TestReport.check(unconstrained.accepted, "explicit unconstrained is present")
		if unconstrained.accepted:
			await wait_for_event(chorus, terminals, unconstrained.request_id)
		ProjectSettings.set_setting("chorus/generation/constraint", {})
		ProjectSettings.set_setting("chorus/generation/chat_template", {"value": ""})
		var chat := ChorusRequest.chat(&"project-template", "hello")
		var invalid := chorus.generate(chat)
		TestReport.check(invalid.accepted, "invalid effective template reaches provider")
		if invalid.accepted:
			var event := await wait_for_event(chorus, terminals, invalid.request_id)
			TestReport.check(event[2] == GodotChorus.ERR_INVALID_REQUEST, "empty effective template fails")
		chat.chat_template = "request template"
		var override := chorus.generate(chat)
		TestReport.check(override.accepted, "request template overrides project template")
		if override.accepted:
			var event := await wait_for_event(chorus, terminals, override.request_id)
			TestReport.check(event[2] == GodotChorus.ERR_UNSUPPORTED_OPTION, "Echo rejects selected request template")
		ProjectSettings.set_setting("chorus/generation/chat_template", {})
	)

	await TestReport.run("batch freezes shared project choices", func():
		ProjectSettings.set_setting("chorus/generation/max_tokens", {"value": 0})
		var batch := chorus.generate_batch([ChorusRequest.stateless("first"), ChorusRequest.stateless("second")])
		ProjectSettings.set_setting("chorus/generation/max_tokens", {})
		for result in batch:
			TestReport.check(result.accepted, "batch entry admitted")
			if result.accepted:
				var event := await wait_for_event(chorus, terminals, result.request_id)
				TestReport.check(event[3] == "", "batch frozen zero")
	)

	await TestReport.run("typed collection metadata names every resource element", func():
		var methods := ClassDB.class_get_method_list("GodotChorus")
		var generate_batch: Dictionary = methods.filter(func(method): return method.name == &"generate_batch")[0]
		var import_history: Dictionary = methods.filter(func(method): return method.name == &"import_conversation_history")[0]
		var export_history: Dictionary = methods.filter(func(method): return method.name == &"export_conversation_history")[0]
		var inject: Dictionary = ClassDB.class_get_property_list("ChorusRequest").filter(func(property): return property.name == &"inject")[0]
		TestReport.check(generate_batch.args[0].hint_string == "ChorusRequest" and generate_batch["return"].hint_string == "ChorusSubmitResult", "expected typed generation batch metadata")
		TestReport.check(import_history.args[1].hint_string == "ChorusMessage" and export_history["return"].hint_string == "ChorusMessage" and inject.hint_string == "ChorusInjectedMessage", "expected typed history and injection metadata")
	)

	await TestReport.run("every session crosses the GodotChorus API as a StringName", func():
		var methods := ClassDB.class_get_method_list("GodotChorus")
		var checked := 0
		for member in ClassDB.class_get_signal_list("GodotChorus") + methods:
			for argument in member.args:
				if argument.name == "session":
					checked += 1
					TestReport.check(argument.type == TYPE_STRING_NAME, "expected StringName session in %s" % member.name)
		TestReport.check(checked > 0, "expected session arguments to check")
		var conversations: Dictionary = methods.filter(func(method): return method.name == &"list_conversations")[0]
		TestReport.check(conversations["return"].hint_string == "StringName", "expected StringName conversation names")
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

	await TestReport.run("request assignment selects and clearing invalid choices restores absence", func():
		var request := ChorusRequest.stateless("request choices")
		request.max_tokens = 0
		request.temperature = 0.0
		request.top_k = 0
		request.top_p = 0.0
		request.seed = 0
		request.frequency_penalty = 0.0
		request.presence_penalty = 0.0
		request.stop = PackedStringArray()
		request.show_thinking = false
		for name in ["max_tokens", "temperature", "top_k", "top_p", "seed", "frequency_penalty", "presence_penalty", "stop", "show_thinking"]:
			TestReport.check(request.call("has_" + name), "assignment must select " + name)
			request.call("clear_" + name)
			TestReport.check(not request.call("has_" + name), "clear must remove " + name)
		request.seed = -1
		TestReport.check(not chorus.generate(request).accepted, "selected invalid seed must reject admission")
		request.clear_seed()
		request.max_tokens = 1 << 40
		TestReport.check(not chorus.generate(request).accepted, "selected out-of-range max_tokens must reject admission")
		request.clear_max_tokens()
		request.top_k = 1 << 40
		TestReport.check(not chorus.generate(request).accepted, "selected out-of-range top_k must reject admission")
		request.clear_top_k()
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "cleared invalid selections must not affect admission")
		if result.accepted:
			await wait_for_event(chorus, terminals, result.request_id)
		request.execution = 99
		TestReport.check(not chorus.generate(request).accepted, "invalid execution must still reject")
	)

	await TestReport.run("request selected zero, empty stop and false reasoning reach Echo", func():
		var request := ChorusRequest.chat(&"request-values", "selected choices")
		request.max_tokens = 0
		request.stop = PackedStringArray()
		request.show_thinking = false
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "zero and empty selections must pass admission")
		if result.accepted:
			var event := await wait_for_event(chorus, terminals, result.request_id)
			TestReport.check(event.size() == 6 and event[3] == "" and event[4] == "", "zero tokens must produce empty output and no reasoning")
		request.stop = PackedStringArray(["selected stop"])
		var unsupported := chorus.generate(request)
		TestReport.check(unsupported.accepted, "unsupported selected stop is a provider error after admission")
		if unsupported.accepted:
			var error_event := await wait_for_event(chorus, terminals, unsupported.request_id)
			TestReport.check(error_event.size() == 4 and error_event[2] == GodotChorus.ERR_UNSUPPORTED_OPTION, "Echo must reject nonempty selected stop")
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

	await TestReport.run("regenerate reads project defaults", func():
		var initial := chorus.generate(ChorusRequest.chat(&"regen-defaults", "original"))
		TestReport.check(initial.accepted, "initial turn admitted")
		if initial.accepted:
			await wait_for_event(chorus, terminals, initial.request_id)
		ProjectSettings.set_setting("chorus/generation/max_tokens", {"value": 0})
		var regenerated := chorus.regenerate(ChorusRequest.regeneration(&"regen-defaults"))
		ProjectSettings.set_setting("chorus/generation/max_tokens", {})
		TestReport.check(regenerated.accepted, "regeneration admitted")
		if regenerated.accepted:
			var event := await wait_for_event(chorus, terminals, regenerated.request_id)
			TestReport.check(event[3] == "", "regeneration retains admission zero")
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

	await TestReport.run("invalid project defaults reject each batch entry", func():
		ProjectSettings.set_setting("chorus/generation/provider_options", {"llama": {"repeat_penalty": Vector2.ONE}})
		var results := chorus.generate_batch([ChorusRequest.stateless("first"), ChorusRequest.stateless("second")])
		TestReport.check(results.size() == 2 and not results[0].accepted and not results[1].accepted, "invalid project options reject batch")
		ProjectSettings.set_setting("chorus/generation/provider_options", {})
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
		TestReport.check(render["return"].class_name == &"ChorusSubmitResult", "preview returns admission metadata")
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
