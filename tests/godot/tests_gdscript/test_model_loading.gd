class_name TestModelLoading
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	
	
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	parent.add_child(chorus)

	await TestReport.run("not loaded before load_model()", func():
		TestReport.check(not chorus.is_loaded(), "expected is_loaded() == false before any load")
	)

	await TestReport.run("unset path rejects admission without a terminal", func():
		var terminals := []
		var on_loaded := func(id, model_id): terminals.append(id)
		var on_failed := func(id, model_id, error, message): terminals.append(id)
		chorus.model_loaded.connect(on_loaded)
		chorus.model_load_failed.connect(on_failed)
		var rejected := chorus.load_model()
		await chorus.get_tree().process_frame
		chorus.model_loaded.disconnect(on_loaded)
		chorus.model_load_failed.disconnect(on_failed)
		TestReport.check(not rejected.accepted and rejected.load_id == -1 and rejected.error == GodotChorus.ERR_INVALID_REQUEST and not rejected.message.is_empty() and terminals.is_empty(), "expected no signal for rejected admission")
	)

	await TestReport.run("load_model() succeeds on a real GGUF", func():
		chorus.model_path = ModelPaths.valid_gguf()
		await LoadWaiter.load(chorus)
		TestReport.check(chorus.is_loaded(), "expected is_loaded() == true after a successful load")
		TestReport.check(chorus.model_path == ModelPaths.valid_gguf(), "expected the absolute filesystem path to remain unchanged")
	)

	await TestReport.run("load_model() preserves ordinary relative filesystem paths", func():
		var working_directory := DirAccess.open(".")
		TestReport.check(working_directory != null, "expected access to the process working directory")
		if working_directory == null:
			return
		var working_parts := working_directory.get_current_dir().split("/", false)
		var model_parts := ModelPaths.valid_gguf().split("/", false)
		var common_parts := 0
		while common_parts < mini(working_parts.size(), model_parts.size()) and working_parts[common_parts] == model_parts[common_parts]:
			common_parts += 1
		var relative_path := "../".repeat(working_parts.size() - common_parts) + "/".join(model_parts.slice(common_parts))
		TestReport.check(not relative_path.is_absolute_path(), "expected an ordinary relative filesystem path")
		chorus.model_path = relative_path
		await LoadWaiter.load(chorus)
		TestReport.check(chorus.model_path == relative_path, "expected the relative path property to remain unchanged")
	)

	await TestReport.run("load_model() resolves a Godot resource path without changing the property", func():
		var resource_path := "res://../" + ModelPaths.VALID_GGUF
		chorus.model_path = resource_path
		await LoadWaiter.load(chorus)
		TestReport.check(chorus.is_loaded(), "expected the resolved model to be ready")
		TestReport.check(chorus.model_path == resource_path, "expected model_path to retain its Godot spelling")
		if chorus.is_loaded():
			var request := ChorusRequest.stateless("Hello")
			request.set_max_tokens(1)
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected the resource-path model to accept generation")
			if result.accepted:
				var event: Array = await chorus.generation_complete
				TestReport.check(event[0] == result.request_id, "expected the resource-path request to complete")
	)

	await TestReport.run("load_model() resolves a loose user file without changing the property", func():
		var user_path := "user://chorus-path-test-%d.gguf" % Time.get_ticks_usec()
		var filesystem_path := ProjectSettings.globalize_path(user_path)
		var copied := DirAccess.copy_absolute(ModelPaths.valid_gguf(), filesystem_path)
		TestReport.check(copied == OK, "expected the user model fixture to copy")
		if copied != OK:
			return
		chorus.model_path = user_path
		var loaded := await LoadWaiter.load(chorus)
		TestReport.check(loaded.get("success", false) and chorus.is_loaded(), "expected a user-relative GGUF to load")
		TestReport.check(chorus.model_path == user_path, "expected model_path to retain its user spelling")
		if loaded.get("success", false):
			var request := ChorusRequest.stateless("Hello")
			request.set_max_tokens(1)
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected the user-path model to accept generation")
			if result.accepted:
				var event: Array = await chorus.generation_complete
				TestReport.check(event[0] == result.request_id, "expected the user-path request to complete")
		chorus.stop_all()
		TestReport.check(DirAccess.remove_absolute(filesystem_path) == OK, "expected the user model fixture to be removed after unload")
	)

	await TestReport.run("load_model() replaces cleanly on a second valid load", func():
		chorus.model_path = ModelPaths.valid_gguf()
		await LoadWaiter.load(chorus)
		TestReport.check(chorus.is_loaded(), "expected is_loaded() == true after reload")
	)

	await TestReport.run("failed reload leaves is_loaded() false (destructive replace)", func():
		chorus.model_path = ModelPaths.missing_gguf()
		var failed := await LoadWaiter.load(chorus, false)
		TestReport.check(failed.get("error") == GodotChorus.ERR_MODEL_LOAD and not failed.get("message", "").is_empty(), "expected identified missing-path diagnostic")
		TestReport.check(not chorus.is_loaded(), "expected is_loaded() == false after a failed reload")
	)


	await TestReport.run("failed replacement retries with real generation", func():
		chorus.model_path = ModelPaths.valid_gguf()
		await LoadWaiter.load(chorus)
		var request := ChorusRequest.stateless("Retry")
		request.set_max_tokens(1)
		var result := chorus.generate(request)
		TestReport.check(result.accepted, "expected retry model to accept generation")
		if result.accepted:
			var event: Array = await chorus.generation_complete
			TestReport.check(event[0] == result.request_id, "expected retry generation")
	)


	await TestReport.run("scene heartbeat during real GGUF replacement", func():
		chorus.model_path = ModelPaths.valid_gguf()
		var terminals := {}
		var progress := []
		var on_progress := func(id, model_id, phase, has_fraction, fraction): progress.append([id, phase, has_fraction, fraction])
		var on_loaded := func(id, model_id): terminals[id] = true
		var on_failed := func(id, model_id, error, message): terminals[id] = false
		chorus.model_load_progress.connect(on_progress)
		chorus.model_loaded.connect(on_loaded)
		chorus.model_load_failed.connect(on_failed)
		var admission := chorus.load_model()
		TestReport.check(admission.accepted and not chorus.is_loaded(), "expected asynchronous replacement admission")
		var frames := 0
		var deadline := Time.get_ticks_msec() + 30000
		while not terminals.has(admission.load_id) and Time.get_ticks_msec() < deadline:
			await chorus.get_tree().process_frame
			frames += 1
		chorus.model_load_progress.disconnect(on_progress)
		chorus.model_loaded.disconnect(on_loaded)
		chorus.model_load_failed.disconnect(on_failed)
		TestReport.check(terminals.get(admission.load_id, false), "expected replacement success")
		for sample in progress:
			TestReport.check(sample[0] == admission.load_id and sample[1] >= ChorusLoadPhase.RELEASING_ENGINE and sample[1] <= ChorusLoadPhase.INITIALIZING_ENGINE and (not sample[2] or (sample[3] >= 0.0 and sample[3] <= 1.0)), "expected identified, valid phase progress")
		print("GGUF replacement heartbeat: %d scene frames before terminal, %d progress samples" % [frames, progress.size()])
	)

	await TestReport.run("identified cancellation drains before retry", func():
		chorus.set_process(false)
		var seen := []
		var failed := func(id, model_id, error, message): seen.append([id, model_id, error, message])
		chorus.model_load_failed.connect(failed)
		var admission := chorus.load_model()
		TestReport.check(admission.accepted and chorus.active_load_id == admission.load_id, "expected active identified load")
		TestReport.check(chorus.cancel_load(admission.load_id), "expected cancellation to be accepted")
		chorus.set_process(true)
		var deadline := Time.get_ticks_msec() + 30000
		while seen.is_empty() and Time.get_ticks_msec() < deadline:
			await chorus.get_tree().process_frame
		chorus.model_load_failed.disconnect(failed)
		TestReport.check(seen.size() == 1 and seen[0][0] == admission.load_id and seen[0][2] == GodotChorus.ERR_CANCELLED, "expected one cancelled terminal")
		await LoadWaiter.load(chorus)
	)

	await TestReport.run("signal handlers retain historical success after stop", func():
		chorus.provider = GodotChorus.PROVIDER_ECHO
		var observed := []
		var first := func(id, model_id): chorus.stop_all()
		var second := func(id, model_id): observed.append([id, model_id, chorus.is_loaded(), chorus.generate(ChorusRequest.stateless("after stop")).accepted])
		chorus.model_loaded.connect(first)
		chorus.model_loaded.connect(second)
		var admission := chorus.load_model()
		var deadline := Time.get_ticks_msec() + 30000
		while observed.is_empty() and Time.get_ticks_msec() < deadline:
			await chorus.get_tree().process_frame
		chorus.model_loaded.disconnect(first)
		chorus.model_loaded.disconnect(second)
		TestReport.check(observed.size() == 1 and observed[0][0] == admission.load_id and not observed[0][2] and not observed[0][3], "expected original success once despite changed state")
	)


	await TestReport.run("old request callback cannot erase published replacement success", func():
		chorus.provider = GodotChorus.PROVIDER_ECHO
		await LoadWaiter.load(chorus)
		chorus.set_process(false)
		var old := chorus.generate(ChorusRequest.stateless("old request"))
		TestReport.check(old.accepted, "expected old request admission")
		var history := []
		var on_old := func(id, _session, _message_id, _text, _reasoning):
			if id == old.request_id:
				history.append("old")
				chorus.stop_all()
		var on_old_error := func(id, _session, _error, _message):
			if id == old.request_id:
				history.append("old")
				chorus.stop_all()
		var on_loaded := func(id, _model_id): history.append(["load", id, chorus.is_loaded()])
		chorus.generation_complete.connect(on_old)
		chorus.generation_error.connect(on_old_error)
		chorus.model_loaded.connect(on_loaded)
		var admission := chorus.load_model()
		for frame in range(20):
			await chorus.get_tree().process_frame
		chorus.set_process(true)
		var deadline := Time.get_ticks_msec() + 30000
		while history.size() < 2 and Time.get_ticks_msec() < deadline:
			await chorus.get_tree().process_frame
		chorus.generation_complete.disconnect(on_old)
		chorus.generation_error.disconnect(on_old_error)
		chorus.model_loaded.disconnect(on_loaded)
		TestReport.check(history.size() == 2 and history[0] == "old" and history[1] == ["load", admission.load_id, false], "expected old terminal before historical load success despite stop")
	)

	await TestReport.run("disabled processing freezes input and delays delivery", func():
		chorus.set_process(false)
		chorus.provider = GodotChorus.PROVIDER_LLAMA
		chorus.model_path = ModelPaths.valid_gguf()
		var observed := []
		var on_loaded := func(id, model_id): observed.append([id, model_id])
		chorus.model_loaded.connect(on_loaded)
		var admission := chorus.load_model()
		chorus.model_path = ModelPaths.missing_gguf()
		chorus.provider = GodotChorus.PROVIDER_ECHO
		for frame in range(5):
			await chorus.get_tree().process_frame
		TestReport.check(observed.is_empty(), "disabled node must not deliver worker signals")
		chorus.set_process(true)
		var deadline := Time.get_ticks_msec() + 30000
		while observed.is_empty() and Time.get_ticks_msec() < deadline:
			await chorus.get_tree().process_frame
		chorus.model_loaded.disconnect(on_loaded)
		TestReport.check(observed.size() == 1 and observed[0] == [admission.load_id, ModelPaths.valid_gguf().get_file().get_basename()], "expected frozen load identity")
	)

	await TestReport.run("freeing an in-flight node fences signals", func():
		var temporary: GodotChorus = ChorusNodeScene.instantiate()
		temporary.provider = GodotChorus.PROVIDER_ECHO
		parent.add_child(temporary)
		temporary.set_process(false)
		var observed := []
		temporary.model_loaded.connect(func(id, model_id): observed.append(id))
		var admission := temporary.load_model()
		TestReport.check(admission.accepted, "expected temporary load admission")
		temporary.queue_free()
		await parent.get_tree().process_frame
		TestReport.check(observed.is_empty() and not is_instance_valid(temporary), "freed node emitted no load signal")
	)

	chorus.queue_free()
