extends Node

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


func _ready() -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	add_child(chorus)
	await LoadWaiter.load(chorus)

	var progress := []
	var terminals := []
	chorus.model_load_progress.connect(func(id, model_id, phase, has_fraction, fraction): progress.append([id, phase, has_fraction, fraction]))
	chorus.model_loaded.connect(func(id, model_id): terminals.append([id, true]))
	chorus.model_load_failed.connect(func(id, model_id, error, message): terminals.append([id, false, error, message]))
	chorus.call("test_hold_next_retirement")
	chorus.provider = GodotChorus.PROVIDER_LLAMA
	chorus.model_path = ModelPaths.valid_gguf()
	var admission := chorus.load_model()
	TestReport.check(admission.accepted, "expected held real-adapter replacement admission")
	var deadline := Time.get_ticks_msec() + 30000
	while not chorus.call("test_retirement_held") and Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
	var held: bool = chorus.call("test_retirement_held")
	var frames := 0
	if held:
		for frame in range(6):
			await get_tree().process_frame
			frames += 1
	var active := chorus.active_load_id == admission.load_id
	var ready := chorus.is_loaded()
	var rejected := not chorus.generate(ChorusRequest.stateless("too soon")).accepted
	var progress_seen := progress.any(func(sample): return sample[0] == admission.load_id and sample[1] == ChorusLoadPhase.RELEASING_ENGINE and not sample[2])
	var no_terminal := terminals.is_empty()
	var before_cancel := Time.get_ticks_msec()
	var cancelled := chorus.cancel_load(admission.load_id)
	var cancel_elapsed := Time.get_ticks_msec() - before_cancel
	chorus.call("test_release_retirement")
	deadline = Time.get_ticks_msec() + 30000
	while terminals.is_empty() and Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
	TestReport.check(held and frames == 6 and active and not ready and rejected and progress_seen and no_terminal, "host frames and phase progress must advance while lifecycle retirement is held")
	TestReport.check(cancelled and cancel_elapsed < 1000, "cancellation admission must not wait for retirement")
	TestReport.check(terminals.size() == 1 and terminals[0][0] == admission.load_id and not terminals[0][1] and terminals[0][2] == GodotChorus.ERR_CANCELLED, "expected exactly one identified cancellation terminal")
	await get_tree().process_frame
	TestReport.check(terminals.size() == 1, "no duplicate terminal on later poll")
	print("CONTROLLED HOST GATE: %d live frames, %d cancellation admission ms, %d progress samples" % [frames, cancel_elapsed, progress.size()])
	chorus.queue_free()
	get_tree().quit(1 if TestReport.failed > 0 else 0)
