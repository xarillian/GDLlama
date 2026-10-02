class_name LoadWaiter
extends RefCounted


static func load(chorus: GodotChorus, expect_success: bool = true) -> Dictionary:
	var terminal := {}
	var on_loaded := func(id: int, model_id: String): terminal[id] = {"success": true, "model_id": model_id}
	var on_failed := func(id: int, model_id: String, error: int, message: String): terminal[id] = {"success": false, "model_id": model_id, "error": error, "message": message}
	chorus.model_loaded.connect(on_loaded)
	chorus.model_load_failed.connect(on_failed)
	var admission := chorus.load_model()
	if admission.accepted:
		var deadline := Time.get_ticks_msec() + 30000
		while not terminal.has(admission.load_id) and Time.get_ticks_msec() < deadline:
			await chorus.get_tree().process_frame
	chorus.model_loaded.disconnect(on_loaded)
	chorus.model_load_failed.disconnect(on_failed)
	TestReport.check(admission.accepted, "expected identified load admission: %s" % admission.message)
	TestReport.check(not admission.accepted or terminal.has(admission.load_id), "load %d timed out" % admission.load_id)
	var outcome: Dictionary = terminal.get(admission.load_id, {})
	if admission.accepted and not outcome.is_empty():
		TestReport.check(outcome.success == expect_success, "load %d: %s" % [admission.load_id, outcome.get("message", "unexpected terminal")])
	outcome["admission"] = admission
	return outcome
