class_name TestCancellation
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	await LoadWaiter.load(chorus)

	await TestReport.run("cancel_request() reaches one valid terminal outcome", func():
		var terminal := {"request_id": -1}
		chorus.generation_complete.connect(func(completed_id: int, _session: StringName, _message_id: int, _content: String, _reasoning: String):
			if completed_id == terminal.request_id:
				terminal["kind"] = "complete"
		)
		chorus.generation_error.connect(func(errored_id: int, _session: String, error_code: int, _message: String):
			if errored_id == terminal.request_id:
				terminal["kind"] = "error"
				terminal["error"] = error_code
		)

		var result := chorus.generate(ChorusRequest.stateless("a rather long sentence to leave a window for cancellation"))
		terminal["request_id"] = result.request_id
		TestReport.check(result.accepted, "expected the cancellation target to be accepted")
		TestReport.check(chorus.is_request_active(result.request_id), "expected the request to be active immediately after generate()")
		TestReport.check(chorus.cancel_request(result.request_id), "expected cancel_request() to return true for a live request")

		var deadline := Time.get_ticks_msec() + 1000
		while not terminal.has("kind") and Time.get_ticks_msec() < deadline:
			await parent.get_tree().process_frame

		TestReport.check(terminal.has("kind"), "expected cancellation to reach a terminal outcome within one second")
		if terminal.has("kind"):
			TestReport.check(terminal.kind == "complete" or terminal.error == GodotChorus.ERR_CANCELLED, "expected cancellation to complete normally or report ERR_CANCELLED")
	)

	chorus.stop_all()
	chorus.queue_free()
