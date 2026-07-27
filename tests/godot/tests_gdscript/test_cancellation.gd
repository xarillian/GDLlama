class_name TestCancellation
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.backend = GodotChorus.BACKEND_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("cancel_request() yields generation_error with ERR_CANCELLED", func():
		var request_id := chorus.generate({"prompt": "a rather long sentence to leave a window for cancellation"})
		TestReport.check(chorus.is_request_active(request_id), "expected the request to be active immediately after generate()")

		TestReport.check(chorus.cancel_request(request_id), "expected cancel_request() to return true for a live request")

		var event: Array = await chorus.generation_error
		var errored_id: int = event[0]
		var error_code: int = event[2]

		TestReport.check(errored_id == request_id, "expected generation_error's request_id to match the cancelled request")
		TestReport.check(error_code == GodotChorus.ERR_CANCELLED, "expected error_code == ERR_CANCELLED")
	)

	# TODO: cancel_request() on an unknown/already-finished id returns false
	# TODO: active_request_for_session() reflects the live request, then -1 after it drains
	# TODO: stop_all() cancels every live request across sessions

	chorus.stop_all()
	chorus.queue_free()
