class_name TestGenerate
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("stateless generate() fires generation_complete with a matching id", func():
		var request_id := chorus.generate({"prompt": "hello world"})
		TestReport.check(request_id >= 0, "expected generate() to return a request id >= 0")

		var event: Array = await chorus.generation_complete
		var completed_id: int = event[0]
		var text: String = event[2]

		TestReport.check(completed_id == request_id, "expected generation_complete's request_id to match generate()'s return value")
		TestReport.check(not text.is_empty(), "expected Echo to return non-empty text")
	)

	await TestReport.run("session identifies which request a signal belongs to", func():
		var expected_session_1 = "test-1"
		var expected_session_2 = "test-2"
		
		var first_id := chorus.generate({"prompt": "hello 1", "session": expected_session_1})
		var second_id := chorus.generate({"prompt": "hello 2", "session": expected_session_2})
		TestReport.check(first_id >= 0, "expected the first sessioned request to be accepted")
		TestReport.check(second_id >= 0, "expected the second sessioned request to be accepted")

		var session_by_request_id := {}
		for _i in range(2):
			var event: Array = await chorus.generation_complete
			session_by_request_id[event[0]] = event[1]

		TestReport.check(session_by_request_id.get(first_id, "") == expected_session_1, "expected the first request's signal to report session" + expected_session_1)
		TestReport.check(session_by_request_id.get(second_id, "") == expected_session_2, "expected the second request's signal to report session" + expected_session_2)
	)

	# TODO: missing 'prompt' key returns -1, no signal fires
	# TODO: stream: true fires token_generated one or more times before generation_complete
	# TODO: an unknown provider_options key is rejected (generate() returns -1)
	# TODO: regenerate(session) with no prior turn fails cleanly

	chorus.stop_all()
	chorus.queue_free()
