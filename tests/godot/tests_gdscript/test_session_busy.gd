class_name TestSessionBusy
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	await LoadWaiter.load(chorus)

	await TestReport.run("a second generate() on a busy session is rejected", func():
		var session := "npc_test/busy"
		var first_id := chorus.generate(ChorusRequest.chat(session, "first turn")).request_id
		TestReport.check(first_id >= 0, "expected the first request on the session to be accepted")

		var second_id := chorus.generate(ChorusRequest.chat(session, "second turn")).request_id
		TestReport.check(second_id == -1, "expected a second concurrent request on the same session to be rejected")

		var reset := chorus.reset_context()
		TestReport.check(not reset.ok and reset.error == GodotChorus.ERR_SESSION_BUSY, "expected reset_context to return a typed busy diagnostic")
		await chorus.generation_complete
	)

	chorus.stop_all()
	chorus.queue_free()
