class_name TestSessionBusy
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("a second generate() on a busy session is rejected", func():
		var session := "npc_test/busy"
		var first_id := chorus.generate({"prompt": "first turn", "session": session})
		TestReport.check(first_id >= 0, "expected the first request on the session to be accepted")

		var second_id := chorus.generate({"prompt": "second turn", "session": session})
		TestReport.check(second_id == -1, "expected a second concurrent request on the same session to be rejected")

		await chorus.generation_complete
	)

	chorus.stop_all()
	chorus.queue_free()
