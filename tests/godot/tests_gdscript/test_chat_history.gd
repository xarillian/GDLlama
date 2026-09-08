class_name TestChatHistory
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("typed history preserves durable ids", func():
		var session := &"npc_test/dialogue"
		var submit := chorus.generate(ChorusRequest.chat(session, "hello there"))
		TestReport.check(submit.accepted, "expected typed chat request to be accepted")
		await chorus.generation_complete
		var history := chorus.export_conversation_history(session)
		TestReport.check(history.size() == 2, "expected one user and assistant message")
		TestReport.check(history[0] is ChorusMessage and history[0].id == submit.request_message_id and history[1].id == submit.response_message_id, "expected exported identities")
		var edited := chorus.edit_message(session, history[0].id, "edited")
		TestReport.check(edited.ok and chorus.export_conversation_history(session)[0].content == "edited", "expected edit by id")
		TestReport.check(chorus.clear_conversation_history(session).ok, "expected typed clear result")
	)

	await TestReport.run("typed history rejects duplicate and negative ids atomically", func():
		var session := &"atomic-history"
		var original := ChorusMessage.create(8, ChorusRole.USER, "keep")
		TestReport.check(chorus.import_conversation_history(session, [original]).ok, "expected baseline history import")
		var duplicate := ChorusMessage.create(9, ChorusRole.USER, "duplicate")
		TestReport.check(not chorus.import_conversation_history(session, [duplicate, ChorusMessage.create(9, ChorusRole.ASSISTANT, "duplicate")]).ok, "expected duplicate id rejection")
		TestReport.check(not chorus.import_conversation_history(session, [ChorusMessage.create(-1, ChorusRole.USER, "negative")]).ok, "expected negative id rejection")
		var history := chorus.export_conversation_history(session)
		TestReport.check(history.size() == 1 and history[0].id == original.id, "expected rejected imports to preserve prior history")
	)

	await TestReport.run("history rejects invalid enum resources", func():
		var message := ChorusMessage.create(3, 99, "bad")
		TestReport.check(not chorus.import_conversation_history(&"invalid", [message]).ok, "expected invalid role rejection")
	)

	chorus.stop_all()
	chorus.queue_free()
