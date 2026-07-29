class_name TestChatHistory
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("a sessioned turn appears in exported history", func():
		var session := "npc_test/dialogue"
		var request_id := chorus.generate({"prompt": "hello there", "session": session})
		TestReport.check(request_id >= 0, "expected generate() to accept a sessioned request")

		await chorus.generation_complete

		var history := chorus.export_conversation_history(session)
		TestReport.check(history.size() == 2, "expected one user + one assistant message in history")
		TestReport.check(history[0]["role"] == "user", "expected the first message to be the user turn")
		TestReport.check(history[1]["role"] == "assistant", "expected the second message to be the assistant reply")

		TestReport.check(chorus.clear_conversation_history(session), "expected clear_conversation_history() to succeed")
		TestReport.check(chorus.export_conversation_history(session).is_empty(), "expected history to be empty after clearing")
	)

	# TODO: edit_message index matrix: -1 and -size succeed, -size-1 and size reject
	# TODO: list_conversations() reflects active sessions
	# TODO: reset_context() clears every session at once
	# TODO: last_turn_outcome() reports TURN_COMPLETED after a normal turn, TURN_CANCELLED after cancel_request()
	# TODO: regenerate(session) replaces the last assistant line

	chorus.stop_all()
	chorus.queue_free()
