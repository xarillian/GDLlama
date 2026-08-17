class_name TestLogging
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_ECHO
	parent.add_child(chorus)
	chorus.load_model()

	await TestReport.run("structured log records reach Godot unchanged", func():
		var request_id := chorus.generate({"prompt": "hello", "temperature": 0.5})
		TestReport.check(request_id >= 0, "expected generate() to accept the request")

		var record: Array = await chorus.log_record
		TestReport.check(record[0] == GodotChorus.LOG_WARN, "expected the Echo diagnostic to retain its level")
		TestReport.check(
			record[1] == "Ignoring content controls; echoed output makes no content claims",
			"expected the provider's message verbatim"
		)
		TestReport.check(record[2].get("controls") == "temperature", "expected typed fields to reach Godot")
	)

	chorus.stop_all()
	chorus.queue_free()
