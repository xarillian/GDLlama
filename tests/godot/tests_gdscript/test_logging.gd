class_name TestLogging
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	chorus.provider = GodotChorus.PROVIDER_LLAMA
	chorus.model_path = ModelPaths.missing_gguf()
	parent.add_child(chorus)

	await TestReport.run("structured load error records reach Godot unchanged", func():
		var records: Array[Array] = []
		chorus.log_record.connect(func(level, message, fields, request_id, session, produced_at):
			records.append([level, message, fields, request_id, session, produced_at])
		)
		await LoadWaiter.load(chorus, false)
		var deadline := Time.get_ticks_msec() + 5000
		var matched: Array = []
		while matched.is_empty() and Time.get_ticks_msec() < deadline:
			for record in records:
				if record[1] == "Failed to load model weights":
					matched = record
					break
			if matched.is_empty():
				await parent.get_tree().process_frame
		TestReport.check(not matched.is_empty(), "expected bounded structured provider load diagnostic delivery")
		if not matched.is_empty():
			var record: Array = matched
			TestReport.check(record[0] == GodotChorus.LOG_ERROR, "expected provider error level")
			TestReport.check(record[1] == "Failed to load model weights", "expected provider message verbatim")
			TestReport.check(record[2].get("path") == chorus.model_path, "expected typed path field unchanged")
			TestReport.check(record[3] == -1 and record[4] == "" and record[5] > 0, "expected unscoped timestamped log")
	)

	chorus.stop_all()
	chorus.queue_free()
