class_name TestModelLoading
extends RefCounted

const ChorusNodeScene := preload("res://tests_gdscript/chorus_node.tscn")


static func run_tests(parent: Node) -> void:
	
	
	var chorus: GodotChorus = ChorusNodeScene.instantiate()
	parent.add_child(chorus)

	await TestReport.run("not loaded before load_model()", func():
		TestReport.check(not chorus.is_loaded(), "expected is_loaded() == false before any load")
	)

	await TestReport.run("load_model() succeeds on a real GGUF", func():
		chorus.model_path = ModelPaths.valid_gguf()
		TestReport.check(chorus.load_model(), "expected load_model() == true")
		TestReport.check(chorus.is_loaded(), "expected is_loaded() == true after a successful load")
	)

	await TestReport.run("load_model() replaces cleanly on a second valid load", func():
		TestReport.check(chorus.load_model(), "expected reload to return true")
		TestReport.check(chorus.is_loaded(), "expected is_loaded() == true after reload")
	)

	await TestReport.run("failed reload leaves is_loaded() false (destructive replace)", func():
		chorus.model_path = ModelPaths.missing_gguf()
		TestReport.check(not chorus.load_model(), "expected load_model() == false on a missing path")
		TestReport.check(not chorus.is_loaded(), "expected is_loaded() == false after a failed reload")
	)

	chorus.queue_free()
