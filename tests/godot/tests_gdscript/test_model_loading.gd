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
		TestReport.check(chorus.model_path == ModelPaths.valid_gguf(), "expected the absolute filesystem path to remain unchanged")
	)

	await TestReport.run("load_model() preserves ordinary relative filesystem paths", func():
		var working_directory := DirAccess.open(".")
		TestReport.check(working_directory != null, "expected access to the process working directory")
		if working_directory == null:
			return
		var working_parts := working_directory.get_current_dir().split("/", false)
		var model_parts := ModelPaths.valid_gguf().split("/", false)
		var common_parts := 0
		while common_parts < mini(working_parts.size(), model_parts.size()) and working_parts[common_parts] == model_parts[common_parts]:
			common_parts += 1
		var relative_path := "../".repeat(working_parts.size() - common_parts) + "/".join(model_parts.slice(common_parts))
		TestReport.check(not relative_path.is_absolute_path(), "expected an ordinary relative filesystem path")
		chorus.model_path = relative_path
		TestReport.check(chorus.load_model(), "expected the model path to remain relative to the process working directory")
		TestReport.check(chorus.model_path == relative_path, "expected the relative path property to remain unchanged")
	)

	await TestReport.run("load_model() resolves a Godot resource path without changing the property", func():
		var resource_path := "res://../" + ModelPaths.VALID_GGUF
		chorus.model_path = resource_path
		TestReport.check(chorus.load_model(), "expected a resource-relative GGUF to load")
		TestReport.check(chorus.is_loaded(), "expected the resolved model to be ready")
		TestReport.check(chorus.model_path == resource_path, "expected model_path to retain its Godot spelling")
		if chorus.is_loaded():
			var request := ChorusRequest.stateless("Hello")
			request.set_max_tokens(0)
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected the resource-path model to accept generation")
			if result.accepted:
				var event: Array = await chorus.generation_complete
				TestReport.check(event[0] == result.request_id, "expected the resource-path request to complete")
	)

	await TestReport.run("load_model() resolves a loose user file without changing the property", func():
		var user_path := "user://chorus-path-test-%d.gguf" % Time.get_ticks_usec()
		var filesystem_path := ProjectSettings.globalize_path(user_path)
		var copied := DirAccess.copy_absolute(ModelPaths.valid_gguf(), filesystem_path)
		TestReport.check(copied == OK, "expected the user model fixture to copy")
		if copied != OK:
			return
		chorus.model_path = user_path
		var loaded := chorus.load_model()
		TestReport.check(loaded and chorus.is_loaded(), "expected a user-relative GGUF to load")
		TestReport.check(chorus.model_path == user_path, "expected model_path to retain its user spelling")
		if loaded:
			var request := ChorusRequest.stateless("Hello")
			request.set_max_tokens(0)
			var result := chorus.generate(request)
			TestReport.check(result.accepted, "expected the user-path model to accept generation")
			if result.accepted:
				var event: Array = await chorus.generation_complete
				TestReport.check(event[0] == result.request_id, "expected the user-path request to complete")
		chorus.stop_all()
		TestReport.check(DirAccess.remove_absolute(filesystem_path) == OK, "expected the user model fixture to be removed after unload")
	)

	await TestReport.run("load_model() replaces cleanly on a second valid load", func():
		chorus.model_path = ModelPaths.valid_gguf()
		TestReport.check(chorus.load_model(), "expected reload to return true")
		TestReport.check(chorus.is_loaded(), "expected is_loaded() == true after reload")
	)

	await TestReport.run("failed reload leaves is_loaded() false (destructive replace)", func():
		chorus.model_path = ModelPaths.missing_gguf()
		TestReport.check(not chorus.load_model(), "expected load_model() == false on a missing path")
		TestReport.check(not chorus.is_loaded(), "expected is_loaded() == false after a failed reload")
	)

	chorus.queue_free()
