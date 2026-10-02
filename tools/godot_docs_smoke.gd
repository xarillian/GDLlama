@tool
extends EditorPlugin


func _enter_tree() -> void:
	call_deferred("_exercise_help")


func _exercise_help() -> void:
	await get_tree().create_timer(1.0).timeout
	var editor := EditorInterface.get_script_editor()
	var search := editor.find_child("*EditorHelpSearch*", true, false)
	if search == null or not search.has_signal("go_to_help"):
		_fail("Cannot find the editor help navigation control")
		return
	if DirAccess.make_dir_recursive_absolute("res://.help-output") != OK:
		_fail("Cannot create help evidence directory")
		return
	EditorInterface.set_main_screen_editor("Script")
	var manifest: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://help_expectations.json"))
	for class_name_text in manifest:
		if not ClassDB.class_exists(class_name_text):
			_fail("Class is not registered: " + class_name_text)
			return
		search.emit_signal("go_to_help", "class_name:" + class_name_text)
		await get_tree().create_timer(0.3).timeout
		var page := editor.find_child(class_name_text, true, false)
		if page == null:
			_fail("Help page did not open: " + class_name_text)
			return
		var text := _help_text(page)
		var output := FileAccess.open("res://.help-output/" + class_name_text + ".txt", FileAccess.WRITE)
		if output == null:
			_fail("Cannot save rendered help: " + class_name_text)
			return
		output.store_string(text)
		output.close()
		if "There is currently no description" in text:
			_fail("Undocumented entry on help page: " + class_name_text)
			return
		var normalized := _normalize(text)
		for expected in manifest[class_name_text]:
			if not _normalize(expected) in normalized:
				_fail("Missing rendered help in " + class_name_text + ": " + expected)
				return
		if DisplayServer.get_name() != "headless":
			RenderingServer.force_draw()
			var image := EditorInterface.get_base_control().get_viewport().get_texture().get_image()
			if image.save_png("res://.help-output/" + class_name_text + ".png") != OK:
				_fail("Could not capture rendered help: " + class_name_text)
				return
		print("CHORUS HELP RENDERED: " + class_name_text)
	print("CHORUS EDITOR DOCUMENTATION PASSED")
	EditorInterface.get_base_control().get_parent().notification.call_deferred(NOTIFICATION_WM_CLOSE_REQUEST)


func _help_text(node: Node) -> String:
	var text := ""
	if node is RichTextLabel:
		text += node.get_parsed_text() + "\n"
	for child in node.get_children():
		text += _help_text(child)
	return text


func _normalize(text: String) -> String:
	return " ".join(text.replace("\u2060", "").replace("\r", " ").replace("\t", " ").replace("\n", " ").split(" ", false))


func _fail(message: String) -> void:
	push_error(message)
	get_tree().quit(1)
