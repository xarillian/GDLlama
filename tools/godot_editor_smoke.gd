@tool
extends EditorPlugin

const CHOICES := {"max_tokens": 128, "temperature": 0.25, "show_thinking": false, "seed": "12345"}
const DOCUMENT := "res://chorus/settings.json"


func _enter_tree() -> void:
	call_deferred("_exercise_settings")


func _exercise_settings() -> void:
	if not ClassDB.can_instantiate("GodotChorus"):
		_fail("GodotChorus is not registered")
		return
	var chorus := ClassDB.instantiate("GodotChorus") as Node
	add_child(chorus)
	chorus.queue_free()
	await get_tree().process_frame

	if "--chorus-read-settings" in OS.get_cmdline_user_args():
		for field in CHOICES:
			if ProjectSettings.get_setting("chorus/generation/" + field) != {"value": CHOICES[field]}:
				_fail("Reopened setting differs: " + field)
				return
		print("CHORUS EDITOR SETTINGS RESTORED")
		get_tree().quit()
		return

	for field in CHOICES:
		ProjectSettings.set_setting("chorus/generation/" + field, {"value": CHOICES[field]})
	var deadline := Time.get_ticks_msec() + 10000
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		var document = JSON.parse_string(FileAccess.get_file_as_string(DOCUMENT))
		if document is not Dictionary:
			continue
		var generation: Dictionary = document.get("generation", {})
		if generation.get("max_tokens") == 128 and generation.get("temperature") == 0.25 and generation.get("show_thinking") == false and generation.get("seed") == 12345:
			print("CHORUS EDITOR SETTINGS SAVED")
			get_tree().quit()
			return
	_fail("The Chorus editor plugin did not save settings")


func _fail(message: String) -> void:
	push_error(message)
	get_tree().quit(1)
