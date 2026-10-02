@tool
extends EditorPlugin

const FIELDS := ["max_tokens", "temperature", "top_k", "top_p", "seed", "frequency_penalty", "presence_penalty", "stop", "show_thinking", "constraint", "chat_template", "provider_options"]
const SELECTOR := "chorus/generation/settings_path"
const DEBOUNCE_SECONDS := 0.2

var _bridge: ChorusProjectSettings
var _snapshot: Dictionary
var _pending := -1.0
var _blocked := false
var _invalid_message := ""
var _conflict: ConfirmationDialog


func _capture() -> Dictionary:
	var values := {SELECTOR: ProjectSettings.get_setting(SELECTOR)}
	for field in FIELDS:
		var key: String = "chorus/generation/" + field
		values[key] = ProjectSettings.get_setting(key).duplicate(true)
	return values


func _enter_tree() -> void:
	_bridge = ChorusProjectSettings.new()
	var initial := _bridge.sync_generation_defaults()
	_snapshot = _capture() if initial.status == ChorusProjectSettings.OK else {SELECTOR: ProjectSettings.get_setting(SELECTOR)}
	_conflict = ConfirmationDialog.new()
	_conflict.title = "Chorus generation settings changed on disk"
	_conflict.dialog_text = "Reload the JSON and discard unsaved edits, or Cancel to keep edits in memory. Saving requires a reload."
	_conflict.ok_button_text = "Reload"
	_conflict.cancel_button_text = "Cancel"
	_conflict.confirmed.connect(_reload_after_conflict)
	add_child(_conflict)
	set_process(true)


func _exit_tree() -> void:
	set_process(false)
	_conflict.queue_free()
	_bridge = null


func _reload_after_conflict() -> void:
	var outcome := _bridge.reload_generation_defaults()
	if outcome.status != ChorusProjectSettings.OK:
		push_error("Chorus settings reload failed: " + outcome.message)
		return
	_blocked = false
	_pending = -1.0
	_snapshot = _capture()


func _process(delta: float) -> void:
	if not Engine.is_editor_hint():
		return
	if ProjectSettings.get_setting(SELECTOR) != _snapshot[SELECTOR]:
		var loaded := _bridge.reload_generation_defaults()
		_pending = -1.0
		if loaded.status != ChorusProjectSettings.OK:
			push_error("Chorus settings import failed: " + loaded.message)
		else:
			_blocked = false
			_invalid_message = ""
			_snapshot = _capture()
		return
	var synced := _bridge.sync_generation_defaults()
	if synced.status != ChorusProjectSettings.OK:
		_pending = -1.0
		if _invalid_message != synced.message:
			_invalid_message = synced.message
			push_error("Chorus settings edit rejected: " + synced.message)
		return
	_invalid_message = ""
	var current := _capture()
	if current != _snapshot:
		_snapshot = current
		if not _blocked:
			_pending = DEBOUNCE_SECONDS
	if _pending < 0.0 or _blocked:
		return
	_pending -= delta
	if _pending > 0.0:
		return
	_pending = -1.0
	var saved := _bridge.save_generation_defaults()
	if saved.status == ChorusProjectSettings.CONFLICT:
		_blocked = true
		_conflict.popup_centered()
	elif saved.status != ChorusProjectSettings.OK:
		push_error("Chorus settings save failed: " + saved.message)
