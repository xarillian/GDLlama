class_name TestReport
extends RefCounted

static var passed := 0
static var failed := 0
static var filter := ""


static func matches(name: String) -> bool:
	return filter.is_empty() or name.contains(filter)


static func run(name: String, test_func: Callable) -> void:
	if not matches(name):
		return

	var failed_before := failed
	await test_func.call()
	if failed == failed_before:
		passed += 1
		print("[PASSED] %s" % name)


static func check(condition: bool, message: String) -> void:
	if not condition:
		failed += 1
		print("[FAILED] %s" % message)
