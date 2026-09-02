extends Node

const LoggingTests := preload("res://tests_gdscript/test_logging.gd")

@export var filter := ""


func _ready() -> void:
	TestReport.filter = filter

	print("======================================")
	print("      CHORUS GODOT TEST SUITE         ")
	print("======================================")

	print("TEST: Model Loading")
	await TestModelLoading.run_tests(self)
	print("TEST: Basic Generation")
	await TestGenerate.run_tests(self)
	print("TEST: Cancellation")
	await TestCancellation.run_tests(self)
	print("TEST: Chat History")
	await TestChatHistory.run_tests(self)
	print("TEST: Session Busy")
	await TestSessionBusy.run_tests(self)
	print("TEST: Basic Properties")
	await TestProperties.run_tests(self)
	print("TEST: Structured Logging")
	await LoggingTests.run_tests(self)

	print("\n======================================")
	if TestReport.failed > 0:
		print("FINAL SUMMARY: %d FAILED, %d PASSED." % [TestReport.failed, TestReport.passed])
	else:
		print("FINAL SUMMARY: ALL TESTS PASSED (%d)." % TestReport.passed)

	get_tree().quit(1 if TestReport.failed > 0 else 0)
