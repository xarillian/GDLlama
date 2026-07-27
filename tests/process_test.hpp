#pragma once

#include <chrono>
#include <string>

bool run_isolated_test_child(const std::string& child_name, std::chrono::milliseconds timeout);
