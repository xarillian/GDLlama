#pragma once

#include <chrono>
#include <string>

void set_process_test_executable_path(std::string path);
const std::string& process_test_executable_path();

bool run_isolated_test_child(const std::string& child_name, std::chrono::milliseconds timeout);
