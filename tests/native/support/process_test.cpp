#include "process_test.hpp"

#include "gtest_utils.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <limits>
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;
#else
#error "The isolated test process helper requires Windows or POSIX"
#endif

namespace {

std::string& process_test_executable_path_storage() {
    static std::string path;
    return path;
}

#if defined(_WIN32)

std::string quote_windows_argument(const std::string& argument) {
    std::string quoted = "\"";
    size_t pending_backslashes = 0;
    for (const char character : argument) {
        if (character == '\\') {
            ++pending_backslashes;
            continue;
        }
        if (character == '"') {
            quoted.append(pending_backslashes * 2 + 1, '\\');
            quoted += character;
        } else {
            quoted.append(pending_backslashes, '\\');
            quoted += character;
        }
        pending_backslashes = 0;
    }
    quoted.append(pending_backslashes * 2, '\\');
    quoted += '"';
    return quoted;
}

#elif defined(__unix__) || defined(__APPLE__)

void terminate_and_reap(pid_t child) {
    kill(child, SIGKILL);

    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
}

#endif

} // namespace

void set_process_test_executable_path(std::string path) {
    process_test_executable_path_storage() = std::move(path);
}

const std::string& process_test_executable_path() {
    return process_test_executable_path_storage();
}

bool run_isolated_test_child(const std::string& child_name, std::chrono::milliseconds timeout) {
    const std::string& executable_path = process_test_executable_path();
    if (executable_path.empty()) {
        std::cerr << "[process test] Cannot launch child '" << child_name << "': the test executable path is empty.\n";
        return false;
    }

#if defined(_WIN32)
    std::string command_line =
        quote_windows_argument(executable_path) + " __chorus_child " + quote_windows_argument(child_name);
    STARTUPINFOA startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};
    if (!CreateProcessA(
            executable_path.c_str(),
            command_line.data(),
            nullptr,
            nullptr,
            TRUE,
            0,
            nullptr,
            nullptr,
            &startup_info,
            &process_info
        )) {
        std::cerr << "[process test] Failed to launch child '" << child_name << "': CreateProcess error "
                  << GetLastError() << ".\n";
        return false;
    }

    CloseHandle(process_info.hThread);
    const auto timeout_count = std::clamp<long long>(timeout.count(), 0, std::numeric_limits<DWORD>::max() - 1);
    const DWORD wait_result = WaitForSingleObject(process_info.hProcess, static_cast<DWORD>(timeout_count));
    if (wait_result != WAIT_OBJECT_0) {
        const DWORD wait_error = wait_result == WAIT_FAILED ? GetLastError() : ERROR_SUCCESS;
        TerminateProcess(process_info.hProcess, 1);
        WaitForSingleObject(process_info.hProcess, INFINITE);
        CloseHandle(process_info.hProcess);
        if (wait_result == WAIT_TIMEOUT) {
            std::cerr << "[process test] Child '" << child_name << "' timed out after " << timeout.count() << " ms.\n";
        } else if (wait_result == WAIT_FAILED) {
            std::cerr << "[process test] Failed while waiting for child '" << child_name
                      << "': WaitForSingleObject error " << wait_error << ".\n";
        } else {
            std::cerr << "[process test] Unexpected wait result " << wait_result << " for child '" << child_name
                      << "'.\n";
        }
        return false;
    }

    DWORD exit_code = 1;
    if (!GetExitCodeProcess(process_info.hProcess, &exit_code)) {
        const DWORD exit_error = GetLastError();
        CloseHandle(process_info.hProcess);
        std::cerr << "[process test] Failed to read exit status for child '" << child_name
                  << "': GetExitCodeProcess error " << exit_error << ".\n";
        return false;
    }
    CloseHandle(process_info.hProcess);
    if (exit_code != 0) {
        std::cerr << "[process test] Child '" << child_name << "' exited with status " << exit_code << ".\n";
        return false;
    }
    return true;
#else
    std::string child_marker = "__chorus_child";
    std::string mutable_child_name = child_name;
    std::string mutable_executable_path = executable_path;
    char* child_argv[] = {mutable_executable_path.data(), child_marker.data(), mutable_child_name.data(), nullptr};

    pid_t child = 0;
    const int spawn_error = posix_spawnp(&child, executable_path.c_str(), nullptr, nullptr, child_argv, environ);
    if (spawn_error != 0) {
        std::cerr << "[process test] Failed to launch child '" << child_name << "': posix_spawnp error " << spawn_error
                  << ".\n";
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    int status = 0;
    for (bool first_poll = true; first_poll || std::chrono::steady_clock::now() < deadline; first_poll = false) {
        const pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            if (WIFEXITED(status)) {
                const int exit_code = WEXITSTATUS(status);
                if (exit_code == 0)
                    return true;
                std::cerr << "[process test] Child '" << child_name << "' exited with status " << exit_code << ".\n";
            } else if (WIFSIGNALED(status)) {
                std::cerr << "[process test] Child '" << child_name << "' terminated by signal " << WTERMSIG(status)
                          << ".\n";
            } else {
                std::cerr << "[process test] Child '" << child_name << "' ended with unexpected wait status " << status
                          << ".\n";
            }
            return false;
        }
        if (result < 0 && errno != EINTR) {
            const int wait_error = errno;
            if (wait_error != ECHILD)
                terminate_and_reap(child);
            std::cerr << "[process test] Failed while waiting for child '" << child_name << "': waitpid error "
                      << wait_error << ".\n";
            return false;
        }
        if (std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    terminate_and_reap(child);
    std::cerr << "[process test] Child '" << child_name << "' timed out after " << timeout.count() << " ms.\n";
    return false;
#endif
}

namespace {

TEST(ProcessTest, Isolated_child_reports_nonzero_exit) {
    std::ostringstream captured_error;
    auto* original_error_buffer = std::cerr.rdbuf(captured_error.rdbuf());
    const bool succeeded = run_isolated_test_child("unknown_isolated_child", std::chrono::seconds(1));
    std::cerr.rdbuf(original_error_buffer);

    ASSERT_TRUE(!succeeded);
    ASSERT_TRUE(captured_error.str().find("unknown_isolated_child") != std::string::npos);
    ASSERT_TRUE(captured_error.str().find("64") != std::string::npos);
}

} // namespace
