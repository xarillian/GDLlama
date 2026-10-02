#include "chorus_c/chorus_c_internal.hpp"
#include "host_settings/generation_defaults_codec.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace chorus_c;

namespace {

namespace fs = std::filesystem;

// The C ABI takes UTF-8 paths; a narrow fs::path would read them in Windows' ANSI code page.
fs::path utf8_path(std::string_view text) {
    return fs::path(std::u8string(text.begin(), text.end()));
}

enum class FileState { Missing, Present, Error };

struct FileRead {
    FileState state = FileState::Error;
    std::string bytes;
    std::string error;
};

FileRead read_settings_file(const fs::path& path) {
    std::error_code ec;
    const auto status = fs::status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || (!ec && !fs::exists(status)))
        return {FileState::Missing, {}, {}};
    if (ec)
        return {FileState::Error, {}, "Cannot inspect settings file: " + ec.message()};
    if (!fs::is_regular_file(status))
        return {FileState::Error, {}, "Settings destination is not a regular file."};
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {FileState::Error, {}, "Cannot open settings file for reading."};
    const auto length = fs::file_size(path, ec);
    if (ec || length > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        return {FileState::Error, {}, "Cannot size settings file."};
    std::string bytes(static_cast<size_t>(length), '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size()) || input.peek() != std::char_traits<char>::eof() ||
        input.bad())
        return {FileState::Error, {}, "Cannot read stable settings file."};
    return {FileState::Present, std::move(bytes), {}};
}

int open_exclusive(const fs::path& path) {
#if defined(_WIN32)
    return _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
#endif
}

bool write_all(int fd, std::string_view bytes) {
    while (!bytes.empty()) {
#if defined(_WIN32)
        const auto written = _write(fd, bytes.data(), static_cast<unsigned>(std::min(bytes.size(), size_t{INT_MAX})));
#else
        const auto written = ::write(fd, bytes.data(), bytes.size());
#endif
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0)
            return false;
        bytes.remove_prefix(static_cast<size_t>(written));
    }
    return true;
}

void close_file(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    ::close(fd);
#endif
}

#if defined(_WIN32)
bool same_windows_file(const fs::path& path, const BY_HANDLE_FILE_INFORMATION& owned) {
    const HANDLE current = CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr
    );
    if (current == INVALID_HANDLE_VALUE)
        return false;
    BY_HANDLE_FILE_INFORMATION observed{};
    const bool matched =
        GetFileInformationByHandle(current, &observed) && owned.dwVolumeSerialNumber == observed.dwVolumeSerialNumber &&
        owned.nFileIndexHigh == observed.nFileIndexHigh && owned.nFileIndexLow == observed.nFileIndexLow;
    CloseHandle(current);
    return matched;
}
#endif

bool remove_owned_file(const fs::path& path, int fd) {
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION owned{};
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (!GetFileInformationByHandle(handle, &owned) || !same_windows_file(path, owned))
        return false;
#else
    struct stat owned{}, current{};
    if (::fstat(fd, &owned) != 0 || ::lstat(path.c_str(), &current) != 0 || owned.st_dev != current.st_dev ||
        owned.st_ino != current.st_ino)
        return false;
#endif
    std::error_code ec;
    return fs::remove(path, ec) && !ec;
}

bool finish_owned_file(int fd, const fs::path& path) {
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION owned{};
    const bool identified = GetFileInformationByHandle(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), &owned);
    const bool synced = _commit(fd) == 0;
#else
    struct stat owned{}, current{};
    const bool identified = ::fstat(fd, &owned) == 0;
    const bool synced = ::fsync(fd) == 0;
#endif
    if (!synced)
        remove_owned_file(path, fd);
#if defined(_WIN32)
    const bool closed = _close(fd) == 0;
    if ((!closed || !synced) && identified && same_windows_file(path, owned)) {
        std::error_code ignored;
        fs::remove(path, ignored);
    }
#else
    const bool closed = ::close(fd) == 0;
    if (!closed && identified && ::lstat(path.c_str(), &current) == 0 && owned.st_dev == current.st_dev &&
        owned.st_ino == current.st_ino)
        ::unlink(path.c_str());
#endif
    return synced && closed;
}

void replace_file(const fs::path& from, const fs::path& to, std::error_code& ec) {
#if defined(_WIN32)
    if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
#else
    fs::rename(from, to, ec);
#endif
}

bool make_parent(const fs::path& path, std::string& error) {
    std::error_code ec;
    const auto parent = path.parent_path();
    if (!parent.empty())
        fs::create_directories(parent, ec);
    if (ec) {
        error = "Cannot create settings directory: " + ec.message();
        return false;
    }
    return true;
}

chorus_error inject_defaults(chorus_runtime* rt, Chorus::GenerationDefaults defaults) {
    // Prepare the adapter copy before changing the runtime, so neither side can diverge on allocation failure.
    Chorus::GenerationDefaults next = defaults;
    rt->value.set_generation_defaults(std::move(defaults));
    rt->generation_defaults = std::move(next);
    clear_last_error(rt);
    return CHORUS_OK;
}

chorus_error parse_and_inject(chorus_runtime* rt, std::string_view bytes) {
    auto parsed = chorus_host_settings::parse_generation_defaults(bytes);
    if (!parsed.ok())
        return invalid_request(rt, parsed.path + ": " + parsed.error);
    return inject_defaults(rt, std::move(parsed.defaults));
}

int open_temporary_file(const fs::path& path, fs::path& temporary) {
    static std::atomic<uint64_t> counter{0};
    for (int attempt = 0; attempt < 16; ++attempt) {
        temporary = path;
        temporary += ".chorus-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                     std::to_string(counter.fetch_add(1));
        const int fd = open_exclusive(temporary);
        if (fd >= 0 || errno != EEXIST)
            return fd;
    }
    return -1;
}

chorus_error load_defaults_file(chorus_runtime* rt, const fs::path& path) {
    auto read = read_settings_file(path);
    if (read.state == FileState::Error)
        return unknown_exception(rt, read.error.c_str());
    if (read.state == FileState::Present)
        return parse_and_inject(rt, read.bytes);

    std::string error;
    if (!make_parent(path, error))
        return unknown_exception(rt, error.c_str());
    fs::path temporary;
    const int fd = open_temporary_file(path, temporary);
    if (fd < 0)
        return unknown_exception(rt, "Cannot create temporary settings file.");
    constexpr std::string_view empty = "{\"version\":1,\"generation\":{}}";
    if (!write_all(fd, empty)) {
        remove_owned_file(temporary, fd);
        close_file(fd);
        return unknown_exception(rt, "Cannot write empty settings file.");
    }
    if (!finish_owned_file(fd, temporary))
        return unknown_exception(rt, "Cannot finish empty settings file.");
    std::error_code ec;
    fs::create_hard_link(temporary, path, ec);
    std::error_code cleanup;
    fs::remove(temporary, cleanup);
    if (cleanup)
        return unknown_exception(rt, ("Cannot remove settings temporary file: " + cleanup.message()).c_str());
    if (ec) {
        if (ec == std::errc::file_exists) {
            read = read_settings_file(path);
            if (read.state == FileState::Present)
                return parse_and_inject(rt, read.bytes);
            if (read.state == FileState::Error)
                return unknown_exception(rt, read.error.c_str());
        }
        return unknown_exception(rt, ("Cannot exclusively create settings file: " + ec.message()).c_str());
    }
    return inject_defaults(rt, {});
}

chorus_error save_defaults_file(chorus_runtime* rt, const fs::path& path) {
    const auto encoded = chorus_host_settings::serialize_generation_defaults(rt->generation_defaults);
    if (!encoded.ok())
        return invalid_request(rt, encoded.path + ": " + encoded.error);
    auto before = read_settings_file(path);
    if (before.state == FileState::Error)
        return unknown_exception(rt, before.error.c_str());
    if (before.state == FileState::Present) {
        auto parsed = chorus_host_settings::parse_generation_defaults(before.bytes);
        if (!parsed.ok())
            return invalid_request(rt, parsed.path + ": " + parsed.error);
    }
    std::string error;
    if (!make_parent(path, error))
        return unknown_exception(rt, error.c_str());

    fs::path temporary;
    const int fd = open_temporary_file(path, temporary);
    if (fd < 0)
        return unknown_exception(rt, "Cannot create temporary settings file.");
    if (!write_all(fd, encoded.json)) {
        remove_owned_file(temporary, fd);
        close_file(fd);
        return unknown_exception(rt, "Cannot write temporary settings file.");
    }
    if (!finish_owned_file(fd, temporary))
        return unknown_exception(rt, "Cannot finish temporary settings file.");
    const auto current = read_settings_file(path);
    if (current.state != before.state || (current.state == FileState::Present && current.bytes != before.bytes)) {
        std::error_code ec;
        fs::remove(temporary, ec);
        return unknown_exception(rt, "Settings destination changed before save; reload it first.");
    }
    std::error_code ec;
    replace_file(temporary, path, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        return unknown_exception(rt, ("Cannot replace settings file: " + ec.message()).c_str());
    }
    clear_last_error(rt);
    return CHORUS_OK;
}

} // namespace

extern "C" {

chorus_error chorus_generation_defaults_load_file(chorus_runtime* rt, const char* path) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!path || !*path)
        return invalid_request(rt, "A nonempty settings path is required.");
    try {
        return load_defaults_file(rt, utf8_path(path));
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot load settings file.");
    }
}

chorus_error chorus_generation_defaults_apply_json(chorus_runtime* rt, const char* json, size_t byte_count) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!json || !byte_count)
        return invalid_request(rt, "Nonempty JSON bytes are required.");
    try {
        return parse_and_inject(rt, std::string_view(json, byte_count));
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot apply settings JSON.");
    }
}

chorus_error chorus_generation_defaults_export_json(const chorus_runtime* rt, char** out_json) {
    if (out_json)
        *out_json = nullptr;
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!out_json)
        return invalid_request(rt, "out_json is required.");
    try {
        auto encoded = chorus_host_settings::serialize_generation_defaults(rt->generation_defaults);
        if (!encoded.ok())
            return invalid_request(rt, encoded.path + ": " + encoded.error);
        *out_json = copy_owned_string(encoded.json);
        if (!*out_json)
            return unknown_exception(rt, "Cannot allocate exported JSON.");
        clear_last_error(rt);
        return CHORUS_OK;
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot export settings JSON.");
    }
}

chorus_error chorus_generation_defaults_save_file(chorus_runtime* rt, const char* path) {
    if (!rt)
        return CHORUS_ERR_INVALID_REQUEST;
    if (!path || !*path)
        return invalid_request(rt, "A nonempty settings path is required.");
    try {
        return save_defaults_file(rt, utf8_path(path));
    } catch (const std::exception& error) {
        return unknown_exception(rt, error.what());
    } catch (...) {
        return unknown_exception(rt, "Cannot save settings file.");
    }
}

} // extern "C"
