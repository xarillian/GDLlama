# ============================================================================
# Basic Configuration
# ============================================================================
# Minimum of 3.14 for:
# - `add_link_options`
# - implicit target dirs
cmake_minimum_required(VERSION 3.14)
project("godot_llm" C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# For Windows, link the C/C++ runtime library to avoid VC++ redistributables
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# For macOS, build a universal binary that runs on both x86_64 and arm64 (Apple Silicon)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures for OSX" FORCE)
endif ()

# ============================================================================
# Platform and Compiler Settings
# ============================================================================
add_definitions(-DSQLITE_CORE)

if (MSVC)
  # Disable MSVC fopen warning
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  # Fix debug build iterator bound check bug https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
  add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
endif()


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # A small attempt to keep the build output clean.
    add_compile_options(
      /vmg
      /vmv
      -Wno-c++98-compat
      -Wno-c++98-compat-pedantic
      -Wno-double-promotion
      -Wno-old-style-cast
      -Wno-implicit-float-conversion
      -Wno-unsafe-buffer-usage
      -Wno-unused-command-line-argument
      -Wno-newline-eof
      -Wno-float-equal
      -Wno-unknown-argument
      -Wno-shorten-64-to-32
      -Wno-sign-conversion
      -Wno-float-conversion
      -Wno-cast-align
      -Wno-cast-qual
      -Wno-extra-semi-stmt
      -Wno-cast-function-type-strict
      -Wno-implicit-int-conversion
      -Wno-missing-prototypes
      -Wno-switch-default
      -Wno-exit-time-destructors
      -Wno-covered-switch-default
      -Wno-global-constructors
      "$<$<CONFIG:Debug>:-fstandalone-debug>"
    )
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fPIC)
endif()

# ============================================================================
# Dependencies
# ============================================================================

add_subdirectory(godot-cpp)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

# ============================================================================
# Target Library (libgodot_llm)
# ============================================================================
set(src_files
  "src/sqlite3.c"
  "src/sqlite-vec.c"
  "src/conversion.cpp"
  "src/gdembedding.cpp"
  "src/gdllama.cpp"
  "src/embedding_runner.cpp"
  "src/llama_runner.cpp"
  "src/llm_db.cpp"
  "src/register_types.cpp"
  "src/logging_utils.cpp"
)

if (APPLE)
  set(ARCH_NAME "universal")
else()
  set(ARCH_NAME ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(LIB_NAME libgodot_llm.${CMAKE_SYSTEM_NAME}.${CMAKE_BUILD_TYPE}.${ARCH_NAME})
string(TOLOWER ${LIB_NAME} LIB_NAME)

set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${LIB_NAME} SHARED ${src_files})

target_link_libraries(${LIB_NAME} PRIVATE godot-cpp common llama)

target_include_directories(${LIB_NAME} PRIVATE
  src
  llama.cpp
  llama.cpp/common
  llama.cpp/src
  llama.cpp/vendor
)

# If the build type isn't DEBUG, we don't want to see debug logs
target_compile_definitions(${LIB_NAME} PRIVATE
  $<$<NOT:$<CONFIG:Debug>>:LOG_DISABLE_LOGS>
)

# ============================================================================
# Install Rules
# ============================================================================
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

if (LLAMA_VULKAN OR LLAMA_METAL)
  # GPU Build
  set(LLM_LIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/gpu/addons/godot_llm)
else()
  # CPU Build
  set(LLM_LIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/cpu/addons/godot_llm)
endif()

if (WIN32)
  install(TARGETS ${LIB_NAME}
    RUNTIME
    DESTINATION ${LLM_LIB_INSTALL_DIR}/bin
  )
else()
  install(TARGETS ${LIB_NAME}
    LIBRARY
    DESTINATION ${LLM_LIB_INSTALL_DIR}/bin
  )
endif()

install(FILES "godot_llm.gdextension" DESTINATION ${LLM_LIB_INSTALL_DIR})

# ============================================================================
# Testing Targets
# ============================================================================
# --- llama_runner_test ---
set(llama_runnner_test_files
  "test/src/llama_runner_test.cpp"
)
add_executable(llama_runner_test ${llama_runnner_test_files} ${src_files})
target_link_libraries(llama_runner_test PRIVATE godot-cpp common llama)
target_include_directories(llama_runner_test PRIVATE
  src
  llama.cpp
  llama.cpp/common
  llama.cpp/src
  llama.cpp/vendor
  godot-cpp
)
target_compile_definitions(llama_runner_test PRIVATE
  SQLITE_CORE
  $<$<CXX_COMPILER_ID:MSVC>:-D_CRT_SECURE_NO_WARNINGS>
  $<$<CXX_COMPILER_ID:MSVC>:-D_ITERATOR_DEBUG_LEVEL=0>
)
add_test(
  NAME llama_runner_test
  COMMAND llama_runner_test
)


# --- embedding_runner_test ---
set(embedding_runnner_test_files
  "test/src/embedding_runner_test.cpp"
)
add_executable(embedding_runner_test ${embedding_runnner_test_files} ${src_files})
target_link_libraries(embedding_runner_test PRIVATE godot-cpp common llama)
target_include_directories(embedding_runner_test PRIVATE
  src
  llama.cpp
  llama.cpp/common
  llama.cpp/src
  llama.cpp/vendor
  godot-cpp
)
target_compile_definitions(llama_runner_test PRIVATE
  SQLITE_CORE
  $<$<CXX_COMPILER_ID:MSVC>:-D_CRT_SECURE_NO_WARNINGS>
  $<$<CXX_COMPILER_ID:MSVC>:-D_ITERATOR_DEBUG_LEVEL=0>
)
add_test(
  NAME embedding_runner_test
  COMMAND embedding_runner_test
)

# --- sqlite_test ---
set(sqlite_test_files
  "test/src/sqlite_test.cpp"
  "src/sqlite3.c"
  "src/sqlite-vec.c"
)
add_executable(sqlite_test ${sqlite_test_files})
target_link_libraries(sqlite_test)
target_compile_definitions(sqlite_test PRIVATE
  SQLITE_CORE
  $<$<CXX_COMPILER_ID:MSVC>:-D_CRT_SECURE_NO_WARNINGS>
  $<$<CXX_COMPILER_ID:MSVC>:-D_ITERATOR_DEBUG_LEVEL=0>
)
add_test(
  NAME sqlite_test
  COMMAND sqlite_test
)
