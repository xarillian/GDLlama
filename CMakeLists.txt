# ============================================================================
# Basic Configuration
# ============================================================================
message("Starting CMake configuration for godot_llm")
# Minimum of 3.14 for:
# - `add_link_options`
# - implicit target dirs
cmake_minimum_required(VERSION 3.14)
project("godot_llm" C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")

# For macOS, build a universal binary that runs on both x86_64 and arm64 (Apple Silicon)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures for OSX" FORCE)
endif ()

# ============================================================================
# Platform and Compiler Settings
# ============================================================================
if (MSVC)
  # Disable MSVC fopen warning
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  # Fix debug build iterator bound check bug:
  # https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
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
      -Wno-unused-function
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

    if (DEFINED ENV{CI})
      message(STATUS "In a CI environment, adding -Wno-character-conversion")
      add_compile_options(-Wno-character-conversion)
    endif()
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fPIC)
endif()

# ============================================================================
# Dependencies
# ============================================================================

add_subdirectory(godot-cpp)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${godot-cpp_SOURCE_DIR}/cmake")
include(GodotCPPModule)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

# ===========================================================================
# Logging Configuration
# ===========================================================================
option(GDLLAMA_DEBUG_LOGGING "Enable verbose debug level logging" OFF)

# ============================================================================
# Target Library (libgodot_llm)
# ============================================================================
set(src_files
  "src/conversion.cpp"
  "src/gdllama.cpp"
  "src/llama_runner.cpp"
  "src/register_types.cpp"
  "src/logging_utils.cpp"
  "src/llama_controller.cpp"
  "src/llama_state.cpp"
)

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_LOWER)

set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

if (APPLE)
  set(ARCH_SUFFIX "universal")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(ARCH_SUFFIX "amd64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ARCH_SUFFIX "aarch64")
else()
  set(ARCH_SUFFIX ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(LIB_NAME godot_llm)

add_library(${LIB_NAME} SHARED ${src_files})

if (GDLLAMA_DEBUG_LOGGING)
  message(STATUS "Enabling GDLlama debug logging")
  target_compile_definitions(${LIB_NAME} PRIVATE GDLLAMA_ENABLE_DEBUG_LOGS)
endif()

if(APPLE)
  set_target_properties(${LIB_NAME} PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
elseif(UNIX)
  set_target_properties(${LIB_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "libgodot_llm"
    DEBUG_POSTFIX ".${SYSTEM_NAME_LOWER}.debug.${ARCH_SUFFIX}"
    RELEASE_POSTFIX ".${SYSTEM_NAME_LOWER}.release.${ARCH_SUFFIX}"
    MINSIZEREL_POSTFIX ".${SYSTEM_NAME_LOWER}.minsizerel.${ARCH_SUFFIX}"
    RELWITHDEBINFO_POSTFIX ".${SYSTEM_NAME_LOWER}.relwithdebinfo.${ARCH_SUFFIX}"
)

target_link_libraries(${LIB_NAME} PRIVATE godot-cpp common llama)

target_include_directories(${LIB_NAME}
  PUBLIC
    include
  PRIVATE
    llama.cpp
    llama.cpp/common
    llama.cpp/src
    llama.cpp/vendor
)

# ============================================================================
# Documentation
# ============================================================================
get_target_property(GODOTCPP_TARGET godot::cpp GODOTCPP_TARGET)

file(
  GLOB_RECURSE DOC_XML
  LIST_DIRECTORIES NO 
  CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/plugin/doc_classes/*.xml"
)

if(DOC_XML)
  message(STATUS "Found documentation files: ${DOC_XML}")
  if(GODOTCPP_TARGET MATCHES "editor|template_debug")
    target_doc_sources(${LIB_NAME} ${DOC_XML})
    message(STATUS "Documentation embedded for target: ${GODOTCPP_TARGET}")
  else()
    message(STATUS "Skipping documentation (target: ${GODOTCPP_TARGET})")
  endif()
endif()

# ============================================================================
# Testing
# ============================================================================
if (BUILD_TESTING)
  enable_testing()
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip # v1.14.0
    DOWNLOAD_EXTRACT_TIMESTAMP true 
  )

  # For Windows: Prevent gtest from overriding /MT /MD settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_library(godot_llm_test_lib STATIC
    src/conversion.cpp
    src/logging_utils.cpp
    src/llama_controller.cpp
    src/llama_state.cpp
    src/llama_runner.cpp
  )

  target_compile_definitions(godot_llm_test_lib PRIVATE LOG_DISABLE_LOGS)

  target_include_directories(godot_llm_test_lib PUBLIC
    include
    godot-cpp
    godot-cpp/include
    godot-cpp/gdextension/include
    godot-cpp/gen/include 
    llama.cpp
    llama.cpp/include
    llama.cpp/common
    llama.cpp/src
    llama.cpp/vendor
    llama.cpp/ggml/src
    llama.cpp/ggml/include
  )

  target_link_libraries(godot_llm_test_lib PUBLIC
    godot-cpp
    common
    llama
  )

  add_executable(run_tests
    tests/main.cpp
    tests/test_llama_state.cpp
    tests/test_llama_controller.cpp
  )

  target_link_libraries(run_tests PRIVATE
    gtest_main
    godot_llm_test_lib
  )

  if (WIN32)
    add_custom_command(TARGET run_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:gtest>
        $<TARGET_FILE:ggml-base>
        $<TARGET_FILE:ggml>
        $<TARGET_FILE:ggml-cpu>
        $<TARGET_FILE:llama>
        $<TARGET_FILE_DIR:run_tests>
      COMMENT "Copying DLLs to test directory"
    )
  endif()

  add_custom_command(TARGET run_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_BINARY_DIR}/tests
    COMMENT "Creating symlink to tests directory"
  )

  add_test(NAME run_tests COMMAND run_tests)
endif()

# ============================================================================
# Install Rules
# ============================================================================
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

if (LLAMA_VULKAN OR LLAMA_METAL)
  set(LLM_LIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/gpu/addons/godot_llm)
  message(STATUS "Configuring GPU build install to: ${LLM_LIB_INSTALL_DIR}")
else()
  set(LLM_LIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/cpu/addons/godot_llm)
  message(STATUS "Configuring CPU build install to: ${LLM_LIB_INSTALL_DIR}")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/plugin/godot_llm.gdextension" GDEXTENSION_CONTENT)
string(REPLACE "@POSTFIX_TYPE@" "$<LOWER_CASE:$<CONFIG>>" GDEXTENSION_CONTENT "${GDEXTENSION_CONTENT}")
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/godot_llm.gdextension"
  CONTENT "${GDEXTENSION_CONTENT}"
)

# Install the base gdextension
install(TARGETS ${LIB_NAME}
  RUNTIME DESTINATION ${LLM_LIB_INSTALL_DIR}/bin  # .dll on Windows
  LIBRARY DESTINATION ${LLM_LIB_INSTALL_DIR}/bin  # .so/.dylib on Unix
  ARCHIVE DESTINATION ${LLM_LIB_INSTALL_DIR}/bin  # .lib/.a
)

# Install shared library dependencies
install(FILES
  $<TARGET_FILE:ggml-base>
  $<TARGET_FILE:ggml>
  $<TARGET_FILE:ggml-cpu>
  $<TARGET_FILE:llama>
  DESTINATION ${LLM_LIB_INSTALL_DIR}/bin
)

# Install plugin bits
install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/godot_llm.gdextension"
  "plugin/plugin.cfg"
  "plugin/plugin.gd"
  "plugin/icon.png"
  DESTINATION ${LLM_LIB_INSTALL_DIR}
)

message(STATUS "Install directory configured: ${LLM_LIB_INSTALL_DIR}")
