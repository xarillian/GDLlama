default:
    @just --list

[arg('cpu', long, value='true')]
build cpu='false':
    if [ '{{cpu}}' = 'true' ]; then \
        scons; \
    else \
        scons use_vulkan=yes; \
    fi

[arg('cpu', long, value='true')]
release cpu='false':
    if [ '{{cpu}}' = 'true' ]; then \
        scons target=template_release; \
    else \
        scons target=template_release use_vulkan=yes; \
    fi

godot: build
    python tools/stage_godot.py

clean:
    scons -c

build-tests:
    scons test

test-model:
    ./bin/run_tests --gtest_filter='*ModelTest*:*Llama/EngineContractTest*'

check-model: build-tests
    just test-model

build-gpu-tests:
    scons test use_vulkan=yes

test-gpu:
    ./bin/run_tests --gtest_filter='*GpuModelTest*'

check-gpu: build-gpu-tests
    just test-gpu

# Run the test suite.
#
# Skip model tests: `just test --quick`
# Optionally filter by test name, e.g. `just test Echo`
[arg('quick', long, value='true')]
test quick='false' filter='':
    if [ '{{quick}}' = 'true' ]; then \
        CHORUS_SKIP_MODEL_TESTS=1 ./bin/run_tests {{filter}}; \
    else \
        ./bin/run_tests {{filter}}; \
    fi

# Build the test binary, then run the test suite.
[arg('quick', long, value='true')]
check quick='false' filter='': build-tests
    if [ '{{quick}}' = 'true' ]; then \
        just test --quick {{filter}}; \
    else \
        just test {{filter}}; \
    fi

compiledb:
    scons compiledb
