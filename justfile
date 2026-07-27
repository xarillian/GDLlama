default:
    @just --list

build:
    scons

release:
    scons target=template_release

clean:
    scons -c

build-tests:
    scons test

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
