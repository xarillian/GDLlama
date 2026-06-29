default:
    @just --list

build:
    scons

build-tests:
    scons test

test:
    just build-tests
    ./bin/run_tests

test-quick:
    just build-tests
    CHORUS_SKIP_MODEL_TESTS=1 ./bin/run_tests

compiledb:
    scons compiledb

clean:
    scons -c
