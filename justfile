default:
    @just --list

build:
    scons

build-tests:
    scons test

# The full gate: every suite, model included. Run once per change-set; trust its FINAL SUMMARY line.
test:
    just build-tests
    ./bin/run_tests

# Fast inner loop: model suites skipped; optional test-name filter, e.g. `just test-quick Echo`
test-quick filter="":
    just build-tests
    CHORUS_SKIP_MODEL_TESTS=1 ./bin/run_tests {{filter}}

# One test/suite by name substring with the model available, e.g. `just test-only Batch_demand`
test-only filter:
    just build-tests
    ./bin/run_tests {{filter}}

compiledb:
    scons compiledb

clean:
    scons -c
