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
    rm -rf plugin/addons/chorus
    mkdir -p plugin/addons/chorus/bin
    cp plugin/chorus.gdextension plugin/plugin.cfg plugin/plugin.gd plugin/icon.png plugin/addons/chorus/
    cp -r plugin/doc_classes plugin/addons/chorus/
    cp bin/libgodot_chorus.so plugin/addons/chorus/bin/
    rm -rf tests/godot/addons/chorus
    mkdir -p tests/godot/addons
    ln -s ../../../plugin/addons/chorus tests/godot/addons/chorus

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
