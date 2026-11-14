#!/bin/sh
# Builds the tests and runs them. Stops if the build fails, so a stale binary never runs.
# Usage: ./test.sh                      run all tests
#        ./test.sh --gtest_filter='Add.*'   extra arguments go to the test program

cd "$(dirname "$0")" || exit 1

cmake --build cmake-build-debug --target loganlm_tests || exit 1
./cmake-build-debug/loganlm_tests "$@"
