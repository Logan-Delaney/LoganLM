# LoganLM

An LLM built from scratch in C++20. The goal is to understand every layer, from tensors up to a language model.

## Requirements

- A C++20 compiler
- CMake 3.20 or newer

GoogleTest is downloaded automatically by CMake the first time you build.

## Build and test

```sh
./test.sh
```

This builds the tests and runs them. It stops if the build fails. Extra arguments go to the test program, for example `./test.sh --gtest_filter='Add.*'` to run only the `Add` tests.

Debug builds enable AddressSanitizer and UndefinedBehaviorSanitizer, and all code is compiled with `-Wall -Wextra -Wpedantic -Wconversion`.

## Formatting

Code style is defined in `.clang-format`, which requires `clang-format` to be installed.

```sh
./format.sh          # format all source files in place
./format.sh check    # report files that need formatting, without changing them
```
