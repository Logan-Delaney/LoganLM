#!/bin/sh
# Formats every source file in the project using .clang-format.
# Usage: ./format.sh          format files in place
#        ./format.sh check    only report files that need formatting
#
# Uses the clang-format on your PATH, or the one named by $CLANG_FORMAT.

cd "$(dirname "$0")" || exit 1

CF=${CLANG_FORMAT:-$(command -v clang-format)}
if [ -z "$CF" ]; then
    echo "clang-format not found. Install it, or set CLANG_FORMAT to its path." >&2
    exit 1
fi

FILES=$(find include src tests -name '*.h' -o -name '*.cpp'; echo main.cpp)

if [ "$1" = "check" ]; then
    # shellcheck disable=SC2086
    "$CF" --dry-run --Werror $FILES
else
    # shellcheck disable=SC2086
    "$CF" -i $FILES
fi
