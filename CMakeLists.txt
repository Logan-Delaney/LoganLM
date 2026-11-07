cmake_minimum_required(VERSION 3.20)
project(LoganLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sanitizers apply to everything, including GoogleTest, so all code is built the same way.
add_compile_options(
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)
add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)

# Warnings apply only to targets that link this, so GoogleTest's own code isn't affected.
add_library(loganlm_warnings INTERFACE)
target_compile_options(loganlm_warnings INTERFACE -Wall -Wextra -Wpedantic -Wconversion)

# The engine
add_library(loganlm_core src/version.cpp src/storage.cpp)
target_include_directories(loganlm_core PUBLIC include)
target_link_libraries(loganlm_core PRIVATE loganlm_warnings)

# The program
add_executable(LoganLM main.cpp)
target_link_libraries(LoganLM PRIVATE loganlm_core loganlm_warnings)

# The tests
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(loganlm_tests tests/test_version.cpp tests/test_storage.cpp)
target_link_libraries(loganlm_tests PRIVATE loganlm_core loganlm_warnings GTest::gtest_main)
gtest_discover_tests(loganlm_tests)
