cmake_minimum_required(VERSION 3.20)
project(LoganLM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(LoganLM main.cpp)

target_compile_options(LoganLM PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)
target_link_options(LoganLM PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)
