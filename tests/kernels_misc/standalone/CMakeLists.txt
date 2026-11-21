# Private build of ONLY the kernels_misc module (the root build also configures other agents' in-progress modules).
#   cmake -S tests/kernels_misc/standalone -B build-b2s -G Ninja && ninja -C build-b2s
cmake_minimum_required(VERSION 3.20)
project(b2_standalone LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=skylake-avx512 -mfma -mf16c -fno-math-errno -Wall -Wextra -Wno-unused-parameter)
find_package(Threads REQUIRED)
enable_testing()
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../src/kernels_misc ${CMAKE_BINARY_DIR}/kernels_misc)
