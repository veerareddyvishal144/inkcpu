# Standalone top-level for building ONLY the core module (does not touch the other agents' modules):
#   export PATH=/usr/bin:/bin
#   cmake -S src/core/standalone -B build-core -G Ninja && cmake --build build-core
cmake_minimum_required(VERSION 3.20)
project(ink_core_standalone LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# same flags as the root CMakeLists: Skylake-SP, AVX-512 F/BW/DQ/VL only (no VNNI / BF16 / AMX)
add_compile_options(-O3 -march=skylake-avx512 -mfma -mf16c -fno-math-errno -Wall -Wextra -Wno-unused-parameter)
get_filename_component(INK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
set(CMAKE_SOURCE_DIR_OVERRIDE ${INK_ROOT})
include_directories(${INK_ROOT}/include)
enable_testing()
add_subdirectory(${INK_ROOT}/src/core core)
