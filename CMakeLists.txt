cmake_minimum_required(VERSION 3.20)
project(inkcpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Target CPU: Xeon Platinum 8171M (Skylake-SP). AVX-512 F/BW/DQ/VL only: NO VNNI, NO BF16, NO AMX, NO VBMI.
# Using any of those instructions raises SIGILL on this machine.
add_compile_options(-O3 -march=skylake-avx512 -mfma -mf16c -fno-math-errno -Wall -Wextra -Wno-unused-parameter)
include_directories(${CMAKE_SOURCE_DIR}/include)
enable_testing()

find_package(Threads REQUIRED)

# Modules are optional so each owner can add theirs independently. Order = dependency order.
foreach(m kernels_gemv kernels_misc core engine capi)
  if(EXISTS ${CMAKE_SOURCE_DIR}/src/${m}/CMakeLists.txt)
    add_subdirectory(src/${m})
  endif()
endforeach()
if(EXISTS ${CMAKE_SOURCE_DIR}/tools/CMakeLists.txt)
  add_subdirectory(tools)
endif()
