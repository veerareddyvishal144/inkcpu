# tools/ (agent E): oracle tools built against the STOCK llama.cpp libs. Works standalone and when added by the root CMakeLists.
#   cmake -S tools -B tools/build && cmake --build tools/build -j4        (use PATH=/usr/bin:/bin)
# build-sys is only linked, never rebuilt or modified from here.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ink_tools LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
endif()

set(INK_LLAMA_DIR "/data/inkling-mini/llama.cpp" CACHE PATH "llama.cpp checkout (the oracle)")
set(INK_LLAMA_BIN "${INK_LLAMA_DIR}/build-sys/bin" CACHE PATH "stock llama.cpp build: libs are linked, never rebuilt")

if(EXISTS "${INK_LLAMA_BIN}/libllama.so" AND EXISTS "${INK_LLAMA_DIR}/include/llama.h")
  execute_process(COMMAND git -C ${INK_LLAMA_DIR} rev-parse --short=9 HEAD
                  OUTPUT_VARIABLE INK_LLAMA_HASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  foreach(t oracle_logits oracle_greedy)
    add_executable(${t} ${t}.cpp)
    target_include_directories(${t} PRIVATE ${INK_LLAMA_DIR}/include ${INK_LLAMA_DIR}/ggml/include)
    target_compile_definitions(${t} PRIVATE ORACLE_LLAMA_HASH="${INK_LLAMA_HASH}")
    target_compile_options(${t} PRIVATE -O2 -Wno-deprecated-declarations)
    target_link_libraries(${t} PRIVATE ${INK_LLAMA_BIN}/libllama.so ${INK_LLAMA_BIN}/libggml.so ${INK_LLAMA_BIN}/libggml-base.so)
    # DT_RPATH (not RUNPATH) so the transitive libs (libggml-cpu, libllama-common) are found next to libllama too
    target_link_options(${t} PRIVATE -Wl,--disable-new-dtags -Wl,-rpath,${INK_LLAMA_BIN} -Wl,-rpath-link,${INK_LLAMA_BIN})
  endforeach()
else()
  message(STATUS "tools: llama.cpp libs not found in ${INK_LLAMA_BIN}; oracle tools skipped")
endif()
