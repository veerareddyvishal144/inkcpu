# Module engine (agent C1): packed weights + decode executor, tests (tests/engine), bench (bench/).
get_filename_component(INK_ROOT ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
# gguf.cpp / model.cpp are compiled in directly (not via ink_core) so libink does not need OpenMP
add_library(ink_engine STATIC engine_load.cpp engine_decode.cpp ${INK_ROOT}/src/core/gguf.cpp ${INK_ROOT}/src/core/model.cpp)
target_include_directories(ink_engine PUBLIC ${INK_ROOT}/include PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(ink_engine PUBLIC ink_kernels_gemv ink_kernels_misc Threads::Threads)
# no mul+add contraction: the expert combine and the scalar tails must round exactly like the reference
target_compile_options(ink_engine PRIVATE -ffp-contract=off)

if(EXISTS ${INK_ROOT}/bench/bench_decode.cpp)
  add_executable(bench_decode ${INK_ROOT}/bench/bench_decode.cpp)
  target_link_libraries(bench_decode ink_engine)
endif()

if(EXISTS ${INK_ROOT}/bench/bench_stream.cpp)
  add_executable(bench_stream ${INK_ROOT}/bench/bench_stream.cpp)
  target_link_libraries(bench_stream ink_engine)
endif()

if(EXISTS ${INK_ROOT}/tests/engine/test_engine.cpp)
  find_package(OpenMP REQUIRED)
  # the scalar reference (agent A) is compiled into the test only
  add_executable(test_engine ${INK_ROOT}/tests/engine/test_engine.cpp ${INK_ROOT}/src/core/ref_forward.cpp)
  target_include_directories(test_engine PRIVATE ${INK_ROOT}/src/core)
  target_compile_definitions(test_engine PRIVATE INK_SOURCE_DIR="${INK_ROOT}")
  target_compile_options(test_engine PRIVATE -ffp-contract=off)
  target_link_libraries(test_engine ink_engine OpenMP::OpenMP_CXX)
endif()
