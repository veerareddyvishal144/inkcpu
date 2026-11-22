# Module engine (agent C1): packed weights + decode executor, tests (tests/engine), bench (bench/).
get_filename_component(INK_ROOT ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
# gguf.cpp / model.cpp are compiled in directly (not via ink_core) so libink does not need OpenMP
add_library(ink_engine STATIC engine_load.cpp engine_decode.cpp engine_rewind.cpp ${INK_ROOT}/src/core/gguf.cpp ${INK_ROOT}/src/core/model.cpp)
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

# ---- agent P: batched prefill executor (needs agent G's GEMM library; skipped in builds that do not have it, e.g. src/engine/standalone) ----
if(TARGET ink_kernels_gemm AND EXISTS ${INK_ROOT}/src/engine/engine_prefill.cpp)
  target_sources(ink_engine PRIVATE engine_prefill.cpp)
  target_link_libraries(ink_engine PUBLIC ink_kernels_gemm)
  target_compile_definitions(ink_engine PUBLIC INK_HAVE_PREFILL=1)   # Engine::forward(n >= 3) -> engine_prefill (orchestrator wiring)
  if(EXISTS ${INK_ROOT}/tests/engine/test_prefill.cpp)
    add_executable(test_prefill ${INK_ROOT}/tests/engine/test_prefill.cpp)
    target_compile_definitions(test_prefill PRIVATE INK_SOURCE_DIR="${INK_ROOT}")
    target_compile_options(test_prefill PRIVATE -ffp-contract=off)
    target_link_libraries(test_prefill ink_engine)
  endif()
  if(EXISTS ${INK_ROOT}/bench/bench_prefill.cpp)
    add_executable(bench_prefill ${INK_ROOT}/bench/bench_prefill.cpp)
    target_link_libraries(bench_prefill ink_engine)
  endif()
  # ---- agent S3: slot checkpoints + rewind gates (docs/REWIND.md); links ink_spec when present for the speculation interplay mode
  if(EXISTS ${INK_ROOT}/tests/engine/test_rewind.cpp)
    add_executable(test_rewind ${INK_ROOT}/tests/engine/test_rewind.cpp)
    target_compile_definitions(test_rewind PRIVATE INK_SOURCE_DIR="${INK_ROOT}")
    target_compile_options(test_rewind PRIVATE -ffp-contract=off)
    target_link_libraries(test_rewind ink_engine)
  endif()
endif()
