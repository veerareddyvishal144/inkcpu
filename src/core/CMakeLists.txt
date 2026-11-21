# ink_core: GGUF reader, ModelConfig/Weights/visit plan, scalar reference forward. Owner: agent A.
find_package(OpenMP REQUIRED)
add_library(ink_core STATIC gguf.cpp model.cpp ref_forward.cpp)
target_include_directories(ink_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ink_core PUBLIC OpenMP::OpenMP_CXX)
# ggml's CPU ops are NOT FMA-contracted (measured: ssm_conv); explicit std::fmaf marks the places where ggml uses FMA intrinsics
target_compile_options(ink_core PRIVATE -ffp-contract=off)

set(INK_CORE_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/core)
foreach(t test_gguf test_ref_state test_ref_vs_oracle)
  if(EXISTS ${INK_CORE_TESTS}/${t}.cpp)
    add_executable(${t} ${INK_CORE_TESTS}/${t}.cpp)
    target_link_libraries(${t} PRIVATE ink_core)
    target_compile_definitions(${t} PRIVATE INK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../..")
    # tests read the real 17.6 GB GGUF and run the scalar reference: 8 threads max, generous timeout
    add_test(NAME core_${t} COMMAND ${t})
    set_tests_properties(core_${t} PROPERTIES TIMEOUT 3600 ENVIRONMENT "OMP_NUM_THREADS=8")
  endif()
endforeach()

# Debug tools (optional): dump llama.cpp intermediates through the eval callback and compare layer by layer.
set(INK_LLAMA_DIR "/data/inkling-mini/llama.cpp" CACHE PATH "llama.cpp checkout (the oracle)")
set(INK_LLAMA_BIN "${INK_LLAMA_DIR}/build-sys/bin" CACHE PATH "stock llama.cpp build (linked, never rebuilt)")
if(EXISTS ${INK_CORE_TESTS}/compare_acts.cpp)
  add_executable(compare_acts ${INK_CORE_TESTS}/compare_acts.cpp)
  target_link_libraries(compare_acts PRIVATE ink_core)
endif()
if(EXISTS "${INK_LLAMA_BIN}/libllama.so" AND EXISTS ${INK_CORE_TESTS}/dump_llama_acts.cpp)
  add_executable(dump_llama_acts ${INK_CORE_TESTS}/dump_llama_acts.cpp)
  target_include_directories(dump_llama_acts PRIVATE ${INK_LLAMA_DIR}/include ${INK_LLAMA_DIR}/ggml/include ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
  target_compile_options(dump_llama_acts PRIVATE -O2 -Wno-deprecated-declarations)
  target_link_libraries(dump_llama_acts PRIVATE ${INK_LLAMA_BIN}/libllama.so ${INK_LLAMA_BIN}/libggml.so ${INK_LLAMA_BIN}/libggml-base.so)
  target_link_options(dump_llama_acts PRIVATE -Wl,--disable-new-dtags -Wl,-rpath,${INK_LLAMA_BIN} -Wl,-rpath-link,${INK_LLAMA_BIN})
endif()
