# libink.so: the C ABI of include/ink/ink_abi.h (agent C1). Python binds it with ctypes.
get_filename_component(INK_ROOT ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
add_library(ink SHARED ink_capi.cpp)
target_link_libraries(ink PRIVATE ink_engine)
set_target_properties(ink PROPERTIES OUTPUT_NAME ink)
