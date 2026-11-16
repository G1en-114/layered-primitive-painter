cmake_minimum_required(VERSION 3.20)
project(lpp_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime: headers vendored in third_party (v1.32.0), library from the onnxruntime-gpu wheel
# of the ML environment (same version). Override ORT_LIB_DIR for another installation.
set(ORT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-1.32.0/include)
if(NOT ORT_LIB_DIR)
  execute_process(COMMAND "$ENV{HOME}/AI infra/.venv/bin/python" -c
                  "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))"
                  OUTPUT_VARIABLE ORT_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_file(ORT_LIB NAMES libonnxruntime.so libonnxruntime.so.1.32.0 PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
message(STATUS "onnxruntime: ${ORT_LIB}")

add_executable(lpp_paint src/main.cpp src/protocol.cpp src/grammar.cpp src/raster.cpp src/session.cpp src/painter.cpp)
target_include_directories(lpp_paint PRIVATE ${ORT_INCLUDE_DIR} src)
target_link_libraries(lpp_paint PRIVATE ${ORT_LIB})
# the wheel ships libonnxruntime.so.1.32.0 but the library's SONAME is libonnxruntime.so.1:
# give the loader that name next to the binary
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ortlib)
file(CREATE_LINK ${ORT_LIB} ${CMAKE_BINARY_DIR}/ortlib/libonnxruntime.so.1 SYMBOLIC)
set_target_properties(lpp_paint PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/ortlib;${ORT_LIB_DIR}")
target_compile_options(lpp_paint PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
