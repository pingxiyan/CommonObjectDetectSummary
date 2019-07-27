cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(TARGET_NAME testapp)
project(${TARGET_NAME})

find_package(OpenCV REQUIRED)

message(STATUS "=============================")
message(STATUS "${OpenCV_LIBS}")
message(STATUS "=============================")
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "=============================")

find_package(Torch REQUIRED)

add_executable(${TARGET_NAME} main.cpp)

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY GLIBCXX_USE_CXX11_ABI 1)