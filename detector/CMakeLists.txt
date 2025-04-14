cmake_minimum_required(VERSION 3.10)
project(Detector)

# Find required packages
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime paths
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.21.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Common library path
set(COMMON_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../common/include")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${COMMON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Detector library
add_library(detector
    src/onnx_detector.cpp
    src/preprocessor.cpp
)

target_include_directories(detector
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
        ${COMMON_INCLUDE_DIRS}
)

# Check if common library exists before linking
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../common/build/libcommon.a")
    target_link_libraries(detector
        PUBLIC 
            ${OpenCV_LIBS}
            ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/build/libcommon.a
    )
else()
    target_link_libraries(detector
        PUBLIC 
            ${OpenCV_LIBS}
            ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    )
endif()

# Add executable for simple_onnx_detector - self-contained, not using the detector library
add_executable(simple_onnx_detector src/simple_onnx_detector.cpp)
target_include_directories(simple_onnx_detector
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
)
target_link_libraries(simple_onnx_detector 
    PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

# Print information during build
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library dir: ${ONNXRUNTIME_LIB_DIR}") 