cmake_minimum_required(VERSION 3.10)
project(Camera_Calibration_Toolkit VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# ONNX Runtime paths
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.21.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(calibration)
add_subdirectory(common)
add_subdirectory(detector)
add_subdirectory(src)

# Print OpenCV information
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
message(STATUS "OpenCV_VERSION = ${OpenCV_VERSION}")
message(STATUS "ONNXRUNTIME_INCLUDE_DIRS = ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNXRUNTIME_LIB_DIR = ${ONNXRUNTIME_LIB_DIR}")

# Enable compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()