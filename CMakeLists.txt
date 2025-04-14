cmake_minimum_required(VERSION 3.10)
project(Camera_Calibration_Toolkit VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform detection and path handling
if(WIN32)
    set(PLATFORM_WINDOWS TRUE)
    message(STATUS "Building on Windows platform")
    
    # Define Windows-specific paths with forward slashes
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-1.21.0")
    
    # Add platform-specific preprocessor definitions
    add_compile_definitions(PLATFORM_WINDOWS)
else()
    set(PLATFORM_WINDOWS FALSE)
    message(STATUS "Building on Unix-like platform")
    
    # Use standard Linux paths
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.21.0")
    
    # Add platform-specific preprocessor definitions
    add_compile_definitions(PLATFORM_LINUX)
endif()

# Ensure paths use forward slashes on all platforms
file(TO_CMAKE_PATH "${ONNXRUNTIME_DIR}" ONNXRUNTIME_DIR)

# Find required packages
if(WIN32)
    # Set OpenCV search hints for Windows (can be overridden by -DOPENCV_DIR=...)
    set(OPENCV_DIR "C:/opencv/build" CACHE PATH "OpenCV installation directory")
    list(APPEND CMAKE_PREFIX_PATH "${OPENCV_DIR}")
endif()
find_package(OpenCV REQUIRED)

# ONNX Runtime configuration
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")

if(WIN32)
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
    set(ONNXRUNTIME_LIB_NAME "onnxruntime.lib")
else()
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
    set(ONNXRUNTIME_LIB_NAME "libonnxruntime.so")
endif()

# Create imported target for ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
)

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

# Visual Studio-specific settings
if(MSVC)
    # Disable warnings about unsafe functions
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    
    # Reduce warnings about STL usage
    add_compile_definitions(_SCL_SECURE_NO_WARNINGS)
    
    # Use multiple processors for compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()