cmake_minimum_required(VERSION 3.10)
project(Detector)

# Find required packages
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get ONNX Runtime paths from parent scope or use default if not set
if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIRS OR NOT DEFINED ONNXRUNTIME_LIB_DIR)
    message(WARNING "ONNX Runtime paths not defined in parent scope, using defaults")
    # This is just a fallback - the actual paths should come from the parent CMakeLists.txt
    if(WIN32)
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-win-x64-1.21.0")
    else()
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.21.0")
    endif()
    
    # Normalize path
    file(TO_CMAKE_PATH "${ONNXRUNTIME_DIR}" ONNXRUNTIME_DIR)
    
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
endif()

# Define the ONNX Runtime library path directly
if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
endif()

# Ensure paths are normalized
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../common/include" COMMON_INCLUDE_DIRS)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${COMMON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add library for simple_onnx_detector without main function
add_library(simple_onnx_detector_lib STATIC src/simple_onnx_detector.cpp)
target_compile_definitions(simple_onnx_detector_lib PRIVATE EXCLUDE_MAIN_FUNCTION)
target_include_directories(simple_onnx_detector_lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
)
target_link_libraries(simple_onnx_detector_lib 
    PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}  # Direct library path instead of target name
)

# Also keep the original executable for standalone use
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
        ${ONNXRUNTIME_LIB}  # Direct library path instead of target name
)

# Print information during build
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library dir: ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}") 