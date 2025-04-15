cmake_minimum_required(VERSION 3.10)
project(TestDetector)

# Find required packages
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime paths - use file(TO_CMAKE_PATH) to handle spaces in paths
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.21.0")
file(TO_CMAKE_PATH "${ONNXRUNTIME_DIR}" ONNXRUNTIME_DIR)
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Detector library path
set(DETECTOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../detector")
file(TO_CMAKE_PATH "${DETECTOR_DIR}" DETECTOR_DIR)
set(DETECTOR_INCLUDE_DIRS "${DETECTOR_DIR}/include")
set(DETECTOR_LIB_DIR "${DETECTOR_DIR}/build")

# Calibration library path
set(CALIBRATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../calibration")
file(TO_CMAKE_PATH "${CALIBRATION_DIR}" CALIBRATION_DIR)
set(CALIBRATION_INCLUDE_DIRS "${CALIBRATION_DIR}/include")
set(CALIBRATION_LIB_DIR "${CALIBRATION_DIR}/build")

# Common library path
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")
file(TO_CMAKE_PATH "${COMMON_DIR}" COMMON_DIR)
set(COMMON_INCLUDE_DIRS "${COMMON_DIR}/include")
set(COMMON_LIB_DIR "${COMMON_DIR}/build")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${DETECTOR_INCLUDE_DIRS}
    ${COMMON_INCLUDE_DIRS}
    ${CALIBRATION_INCLUDE_DIRS}
)

# Test application
add_executable(test_detector
    src/test_detector.cpp
)

# Calibration test application
add_executable(test_calibration_image
    src/test_calibration_image.cpp
)

target_link_libraries(test_detector
    PRIVATE
        ${DETECTOR_LIB_DIR}/libdetector.a
        ${COMMON_LIB_DIR}/libcommon.a
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.21.0
)

target_link_libraries(test_calibration_image
    PRIVATE
        ${DETECTOR_LIB_DIR}/libdetector.a
        ${COMMON_LIB_DIR}/libcommon.a
        ${CALIBRATION_LIB_DIR}/libcalibration.a
        ${OpenCV_LIBS}
)

# Set RPATH for the test application (preferred way)
set_target_properties(test_detector PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print information during build
message(STATUS "Test detector: OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Test detector: OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Test detector: OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Test detector: ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "Test detector: ONNX Runtime library dir: ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "Test detector: Detector library dir: ${DETECTOR_LIB_DIR}")
message(STATUS "Test detector: Common library dir: ${COMMON_LIB_DIR}")
message(STATUS "Test detector: Calibration library dir: ${CALIBRATION_LIB_DIR}") 