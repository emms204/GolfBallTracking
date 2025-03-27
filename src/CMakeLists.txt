cmake_minimum_required(VERSION 3.10)
project(GolfBallDetector)

# Find required packages
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime paths
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.21.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

# Add executables
add_executable(onnx_detector onnx_detector.cpp)

# Link libraries
target_link_libraries(onnx_detector ${OpenCV_LIBS} ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

# Set RPATH for the ONNX Runtime-based executable
set_target_properties(onnx_detector PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print OpenCV version information during cmake configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library dir: ${ONNXRUNTIME_LIB_DIR}")

# Install target
install(TARGETS onnx_detector DESTINATION bin) 