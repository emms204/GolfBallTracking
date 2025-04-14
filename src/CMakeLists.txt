cmake_minimum_required(VERSION 3.10)
project(GolfBallDetector)

# Find required packages
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime paths
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.21.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Calibration library path
set(CALIBRATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../calibration")
set(CALIBRATION_INCLUDE_DIRS "${CALIBRATION_DIR}/include")
set(CALIBRATION_LIB_DIR "${CMAKE_BINARY_DIR}/../calibration/build")

# Make sure calibration library exists or has been built
if(NOT TARGET calibration)
    find_library(CALIBRATION_LIB calibration 
                 PATHS ${CALIBRATION_LIB_DIR} ${CMAKE_BINARY_DIR}/lib
                 NO_DEFAULT_PATH)
    if(NOT CALIBRATION_LIB)
        message(WARNING "Calibration library not found. Make sure to build calibration first.")
    endif()
endif()

# Detector library path
set(DETECTOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../detector")
set(DETECTOR_INCLUDE_DIRS "${DETECTOR_DIR}/include")

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${ONNXRUNTIME_INCLUDE_DIRS} 
    ${CALIBRATION_INCLUDE_DIRS}
    ${DETECTOR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
)

# Add executables
add_executable(detection_test_app detection_test_app.cpp)
add_executable(enhanced_detector_main enhanced_detector_main.cpp)
add_executable(preprocessor_main preprocessor_main.cpp)
add_executable(test_calibration_image ../test/src/test_calibration_image.cpp)

# Link libraries
target_link_libraries(detection_test_app 
    detector
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    calibration
)

target_link_libraries(enhanced_detector_main 
    detector
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

target_link_libraries(preprocessor_main 
    detector
    ${OpenCV_LIBS}
)

target_link_libraries(test_calibration_image
    detector
    ${OpenCV_LIBS}
    calibration
)

# Set RPATH for the executables
set_target_properties(detection_test_app PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(enhanced_detector_main PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print version information during cmake configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library dir: ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "Calibration include dirs: ${CALIBRATION_INCLUDE_DIRS}")
message(STATUS "Detector include dirs: ${DETECTOR_INCLUDE_DIRS}")

# Install targets
install(TARGETS detection_test_app enhanced_detector_main preprocessor_main test_calibration_image DESTINATION bin) 