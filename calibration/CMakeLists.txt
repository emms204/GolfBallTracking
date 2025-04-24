cmake_minimum_required(VERSION 3.10)
project(CameraCalibration)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Print OpenCV version information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add path to common library
set(COMMON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common/include")
include_directories(${COMMON_INCLUDE_DIR})

# Calibration library
add_library(calibration
    src/calibrator.cpp
    src/stereo_calibrator.cpp
)

target_include_directories(calibration
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${COMMON_INCLUDE_DIR}
)

target_link_libraries(calibration
    PUBLIC 
        ${OpenCV_LIBS}
        common
)

# Calibration tool executable
add_executable(calibrate_camera
    src/calibrate_camera_main.cpp
)

target_link_libraries(calibrate_camera
    PRIVATE
        calibration
        common
        ${OpenCV_LIBS}
)

# Stereo calibration tool executable
add_executable(stereo_calibration
    src/stereo_calibration_main.cpp
)

target_link_libraries(stereo_calibration
    PRIVATE
        calibration
        common
        ${OpenCV_LIBS}
)

# Platform-specific filesystem library linkage
if(NOT WIN32)
    target_link_libraries(calibrate_camera PRIVATE stdc++fs)
endif()

# Install targets
install(TARGETS calibration calibrate_camera
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
) 