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

# Calibration library
add_library(calibration
    src/calibrator.cpp
)

target_include_directories(calibration
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(calibration
    PUBLIC 
        ${OpenCV_LIBS}
)

# Calibration tool executable
add_executable(calibrate_camera
    src/calibrate_camera_main.cpp
)

target_link_libraries(calibrate_camera
    PRIVATE
        calibration
        ${OpenCV_LIBS}
        stdc++fs
)

# Install targets
install(TARGETS calibration calibrate_camera
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
) 