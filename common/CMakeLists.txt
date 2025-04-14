cmake_minimum_required(VERSION 3.10)
project(Common)

# Find required packages
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Common library for shared functionality
add_library(common
    src/image_utils.cpp
    src/camera_params.cpp
    src/path_utils.cpp
)

target_include_directories(common
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(common
    PUBLIC 
        ${OpenCV_LIBS}
)

# Print information during build
message(STATUS "Common library: OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Common library: OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Common library: OpenCV libraries: ${OpenCV_LIBS}") 