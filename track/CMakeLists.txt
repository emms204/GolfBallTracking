cmake_minimum_required(VERSION 3.14)
project(yolo_bytetrack)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("Build type: " ${CMAKE_BUILD_TYPE})

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Find ONNX Runtime
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "ONNX Runtime root directory")
if(ONNXRUNTIME_ROOT_DIR)
  message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT_DIR}")
  set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
  if(WIN32)
    set(ONNXRUNTIME_LIBS "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib")
  else()
    set(ONNXRUNTIME_LIBS "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so")
  endif()
else()
  message(FATAL_ERROR "ONNX Runtime directory not specified. Please set ONNXRUNTIME_ROOT_DIR.")
endif()

# Include ByteTrack library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ByteTrack-cpp ${CMAKE_CURRENT_BINARY_DIR}/ByteTrack-cpp)

# Add OnnxDetector source
add_library(onnx_detector SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/onnx_detector.cpp
)

target_include_directories(onnx_detector PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(onnx_detector
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBS}
)

# Add main integration executable
add_executable(yolo_bytetrack
  ${CMAKE_CURRENT_SOURCE_DIR}/yolo_bytetrack_integration.cpp
)

target_include_directories(yolo_bytetrack PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../ByteTrack-cpp/include
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(yolo_bytetrack
  ${OpenCV_LIBS}
  onnx_detector
  bytetrack
)

# Installation
install(TARGETS yolo_bytetrack onnx_detector
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)

# Print summary
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen3 library status:")
message(STATUS "    version: ${EIGEN3_VERSION}")
message(STATUS "    include path: ${EIGEN3_INCLUDE_DIRS}") 