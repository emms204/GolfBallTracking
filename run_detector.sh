#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define paths
BUILD_DIR="${SCRIPT_DIR}/build"
DETECTOR_TOOL="${BUILD_DIR}/bin/test_detector"
ONNXRUNTIME_DIR="${SCRIPT_DIR}/onnxruntime-linux-x64-1.21.0"
DEFAULT_MODEL="${SCRIPT_DIR}/best.onnx"
DEFAULT_CLASSES="${SCRIPT_DIR}/classes.txt"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Build directory not found. Building project..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake ..
    make -j$(nproc)
    cd "${SCRIPT_DIR}"
fi

# Check if detector tool exists
if [ ! -f "${DETECTOR_TOOL}" ]; then
    echo "Detector tool not found. Please build the project first."
    exit 1
fi

# Check if model exists
if [ ! -f "${DEFAULT_MODEL}" ]; then
    echo "Warning: Default model file not found at ${DEFAULT_MODEL}"
    echo "Please specify a model file with --model=<path>"
fi

# Check if classes file exists
if [ ! -f "${DEFAULT_CLASSES}" ]; then
    echo "Warning: Default classes file not found at ${DEFAULT_CLASSES}"
    echo "Please specify a classes file with --classes=<path>"
fi

# Set LD_LIBRARY_PATH to include ONNX Runtime
export LD_LIBRARY_PATH="${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH}"

# Run the detector tool with arguments
"${DETECTOR_TOOL}" "$@" 