#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define paths
BUILD_DIR="${SCRIPT_DIR}/build"
CALIBRATION_TOOL="${BUILD_DIR}/bin/calibrate_camera"
ONNXRUNTIME_DIR="${SCRIPT_DIR}/onnxruntime-linux-x64-1.21.0"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Build directory not found. Building project..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake ..
    make -j$(nproc)
    cd "${SCRIPT_DIR}"
fi

# Check if calibration tool exists
if [ ! -f "${CALIBRATION_TOOL}" ]; then
    echo "Calibration tool not found. Please build the project first."
    exit 1
fi

# Set LD_LIBRARY_PATH to include ONNX Runtime
export LD_LIBRARY_PATH="${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH}"

# Run the calibration tool with arguments
"${CALIBRATION_TOOL}" "$@" 