#!/bin/bash

# Script to run the Enhanced ONNX Runtime-based detector

# Check if a video file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video_file> [model_file] [classes_file] [conf_threshold] [nms_threshold]"
    echo "Example: $0 videos/input.mp4 best.onnx classes.txt 0.25 0.45"
    exit 1
fi

INPUT_VIDEO="$1"
# Convert to absolute paths
if [[ "$2" == /* ]]; then
    MODEL_FILE="$2"
else
    MODEL_FILE="${2:-best.onnx}"
    if [[ "$MODEL_FILE" != /* ]]; then
        MODEL_FILE="$(readlink -f $(dirname $0)/../$MODEL_FILE)"
    fi
fi

if [[ "$3" == /* ]]; then
    CLASSES_FILE="$3"
else
    CLASSES_FILE="${3:-classes.txt}"
    if [[ "$CLASSES_FILE" != /* ]]; then
        CLASSES_FILE="$(readlink -f $(dirname $0)/../$CLASSES_FILE)"
    fi
fi

CONF_THRESHOLD="${4:-0.25}"  # 4th param, default 0.25
NMS_THRESHOLD="${5:-0.45}"   # 5th param, default 0.45

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video file '$INPUT_VIDEO' does not exist."
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file '$MODEL_FILE' does not exist."
    exit 1
fi

# Check if classes file exists
if [ ! -f "$CLASSES_FILE" ]; then
    echo "Warning: Classes file '$CLASSES_FILE' does not exist. Creating default..."
    mkdir -p "$(dirname "$CLASSES_FILE")"
    echo "ball" > "$CLASSES_FILE"
    echo "club" >> "$CLASSES_FILE"
fi

# Get the project root directory
PROJECT_ROOT="$(readlink -f $(dirname $0)/..)"

# Set the ONNX Runtime library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROJECT_ROOT/onnxruntime-linux-x64-1.21.0/lib

# Move to project root directory and build the project if not already built
cd "$PROJECT_ROOT"
mkdir -p build
cd build

# Run CMake if needed
if [ ! -f "CMakeCache.txt" ]; then
    echo "Configuring project with CMake..."
    cmake ..
fi

# Build the project
echo "Building the project..."
make -j$(nproc)

if [ ! -f "src/enhanced_detector_main" ]; then
    echo "Failed to build the enhanced detector. Check the compile errors above."
    exit 1
fi

# Run the enhanced detector
echo "Running Enhanced ONNX detector on '$INPUT_VIDEO'..."
echo "Using model: $MODEL_FILE"
echo "Using classes: $CLASSES_FILE"

./src/enhanced_detector_main --video="$INPUT_VIDEO" --model="$MODEL_FILE" --classes="$CLASSES_FILE" --conf="$CONF_THRESHOLD" --nms="$NMS_THRESHOLD"

if [ $? -eq 0 ]; then
    echo "Detection completed successfully!"
else
    echo "Detection failed. Check the error messages above."
fi