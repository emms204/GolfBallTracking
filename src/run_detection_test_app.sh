#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(readlink -f $(dirname $0)/..)"

# Default calibration file
CALIBRATION_FILE="$PROJECT_ROOT/calibration_results/master_calibration.yaml"
# Default video file (empty means use camera)
VIDEO_FILE=""
# Default to using calibration if available
NO_CALIBRATION=0

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --calibration)
            CALIBRATION_FILE="$2"
            shift
            ;;
        --no-calibration)
            NO_CALIBRATION=1
            ;;
        --video)
            VIDEO_FILE="$2"
            shift
            ;;
        --debug)
            USE_DEBUG=1
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --calibration <file>  Path to calibration file (default: $CALIBRATION_FILE)"
            echo "  --no-calibration      Disable calibration even if a calibration file exists"
            echo "  --video <file>        Path to video file (if not specified, uses camera)"
            echo "  --debug               Run with GDB to debug crashes"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
    shift
done

echo -e "${GREEN}Starting detection test app build and run script...${NC}"

# Check for ONNX Runtime
ONNXRUNTIME_DIR="$PROJECT_ROOT/onnxruntime-linux-x64-1.21.0"
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo -e "${YELLOW}ONNX Runtime not found at $ONNXRUNTIME_DIR${NC}"
    echo -e "${YELLOW}Downloading ONNX Runtime 1.21.0...${NC}"
    
    mkdir -p "$PROJECT_ROOT/onnxruntime-linux-x64-1.21.0"
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz -O "$PROJECT_ROOT/onnxruntime.tgz"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download ONNX Runtime. Please check your internet connection.${NC}"
        exit 1
    fi
    
    tar -xzf "$PROJECT_ROOT/onnxruntime.tgz" -C "$PROJECT_ROOT"
    rm "$PROJECT_ROOT/onnxruntime.tgz"
    
    echo -e "${GREEN}Successfully downloaded and extracted ONNX Runtime.${NC}"
fi

# Set up environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNXRUNTIME_DIR/lib

# Check for model file
MODEL_PATH="$PROJECT_ROOT/best.onnx"
CLASSES_PATH="$PROJECT_ROOT/classes.txt"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Model file best.onnx not found at $MODEL_PATH.${NC}"
    echo -e "${YELLOW}Please ensure you have the model file in the correct location.${NC}"
else
    echo -e "${GREEN}Model file found.${NC}"
fi

# Check for video file if specified
if [ -n "$VIDEO_FILE" ] && [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${RED}Video file not found at $VIDEO_FILE${NC}"
    exit 1
fi

# Move to project root directory and create build directory
cd "$PROJECT_ROOT"
mkdir -p build
cd build

# Run CMake
echo -e "${GREEN}Configuring project with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed. Please check error messages above.${NC}"
    exit 1
fi

# Build the project
echo -e "${GREEN}Building the project...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed. Please check error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully.${NC}"

# Check for necessary files
if [ ! -f "./src/detection_test_app" ]; then
    echo -e "${RED}The executable was not built correctly.${NC}"
    exit 1
fi

# Run the application
echo -e "${GREEN}Running detection test app...${NC}"
echo -e "${YELLOW}Press 'q' to quit, 'c' to toggle calibration, 'p' to pause/resume${NC}"

# Build command arguments using arrays for proper space handling
cmd=("./src/detection_test_app" "--model" "$MODEL_PATH" "--classes" "$CLASSES_PATH")

# Handle calibration options
if [ "$NO_CALIBRATION" -eq 1 ]; then
    echo -e "${YELLOW}Calibration explicitly disabled with --no-calibration flag${NC}"
    cmd+=("--no-calibration")
elif [ ! -f "$CALIBRATION_FILE" ]; then
    echo -e "${YELLOW}Calibration file not found at $CALIBRATION_FILE${NC}"
    echo -e "${YELLOW}Running without calibration. You can generate a calibration file using the calibration tool.${NC}"
else
    echo -e "${GREEN}Calibration file found at $CALIBRATION_FILE. Using for undistortion.${NC}"
    cmd+=("--calibration" "$CALIBRATION_FILE")
fi

# Set up video argument if specified
if [ -n "$VIDEO_FILE" ]; then
    echo -e "${GREEN}Using video file: $VIDEO_FILE${NC}"
    cmd+=("--video" "$VIDEO_FILE")
else
    echo -e "${YELLOW}Using camera as input source${NC}"
fi

# Execute the command with proper argument handling
if [ "${USE_DEBUG}" = "1" ]; then
    echo -e "${YELLOW}Running with GDB for debugging...${NC}"
    echo -e "${YELLOW}When GDB starts, type 'run' to start the program${NC}"
    echo -e "${YELLOW}If the program crashes, type 'bt' to get a backtrace${NC}"
    echo -e "${YELLOW}Type 'quit' to exit GDB${NC}"
    gdb --args "${cmd[@]}"
else
    "${cmd[@]}"
fi

echo -e "${GREEN}Application exited.${NC}" 