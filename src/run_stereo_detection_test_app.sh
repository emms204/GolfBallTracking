#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(readlink -f $(dirname $0)/..)"

# Default stereo calibration file
STEREO_CALIBRATION_FILE="$PROJECT_ROOT/calibration_results/stereo_calibration.yaml"
# Default video files (empty means use cameras)
MASTER_VIDEO_FILE=""
SLAVE_VIDEO_FILE=""
# Default to using calibration if available
NO_CALIBRATION=0
# Default trajectory output file
TRAJECTORY_FILE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --stereo_calibration=*)
            STEREO_CALIBRATION_FILE="${1#*=}"
            ;;
        --no-calibration)
            NO_CALIBRATION=1
            ;;
        --master_video=*)
            MASTER_VIDEO_FILE="${1#*=}"
            ;;
        --slave_video=*)
            SLAVE_VIDEO_FILE="${1#*=}"
            ;;
        --master_camera=*)
            MASTER_CAMERA_ID="${1#*=}"
            ;;
        --slave_camera=*)
            SLAVE_CAMERA_ID="${1#*=}"
            ;;
        --save_trajectory=*)
            TRAJECTORY_FILE="${1#*=}"
            ;;
        --debug)
            USE_DEBUG=1
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --stereo_calibration=<file>  Path to stereo calibration file (default: $STEREO_CALIBRATION_FILE)"
            echo "  --no-calibration             Disable calibration even if a calibration file exists"
            echo "  --master_video=<file>        Path to master camera video file"
            echo "  --slave_video=<file>         Path to slave camera video file"
            echo "  --master_camera=<id>         Master camera ID (default: 0)"
            echo "  --slave_camera=<id>          Slave camera ID (default: 1)"
            echo "  --save_trajectory=<file>     Save trajectory to CSV file"
            echo "  --debug                      Run with GDB to debug crashes"
            echo "  --help                       Show this help message"
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

echo -e "${GREEN}Starting stereo detection test app build and run script...${NC}"

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

# Check for video files if specified
if [ -n "$MASTER_VIDEO_FILE" ] && [ ! -f "$MASTER_VIDEO_FILE" ]; then
    echo -e "${RED}Master video file not found at $MASTER_VIDEO_FILE${NC}"
    exit 1
fi

if [ -n "$SLAVE_VIDEO_FILE" ] && [ ! -f "$SLAVE_VIDEO_FILE" ]; then
    echo -e "${RED}Slave video file not found at $SLAVE_VIDEO_FILE${NC}"
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
if [ ! -f "./src/stereo_detection_test_app" ]; then
    echo -e "${RED}The stereo_detection_test_app executable was not built correctly.${NC}"
    exit 1
fi

# Run the application
echo -e "${GREEN}Running stereo detection test app...${NC}"
echo -e "${YELLOW}Press 'q' to quit, 's' to save trajectory, 'r' to reset tracking, 'p' to pause/resume${NC}"

# Build command with equals format arguments
cmd=("./src/stereo_detection_test_app" "--model=$MODEL_PATH" "--classes=$CLASSES_PATH")

# Handle calibration options
if [ "$NO_CALIBRATION" -eq 1 ]; then
    echo -e "${YELLOW}Stereo calibration explicitly disabled with --no-calibration flag${NC}"
elif [ ! -f "$STEREO_CALIBRATION_FILE" ]; then
    echo -e "${YELLOW}Stereo calibration file not found at $STEREO_CALIBRATION_FILE${NC}"
    echo -e "${YELLOW}Running without stereo calibration. 3D tracking will not be available.${NC}"
else
    echo -e "${GREEN}Stereo calibration file found at $STEREO_CALIBRATION_FILE. Using for stereo reconstruction.${NC}"
    cmd+=("--stereo_calibration=$STEREO_CALIBRATION_FILE")
fi

# Set up video arguments if specified
if [ -n "$MASTER_VIDEO_FILE" ] && [ -n "$SLAVE_VIDEO_FILE" ]; then
    echo -e "${GREEN}Using video files:${NC}"
    echo -e "${GREEN}  Master: $MASTER_VIDEO_FILE${NC}"
    echo -e "${GREEN}  Slave: $SLAVE_VIDEO_FILE${NC}"
    cmd+=("--master_video=$MASTER_VIDEO_FILE" "--slave_video=$SLAVE_VIDEO_FILE")
else
    echo -e "${YELLOW}Using cameras as input sources${NC}"
    
    # If camera IDs are specified, use them
    if [ -n "$MASTER_CAMERA_ID" ]; then
        cmd+=("--master_camera=$MASTER_CAMERA_ID")
    fi
    
    if [ -n "$SLAVE_CAMERA_ID" ]; then
        cmd+=("--slave_camera=$SLAVE_CAMERA_ID")
    fi
fi

# Add trajectory file if specified
if [ -n "$TRAJECTORY_FILE" ]; then
    echo -e "${GREEN}Trajectory will be saved to: $TRAJECTORY_FILE${NC}"
    cmd+=("--save_trajectory=$TRAJECTORY_FILE")
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

# Check if trajectory file was created
if [ -n "$TRAJECTORY_FILE" ] && [ -f "$TRAJECTORY_FILE" ]; then
    echo -e "${GREEN}Trajectory saved to: $TRAJECTORY_FILE${NC}"
    echo -e "${YELLOW}To visualize the trajectory, you can use tools like Matplotlib or Excel.${NC}"
fi 