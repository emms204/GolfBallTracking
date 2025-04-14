#!/bin/bash

# Build script for the calibration tool

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
BUILD_DIR="build"
CMAKE_OPTIONS=""
BUILD_ONLY=true
INPUT_SOURCE=""
PATTERN_SIZE="9x6"
SQUARE_SIZE=20
OUTPUT_FILE="camera_calibration.yaml"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build-dir=*)
            BUILD_DIR="${1#*=}"
            ;;
        --run)
            BUILD_ONLY=false
            ;;
        --input=*)
            INPUT_SOURCE="${1#*=}"
            ;;
        --pattern=*)
            PATTERN_SIZE="${1#*=}"
            ;;
        --square=*)
            SQUARE_SIZE="${1#*=}"
            ;;
        --output=*)
            OUTPUT_FILE="${1#*=}"
            ;;
        --cmake-options=*)
            CMAKE_OPTIONS="${1#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build-dir=<dir>      Build directory (default: $BUILD_DIR)"
            echo "  --run                  Build and run the calibration tool"
            echo "  --input=<source>       Input source (camera index or directory path)"
            echo "  --pattern=<size>       Chessboard pattern size (e.g., 9x6) (default: $PATTERN_SIZE)"
            echo "  --square=<size>        Square size in mm (default: $SQUARE_SIZE)"
            echo "  --output=<file>        Output calibration file (default: $OUTPUT_FILE)"
            echo "  --cmake-options=<opts> Additional CMake options"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --run --input=0 --pattern=9x6 --square=20 --output=my_camera.yaml"
            echo "  $0 --run --input=/path/to/images --pattern=4x4 --square=100"
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

# Create a build directory inside the calibration folder
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure and build the project
echo -e "${YELLOW}Configuring calibration with CMake...${NC}"
cmake ${CMAKE_OPTIONS} ..

echo -e "${YELLOW}Building the calibration tool...${NC}"
make -j$(nproc) calibrate_camera

echo -e "${GREEN}Build complete!${NC}"

# Check if we should run the application
if [ "$BUILD_ONLY" = false ]; then
    if [ -z "$INPUT_SOURCE" ]; then
        echo -e "${RED}Error: Input source not specified. Use --input=<source> parameter.${NC}"
        echo "Use --help for usage information."
        exit 1
    fi

    echo -e "${GREEN}Running camera calibration with:${NC}"
    echo -e "${YELLOW}  Input source: $INPUT_SOURCE${NC}"
    echo -e "${YELLOW}  Pattern size: $PATTERN_SIZE${NC}"
    echo -e "${YELLOW}  Square size: $SQUARE_SIZE mm${NC}"
    echo -e "${YELLOW}  Output file: $OUTPUT_FILE${NC}"
    
    # Run the application with space-separated arguments
    ./calibrate_camera --input "$INPUT_SOURCE" --pattern_size "$PATTERN_SIZE" --square_size "$SQUARE_SIZE" --output "$OUTPUT_FILE"
fi

cd ..
echo -e "${GREEN}Done.${NC}" 