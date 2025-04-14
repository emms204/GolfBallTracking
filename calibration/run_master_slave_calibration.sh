#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default paths
MASTER_IMAGES_DIR="/home/emms/Downloads/Test Video/Ignore/Combined/Master"
SLAVE_IMAGES_DIR="/home/emms/Downloads/Test Video/Ignore/Combined/Slave"
OUTPUT_DIR="../calibration_results"
PATTERN_SIZE="4x4"
SQUARE_SIZE=100  # in mm
MIN_IMAGES=10

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --master=*)
            MASTER_IMAGES_DIR="${1#*=}"
            ;;
        --slave=*)
            SLAVE_IMAGES_DIR="${1#*=}"
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"
            ;;
        --pattern=*)
            PATTERN_SIZE="${1#*=}"
            ;;
        --square=*)
            SQUARE_SIZE="${1#*=}"
            ;;
        --min-images=*)
            MIN_IMAGES="${1#*=}"
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --master=<dir>       Path to master camera chessboard images (default: $MASTER_IMAGES_DIR)"
            echo "  --slave=<dir>        Path to slave camera chessboard images (default: $SLAVE_IMAGES_DIR)"
            echo "  --output=<dir>       Path to output directory for calibration files (default: $OUTPUT_DIR)"
            echo "  --pattern=<size>     Chessboard pattern size (e.g., 4x4) (default: $PATTERN_SIZE)"
            echo "  --square=<size>      Square size in mm (default: $SQUARE_SIZE)"
            echo "  --min-images=<num>   Minimum number of images required (default: $MIN_IMAGES)"
            echo "  --help               Show this help message"
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

echo -e "${GREEN}Starting Master and Slave Camera Calibration...${NC}"

# Build the calibration tool
echo -e "${YELLOW}Building the calibration tool...${NC}"
./build_calibration.sh

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build the calibration tool. Aborting.${NC}"
    exit 1
fi

# Check if image directories exist
if [ ! -d "$MASTER_IMAGES_DIR" ]; then
    echo -e "${RED}Master images directory not found: $MASTER_IMAGES_DIR${NC}"
    exit 1
fi

if [ ! -d "$SLAVE_IMAGES_DIR" ]; then
    echo -e "${RED}Slave images directory not found: $SLAVE_IMAGES_DIR${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define calibration file paths
MASTER_CALIB_FILE="${OUTPUT_DIR}/master_calibration.yaml"
SLAVE_CALIB_FILE="${OUTPUT_DIR}/slave_calibration.yaml"

echo -e "${GREEN}Calibrating Master camera...${NC}"
echo -e "${YELLOW}Using chessboard pattern: $PATTERN_SIZE with square size: $SQUARE_SIZE mm${NC}"
echo -e "${YELLOW}Images directory: $MASTER_IMAGES_DIR${NC}"
echo -e "${YELLOW}Output file: $MASTER_CALIB_FILE${NC}"

# Note: Using space-separated arguments for C++ application
./build/calibrate_camera \
    --input "$MASTER_IMAGES_DIR" \
    --pattern_size "$PATTERN_SIZE" \
    --square_size "$SQUARE_SIZE" \
    --output "$MASTER_CALIB_FILE" \
    --min_images "$MIN_IMAGES"

# Check if master calibration was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Master camera calibration failed. Check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Master camera calibration completed.${NC}"

echo -e "${GREEN}Calibrating Slave camera...${NC}"
echo -e "${YELLOW}Using chessboard pattern: $PATTERN_SIZE with square size: $SQUARE_SIZE mm${NC}"
echo -e "${YELLOW}Images directory: $SLAVE_IMAGES_DIR${NC}"
echo -e "${YELLOW}Output file: $SLAVE_CALIB_FILE${NC}"

# Note: Using space-separated arguments for C++ application
./build/calibrate_camera \
    --input "$SLAVE_IMAGES_DIR" \
    --pattern_size "$PATTERN_SIZE" \
    --square_size "$SQUARE_SIZE" \
    --output "$SLAVE_CALIB_FILE" \
    --min_images "$MIN_IMAGES"

# Check if slave calibration was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Slave camera calibration failed. Check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Slave camera calibration completed.${NC}"

echo -e "${GREEN}Calibration files saved to $OUTPUT_DIR/${NC}"
echo -e "${GREEN}You can now use the detection_test_app with the calibration files:${NC}"
echo -e "${YELLOW}cd ../src${NC}"
echo -e "${YELLOW}./run_detection_test_app.sh --calibration=$MASTER_CALIB_FILE${NC}"

echo -e "${GREEN}Done!${NC}" 