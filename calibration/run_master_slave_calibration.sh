#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Master and Slave Camera Calibration...${NC}"

# Build the calibration tool
echo -e "${YELLOW}Building the calibration tool...${NC}"
./build_calibration.sh

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build the calibration tool. Aborting.${NC}"
    exit 1
fi

# Define paths
MASTER_IMAGES_DIR="/home/emms/Downloads/Test Video/Ignore/Combined/Master"
SLAVE_IMAGES_DIR="/home/emms/Downloads/Test Video/Ignore/Combined/Slave"
MASTER_CALIB_FILE="master_calibration.yaml"
SLAVE_CALIB_FILE="slave_calibration.yaml"

# Check if image directories exist
if [ ! -d "$MASTER_IMAGES_DIR" ]; then
    echo -e "${RED}Master images directory not found: $MASTER_IMAGES_DIR${NC}"
    exit 1
fi

if [ ! -d "$SLAVE_IMAGES_DIR" ]; then
    echo -e "${RED}Slave images directory not found: $SLAVE_IMAGES_DIR${NC}"
    exit 1
fi

# Pattern size (4x4 internal corners in a 5x5 chessboard)
PATTERN_SIZE="4x4"
SQUARE_SIZE=100  # in mm

echo -e "${GREEN}Calibrating Master camera...${NC}"
echo -e "${YELLOW}Using chessboard pattern: $PATTERN_SIZE with square size: $SQUARE_SIZE mm${NC}"
echo -e "${YELLOW}Images directory: $MASTER_IMAGES_DIR${NC}"
echo -e "${YELLOW}Output file: $MASTER_CALIB_FILE${NC}"

./build/calibrate_camera \
    --input "$MASTER_IMAGES_DIR" \
    --pattern_size "$PATTERN_SIZE" \
    --square_size "$SQUARE_SIZE" \
    --output "$MASTER_CALIB_FILE" \
    --min_images 10

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

./build/calibrate_camera \
    --input "$SLAVE_IMAGES_DIR" \
    --pattern_size "$PATTERN_SIZE" \
    --square_size "$SQUARE_SIZE" \
    --output "$SLAVE_CALIB_FILE" \
    --min_images 10

# Check if slave calibration was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Slave camera calibration failed. Check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Slave camera calibration completed.${NC}"

# Create directory for our detection test app
mkdir -p "../calibration_results"

# Copy calibration files to the detection test app directory
cp "$MASTER_CALIB_FILE" "../calibration_results/"
cp "$SLAVE_CALIB_FILE" "../calibration_results/"

echo -e "${GREEN}Calibration files copied to ../calibration_results/${NC}"
echo -e "${GREEN}You can now use the detection_test_app with the calibration files:${NC}"
echo -e "${YELLOW}cd ../src${NC}"
echo -e "${YELLOW}./run_detection_test_app.sh --calibration ../calibration_results/master_calibration.yaml${NC}"

echo -e "${GREEN}Done!${NC}" 