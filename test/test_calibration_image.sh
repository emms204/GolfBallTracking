#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(readlink -f $(dirname $0)/..)"

# Default values
CALIBRATION_FILE=""
INPUT_IMAGE=""
OUTPUT_IMAGE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --calibration)
            CALIBRATION_FILE="$2"
            shift
            ;;
        --input)
            INPUT_IMAGE="$2"
            shift
            ;;
        --output)
            OUTPUT_IMAGE="$2"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --calibration <file>    Path to calibration file (required)"
            echo "  --input <file>          Path to input image file (required)"
            echo "  --output <file>         Path to output image file (default: calibration_test_output.jpg)"
            echo "  --help                  Show this help message"
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

# Check for required arguments
if [ -z "$CALIBRATION_FILE" ]; then
    echo -e "${RED}Error: Calibration file not specified. Use --calibration.${NC}"
    exit 1
fi

if [ -z "$INPUT_IMAGE" ]; then
    echo -e "${RED}Error: Input image file not specified. Use --input.${NC}"
    exit 1
fi

# Set default output if not specified
if [ -z "$OUTPUT_IMAGE" ]; then
    OUTPUT_IMAGE="calibration_test_output.jpg"
fi

# Check if calibration file exists
if [ ! -f "$CALIBRATION_FILE" ]; then
    echo -e "${RED}Error: Calibration file not found: $CALIBRATION_FILE${NC}"
    exit 1
fi

# Check if input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo -e "${RED}Error: Input image file not found: $INPUT_IMAGE${NC}"
    exit 1
fi

echo -e "${GREEN}Starting calibration test...${NC}"
echo -e "${GREEN}Calibration file: $CALIBRATION_FILE${NC}"
echo -e "${GREEN}Input image: $INPUT_IMAGE${NC}"
echo -e "${GREEN}Output image: $OUTPUT_IMAGE${NC}"

# Build the project
echo -e "${GREEN}Building the project...${NC}"
cd "$PROJECT_ROOT"
mkdir -p build
cd build

# Run CMake if needed
if [ ! -f "CMakeCache.txt" ]; then
    echo -e "${GREEN}Configuring project with CMake...${NC}"
    cmake ..
fi

# Build the project
echo -e "${GREEN}Building...${NC}"
make -j$(nproc) test_calibration_image

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed. Please check error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"

# Run the test_calibration_image program
TEST_EXE=""
# Check multiple possible locations
for possible_path in \
    "src/test_calibration_image" \
    "test_calibration_image" \
    "../build/src/test_calibration_image" \
    "../build/test_calibration_image"; do
    if [ -f "$possible_path" ]; then
        TEST_EXE="$possible_path"
        break
    fi
done

if [ -z "$TEST_EXE" ]; then
    echo -e "${RED}Error: test_calibration_image executable not found!${NC}"
    echo -e "${YELLOW}Checking possible locations:${NC}"
    find "$PROJECT_ROOT" -name "test_calibration_image" 2>/dev/null
    exit 1
fi

echo -e "${GREEN}Found executable at: $TEST_EXE${NC}"
echo -e "${GREEN}Running calibration test...${NC}"
"$TEST_EXE" "$INPUT_IMAGE" "$CALIBRATION_FILE" "$OUTPUT_IMAGE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Calibration test completed successfully!${NC}"
    echo -e "${GREEN}Generated output image: $OUTPUT_IMAGE${NC}"
    
    # Attempt to display the image if running in GUI environment
    if command -v display &> /dev/null; then
        display "$OUTPUT_IMAGE" &
    elif command -v eog &> /dev/null; then
        eog "$OUTPUT_IMAGE" &
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$OUTPUT_IMAGE" &
    else
        echo -e "${YELLOW}No image viewer found. Please open the output image manually.${NC}"
    fi
else
    echo -e "${RED}Calibration test failed!${NC}"
fi

exit 0 