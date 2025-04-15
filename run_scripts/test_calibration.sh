#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
TEST_IMAGE=""
TEST_DIRECTORY=""
CALIBRATION_FILE=""
OUTPUT_DIR="calibration_test_results"
PATTERN_SIZE="9x6"
SQUARE_SIZE=20

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --image=*)
      TEST_IMAGE="${arg#*=}"
      ;;
    --dir=*)
      TEST_DIRECTORY="${arg#*=}"
      ;;
    --calibration=*)
      CALIBRATION_FILE="${arg#*=}"
      ;;
    --output=*)
      OUTPUT_DIR="${arg#*=}"
      ;;
    --pattern=*)
      PATTERN_SIZE="${arg#*=}"
      ;;
    --square=*)
      SQUARE_SIZE="${arg#*=}"
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --image=<path>          Test single image with existing calibration"
      echo "  --dir=<path>            Directory with calibration images"
      echo "  --calibration=<path>    Existing calibration file (for testing)"
      echo "  --output=<path>         Output directory (default: calibration_test_results)"
      echo "  --pattern=<WxH>         Chessboard pattern size (default: 9x6)"
      echo "  --square=<mm>           Chessboard square size in mm (default: 20)"
      echo "  --help                  Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $arg${NC}"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if we have a build directory and the necessary tools
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Build directory not found, building project...${NC}"
    ./build_linux.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed. Cannot continue with tests.${NC}"
        exit 1
    fi
fi

# Test a single image with existing calibration
if [ -n "$TEST_IMAGE" ] && [ -n "$CALIBRATION_FILE" ]; then
    if [ ! -f "$TEST_IMAGE" ]; then
        echo -e "${RED}Test image not found: $TEST_IMAGE${NC}"
        exit 1
    fi
    
    if [ ! -f "$CALIBRATION_FILE" ]; then
        echo -e "${RED}Calibration file not found: $CALIBRATION_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Testing calibration with single image...${NC}"
    echo -e "${GREEN}Image: $TEST_IMAGE${NC}"
    echo -e "${GREEN}Calibration: $CALIBRATION_FILE${NC}"
    
    # Run the test_calibration_image tool
    build/test/test_calibration_image --calibration "$CALIBRATION_FILE" --input "$TEST_IMAGE" --output "$OUTPUT_DIR/test_result.jpg"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Testing failed.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Test completed. Result image saved to $OUTPUT_DIR/test_result.jpg${NC}"
fi

# Run calibration on a directory of images
if [ -n "$TEST_DIRECTORY" ]; then
    if [ ! -d "$TEST_DIRECTORY" ]; then
        echo -e "${RED}Test directory not found: $TEST_DIRECTORY${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Running calibration on directory...${NC}"
    echo -e "${GREEN}Directory: $TEST_DIRECTORY${NC}"
    echo -e "${GREEN}Pattern size: $PATTERN_SIZE${NC}"
    echo -e "${GREEN}Square size: $SQUARE_SIZE mm${NC}"
    
    # Run the calibration tool
    build/calibration/calibrate_camera --input "$TEST_DIRECTORY" --pattern_size "$PATTERN_SIZE" --square_size "$SQUARE_SIZE" --output "$OUTPUT_DIR/calibration_result.yaml"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Calibration failed.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Calibration completed. Parameters saved to $OUTPUT_DIR/calibration_result.yaml${NC}"
    
    # Also test the first image with the new calibration
    FIRST_IMAGE=$(find "$TEST_DIRECTORY" -type f \( -name "*.jpg" -o -name "*.png" \) | head -n 1)
    if [ -n "$FIRST_IMAGE" ]; then
        echo -e "${GREEN}Testing new calibration with first image: $FIRST_IMAGE${NC}"
        
        build/test/test_calibration_image --calibration "$OUTPUT_DIR/calibration_result.yaml" --input "$FIRST_IMAGE" --output "$OUTPUT_DIR/calibration_test_result.jpg"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Calibration test completed. Result image saved to $OUTPUT_DIR/calibration_test_result.jpg${NC}"
        fi
    fi
fi

echo -e "${GREEN}All tests completed.${NC}" 