#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT=$(pwd)

# Default values
CALIBRATION_FILE=""
VIDEO_FILE=""
TEST_ENHANCED=1
TEST_DETECTION_APP=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --calibration)
            CALIBRATION_FILE="$2"
            shift
            ;;
        --video)
            VIDEO_FILE="$2"
            shift
            ;;
        --only-enhanced)
            TEST_DETECTION_APP=0
            ;;
        --only-detection-app)
            TEST_ENHANCED=0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --calibration <file>    Path to calibration file"
            echo "  --video <file>          Path to video file"
            echo "  --only-enhanced         Test only the enhanced detector"
            echo "  --only-detection-app    Test only the detection_test_app"
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

if [ -z "$VIDEO_FILE" ]; then
    echo -e "${RED}Error: Video file not specified. Use --video.${NC}"
    exit 1
fi

# Check if files exist
if [ ! -f "$CALIBRATION_FILE" ]; then
    echo -e "${RED}Error: Calibration file not found: $CALIBRATION_FILE${NC}"
    exit 1
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${RED}Error: Video file not found: $VIDEO_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Starting detector testing...${NC}"
echo -e "${GREEN}Project root: $PROJECT_ROOT${NC}"
echo -e "${GREEN}Calibration file: $CALIBRATION_FILE${NC}"
echo -e "${GREEN}Video file: $VIDEO_FILE${NC}"

# Set up environment variables for ONNX Runtime
ONNXRUNTIME_DIR="$PROJECT_ROOT/../onnxruntime-linux-x64-1.21.0/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ONNXRUNTIME_DIR"

# Verify ONNX Runtime library exists
if [ ! -f "$ONNXRUNTIME_DIR/libonnxruntime.so" ]; then
    echo -e "${RED}Error: ONNX Runtime library not found at $ONNXRUNTIME_DIR/libonnxruntime.so${NC}"
    exit 1
fi

# Build the project using build_test.sh
echo -e "${GREEN}Building the project...${NC}"
"$PROJECT_ROOT/build_test.sh"
if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed. Please check error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"

# Test Enhanced Detector
if [ $TEST_ENHANCED -eq 1 ]; then
    echo -e "\n${YELLOW}Testing Test Detector with calibration...${NC}"
    
    # Look for the detector executable in multiple possible locations
    DETECTOR_EXEC=""
    for possible_path in \
        "$PROJECT_ROOT/build/test_detector" \
        "$PROJECT_ROOT/build/test/test_detector" \
        "$PROJECT_ROOT/test/build/test_detector"; do
        if [ -f "$possible_path" ]; then
            DETECTOR_EXEC="$possible_path"
            break
        fi
    done
    
    if [ -z "$DETECTOR_EXEC" ]; then
        echo -e "${RED}Error: test_detector executable not found!${NC}"
        echo -e "${YELLOW}Checking possible locations:${NC}"
        find "$PROJECT_ROOT" -name "test_detector" 2>/dev/null
    else
        echo -e "${GREEN}Found test_detector at: $DETECTOR_EXEC${NC}"
        echo -e "${GREEN}Running Test Detector...${NC}"
        echo -e "${YELLOW}Command: $DETECTOR_EXEC --params=\"$CALIBRATION_FILE\" --video=\"$VIDEO_FILE\" --model=\"$PROJECT_ROOT/../best.onnx\" --classes=\"$PROJECT_ROOT/../classes.txt\"${NC}"
        
        # Check if all files exist
        echo -e "Checking for required files:"
        echo -e "  Model file: $PROJECT_ROOT/../best.onnx - $([ -f "$PROJECT_ROOT/../best.onnx" ] && echo "EXISTS" || echo "MISSING")"
        echo -e "  Classes file: $PROJECT_ROOT/../classes.txt - $([ -f "$PROJECT_ROOT/../classes.txt" ] && echo "EXISTS" || echo "MISSING")"
        
        # Add strace to debug the segfault
        strace -f "$DETECTOR_EXEC" --params="$CALIBRATION_FILE" --video="$VIDEO_FILE" --model="$PROJECT_ROOT/../best.onnx" --classes="$PROJECT_ROOT/../classes.txt" 2>&1 | grep -i "libonnxruntime"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Test Detector test completed successfully!${NC}"
        else
            echo -e "${RED}Test Detector test failed!${NC}"
        fi
    fi
fi

# Test Detection Test App
if [ $TEST_DETECTION_APP -eq 1 ]; then
    echo -e "\n${YELLOW}Testing Detection Test App with calibration...${NC}"
    
    if [ ! -f "src/detection_test_app" ]; then
        echo -e "${RED}Error: detection_test_app executable not found!${NC}"
    else
        echo -e "${GREEN}Running Detection Test App...${NC}"
        ./src/detection_test_app --calibration "$CALIBRATION_FILE" --video "$VIDEO_FILE" --model "$PROJECT_ROOT/best.onnx" --classes "$PROJECT_ROOT/classes.txt"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Detection Test App test completed successfully!${NC}"
        else
            echo -e "${RED}Detection Test App test failed!${NC}"
        fi
    fi
fi

echo -e "\n${GREEN}All tests completed!${NC}"
exit 0 