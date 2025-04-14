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
VENV_PATH="/opt/venv/global/bin/activate"
PATTERN_SIZE="4x4"
SQUARE_SIZE=100

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
        --venv=*)
            VENV_PATH="${1#*=}"
            ;;
        --pattern=*)
            PATTERN_SIZE="${1#*=}"
            ;;
        --square=*)
            SQUARE_SIZE="${1#*=}"
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --master=<dir>    Path to master camera chessboard images (default: $MASTER_IMAGES_DIR)"
            echo "  --slave=<dir>     Path to slave camera chessboard images (default: $SLAVE_IMAGES_DIR)"
            echo "  --output=<dir>    Path to output directory for calibration files (default: $OUTPUT_DIR)"
            echo "  --venv=<path>     Path to virtual environment activate script (default: $VENV_PATH)"
            echo "  --pattern=<size>  Chessboard pattern size (e.g., 4x4) (default: $PATTERN_SIZE)"
            echo "  --square=<size>   Square size in mm (default: $SQUARE_SIZE)"
            echo "  --help            Show this help message"
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

echo -e "${GREEN}Running Python-based Camera Calibration...${NC}"
echo -e "${YELLOW}Master images directory: $MASTER_IMAGES_DIR${NC}"
echo -e "${YELLOW}Slave images directory: $SLAVE_IMAGES_DIR${NC}"
echo -e "${YELLOW}Output directory: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}Pattern size: $PATTERN_SIZE${NC}"
echo -e "${YELLOW}Square size: $SQUARE_SIZE mm${NC}"

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CALIB_SCRIPT="${SCRIPT_DIR}/calibration_script.py"

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

# Activate virtual environment if it exists
if [ -f "$VENV_PATH" ]; then
    echo -e "${YELLOW}Activating virtual environment: $VENV_PATH${NC}"
    source "$VENV_PATH"
else
    echo -e "${YELLOW}Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}Continuing without activating a virtual environment${NC}"
fi

# Check for Python and required packages
echo -e "${YELLOW}Checking for required Python packages...${NC}"

PYTHON_CMD=""
# Try to find Python 3
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        if $cmd -c "import sys; exit(0 if sys.version_info.major >= 3 else 1)" &> /dev/null; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Python 3 not found. Please install Python 3 to use this script.${NC}"
    exit 1
fi

# Check for required packages
$PYTHON_CMD -c "import numpy, cv2, pickle" 2> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing required Python packages...${NC}"
    pip install numpy opencv-python matplotlib
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install required Python packages. Please install them manually:${NC}"
        echo -e "${YELLOW}pip install numpy opencv-python matplotlib${NC}"
        exit 1
    fi
fi

# Run the Python calibration script
echo -e "${GREEN}Running Python calibration script...${NC}"
$PYTHON_CMD "$PYTHON_CALIB_SCRIPT" --master="$MASTER_IMAGES_DIR" --slave="$SLAVE_IMAGES_DIR" --output="$OUTPUT_DIR" --pattern="$PATTERN_SIZE" --square="$SQUARE_SIZE"

# Check if calibration was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Python calibration script failed. Please check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Python-based calibration completed.${NC}"
echo -e "${GREEN}You can now use the detection_test_app with the calibration files:${NC}"
echo -e "${YELLOW}cd ../src${NC}"
echo -e "${YELLOW}./run_detection_test_app.sh --calibration=$OUTPUT_DIR/master_calibration.yaml${NC}"
echo -e "${YELLOW}./run_detection_test_app.sh --calibration=$OUTPUT_DIR/slave_calibration.yaml${NC}"

echo -e "${GREEN}Done!${NC}" 