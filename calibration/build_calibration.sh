#!/bin/bash

# Build script for the calibration tool

set -e

# Create a build directory inside the calibration folder
mkdir -p build
cd build

# Configure and build the project
echo "Configuring calibration with CMake..."
cmake ..

echo "Building the calibration tool..."
make -j$(nproc) calibrate_camera

echo "Build complete!"

# Check if we should run the application
if [ "$1" == "--run" ]; then
    # Default parameters if none provided
    INPUT_SOURCE=${2:-0}  # Default to camera 0
    PATTERN_SIZE=${3:-"9x6"}  # Default pattern size
    SQUARE_SIZE=${4:-20}  # Default square size in mm
    OUTPUT_FILE=${5:-"camera_calibration.yaml"}  # Default output file

    echo "Running camera calibration with:"
    echo "  Input source: $INPUT_SOURCE"
    echo "  Pattern size: $PATTERN_SIZE"
    echo "  Square size: $SQUARE_SIZE mm"
    echo "  Output file: $OUTPUT_FILE"
    
    # Run the application
    ./calibrate_camera --input "$INPUT_SOURCE" --pattern_size "$PATTERN_SIZE" --square_size "$SQUARE_SIZE" --output "$OUTPUT_FILE"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --run [input] [pattern_size] [square_size] [output]  Build and run the calibration tool"
    echo "  --help, -h                                          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --run 0 9x6 20 my_camera.yaml    # Calibrate using camera 0 with 9x6 pattern"
    echo "  $0 --run /path/to/images 9x6 25     # Calibrate using images with 9x6 pattern, 25mm squares"
fi

cd ..
echo "Done." 