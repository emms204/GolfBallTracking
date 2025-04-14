#!/bin/bash

# Build script for camera calibration project

set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build the project
echo "Configuring with CMake..."
cmake ..

echo "Building the project..."
# First build the libraries in dependency order
echo "Building common library..."
make -j$(nproc) common

echo "Building calibration library..."
make -j$(nproc) calibration

echo "Building detector library..."
make -j$(nproc) detector

# Then build everything else
echo "Building applications..."
make -j$(nproc)

echo "Build complete!"

# Check if we should run the calibration tool
if [ "$1" == "--calibrate" ]; then
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
    ./calibration/calibrate_camera --input "$INPUT_SOURCE" --pattern_size "$PATTERN_SIZE" --square_size "$SQUARE_SIZE" --output "$OUTPUT_FILE"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --calibrate [input] [pattern_size] [square_size] [output]  Build and run the calibration tool"
    echo "  --help, -h                                                Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --calibrate 0 9x6 20 my_camera.yaml    # Calibrate using camera 0 with 9x6 pattern"
    echo "  $0 --calibrate /path/to/images 9x6 25     # Calibrate using images with 9x6 pattern, 25mm squares"
fi

cd ..
echo "Done." 