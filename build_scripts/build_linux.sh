#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Camera Calibration Toolkit for Linux...${NC}"

# Check for ONNX Runtime
ONNXRUNTIME_DIR="./onnxruntime-linux-x64-1.21.0"
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo -e "${YELLOW}ONNX Runtime not found. Downloading ONNX Runtime...${NC}"
    
    # Create directory
    mkdir -p "$ONNXRUNTIME_DIR"
    
    # Download ONNX Runtime
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz -O onnxruntime.tgz
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download ONNX Runtime. Please check your internet connection.${NC}"
        exit 1
    fi
    
    # Extract
    tar -xzf onnxruntime.tgz -C .
    rm onnxruntime.tgz
    
    echo -e "${GREEN}Successfully downloaded and extracted ONNX Runtime.${NC}"
fi

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure the project with CMake
echo -e "${GREEN}Configuring project with CMake...${NC}"
cmake ..

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed.${NC}"
    cd ..
    exit 1
fi

# Build the project
echo -e "${GREEN}Building project...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed.${NC}"
    cd ..
    exit 1
fi

echo -e "${GREEN}Build completed successfully.${NC}"
echo -e "${GREEN}Binaries can be found in the build directory.${NC}"

cd .. 