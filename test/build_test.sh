#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")" && cd .. && pwd)"

echo -e "${GREEN}Starting build process...${NC}"
echo -e "${GREEN}Project root: $PROJECT_ROOT${NC}"

# Build the detector library
echo -e "\n${YELLOW}Building detector library...${NC}"
cd "$PROJECT_ROOT/detector"
mkdir -p build
cd build
cmake ..
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build detector library. Aborting.${NC}"
    exit 1
fi

# Build the common library if it exists
if [ -d "$PROJECT_ROOT/common" ]; then
    echo -e "\n${YELLOW}Building common library...${NC}"
    cd "$PROJECT_ROOT/common"
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build common library. Aborting.${NC}"
        exit 1
    fi
fi

# Build the calibration library if it exists
if [ -d "$PROJECT_ROOT/calibration" ]; then
    echo -e "\n${YELLOW}Building calibration library...${NC}"
    cd "$PROJECT_ROOT/calibration"
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build calibration library. Aborting.${NC}"
        exit 1
    fi
fi

# Build the test application
echo -e "\n${YELLOW}Building test application...${NC}"
cd "$PROJECT_ROOT/test"
mkdir -p build
cd build
cmake ..
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build test application. Aborting.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Build completed successfully!${NC}"
echo -e "You can now run the test_detectors.sh script."
exit 0 