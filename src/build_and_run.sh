#!/bin/bash

# Build and run C++ Golf Ball Object Detector

# Check if required packages are installed
if ! command -v pkg-config &> /dev/null; then
    echo "pkg-config is not installed. Please install it first."
    exit 1
fi

if ! pkg-config --exists opencv4; then
    echo "OpenCV 4 not found. Please install OpenCV with DNN module support."
    echo "On Ubuntu/Debian: sudo apt install libopencv-dev"
    echo "On CentOS/RHEL: sudo yum install opencv-devel"
    echo "See https://opencv.org/releases/ for other installation methods."
    exit 1
fi

# Build the project
echo "Building the project..."
mkdir -p build
cd build || exit
cmake ..
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed. Please check the error messages above."
    exit 1
fi

cd ..

# Function to print usage information
print_usage() {
    echo "Usage: $0 [OPTION]"
    echo "Build and run the C++ Golf Ball Object Detector"
    echo ""
    echo "Options:"
    echo "  --run VIDEO OUTPUT   Build and run with specified input VIDEO and OUTPUT path"
    echo "  --build-only         Only build the project without running"
    echo "  --help               Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0 --run test.mp4 output.avi"
    echo "  $0 --build-only"
}

# Parse command line arguments
case "$1" in
    --run)
        if [ $# -lt 3 ]; then
            echo "Error: --run requires VIDEO and OUTPUT paths"
            print_usage
            exit 1
        fi
        VIDEO_PATH="$2"
        OUTPUT_PATH="$3"
        
        if [ ! -f "$VIDEO_PATH" ]; then
            echo "Error: Input video file '$VIDEO_PATH' does not exist."
            exit 1
        fi
        
        echo "Running object detector..."
        echo "Input: $VIDEO_PATH"
        echo "Output: $OUTPUT_PATH"
        
        ./build/object_detector --video="$VIDEO_PATH" --output="$OUTPUT_PATH" --model="TrainResults1/best.onnx"
        ;;
    --build-only)
        echo "Build completed. You can run the detector with:"
        echo "./build/object_detector --video=<input_video_path> --output=<output_video_path>"
        ;;
    --help)
        print_usage
        ;;
    *)
        echo "Error: Unknown option '$1'"
        print_usage
        exit 1
        ;;
esac

exit 0 