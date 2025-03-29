#!/bin/bash

# Function to display help
function display_help {
    echo "ONNX Object Detection Docker Runner"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  video FILE    Run detection on the specified video file"
    echo "  camera        Run detection using camera"
    echo "  build         Only build the Docker image"
    echo "  push TAG      Build and push to Docker Hub with specified tag"
    echo "  help          Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 video sample.mp4    # Run detection on sample.mp4"
    echo "  $0 camera              # Run detection using camera"
    echo "  $0 build               # Only build the Docker image"
    echo "  $0 push v1.0           # Push image as yourusername/onnx-detector:v1.0"
}

# Build the Docker image
function build_image {
    echo "Building Docker image..."
    docker build -t onnx-detector:latest .
    
    if [ $? -eq 0 ]; then
        echo "Docker image built successfully!"
    else
        echo "Error building Docker image."
        exit 1
    fi
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Parse command line arguments
case "$1" in
    video)
        if [ -z "$2" ]; then
            echo "Error: No video file specified."
            echo "Usage: $0 video FILE"
            exit 1
        fi
        
        # Check if the video file exists in the videos directory
        if [ ! -f "videos/$2" ]; then
            echo "Error: Video file 'videos/$2' not found."
            echo "Please place your video files in the 'videos' directory."
            exit 1
        fi
        
        build_image
        
        echo "Running detection on video: $2"
        docker run -it --rm -v "$(pwd)/videos:/app/videos" onnx-detector:latest video "$2"
        ;;
        
    camera)
        build_image
        
        echo "Running detection using camera..."
        docker run -it --rm --device=/dev/video0 onnx-detector:latest camera
        ;;
        
    build)
        build_image
        ;;
        
    push)
        if [ -z "$2" ]; then
            echo "Error: No tag specified."
            echo "Usage: $0 push TAG"
            exit 1
        fi
        
        # Get Docker Hub username
        read -p "Enter your Docker Hub username: " username
        
        build_image
        
        # Tag and push the image
        docker tag onnx-detector:latest "$username/onnx-detector:$2"
        docker push "$username/onnx-detector:$2"
        
        if [ $? -eq 0 ]; then
            echo "Image pushed to Docker Hub as $username/onnx-detector:$2"
        else
            echo "Error pushing image to Docker Hub."
            exit 1
        fi
        ;;
        
    help|*)
        display_help
        ;;
esac 