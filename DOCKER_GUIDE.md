# Docker Guide: Real-Time Object Detection

This guide explains how to run the ONNX-based object detection application using Docker, which allows you to run the application on any platform (Windows, macOS, or Linux) without installing dependencies directly on your system.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
- For camera access: a webcam connected to your computer

## Quick Start

We've provided a convenient script that handles building and running the Docker container for you.

### 1. Make the script executable (Linux/macOS only)

```bash
chmod +x run_docker.sh
```

### 2. Run the object detector

#### With a video file:

1. Place your video files in the `videos` directory
2. Run the script with the video filename:

```bash
./run_docker.sh video sample.mp4
```

#### With a camera:

```bash
./run_docker.sh camera
```

## Manual Docker Commands

If you prefer to use Docker commands directly:

### Building the Docker image

```bash
docker build -t onnx-detector:latest .
```

### Running the container

#### With a video file:

```bash
docker run -it --rm -v $(pwd)/videos:/app/videos onnx-detector:latest video sample.mp4
```

#### With a camera:

```bash
docker run -it --rm --device=/dev/video0 onnx-detector:latest camera
```

## Using Docker Compose

We also provide a `docker-compose.yml` file for convenience:

1. Edit the `docker-compose.yml` file to select video or camera mode
2. Run:

```bash
docker compose up --build
```

## Pushing to Docker Hub

If you want to share the image with others via Docker Hub:

```bash
./run_docker.sh push v1.0
```

You'll be prompted for your Docker Hub username.

## Windows-Specific Notes

### Running the script on Windows

On Windows, you can run the commands from the script directly, or use PowerShell:

```powershell
# Build the image
docker build -t onnx-detector:latest .

# Run with a video
docker run -it --rm -v ${PWD}/videos:/app/videos onnx-detector:latest video sample.mp4

# Run with camera
docker run -it --rm --device=/dev/video0 onnx-detector:latest camera
```

### Camera access on Windows

For Windows, camera access requires additional configuration:

1. In Docker Desktop, go to Settings > Resources > File Sharing
2. Add your project directory to the shared paths
3. For webcam access, you may need to install and use [OBS Virtual Camera](https://obsproject.com/) or similar solution

## Troubleshooting

### Common issues:

1. **Error: Cannot connect to the Docker daemon**
   - Make sure Docker Desktop is running
   - On Linux, you may need to add your user to the docker group:
     ```bash
     sudo usermod -aG docker $USER
     # Then log out and back in
     ```

2. **Error accessing camera**
   - Make sure your webcam is connected and working
   - Try changing the device ID (e.g., `/dev/video1` instead of `/dev/video0`)
   - On Windows, see the Windows-specific notes about camera access

3. **No video output appears**
   - Check that X11 forwarding is properly configured (for GUI output)
   - For headless servers, modify the Dockerfile to save output video files instead

4. **Performance issues**
   - Try running with GPU support if available (requires NVIDIA Docker)
   - Lower the resolution of the input video 