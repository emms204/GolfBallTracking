# YOLO+ByteTrack Integration

This project integrates ONNX YOLO object detection with ByteTrack for object tracking in video streams.

## Prerequisites

- CMake (version 3.14 or higher)
- OpenCV (with C++ support)
- ONNX Runtime
- Eigen3 (used by ByteTrack)
- C++17 compatible compiler

## Building

1. First, clone this repository and update submodules:

```bash
git clone <repository_url>
cd <repository_name>
```

2. Place the ByteTrack-cpp library in the project directory:

```bash
# If ByteTrack-cpp is not already in the project directory
git clone https://github.com/example/ByteTrack-cpp.git
```

3. Build with CMake:

```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
make
```

Replace `/path/to/onnxruntime` with the actual path to your ONNX Runtime installation.

## Running

The program supports both video files and camera input:

```bash
# For video file input
./yolo_bytetrack --video=path/to/video.mp4 --model=path/to/your/model.onnx --classes=path/to/classes.txt

# For camera input
./yolo_bytetrack --camera=0 --model=path/to/your/model.onnx --classes=path/to/classes.txt
```

### Command-line Options

- `--help` - Show help message
- `--video` - Path to input video file
- `--camera` - Use camera as input (specify device ID, e.g., 0)
- `--model` - Path to ONNX model (default: best.onnx)
- `--classes` - Path to class names file (default: TrainResults1/classes.txt)
- `--conf` - Confidence threshold (default: 0.25)
- `--nms` - NMS threshold (default: 0.45)
- `--fps` - Frames per second for tracking (default: 30)
- `--track_buffer` - Track buffer for ByteTrack (default: 30)
- `--track_thresh` - Track threshold for ByteTrack (default: 0.5)
- `--high_thresh` - High detection threshold for ByteTrack (default: 0.6)
- `--match_thresh` - Match threshold for ByteTrack (default: 0.8)

## How It Works

1. The program processes each frame from the video source.
2. YOLO detection model detects objects in each frame.
3. Detections are converted to the format expected by ByteTrack.
4. ByteTrack algorithm tracks objects across frames.
5. Results are displayed with tracking IDs.

## Architecture

- `onnx_detector.cpp` - ONNX-based YOLO object detector
- `yolo_bytetrack_integration.cpp` - Main integration code
- `ByteTrack-cpp` - ByteTrack library (used as a subproject)

## License

This project is licensed under the MIT License - see the LICENSE file for details.