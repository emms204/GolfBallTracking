# Real-Time Object Detection with ONNX Runtime

This repository contains a C++ implementation of real-time object detection using ONNX Runtime and OpenCV. It's designed to work with YOLO-based object detection models exported to ONNX format.

## Features

- Fast object detection using ONNX Runtime
- Support for YOLO models (with built-in NMS)
- Real-time video processing
- Camera feed support
- Letterbox resizing to maintain aspect ratio
- Proper bounding box scaling and coordinate handling
- Configurable confidence and NMS thresholds

## Requirements

- C++ compiler (GCC/Clang)
- CMake (3.10 or higher)
- OpenCV (4.0 or higher)
- ONNX Runtime (1.16 or higher)

## Directory Structure

```
├── src/                          # Source code
│   ├── onnx_detector.cpp         # Main detector implementation
│   ├── onnx_detector.h           # Detector header
│   ├── CMakeLists.txt            # CMake build configuration
│   ├── build_and_run.sh          # Build and run script 
│   └── run_onnx_detector.sh      # Script to run the detector
├── videos/                       # Example videos
├── best.onnx                     # ONNX model file
├── classes.txt                   # Class names file
└── requirements.txt              # Python requirements (for auxiliary scripts)
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/onnx-object-detection.git
cd onnx-object-detection
```

### 2. Install dependencies

#### On Ubuntu/Debian:

```bash
# Install OpenCV
sudo apt-get update
sudo apt-get install -y libopencv-dev

# Install ONNX Runtime
# Download from https://github.com/microsoft/onnxruntime/releases
# Example for Linux x64:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
```

#### On macOS:

```bash
# Install OpenCV
brew install opencv

# Install ONNX Runtime
# Download from https://github.com/microsoft/onnxruntime/releases
# Example for macOS:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-x64-1.16.0.tgz
tar -xzf onnxruntime-osx-x64-1.16.0.tgz
```

### 3. Build the project

```bash
cd src
mkdir -p build
cd build
cmake ..
make
```

Alternatively, use the provided build script:

```bash
cd src
chmod +x build_and_run.sh
./build_and_run.sh
```

## Usage

### Running the detector on a video file

```bash
./src/run_onnx_detector.sh path/to/video.mp4 path/to/model.onnx path/to/classes.txt
```

### With custom confidence and NMS thresholds

```bash
./src/run_onnx_detector.sh path/to/video.mp4 path/to/model.onnx path/to/classes.txt --conf=0.25 --nms=0.45
```

### Using a camera feed

```bash
./src/run_onnx_detector.sh --camera=0 path/to/model.onnx path/to/classes.txt
```

## Example

```bash
# Run detection on a sample video
./src/run_onnx_detector.sh videos/sample.mp4 best.onnx classes.txt

# Run with lower confidence threshold
./src/run_onnx_detector.sh videos/sample.mp4 best.onnx classes.txt --conf=0.2
```

## Configuration

The detector can be configured with the following parameters:

- `--conf`: Confidence threshold (default: 0.25)
- `--nms`: Non-maximum suppression threshold (default: 0.45)
- `--camera`: Use camera as input (specify device ID, e.g., 0)
- `--video`: Path to input video file

## Model Support

The detector is designed to work with YOLO models exported to ONNX format. It supports models with the following output formats:

1. Models with built-in NMS: Output shape [batch, num_detections, 6] where each detection is [x, y, w, h, confidence, class_id]
2. Alternative format: Output shape [num_detections, 6]
3. Legacy format: Output shape [1, num_classes+5, num_boxes]

The detector automatically detects the output format and processes it accordingly.

## Implementation Details

The detector implements the following pipeline:

1. **Preprocessing**: 
   - Resize the input image with letterboxing to maintain aspect ratio
   - Convert BGR to RGB
   - Normalize pixel values to 0-1 range

2. **Inference**:
   - Run the ONNX model using ONNX Runtime
   - Process output based on detected format

3. **Postprocessing**:
   - Handle both normalized (0-1) and absolute coordinates
   - Apply proper scaling to account for letterboxing
   - Convert center coordinates to top-left corner format for OpenCV
   - Apply confidence thresholding
   - Apply NMS if not already done by the model

4. **Visualization**:
   - Draw bounding boxes with class names and confidence scores
   - Display FPS information

## Performance Optimization

- Use the `--conf` parameter to adjust the confidence threshold based on your needs
- Models with built-in NMS are faster than models requiring separate NMS
- Lower resolution videos process faster than higher resolution ones

## Troubleshooting

### Common Issues

1. **"Error loading model"**: Ensure the model file path is correct and accessible.
2. **"Error creating session"**: Check if the ONNX model is compatible with your ONNX Runtime version.
3. **"No detections"**: Try lowering the confidence threshold with `--conf=0.1`.
4. **"Unexpected output tensor shape"**: The model output format may not be supported.

### Checking ONNX Model Information

You can inspect your ONNX model using the following Python script:

```python
import onnx
model = onnx.load("path/to/your/model.onnx")
print(model.graph.input)
print(model.graph.output)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [OpenCV](https://opencv.org/)
- [YOLO](https://github.com/ultralytics/yolov5)
