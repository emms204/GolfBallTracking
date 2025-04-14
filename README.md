# Camera Calibration and Object Detection Toolkit

This repository contains a toolkit for camera calibration and object detection using ONNX Runtime.

## Project Structure

The project is organized into the following directories:

- **calibration**: Contains calibration-related code
- **detector**: Contains the enhanced ONNX detector implementation with proper letterboxing and distortion correction
- **common**: Common utilities shared across the project
- **src**: Applications built using the detector and calibration libraries

## Enhanced Detector

The detector implementation in `detector/` includes several improvements:

1. **Proper Letterboxing**: Preserves aspect ratio when resizing images for the ONNX model
2. **Camera Distortion Correction**: Uses camera calibration parameters to undistort images
3. **Bounding Box Coordinate Transformation**: Correctly scales detection coordinates back to the original image coordinates

The letterboxing and coordinate transformation code is based on the following approach:

```cpp
// Letterbox the input image (maintain aspect ratio with padding)
double ratio = std::min(target_size / (double)input.cols, target_size / (double)input.rows);
int new_width = static_cast<int>(input.cols * ratio);
int new_height = static_cast<int>(input.rows * ratio);
int pad_left = (target_size - new_width) / 2;
int pad_top = (target_size - new_height) / 2;

// Resize and pad
cv::Mat resized, letterboxed;
cv::resize(input, resized, cv::Size(new_width, new_height));
cv::copyMakeBorder(resized, letterboxed, pad_top, pad_bottom, pad_left, pad_right, 
                  cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
```

## Applications

1. **detection_test_app**: Tests the detector with a camera or video file, with optional calibration
2. **enhanced_detector_main**: Standalone application for the enhanced detector

## Running the Applications

### Detection Test App

```bash
./build/detection_test_app --video=your_video.mp4 --model=best.onnx --classes=classes.txt --calibration=your_calibration.yaml
```

### Enhanced Detector

```bash
./build/enhanced_detector_main --video=your_video.mp4 --model=best.onnx --classes=classes.txt --params=your_calibration.yaml --undistort
```

## Building the Project

```bash
mkdir -p build
cd build
cmake ..
make
```

## Detector Architecture

The detector is implemented using a modular architecture:

1. **ONNXDetector**: Core detector class that interfaces with ONNX Runtime
2. **Preprocessor**: Handles image preprocessing (letterboxing, undistortion)
3. **CameraParams**: Manages camera calibration parameters

This design ensures proper coordinate transformations for accurate bounding boxes.

## Testing Detectors with Calibration

A test script is provided to verify that both the enhanced detector and the detection test app work correctly with calibration files. This helps ensure that the calibration process works as expected with our detection code.

### Testing with the test_detectors.sh script

The `test_detectors.sh` script can test both detectors with your calibration files:

```bash
# For master camera videos
./test_detectors.sh --calibration calibration_results/master_calibration.yaml --video /path/to/Driver-Master-1.avi

# For slave camera videos
./test_detectors.sh --calibration calibration_results/slave_calibration.yaml --video /path/to/Driver-Slave-1.avi
```

You can also test only one detector at a time:

```bash
# Test only the enhanced detector
./test_detectors.sh --calibration calibration_results/master_calibration.yaml --video /path/to/Driver-Master-1.avi --only-enhanced

# Test only the detection test app
./test_detectors.sh --calibration calibration_results/master_calibration.yaml --video /path/to/Driver-Master-1.avi --only-detection-app
```

### Original detection test commands

For reference, the original commands for testing with the detection test app are:

```bash
# For master camera videos
./run_detection_test_app.sh --calibration /tmp/master_calibration.yaml --video /tmp/Driver-Master-1.avi

# For slave camera videos
./run_detection_test_app.sh --calibration /tmp/slave_calibration.yaml --video /tmp/Driver-Slave-1.avi
```

### Testing with the enhanced detector

The enhanced detector can be directly tested with:

```bash
./src/enhanced_detector_main --video="/path/to/video.avi" --params="calibration_results/master_calibration.yaml" --undistort
```

This will run the enhanced detector with the specified calibration file and apply undistortion to the video frames.

### Testing calibration visually

You can also test calibration visually with a single image using the provided test tool:

```bash
# Test the master calibration with an image
./test/test_calibration_image.sh --calibration calibration_results/master_calibration.yaml --input /path/to/image.jpg

# Test the slave calibration with an image
./test/test_calibration_image.sh --calibration calibration_results/slave_calibration.yaml --input /path/to/image.jpg
```

This will create a side-by-side comparison of the original and undistorted images with a grid overlay to help visualize distortion correction.
