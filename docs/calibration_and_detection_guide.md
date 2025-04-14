# Camera Calibration and Object Detection Guide

## 1. Using the Calibration Tool

The calibration tool is designed to calculate camera intrinsic parameters and distortion coefficients using a chessboard pattern. This process is essential for correcting lens distortion and enabling accurate object detection.

### Prerequisites
- A chessboard pattern printed on a rigid, flat surface
- Camera or set of calibration images
- The compiled calibration tool

### Calibration Process

#### Step 1: Build the Calibration Tool

**On Linux:**
```bash
./calibration/build_calibration.sh
```

**On Windows:**
```batch
# First ensure you have OpenCV and ONNX Runtime installed
# Then generate the Visual Studio solution
generate_vs_solution.bat

# Open the solution in Visual Studio and build the project
# Or build from command line:
cd build
cmake --build . --config Release
```

#### Step 2: Run the Calibration Tool

**Using a Live Camera:**

Linux:
```bash
./build/calibrate_camera --input 0 --pattern_size 9x6 --square_size 20 --output camera_calibration.yaml
```

Windows:
```batch
build\Release\calibrate_camera.exe --input 0 --pattern_size 9x6 --square_size 20 --output camera_calibration.yaml
```

**Using a Directory of Images:**

Linux:
```bash
./build/calibrate_camera --input /path/to/images --pattern_size 9x6 --square_size 20 --output camera_calibration.yaml
```

Windows:
```batch
build\Release\calibrate_camera.exe --input C:/path/to/images --pattern_size 9x6 --square_size 20 --output camera_calibration.yaml
```

**Parameters Explained:**
- `--input`: Camera index (e.g., 0 for default camera) or directory path containing calibration images
- `--pattern_size`: Dimensions of the internal corners of the chessboard (Width x Height)
- `--square_size`: Physical size of each square in millimeters
- `--output`: Path to save the calibration file
- `--min_images`: Minimum number of images required for calibration (default: 10)
- `--skip_frames`: Number of frames to skip between captures in camera mode (default: 20)

#### Step 3: Capture Calibration Images
When using a camera:
- Press `Space` or `C` to capture the current frame when a chessboard is detected
- Press `A` to toggle auto-capture mode
- Press `Enter` to start calibration once enough images are captured
- Press `ESC` to exit

### Multiple Camera Calibration
For dual-camera setups, you can use the master-slave calibration script (Linux only):

```bash
./calibration/run_master_slave_calibration.sh --master=/path/to/master/images --slave=/path/to/slave/images --pattern=4x4 --square=100
```

For Windows, use the individual calibration tool for each camera and save the results to separate files.

## 2. Integrating Calibration Parameters with Detection

### Loading Calibration in Detection Applications

#### Option 1: Using the Detection Test App

Linux:
```bash
./src/detection_test_app --help
./src/detection_test_app --video=your_video.mp4 --model=best.onnx --classes=classes.txt --calibration=your_calibration.yaml
```

Windows:
```batch
build\Release\detection_test_app.exe --video=your_video.mp4 --model=best.onnx --classes=classes.txt --calibration=your_calibration.yaml
```

### How Calibration Affects Detection

1. **Undistortion Process**: The calibration parameters are used to correct lens distortion in the input frames
2. **Coordinate Transformation**: Bounding box coordinates are properly transformed from model space to the undistorted image space

### Preprocessing and Detection Pipeline

```
┌───────────────┐       ┌───────────────┐       ┌────────────────┐       ┌──────────────┐
│  Input Image  │──────►│  Undistortion │──────►│  Letterboxing  │──────►│  ONNX Model  │
└───────────────┘       └───────────────┘       └────────────────┘       └──────────────┘
                             │                                                   │
                        Uses camera                                              │
                        calibration                                              │
                        parameters                                               │
                                                                                 ▼
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐     ┌──────────────┐
 │  Final Output  │◄─────│ Draw Detections │◄─────│ Coordinate    │◄────│ Raw Model    │
 └────────────────┘      └────────────────┘      │ Transformation │     │ Predictions  │
                                                 └────────────────┘     └──────────────┘
```

## 3. Troubleshooting Common Issues

### Calibration Issues

| Problem | Possible Causes | Solution |
|---------|----------------|----------|
| Chessboard not detected | Poor lighting, blurry image, wrong pattern size | Improve lighting, ensure pattern is in focus, verify pattern dimensions |
| High reprojection error | Inaccurate corner detection, moving chessboard | Use a rigid chessboard, capture from different angles, ensure pattern is stationary |
| Insufficient calibration images | Too few successful detections | Capture more images, position chessboard at different angles and distances |
| Distortion still visible after calibration | Incorrect calibration parameters, extreme lens distortion | Recalibrate with more images, verify calibration file is loaded correctly |
| Windows path issues | Backslashes in paths | Use forward slashes in paths ('/') even on Windows, or use raw strings with double backslashes |

### Detection Issues

| Problem | Possible Causes | Solution |
|---------|----------------|----------|
| Poor detection accuracy with calibration | Incorrect coordinate transformation | Ensure the enhanced detector is being used with letterboxing |
| No detections with calibration enabled | Incompatible calibration file, extreme undistortion | Verify calibration file format, check calibration quality |
| Detection coordinates misaligned | Improper scaling or transformation | Use the enhanced detector with proper letterboxing |
| Slowed performance with calibration | Computationally expensive undistortion | Pre-compute undistortion maps, optimize preprocessing pipeline |
| DLL not found errors (Windows) | Missing path to ONNX Runtime or OpenCV | Add library directories to PATH or copy DLLs to executable directory |

## 4. Guidelines for Accurate Calibration

### Chessboard Preparation
- Use a flat, rigid surface to print or mount your chessboard
- Ensure the pattern has precise dimensions
- Use a matte finish to avoid glare

### Image Capture
- Capture the chessboard from various angles and distances
- Cover the entire field of view, especially the edges
- Maintain consistent lighting conditions
- Keep the chessboard stationary during captures
- Collect at least 15-20 images for reliable calibration

### Calibration Quality Assessment
- **Reprojection Error**: Lower is better, aim for < 0.5 pixels
- **Visual Verification**: Use `test_calibration_image.sh` to visually inspect undistortion
- **Grid Test**: Straight lines in the undistorted image should remain straight

## 5. Bounding Box Accuracy Improvements

The enhanced detector implements several key improvements over basic implementations:

### Proper Letterboxing
Standard resizing can distort aspect ratios, leading to inaccurate detections. Our implementation:
- Maintains the original aspect ratio during resizing
- Adds padding to reach the target dimensions
- Prevents distortion of object shapes

```cpp
// Letterboxing implementation
double ratio = std::min(target_size / (double)input.cols, target_size / (double)input.rows);
int new_width = static_cast<int>(input.cols * ratio);
int new_height = static_cast<int>(input.rows * ratio);
int pad_left = (target_size - new_width) / 2;
int pad_top = (target_size - new_height) / 2;
```

### Coordinate Transformation
Proper bounding box coordinates require careful scaling:
- Model outputs coordinates in its input space (e.g., 640x640)
- These must be transformed back to the original image space
- Our implementation accounts for both letterboxing and undistortion

### Combined Effect
When combined with camera calibration:
1. First, lens distortion is corrected using calibration parameters
2. The undistorted image is letterboxed to maintain aspect ratio
3. Detection is performed on the properly processed image
4. Resulting bounding box coordinates are transformed back to original image space

This comprehensive approach significantly improves detection accuracy, especially for:
- Wide-angle camera lenses with significant distortion
- Applications requiring precise object localization
- Cases where objects appear near the edges of the frame

## Testing Your Calibration

### Visual Testing

Linux:
```bash
./test/test_calibration_image.sh --calibration calibration_results/master_calibration.yaml --input /path/to/image.jpg
```

Windows:
```batch
build\Release\test_calibration_image.exe --calibration calibration_results/master_calibration.yaml --input C:/path/to/image.jpg
```

### Testing with Detection

Linux:
```bash
./test_detectors.sh --calibration calibration_results/master_calibration.yaml --video /path/to/video.avi
```

Windows:
```batch
build\Release\detection_test_app.exe --video=C:/path/to/video.avi --model=best.onnx --classes=classes.txt --calibration=calibration_results/master_calibration.yaml
```

These tests help verify that your calibration is working correctly and improving detection accuracy.

## Platform-Specific Considerations

### Windows Path Handling

When working on Windows, consider these path handling tips:

1. Always use forward slashes in paths, even on Windows: `C:/path/to/file` instead of `C:\path\to\file`
2. If you must use backslashes, double them in strings: `C:\\path\\to\\file`
3. For command-line arguments, enclose paths with spaces in quotes: `"C:/Program Files/My App/data"`
4. The toolkit has been designed to handle all these cases properly

### Environment Variables

On Windows, you may need to set environment variables to help the system find required libraries:

```batch
set PATH=%PATH%;C:\opencv\build\x64\vc16\bin;C:\path\to\onnxruntime-win-x64-1.21.0\lib
```

Alternatively, you can copy required DLLs to the same directory as your executables. 