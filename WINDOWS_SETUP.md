# Windows Setup Guide for ONNX Object Detection

This guide will help you set up and run the real-time object detection codebase on Windows.

## Prerequisites

1. **Visual Studio**: 
   - Install [Visual Studio 2019](https://visualstudio.microsoft.com/vs/older-downloads/) or [Visual Studio 2022](https://visualstudio.microsoft.com/vs/)
   - During installation, select "Desktop development with C++" workload

2. **CMake**:
   - Download and install [CMake](https://cmake.org/download/) (3.10 or higher)
   - Add CMake to your system PATH during installation

3. **OpenCV**:
   - Download pre-built OpenCV for Windows from [OpenCV Releases](https://opencv.org/releases/) (version 4.x recommended)
   - Extract to a location (e.g., `C:\opencv`)
   - Add `C:\opencv\build\x64\vc15\bin` (or similar path depending on your OpenCV version) to your system PATH

4. **ONNX Runtime**:
   - Download Windows release from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) (e.g., `onnxruntime-win-x64-1.16.0.zip` or newer)
   - Extract to a location (e.g., `C:\onnxruntime`)

## Setup Project

1. **Clone or download the repository**:
   - Extract the project to a folder (e.g., `C:\Projects\onnx-detector`)

2. **Prepare the model and classes**:
   - Ensure `best.onnx` and `classes.txt` are in the project root directory

3. **Create a modified CMakeLists.txt for Windows**:
   - Create a file named `CMakeLists-windows.txt` in the `src` folder with the Windows configuration

## Building with CMake (Command Line)

1. Open a command prompt (with admin rights recommended)
2. Navigate to the project's `src` directory
3. Run the following commands:

```batch
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=C:\opencv\build -DONNXRUNTIME_DIR=C:\onnxruntime
cmake --build . --config Release
```

Replace the paths with your actual OpenCV and ONNX Runtime installation paths.

## Running the Detector

Create a batch file `run_detector.bat` in the project root with the following content:

```batch
@echo off
SET PATH=%PATH%;C:\opencv\build\x64\vc15\bin;C:\onnxruntime\bin

src\build\Release\onnx_detector.exe --video=%1 --model=%2 --classes=%3 --conf=%4 --nms=%5
```

Then run it with:

```batch
run_detector.bat videos\your_video.mp4 best.onnx classes.txt 0.25 0.45
```

## Troubleshooting

### Common Issues:

1. **Missing DLLs**:
   - Error: "The program can't start because opencv_world4xx.dll is missing"
   - Solution: Ensure OpenCV bin directory is in your PATH or copy the DLLs to your executable directory

2. **ONNX Runtime errors**:
   - Error: "The program can't start because onnxruntime.dll is missing"
   - Solution: Ensure ONNX Runtime bin directory is in your PATH or copy the DLLs to your executable directory

3. **Build failures**:
   - Ensure all paths in CMakeLists.txt are correct
   - Check that you're using compatible versions of OpenCV and ONNX Runtime
   - Make sure Visual Studio has C++ development tools installed

4. **No detection output**:
   - Try lowering confidence threshold (--conf=0.1)
   - Check if the model and class files are loaded correctly

## Additional Notes

- The executable requires administrator rights if accessing camera devices
- For best performance, use Release build configuration
- If you encounter performance issues, consider using a GPU-enabled version of ONNX Runtime 