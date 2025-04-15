# Windows Setup Guide for GolfBallTracking Project

Here's a step-by-step guide to set up, build, and test the GolfBallTracking project on a fresh Windows installation:

## 1. Install Required Software

### Install Git
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default options
3. Open Git Bash or Command Prompt to confirm installation: `git --version`

### Install Visual Studio Build Tools
1. Download Visual Studio Build Tools from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)
2. During installation, select:
   - "Desktop development with C++"
   - Windows 10/11 SDK
   - C++ CMake tools for Windows

### Install CMake
1. Download CMake from [cmake.org/download](https://cmake.org/download/)
2. Run the installer and select "Add CMake to the system PATH for all users"

### Install OpenCV
1. Download OpenCV 4.8.0 (or latest) from [opencv.org/releases](https://opencv.org/releases/)
2. Extract to a location like `C:\opencv`
3. Add to PATH:
   - Right-click "This PC" → Properties → Advanced system settings → Environment Variables
   - Edit the "Path" variable and add: `C:\opencv\build\x64\vc16\bin`

### Install ONNX Runtime
1. Download ONNX Runtime for Windows from [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)
2. Extract to `C:\onnxruntime-win-x64-1.21.0` (version number may differ)
3. Add to PATH: `C:\onnxruntime-win-x64-1.21.0\lib`

## 2. Clone and Configure the Project

1. Enter the repository:
   ```
   cd GolfBallTracking
   ```


## 3. Configure CMake Build

1. Create a build directory:
   ```
   mkdir build
   cd build
   ```

2. Configure CMake with proper paths:
   ```
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\opencv;C:\onnxruntime-win-x64-1.21.0"
   ```

   If you encounter errors, you may need to specify exact paths:
   ```
   cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:\opencv\build -DOnnxRuntime_DIR=C:\onnxruntime-win-x64-1.21.0\lib\cmake\onnxruntime
   ```

## 4. Build the Project

1. Build the solution:
   ```
   cmake --build . --config Release
   ```

2. If successful, executables will be created in `build\Release` directory

3. Copy required DLLs to executable directory (if not using PATH):
   ```
   copy C:\opencv\build\x64\vc16\bin\opencv_world480.dll build\calibration\Release\  build\detector\Release\  build\src\Release\
   copy C:\onnxruntime-win-x64-1.21.0\lib\onnxruntime.dll build\calibration\Release\  build\detector\Release\  build\src\Release\
   ```

See `README.md` for further steps

## Troubleshooting

### DLL Not Found Errors
If you encounter "DLL not found" errors when running executables:
1. Ensure all required DLLs are in the same directory as the executable or in the system PATH
2. Check for missing dependencies using a tool like Dependency Walker

### Build Errors
If CMake configuration fails:
1. Double-check paths to OpenCV and ONNX Runtime
2. Ensure you have the correct Visual Studio version installed

### Execution Errors
If the application runs but crashes:
1. Verify your model file (best.onnx) is valid and in the correct format
2. Check that classes.txt matches the model's expected classes
3. Make sure any video files you're using are in a supported format

This guide covers the basic setup and testing process for the GolfBallTracking project on a fresh Windows installation.