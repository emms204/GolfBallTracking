@echo off
echo Generating Visual Studio solution...

:: Create build directory if it doesn't exist
if not exist build mkdir build

:: Move to build directory
cd build

:: Set path to OpenCV and ONNX Runtime (modify as needed)
set OPENCV_DIR=C:/opencv/build
set ONNXRUNTIME_DIR=%~dp0/onnxruntime-win-x64-1.21.0

:: Run CMake to generate Visual Studio solution
cmake -G "Visual Studio 17 2022" ^
      -A x64 ^
      -DCMAKE_PREFIX_PATH="%OPENCV_DIR%;%ONNXRUNTIME_DIR%" ^
      ..

:: Check if CMake was successful
if %ERRORLEVEL% neq 0 (
    echo Error: CMake failed to generate the solution.
    exit /b %ERRORLEVEL%
)

echo.
echo Visual Studio solution generated successfully in the 'build' directory.
echo You can now open build/Camera_Calibration_Toolkit.sln in Visual Studio.

cd .. 