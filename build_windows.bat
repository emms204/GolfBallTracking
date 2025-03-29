@echo off
SETLOCAL

echo Building the ONNX Object Detector for Windows...

REM Set default paths - modify these to match your installation
SET "OPENCV_DIR=C:\opencv\build"
SET "ONNXRUNTIME_DIR=C:\onnxruntime"

REM Allow command-line override of paths
if not "%~1"=="" SET "OPENCV_DIR=%~1"
if not "%~2"=="" SET "ONNXRUNTIME_DIR=%~2"

REM Check if paths exist
if not exist "%OPENCV_DIR%" (
    echo Error: OpenCV directory '%OPENCV_DIR%' does not exist.
    echo Usage: %0 [OpenCV_DIR] [ONNXRUNTIME_DIR]
    echo Example: %0 C:\opencv\build C:\onnxruntime
    exit /b 1
)

if not exist "%ONNXRUNTIME_DIR%" (
    echo Error: ONNX Runtime directory '%ONNXRUNTIME_DIR%' does not exist.
    echo Usage: %0 [OpenCV_DIR] [ONNXRUNTIME_DIR]
    echo Example: %0 C:\opencv\build C:\onnxruntime
    exit /b 1
)

REM Create build directory
cd src
if not exist "build" mkdir build
cd build

REM Run CMake
echo Running CMake...
cmake .. -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_PREFIX_PATH="%OPENCV_DIR%" ^
    -DONNXRUNTIME_DIR="%ONNXRUNTIME_DIR%" ^
    -DCMAKE_CONFIGURATION_TYPES=Release ^
    -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed.
    exit /b 1
)

REM Build the project
echo Building the project...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b 1
)

echo Build completed successfully.
echo.
echo You can run the detector using:
echo run_detector.bat videos\your_video.mp4 best.onnx classes.txt
echo.

cd ..\..
ENDLOCAL 