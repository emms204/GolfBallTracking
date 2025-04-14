@echo off
echo Building Camera Calibration Toolkit for Windows...

:: Check if OPENCV_DIR environment variable is set
if "%OPENCV_DIR%"=="" (
    echo Warning: OPENCV_DIR environment variable is not set.
    echo Please set it to your OpenCV installation directory, e.g.:
    echo set OPENCV_DIR=C:\opencv\build
    echo.
    echo Attempting to continue using default paths...
)

:: Check for ONNX Runtime
if not exist "onnxruntime-win-x64-1.21.0" (
    echo ONNX Runtime not found. Downloading ONNX Runtime...
    call download_onnxruntime.bat
    if %ERRORLEVEL% neq 0 (
        echo Failed to download ONNX Runtime.
        exit /b %ERRORLEVEL%
    )
)

:: Create build directory if it doesn't exist
if not exist build mkdir build
cd build

:: Configure the project with CMake
echo Configuring project with CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed.
    cd ..
    exit /b %ERRORLEVEL%
)

:: Build the project
echo Building project...
cmake --build . --config Release --parallel

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    cd ..
    exit /b %ERRORLEVEL%
)

echo.
echo Build completed successfully.
echo Release binaries can be found in build\Release directory.

cd .. 