@echo off
SETLOCAL

REM Set default values for parameters
SET "CAMERA_ID=%~1"
if "%CAMERA_ID%"=="" SET "CAMERA_ID=0"
SET "MODEL_FILE=%~2"
if "%MODEL_FILE%"=="" SET "MODEL_FILE=best.onnx"
SET "CLASSES_FILE=%~3"
if "%CLASSES_FILE%"=="" SET "CLASSES_FILE=classes.txt"
SET "CONF_THRESHOLD=%~4"
if "%CONF_THRESHOLD%"=="" SET "CONF_THRESHOLD=0.25"
SET "NMS_THRESHOLD=%~5"
if "%NMS_THRESHOLD%"=="" SET "NMS_THRESHOLD=0.45"

REM Check if model file exists
if not exist "%MODEL_FILE%" (
    echo Error: Model file '%MODEL_FILE%' does not exist.
    exit /b 1
)

REM Check if classes file exists
if not exist "%CLASSES_FILE%" (
    echo Warning: Classes file '%CLASSES_FILE%' does not exist. Creating default...
    echo ball > "%CLASSES_FILE%"
    echo club >> "%CLASSES_FILE%"
)

REM Check if the executable exists
SET "EXECUTABLE=src\build\Release\onnx_detector.exe"
if not exist "%EXECUTABLE%" (
    echo Error: ONNX detector executable not found at '%EXECUTABLE%'.
    echo Please build the project first using Visual Studio or CMake.
    exit /b 1
)

REM Ensure OpenCV and ONNX Runtime DLLs are in the PATH
REM Modify these paths to match your actual installation locations
SET "PATH=%PATH%;C:\opencv\build\x64\vc15\bin;C:\onnxruntime\bin"

echo Running ONNX detector with camera ID %CAMERA_ID%...
echo Using model: %MODEL_FILE%
echo Using classes: %CLASSES_FILE%
echo Confidence threshold: %CONF_THRESHOLD%
echo NMS threshold: %NMS_THRESHOLD%

REM Run the detector with camera input
"%EXECUTABLE%" --camera=%CAMERA_ID% --model="%MODEL_FILE%" --classes="%CLASSES_FILE%" --conf="%CONF_THRESHOLD%" --nms="%NMS_THRESHOLD%"

if %ERRORLEVEL% equ 0 (
    echo Detection completed successfully!
) else (
    echo Detection failed. Check the error messages above.
)

ENDLOCAL 