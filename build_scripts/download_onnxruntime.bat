@echo off
echo Downloading ONNX Runtime for Windows...

set ONNXRUNTIME_VERSION=1.21.0
set ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ONNXRUNTIME_VERSION%/onnxruntime-win-x64-%ONNXRUNTIME_VERSION%.zip
set DOWNLOAD_DIR=%~dp0
set ONNXRUNTIME_ZIP=%DOWNLOAD_DIR%\onnxruntime.zip

:: Download ONNX Runtime
curl -L %ONNXRUNTIME_URL% -o %ONNXRUNTIME_ZIP%

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to download ONNX Runtime.
    exit /b %ERRORLEVEL%
)

:: Extract the archive
echo Extracting ONNX Runtime...
powershell -command "Expand-Archive -Path '%ONNXRUNTIME_ZIP%' -DestinationPath '%DOWNLOAD_DIR%' -Force"

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to extract ONNX Runtime.
    exit /b %ERRORLEVEL%
)

:: Rename the extracted directory
rename onnxruntime-win-x64-%ONNXRUNTIME_VERSION% onnxruntime-win-x64-%ONNXRUNTIME_VERSION%

:: Clean up the zip file
del %ONNXRUNTIME_ZIP%

echo ONNX Runtime has been downloaded and extracted successfully. 