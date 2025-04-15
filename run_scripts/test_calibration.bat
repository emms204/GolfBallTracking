@echo off
setlocal enabledelayedexpansion

:: Default values
set "TEST_IMAGE="
set "TEST_DIRECTORY="
set "CALIBRATION_FILE="
set "OUTPUT_DIR=calibration_test_results"
set "PATTERN_SIZE=9x6"
set "SQUARE_SIZE=20"

:: Parse command-line arguments
for %%a in (%*) do (
    set "arg=%%a"
    
    if "!arg:~0,8!"=="--image=" (
        set "TEST_IMAGE=!arg:~8!"
    ) else if "!arg:~0,6!"=="--dir=" (
        set "TEST_DIRECTORY=!arg:~6!"
    ) else if "!arg:~0,14!"=="--calibration=" (
        set "CALIBRATION_FILE=!arg:~14!"
    ) else if "!arg:~0,9!"=="--output=" (
        set "OUTPUT_DIR=!arg:~9!"
    ) else if "!arg:~0,10!"=="--pattern=" (
        set "PATTERN_SIZE=!arg:~10!"
    ) else if "!arg:~0,9!"=="--square=" (
        set "SQUARE_SIZE=!arg:~9!"
    ) else if "!arg!"=="--help" (
        echo Usage: %0 [options]
        echo Options:
        echo   --image=^<path^>          Test single image with existing calibration
        echo   --dir=^<path^>            Directory with calibration images
        echo   --calibration=^<path^>    Existing calibration file (for testing)
        echo   --output=^<path^>         Output directory (default: calibration_test_results)
        echo   --pattern=^<WxH^>         Chessboard pattern size (default: 9x6)
        echo   --square=^<mm^>           Chessboard square size in mm (default: 20)
        echo   --help                  Show this help message
        exit /b 0
    ) else (
        echo Unknown option: !arg!
        echo Use --help for usage information.
        exit /b 1
    )
)

:: Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Check if we have a build directory and the necessary tools
if not exist "build" (
    echo Build directory not found, building project...
    call build_windows.bat
    
    if %ERRORLEVEL% neq 0 (
        echo Build failed. Cannot continue with tests.
        exit /b 1
    )
)

:: Test a single image with existing calibration
if defined TEST_IMAGE if defined CALIBRATION_FILE (
    if not exist "%TEST_IMAGE%" (
        echo Test image not found: %TEST_IMAGE%
        exit /b 1
    )
    
    if not exist "%CALIBRATION_FILE%" (
        echo Calibration file not found: %CALIBRATION_FILE%
        exit /b 1
    )
    
    echo Testing calibration with single image...
    echo Image: %TEST_IMAGE%
    echo Calibration: %CALIBRATION_FILE%
    
    :: Run the test_calibration_image tool
    build\Release\test_calibration_image.exe --calibration "%CALIBRATION_FILE%" --input "%TEST_IMAGE%" --output "%OUTPUT_DIR%\test_result.jpg"
    
    if %ERRORLEVEL% neq 0 (
        echo Testing failed.
        exit /b 1
    )
    
    echo Test completed. Result image saved to %OUTPUT_DIR%\test_result.jpg
)

:: Run calibration on a directory of images
if defined TEST_DIRECTORY (
    if not exist "%TEST_DIRECTORY%" (
        echo Test directory not found: %TEST_DIRECTORY%
        exit /b 1
    )
    
    echo Running calibration on directory...
    echo Directory: %TEST_DIRECTORY%
    echo Pattern size: %PATTERN_SIZE%
    echo Square size: %SQUARE_SIZE% mm
    
    :: Run the calibration tool
    build\Release\calibrate_camera.exe --input "%TEST_DIRECTORY%" --pattern_size "%PATTERN_SIZE%" --square_size "%SQUARE_SIZE%" --output "%OUTPUT_DIR%\calibration_result.yaml"
    
    if %ERRORLEVEL% neq 0 (
        echo Calibration failed.
        exit /b 1
    )
    
    echo Calibration completed. Parameters saved to %OUTPUT_DIR%\calibration_result.yaml
    
    :: Also test the first image with the new calibration
    :: Find the first image in the directory
    for /r "%TEST_DIRECTORY%" %%f in (*.jpg *.png) do (
        set "FIRST_IMAGE=%%f"
        goto found_image
    )
    :found_image
    
    if defined FIRST_IMAGE (
        echo Testing new calibration with first image: !FIRST_IMAGE!
        
        build\Release\test_calibration_image.exe --calibration "%OUTPUT_DIR%\calibration_result.yaml" --input "!FIRST_IMAGE!" --output "%OUTPUT_DIR%\calibration_test_result.jpg"
        
        if %ERRORLEVEL% equ 0 (
            echo Calibration test completed. Result image saved to %OUTPUT_DIR%\calibration_test_result.jpg
        )
    )
)

echo All tests completed. 