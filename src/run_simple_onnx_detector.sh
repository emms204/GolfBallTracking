#!/bin/bash

# Save the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Set default values
MODEL="best.onnx"
CLASSES="classes.txt"  # Default classes file in root directory
CONF=0.25
NMS=0.45
CAMERA=0
VIDEO=""
CALIBRATION=""

# Parse command-line arguments
for arg in "$@"; do
  case "$arg" in
    --model=*|--m=*)
      MODEL="${arg#*=}"
      ;;
    --classes=*|--c=*)
      CLASSES="${arg#*=}"
      ;;
    --conf=*)
      CONF="${arg#*=}"
      ;;
    --nms=*)
      NMS="${arg#*=}"
      ;;
    --video=*|--v=*)
      VIDEO="${arg#*=}"
      ;;
    --camera=*)
      CAMERA="${arg#*=}"
      ;;
    --calibration=*)
      CALIBRATION="${arg#*=}"
      ;;
    --help|--h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --model=PATH, --m=PATH       Path to ONNX model file (default: best.onnx)"
      echo "  --classes=PATH, --c=PATH     Path to classes file (default: classes.txt)"
      echo "  --conf=VALUE                 Confidence threshold (default: 0.25)"
      echo "  --nms=VALUE                  NMS threshold (default: 0.45)"
      echo "  --video=PATH, --v=PATH       Path to input video file"
      echo "  --camera=ID                  Camera device ID (default: 0)"
      echo "  --calibration=PATH           Path to camera calibration file"
      echo "  --help, --h                  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Convert relative paths to absolute paths
if [[ ! "$MODEL" = /* ]]; then
  MODEL="$ROOT_DIR/$MODEL"
fi

if [[ ! "$CLASSES" = /* ]]; then
  CLASSES="$ROOT_DIR/$CLASSES"
fi

if [[ -n "$VIDEO" && ! "$VIDEO" = /* ]]; then
  VIDEO="$ROOT_DIR/$VIDEO"
fi

if [[ -n "$CALIBRATION" && ! "$CALIBRATION" = /* ]]; then
  CALIBRATION="$ROOT_DIR/$CALIBRATION"
fi

# Look for the detector directory
DETECTOR_DIR="$ROOT_DIR/detector"
if [ ! -d "$DETECTOR_DIR" ]; then
  echo "Error: Cannot find the detector directory at $DETECTOR_DIR"
  exit 1
fi

# Check if the executable exists
DETECTOR_PATH="$DETECTOR_DIR/build/simple_onnx_detector"
if [ ! -f "$DETECTOR_PATH" ]; then
    echo "ONNX detector executable not found. Building..."
    
    # Navigate to the detector directory and build
    mkdir -p "$DETECTOR_DIR/build"
    cd "$DETECTOR_DIR/build"
    cmake ..
    make -j4
    
    if [ $? -ne 0 ]; then
        echo "Failed to build the ONNX detector. Check the compile errors above."
        # Return to the original directory
        cd "$ROOT_DIR"
        exit 1
    fi
    
    # Return to the original directory
    cd "$ROOT_DIR"
fi

# Check if files exist
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file '$MODEL' not found"
    exit 1
fi

if [ ! -f "$CLASSES" ]; then
    echo "Error: Classes file '$CLASSES' not found"
    exit 1
fi

if [ -n "$VIDEO" ] && [ ! -f "$VIDEO" ]; then
    echo "Error: Video file '$VIDEO' not found"
    exit 1
fi

if [ -n "$CALIBRATION" ] && [ ! -f "$CALIBRATION" ]; then
    echo "Error: Calibration file '$CALIBRATION' not found"
    exit 1
fi

# Build command with proper quoting
CMD="\"$DETECTOR_PATH\""

# Add input source
if [ -n "$VIDEO" ]; then
  CMD="$CMD --video=\"$VIDEO\""
else
  CMD="$CMD --camera=$CAMERA"
fi

# Add model parameters
CMD="$CMD --model=\"$MODEL\" --classes=\"$CLASSES\" --conf=$CONF --nms=$NMS"

# Add calibration if specified
if [ -n "$CALIBRATION" ]; then
  CMD="$CMD --calibration=\"$CALIBRATION\" --undistort=true"
fi

# Run the detector
echo "Running: $CMD"
eval $CMD