#!/bin/bash

# Activate the global Python environment
source /opt/venv/global/bin/activate

echo "=== Running Simple PyTorch Prediction Test ==="
python python_track/track_simple_predict.py \
    --video "videos/i58-Master-1.avi" \
    --output "i58-simple-predict-output.avi" \
    --model "TrainResults3/best.pt" \
    --calib_file "stereo_calibration/calibration_results/master_calibration_8mm.npz" \
    --camera "master" \
    --log_file "python_simple_predict.log"

echo "=== Running ONNX Runtime Test ==="
python python_track/track_onnx.py \
    --video "videos/i58-Master-1.avi" \
    --output "i58-onnx-output.avi" \
    --model "best.onnx" \
    --calib_file "stereo_calibration/calibration_results/master_calibration_8mm.npz" \
    --camera "master" \
    --log_file "python_ort_test.log"

echo "=== Comparison Complete ==="
echo "Simple prediction log: python_simple_predict.log"
echo "ONNX Runtime log: python_ort_test.log"
echo "Compare these logs to see if they produce similar results" 