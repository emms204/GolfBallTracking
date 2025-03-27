# Stereo Camera Calibration

This project provides tools for calibrating a stereo camera system using chessboard images. The calibration process determines the intrinsic parameters (camera matrix, distortion coefficients) for each camera and the extrinsic parameters (rotation and translation) between them.

## Camera Setup

- Master and Slave cameras with a baseline of 500mm (centers of sensors are 500mm apart on the same horizontal line)
- Resolution: 720x540 at 302 fps (images provided at this resolution), trimmed to 400x540 for ball tracking at 400 fps

## Chessboard Specifications

- Each black and white square is 100mm x 100mm (10cm x 10cm)
- The chessboard has 5x5 squares, making the total size 500mm x 500mm
- The calibration looks for 4x4 internal corners

## Directory Structure

```
.
├── Combined/
│   ├── Master/  # 80 images from the master camera
│   └── Slave/   # 80 images from the slave camera
├── stereo_calibration.py  # Main calibration script
├── visualize_calibration.py  # Visualization script
├── stereo_reconstruction.py  # 3D reconstruction script
├── stereo_pipeline.py  # Complete pipeline script
└── stereo_calibration_README.md  # This README file
```

## Requirements

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the Complete Pipeline

The easiest way to run the entire stereo calibration and reconstruction process is to use the unified pipeline script:

```bash
python stereo_pipeline.py [options]
```

Options:
- `--skip_calibration`: Skip the calibration step
- `--skip_visualization`: Skip the visualization step
- `--skip_reconstruction`: Skip the reconstruction step
- `--master_img`: Path to master camera image for visualization/reconstruction (default: 'Chess 8mm/Master/1.jpeg')
- `--slave_img`: Path to slave camera image for visualization/reconstruction (default: 'Chess 8mm/Slave/1.jpeg')

This script will run all steps of the pipeline in sequence, creating all necessary output directories.

## Running Individual Scripts

If you prefer to run scripts separately, follow these steps:

1. To run the stereo calibration, use the following command:

```bash
python stereo_calibration.py
```

This will:
- Detect chessboard corners in all image pairs
- Calibrate each camera individually
- Perform stereo calibration to determine the extrinsic parameters
- Save the calibration results to the `calibration_results` directory

2. To visualize the calibration results:

```bash
python visualize_calibration.py [--calib_file CALIB_FILE] [--master_img MASTER_IMG] [--slave_img SLAVE_IMG] [--output_dir OUTPUT_DIR]
```

Arguments:
- `--calib_file`: Path to the calibration file (default: 'calibration_results/stereo_calibration_8mm.pkl')
- `--master_img`: Path to a master camera image (default: 'Chess 8mm/Master/1.jpeg')
- `--slave_img`: Path to a slave camera image (default: 'Chess 8mm/Slave/1.jpeg')
- `--output_dir`: Directory to save visualization results (default: 'visualization')

3. To perform 3D reconstruction using the calibrated stereo system:

```bash
python stereo_reconstruction.py [--calib_file CALIB_FILE] [--master_img MASTER_IMG] [--slave_img SLAVE_IMG] [--output_dir OUTPUT_DIR]
```

Arguments:
- `--calib_file`: Path to the calibration file (default: 'calibration_results/stereo_calibration_8mm.pkl')
- `--master_img`: Path to a master camera image (default: 'Chess 8mm/Master/1.jpeg')
- `--slave_img`: Path to a slave camera image (default: 'Chess 8mm/Slave/1.jpeg')
- `--output_dir`: Directory to save reconstruction results (default: 'reconstruction')

## Output Files

The calibration process generates the following files:

- `calibration_results/stereo_calibration_8mm.pkl`: Complete stereo calibration results
- `calibration_results/master_calibration_8mm.npz`: Individual calibration results for the master camera
- `calibration_results/slave_calibration_8mm.npz`: Individual calibration results for the slave camera
- `calibration_results/rectified_pair.jpg`: Example of a rectified image pair

The visualization script generates:

- `visualization/original_pair.jpg`: Original image pair
- `visualization/rectified_pair.jpg`: Rectified image pair with horizontal lines
- `visualization/disparity_map.jpg`: Disparity map computed from the rectified images

The reconstruction script generates:

- `reconstruction/rectified_master.jpg` and `reconstruction/rectified_slave.jpg`: Rectified input images
- `reconstruction/disparity_map.jpg`: Disparity map of the scene
- `reconstruction/point_cloud.npy`: 3D point cloud data (as NumPy array)
- `reconstruction/point_cloud.png`: Visualization of the 3D point cloud
- For chessboard images, additional outputs include:
  - `reconstruction/corners_master.jpg` and `reconstruction/corners_slave.jpg`: Images with detected corners
  - `reconstruction/corners_3d.npy`: 3D coordinates of the chessboard corners
  - `reconstruction/corners_3d.png`: Visualization of the 3D chessboard corners

## Verification

The stereo calibration can be verified by:

1. Examining the RMS error of the calibration
2. Checking that the translation vector reflects the 500mm baseline
3. Visually inspecting the rectified images with horizontal lines to ensure proper alignment

## Camera Parameters

The calibration provides the following parameters:

- Camera matrices for both cameras
- Distortion coefficients for both cameras
- Rotation matrix between cameras
- Translation vector between cameras
- Rectification matrices
- Projection matrices
- Disparity-to-depth mapping matrix

These parameters are essential for accurate 3D reconstruction and depth estimation from stereo images.

## 3D Reconstruction

The `stereo_reconstruction.py` script demonstrates how to use the calibration results to:

1. Rectify a pair of stereo images
2. Compute a disparity map
3. Generate a 3D point cloud
4. Triangulate corresponding points in 3D space

For chessboard images, the script also demonstrates how to triangulate the chessboard corners and visualize the 3D structure of the chessboard.

## Notes

- The 100mm square size is suitable for the 8mm lens and 720x540 resolution
- The calibration assumes that both cameras are at the same height and orientation
- For best results, ensure that the chessboard is captured at various angles and distances
- The calibration process requires at least 10-20 good image pairs for accurate results
- The disparity map and resulting 3D reconstruction depend on the quality of the calibration
- The baseline of 500mm provides a good compromise between depth resolution and matching accuracy 