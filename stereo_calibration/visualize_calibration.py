import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_calibration(calib_file):
    """
    Load calibration results from file.
    
    Args:
        calib_file: Path to the calibration file (.pkl)
        
    Returns:
        stereo_calib_result: Dictionary containing stereo calibration results
    """
    with open(calib_file, 'rb') as f:
        stereo_calib_result = pickle.load(f)
    
    return stereo_calib_result

def undistort_rectify_image(img, camera_matrix, dist_coeffs, R, P):
    """
    Undistort and rectify an image.
    
    Args:
        img: Input image
        camera_matrix: Camera matrix
        dist_coeffs: Distortion coefficients
        R: Rectification matrix
        P: Projection matrix
        
    Returns:
        dst: Undistorted and rectified image
    """
    h, w = img.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, R, P, (w, h), cv2.CV_32FC1
    )
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return dst

def draw_horizontal_lines(img, step=50, color=(0, 255, 0)):
    """
    Draw horizontal lines on an image for rectification visualization.
    
    Args:
        img: Input image
        step: Spacing between lines
        color: Line color
        
    Returns:
        img_with_lines: Image with horizontal lines
    """
    img_with_lines = img.copy()
    h, w = img.shape[:2]
    
    for y in range(0, h, step):
        cv2.line(img_with_lines, (0, y), (w, y), color, 1)
    
    return img_with_lines

def process_image_pair(master_img_path, slave_img_path, stereo_calib_result):
    """
    Process an image pair using the calibration results.
    
    Args:
        master_img_path: Path to master camera image
        slave_img_path: Path to slave camera image
        stereo_calib_result: Stereo calibration results
        
    Returns:
        original_pair: Original images side by side
        rectified_pair: Rectified images side by side
    """
    # Read images
    img_master = cv2.imread(master_img_path)
    img_slave = cv2.imread(slave_img_path)
    
    if img_master is None or img_slave is None:
        raise ValueError(f"Failed to read images: {master_img_path}, {slave_img_path}")
    
    # Create composite of original images
    original_pair = np.hstack((img_master, img_slave))
    
    # Rectify images
    rectified_master = cv2.remap(
        img_master, 
        stereo_calib_result['rectification_map_x_master'], 
        stereo_calib_result['rectification_map_y_master'], 
        cv2.INTER_LINEAR
    )
    
    rectified_slave = cv2.remap(
        img_slave, 
        stereo_calib_result['rectification_map_x_slave'], 
        stereo_calib_result['rectification_map_y_slave'], 
        cv2.INTER_LINEAR
    )
    
    # Draw horizontal lines for visualization
    rectified_master_lines = draw_horizontal_lines(rectified_master)
    rectified_slave_lines = draw_horizontal_lines(rectified_slave)
    
    # Create composite of rectified images
    rectified_pair = np.hstack((rectified_master_lines, rectified_slave_lines))
    
    return original_pair, rectified_pair

def display_calibration_parameters(stereo_calib_result):
    """
    Display the key calibration parameters.
    
    Args:
        stereo_calib_result: Stereo calibration results
    """
    print("\n====== Stereo Calibration Results ======")
    print(f"RMS Error: {stereo_calib_result['rms_error']:.6f}")
    
    print("\nCamera Matrix (Master):")
    print(stereo_calib_result['camera_matrix_master'])
    
    print("\nDistortion Coefficients (Master):")
    print(stereo_calib_result['dist_coeffs_master'])
    
    print("\nCamera Matrix (Slave):")
    print(stereo_calib_result['camera_matrix_slave'])
    
    print("\nDistortion Coefficients (Slave):")
    print(stereo_calib_result['dist_coeffs_slave'])
    
    print("\nRotation Matrix (from Master to Slave):")
    print(stereo_calib_result['rotation_matrix'])
    
    print("\nTranslation Vector (from Master to Slave, in mm):")
    print(stereo_calib_result['translation_vector'])
    
    # Calculate baseline from translation vector
    baseline = np.linalg.norm(stereo_calib_result['translation_vector'])
    print(f"\nBaseline (mm): {baseline:.2f}")
    
    # Calculate focal length (average of fx and fy from camera matrix)
    fx_master = stereo_calib_result['camera_matrix_master'][0, 0]
    fy_master = stereo_calib_result['camera_matrix_master'][1, 1]
    focal_length_master = (fx_master + fy_master) / 2
    
    fx_slave = stereo_calib_result['camera_matrix_slave'][0, 0]
    fy_slave = stereo_calib_result['camera_matrix_slave'][1, 1]
    focal_length_slave = (fx_slave + fy_slave) / 2
    
    print(f"\nFocal Length (Master, pixels): {focal_length_master:.2f}")
    print(f"Focal Length (Slave, pixels): {focal_length_slave:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize stereo calibration results')
    parser.add_argument('--calib_file', type=str, default='calibration_results/stereo_calibration_8mm.pkl',
                        help='Path to the calibration file')
    parser.add_argument('--master_img', type=str, default='Chess 8mm/Master/1.jpeg',
                        help='Path to master camera image')
    parser.add_argument('--slave_img', type=str, default='Chess 8mm/Slave/1.jpeg',
                        help='Path to slave camera image')
    parser.add_argument('--output_dir', type=str, default='visualization',
                        help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load calibration results
    stereo_calib_result = load_calibration(args.calib_file)
    
    # Display calibration parameters
    display_calibration_parameters(stereo_calib_result)
    
    # Process and visualize image pair
    original_pair, rectified_pair = process_image_pair(
        args.master_img, args.slave_img, stereo_calib_result
    )
    
    # Save visualization results
    cv2.imwrite(os.path.join(args.output_dir, 'original_pair.jpg'), original_pair)
    cv2.imwrite(os.path.join(args.output_dir, 'rectified_pair.jpg'), rectified_pair)
    
    # Display disparity map (optional, for demonstration)
    rectified_master = cv2.remap(
        cv2.imread(args.master_img), 
        stereo_calib_result['rectification_map_x_master'], 
        stereo_calib_result['rectification_map_y_master'], 
        cv2.INTER_LINEAR
    )
    
    rectified_slave = cv2.remap(
        cv2.imread(args.slave_img), 
        stereo_calib_result['rectification_map_x_slave'], 
        stereo_calib_result['rectification_map_y_slave'], 
        cv2.INTER_LINEAR
    )
    
    # Convert to grayscale for disparity calculation
    gray_master = cv2.cvtColor(rectified_master, cv2.COLOR_BGR2GRAY)
    gray_slave = cv2.cvtColor(rectified_slave, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16*10, blockSize=15)
    disparity = stereo.compute(gray_master, gray_slave)
    
    # Normalize disparity for visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap for better visualization
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    # Save disparity map
    cv2.imwrite(os.path.join(args.output_dir, 'disparity_map.jpg'), disparity_color)
    
    print(f"\nVisualization results saved to {args.output_dir}")
    print("To display the results, run the following command:")
    print(f"  open {args.output_dir}/original_pair.jpg {args.output_dir}/rectified_pair.jpg {args.output_dir}/disparity_map.jpg")

if __name__ == "__main__":
    main() 