import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

def hsv_filter(img, lwr, upr, pattern_size=(4, 4)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)

    res = np.uint8(res)
    ret, corners = cv2.findChessboardCorners(res, pattern_size,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
    return res, ret, corners

def find_chessboard_corners(master_images, slave_images, pattern_size=(4, 4), square_size=100, save_debug=True, debug_dir="debug_corners"):
    """
    Find chessboard corners in the provided images.
    
    Args:
        master_images: List of paths to master images.
        slave_images: List of paths to slave images.
        pattern_size: Size of the chessboard pattern (internal corners).
        square_size: Size of each square in millimeters.
        save_debug: If True, saves images with drawn corners for debugging.
        debug_dir: Directory to save debug images.
        
    Returns:
        objpoints: 3D points in real world space.
        imgpoints: 2D points in image plane.
    """
    # Create debug directory if needed
    if save_debug:
        Path(debug_dir).mkdir(exist_ok=True)
    
   # Prepare object points (0,0,0), (100,0,0), (200,0,0) ... etc
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    found_corners_path = []
    
    # Arrays to store object points and image points
    objpoints_master = []  # 3D points in real world space
    imgpoints_master = []  # 2D points in image plane

    objpoints_slave = []  # 3D points in real world space
    imgpoints_slave = []  # 2D points in image plane
    
    # Criteria for corner detection refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Define flags for improved detection
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE 
    
    # Define a list of HSV thresholds to try
    hsv_thresholds = [
        np.array([0, 0, 90]),   # First threshold to try
        np.array([0, 0, 135]),  # Second threshold to try
        np.array([0, 0, 235])   # Third threshold to try
    ]
    upper_threshold = np.array([179, 255, 255])  # Upper threshold is the same for all

    for i, (mst_path, slv_path) in enumerate(zip(master_images, slave_images)):
        print(f"Processing image pair {i+1}/{len(master_images)}: {os.path.basename(mst_path)} / {os.path.basename(slv_path)}")
        mst_img = cv2.imread(mst_path)
        slv_img = cv2.imread(slv_path)
        
        if mst_img is None:
            print(f"Could not read master image: {mst_path}")
            continue
            
        if slv_img is None:
            print(f"Could not read slave image: {slv_path}")
            continue
            
        found_corners = False
        
        # Try each threshold until we find corners
        for threshold in hsv_thresholds:
            mst_res, mst_ret, mst_corners = hsv_filter(mst_img, threshold, upper_threshold, pattern_size)
            slv_res, slv_ret, slv_corners = hsv_filter(slv_img, threshold, upper_threshold, pattern_size)
            
            if mst_ret and slv_ret:
                objpoints_master.append(objp)
                objpoints_slave.append(objp)

                mst_corners2 = cv2.cornerSubPix(mst_res, mst_corners, (11, 11), (-1, -1), criteria)
                slv_corners2 = cv2.cornerSubPix(slv_res, slv_corners, (11, 11), (-1, -1), criteria)
                imgpoints_master.append(mst_corners2)
                imgpoints_slave.append(slv_corners2)

                mst_img_display = mst_img.copy()
                slv_img_display = slv_img.copy()
                
                cv2.drawChessboardCorners(mst_img_display, pattern_size, mst_corners2, mst_ret)
                cv2.drawChessboardCorners(slv_img_display, pattern_size, slv_corners2, slv_ret)
                
                found_corners = True
                print(f"  Found corners in both images!")
                
                if save_debug:
                    mst_filename = os.path.basename(mst_path)
                    slv_filename = os.path.basename(slv_path)
                    filename = os.path.splitext(mst_filename)[0] + "_" + os.path.splitext(slv_filename)[0] + ".jpg"
                    found_corners_path.append(os.path.splitext(filename)[0])
                    img = np.hstack((mst_img_display, slv_img_display))
                    debug_file = os.path.join(debug_dir, f"corners_{filename}")
                    cv2.imwrite(debug_file, img)
                    print(f"  Saved debug image to {debug_file}")
                break
        
        # If no corners were found with any threshold
        if not found_corners:
            print(f"  Failed to find chessboard corners in this image pair")
    
    print(f"Successfully found corners in {len(objpoints_master)} image pairs out of {len(master_images)}")
    
    return objpoints_master, imgpoints_master, objpoints_slave, imgpoints_slave, found_corners_path    
    
def calibrate_camera(objpoints, imgpoints, img_shape):
    """
    Calibrate a single camera.
    
    Args:
        objpoints: 3D points in real world space
        imgpoints: 2D points in image plane
        img_shape: Shape of the image (height, width)
        
    Returns:
        ret: RMS error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img_shape[1], img_shape[0]), None, None
    )
    
    return ret, mtx, dist, rvecs, tvecs

def save_calibration_yaml(filename, camera_matrix, dist_coeffs, image_width, image_height, reprojection_error=0.0):
    """
    Save calibration parameters in YAML format compatible with the C++ calibration tool.
    """
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_width)
    fs.write("image_height", image_height)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("avg_reprojection_error", reprojection_error)
    fs.write("calibration_time", cv2.getTickCount())
    fs.release()
    print(f"Calibration saved to {filename}")

def main():
    # Parse command-line arguments
    # Note: argparse automatically handles both "--arg=value" and "--arg value" formats
    parser = argparse.ArgumentParser(description='Camera calibration from chessboard images')
    parser.add_argument('--master', required=True, help='Directory containing master camera images')
    parser.add_argument('--slave', required=True, help='Directory containing slave camera images')
    parser.add_argument('--output', default='calibration_results', help='Output directory for calibration files')
    parser.add_argument('--pattern', default='4x4', help='Chessboard pattern internal corners (NxM)')
    parser.add_argument('--square', type=float, default=100.0, help='Chessboard square size in mm')
    
    # Parse args and handle equals format (argparse handles this automatically)
    args = parser.parse_args()
    
    # Parameters
    pattern_parts = args.pattern.split('x')
    pattern_size = (int(pattern_parts[0]), int(pattern_parts[1]))
    square_size = args.square
    output_dir = args.output
    debug_dir = os.path.join(output_dir, "debug_corners")
    
    print(f"Using chessboard pattern: {pattern_size[0]}x{pattern_size[1]} internal corners")
    print(f"Square size: {square_size} mm")
    print(f"Master images directory: {args.master}")
    print(f"Slave images directory: {args.slave}")
    
    # Debug: List master directory contents
    print("Master directory contents:")
    try:
        print(os.listdir(args.master)[:5])
    except Exception as e:
        print(f"Error listing master directory: {e}")
    
    # Paths to the images
    master_pattern = os.path.join(args.master, '*.jpeg')
    slave_pattern = os.path.join(args.slave, '*.jpeg')
    
    print(f"Master glob pattern: {master_pattern}")
    print(f"Slave glob pattern: {slave_pattern}")
    
    master_images = sorted(glob.glob(master_pattern))
    if len(master_images) == 0:
        master_pattern = os.path.join(args.master, '*.jpg')
        master_images = sorted(glob.glob(master_pattern))
        print(f"Tried alternate pattern: {master_pattern}, found: {len(master_images)}")
    if len(master_images) == 0:
        master_pattern = os.path.join(args.master, '*.png')
        master_images = sorted(glob.glob(master_pattern))
        print(f"Tried alternate pattern: {master_pattern}, found: {len(master_images)}")
    
    slave_images = sorted(glob.glob(slave_pattern))
    if len(slave_images) == 0:
        slave_pattern = os.path.join(args.slave, '*.jpg')
        slave_images = sorted(glob.glob(slave_pattern))
        print(f"Tried alternate pattern: {slave_pattern}, found: {len(slave_images)}")
    if len(slave_images) == 0:
        slave_pattern = os.path.join(args.slave, '*.png')
        slave_images = sorted(glob.glob(slave_pattern))
        print(f"Tried alternate pattern: {slave_pattern}, found: {len(slave_images)}")
    
    # Check if we have paired images
    if len(master_images) == 0 or len(slave_images) == 0:
        print(f"Error: No images found in directories: {args.master}, {args.slave}")
        return 1
    
    if len(master_images) != len(slave_images):
        print(f"Warning: Number of master ({len(master_images)}) and slave ({len(slave_images)}) images don't match!")
        min_images = min(len(master_images), len(slave_images))
        master_images = master_images[:min_images]
        slave_images = slave_images[:min_images]
        print(f"Using only the first {min_images} images from each set")
    
    print(f"Found {len(master_images)} master images and {len(slave_images)} slave images")
    if len(master_images) > 0:
        print(f"First master image: {master_images[0]}")
    if len(slave_images) > 0:
        print(f"First slave image: {slave_images[0]}")
    
    # Read an image to get dimensions
    if len(master_images) == 0:
        print("Error: No master images found. Exiting.")
        return 1
        
    img = cv2.imread(master_images[0])
    if img is None:
        print(f"Error: Could not read image {master_images[0]}")
        return 1
        
    img_shape = img.shape
    print(f"Image shape: {img_shape}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find chessboard corners
    print("Finding chessboard corners in master images and slave images")
    objpoints_master, imgpoints_master, objpoints_slave, imgpoints_slave, _ = find_chessboard_corners(
        master_images, slave_images, pattern_size, square_size, True, debug_dir
    )
    
    if len(objpoints_master) == 0 or len(objpoints_slave) == 0:
        print("Error: No corners detected in any images. Cannot perform calibration.")
        return 1
    
    # Calibrate master camera
    print("Calibrating master camera...")
    ret_master, mtx_master, dist_master, rvecs_master, tvecs_master = calibrate_camera(
        objpoints_master, imgpoints_master, img_shape
    )
    print(f"Master camera calibration RMS error: {ret_master}")
    
    # Calibrate slave camera
    print("Calibrating slave camera...")
    ret_slave, mtx_slave, dist_slave, rvecs_slave, tvecs_slave = calibrate_camera(
        objpoints_slave, imgpoints_slave, img_shape
    )
    print(f"Slave camera calibration RMS error: {ret_slave}")
    
    # Save calibration files in YAML format for C++ compatibility
    save_calibration_yaml(
        os.path.join(output_dir, "master_calibration.yaml"),
        mtx_master,
        dist_master,
        img_shape[1],
        img_shape[0],
        ret_master
    )
    
    save_calibration_yaml(
        os.path.join(output_dir, "slave_calibration.yaml"),
        mtx_slave,
        dist_slave,
        img_shape[1],
        img_shape[0],
        ret_slave
    )
    
    print("Calibration completed successfully!")
    print(f"Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
