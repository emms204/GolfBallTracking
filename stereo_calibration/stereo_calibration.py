import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

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

    for mst_path, slv_path in zip(master_images, slave_images):
        mst_img = cv2.imread(mst_path)
        slv_img = cv2.imread(slv_path)
        found_corners = False
        
        # Try each threshold until we find corners
        for threshold in hsv_thresholds:
            mst_res, mst_ret, mst_corners = hsv_filter(mst_img, threshold, upper_threshold)
            slv_res, slv_ret, slv_corners = hsv_filter(slv_img, threshold, upper_threshold)
            # print(f"mst_ret: {mst_ret}, slv_ret: {slv_ret}")
            if mst_ret and slv_ret:
                objpoints_master.append(objp)
                objpoints_slave.append(objp)

                mst_corners2 = cv2.cornerSubPix(mst_res, mst_corners, (11, 11), (-1, -1), criteria)
                slv_corners2 = cv2.cornerSubPix(slv_res, slv_corners, (11, 11), (-1, -1), criteria)
                imgpoints_master.append(mst_corners2)
                imgpoints_slave.append(slv_corners2)

                mst_img = cv2.drawChessboardCorners(mst_img, pattern_size, mst_corners2, mst_ret)
                slv_img = cv2.drawChessboardCorners(slv_img, pattern_size, slv_corners2, slv_ret)
                found_corners = True
                if save_debug:
                    mst_filename = os.path.basename(mst_path)
                    slv_filename = os.path.basename(slv_path)
                    filename = os.path.splitext(mst_filename)[0] + "_" + os.path.splitext(slv_filename)[0] + ".jpg"
                    found_corners_path.append(os.path.splitext(filename)[0])
                    img = np.hstack((mst_img, slv_img))
                    cv2.imwrite(os.path.join(debug_dir, f"corners_{filename}"), img)
                break
        # If no corners were found with any threshold
        if not found_corners:
            print(f"Failed to find chessboard corners in {mst_path} and {slv_path}")
    
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

def stereo_calibrate(objpoints, imgpoints_master, imgpoints_slave, mtx_master, 
                     dist_master, mtx_slave, dist_slave, img_shape):
    """
    Perform stereo calibration.
    
    Args:
        objpoints: 3D points in real world space
        imgpoints_master, imgpoints_slave: 2D points in image planes
        mtx_master, mtx_slave: Camera matrices
        dist_master, dist_slave: Distortion coefficients
        img_shape: Shape of the image (height, width)
        
    Returns:
        stereo_calib_result: Dictionary containing stereo calibration results
    """
    # Convert image shape to dimensions accepted by OpenCV
    img_size = (img_shape[1], img_shape[0])
    
    # Perform stereo calibration
    # flags = cv2.CALIB_FIX_INTRINSIC  # Use the intrinsic parameters we already found
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret, mtx_master, dist_master, mtx_slave, dist_slave, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_master, imgpoints_slave,
        mtx_master, dist_master,
        mtx_slave, dist_slave,
        img_size,
        flags=flags
    )
    
    # Compute rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_master, dist_master,
        mtx_slave, dist_slave,
        img_size, R, T,
        None, None
    )

    
    # Compute rectification maps
    map1_master, map2_master = cv2.initUndistortRectifyMap(
        mtx_master, dist_master, R1, P1, img_size, cv2.CV_32FC1
    )
    
    map1_slave, map2_slave = cv2.initUndistortRectifyMap(
        mtx_slave, dist_slave, R2, P2, img_size, cv2.CV_32FC1
    )
    
    # Store calibration results
    stereo_calib_result = {
        'rms_error': ret,
        'camera_matrix_master': mtx_master,
        'dist_coeffs_master': dist_master,
        'camera_matrix_slave': mtx_slave,
        'dist_coeffs_slave': dist_slave,
        'rotation_matrix': R,
        'translation_vector': T,
        'essential_matrix': E,
        'fundamental_matrix': F,
        'rectification_master': R1,
        'rectification_slave': R2,
        'projection_master': P1,
        'projection_slave': P2,
        'disparity_to_depth_matrix': Q,
        'valid_roi_master': roi1,
        'valid_roi_slave': roi2,
        'rectification_map_x_master': map1_master,
        'rectification_map_y_master': map2_master,
        'rectification_map_x_slave': map1_slave,
        'rectification_map_y_slave': map2_slave
    }
    
    return stereo_calib_result

def verify_calibration(stereo_calib_result, master_image_path, slave_image_path):
    """
    Verify the calibration by rectifying a pair of images.
    
    Args:
        stereo_calib_result: Stereo calibration results
        master_image_path: Path to a master camera image
        slave_image_path: Path to a slave camera image
        
    Returns:
        rectified_img: Composite image showing rectification
    """
    # Read images
    img_master = cv2.imread(master_image_path)
    img_slave = cv2.imread(slave_image_path)
    
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
    
    # Draw horizontal lines every 50 pixels for visual verification
    height, width = img_master.shape[:2]
    rectified_master_with_lines = rectified_master.copy()
    rectified_slave_with_lines = rectified_slave.copy()
    
    for y in range(0, height, 50):
        cv2.line(rectified_master_with_lines, (0, y), (width, y), (0, 255, 0), 1)
        cv2.line(rectified_slave_with_lines, (0, y), (width, y), (0, 255, 0), 1)
    
    # Combine images side by side
    rectified_img = np.hstack((rectified_master_with_lines, rectified_slave_with_lines))
    
    return rectified_img

def main():
    # Parameters
    pattern_size = (4, 4)  # 4x4 internal corners in a 5x5 chessboard
    square_size = 100.0  # Size of each square in millimeters
    
    # Paths to the images
    master_images = sorted(glob.glob(os.path.join('../Combined', 'Master', '*.jpeg')))
    slave_images = sorted(glob.glob(os.path.join('../Combined', 'Slave', '*.jpeg')))
    
    # Check if we have paired images
    if len(master_images) != len(slave_images):
        print("Error: Number of master and slave images don't match!")
        return
    
    print(f"Found {len(master_images)} master images and {len(slave_images)} slave images")
    
    # Read an image to get dimensions
    img = cv2.imread(master_images[0])
    img_shape = img.shape
    print(f"Image shape: {img_shape}")
    
    # Find chessboard corners
    print("Finding chessboard corners in master images and slave images")
    objpoints_master, imgpoints_master, objpoints_slave, imgpoints_slave, _ = find_chessboard_corners(master_images, slave_images, pattern_size, square_size)
    
    # Remove bad images
    bad_idx = [6, 36, 41, 44]

    objpoints_master = [obj for i, obj in enumerate(objpoints_master) if i not in bad_idx]
    imgpoints_master = [img for i, img in enumerate(imgpoints_master) if i not in bad_idx]
    objpoints_slave = [obj for i, obj in enumerate(objpoints_slave) if i not in bad_idx]
    imgpoints_slave = [img for i, img in enumerate(imgpoints_slave) if i not in bad_idx]

    # Make sure we found corners in the same number of images
    common_count = min(len(objpoints_master), len(objpoints_slave))
    objpoints = objpoints_master
    # imgpoints_master = imgpoints_master[:common_count]
    # imgpoints_slave = imgpoints_slave[:common_count]
    
    print(f"Successfully detected corners in {common_count} image pairs")
    
    # Calibrate master camera
    print("Calibrating master camera...")
    ret_master, mtx_master, dist_master, rvecs_master, tvecs_master = calibrate_camera(
        objpoints, imgpoints_master, img_shape
    )
    print(f"Master camera calibration RMS error: {ret_master}")
    
    # Calibrate slave camera
    print("Calibrating slave camera...")
    ret_slave, mtx_slave, dist_slave, rvecs_slave, tvecs_slave = calibrate_camera(
        objpoints, imgpoints_slave, img_shape
    )
    print(f"Slave camera calibration RMS error: {ret_slave}")
    
    # Perform stereo calibration
    print("Performing stereo calibration...")
    stereo_calib_result = stereo_calibrate(
        objpoints, imgpoints_master, imgpoints_slave,
        mtx_master, dist_master,
        mtx_slave, dist_slave,
        img_shape
    )
    
    print(f"Stereo calibration RMS error: {stereo_calib_result['rms_error']}")
    print(f"Rotation matrix: \n{stereo_calib_result['rotation_matrix']}")
    print(f"Translation vector (baseline in mm): \n{stereo_calib_result['translation_vector']}")
    
    # Create output directory if it doesn't exist
    output_dir = "calibration_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save calibration results
    print(f"Saving calibration results to {output_dir}...")
    with open(os.path.join(output_dir, 'stereo_calibration_8mm.pkl'), 'wb') as f:
        pickle.dump(stereo_calib_result, f)
    
    # Save individual camera calibration results as well
    np.savez(os.path.join(output_dir, 'master_calibration_8mm.npz'),
             camera_matrix=mtx_master,
             dist_coeffs=dist_master,
             rvecs=rvecs_master,
             tvecs=tvecs_master)
    
    np.savez(os.path.join(output_dir, 'slave_calibration_8mm.npz'),
             camera_matrix=mtx_slave,
             dist_coeffs=dist_slave,
             rvecs=rvecs_slave,
             tvecs=tvecs_slave)
    
    # Verify calibration
    print("Verifying calibration...")
    # Use the first image pair for verification
    rectified_img = verify_calibration(
        stereo_calib_result, master_images[0], slave_images[0]
    )
    
    # Save the rectified image
    cv2.imwrite(os.path.join(output_dir, 'rectified_pair.jpg'), rectified_img)
    
    print("Calibration completed successfully!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 