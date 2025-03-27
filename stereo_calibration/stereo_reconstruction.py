import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def rectify_images(img_master, img_slave, stereo_calib_result):
    """
    Rectify a pair of stereo images.
    
    Args:
        img_master: Master camera image
        img_slave: Slave camera image
        stereo_calib_result: Stereo calibration results
        
    Returns:
        rectified_master: Rectified master image
        rectified_slave: Rectified slave image
    """
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
    
    return rectified_master, rectified_slave

def compute_disparity_map(rectified_master, rectified_slave, block_size=15, num_disparities=160):
    """
    Compute a disparity map from rectified stereo images.
    
    Args:
        rectified_master: Rectified master image
        rectified_slave: Rectified slave image
        block_size: Block size for stereo matching
        num_disparities: Number of disparities to consider
        
    Returns:
        disparity: Disparity map
    """
    # Convert to grayscale if not already
    if len(rectified_master.shape) == 3:
        gray_master = cv2.cvtColor(rectified_master, cv2.COLOR_BGR2GRAY)
        gray_slave = cv2.cvtColor(rectified_slave, cv2.COLOR_BGR2GRAY)
    else:
        gray_master = rectified_master
        gray_slave = rectified_slave
    
    # Create stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    
    # Compute disparity map
    disparity = stereo.compute(gray_master, gray_slave)
    
    return disparity

def generate_point_cloud(disparity, stereo_calib_result, max_points=10000):
    """
    Generate a 3D point cloud from a disparity map.
    
    Args:
        disparity: Disparity map
        stereo_calib_result: Stereo calibration results
        max_points: Maximum number of points to include
        
    Returns:
        points_3d: 3D point cloud (Nx3 array)
        colors: Colors for each point if available (Nx3 array)
    """
    # Convert disparity to 3D points
    h, w = disparity.shape
    
    # Sample points to avoid too many points
    step = max(1, int(np.sqrt(h * w / max_points)))
    
    # Get valid disparity values
    mask = disparity > disparity.min()
    
    # Sample valid positions
    y_coords, x_coords = np.where(mask)
    indices = np.arange(len(y_coords))
    np.random.shuffle(indices)
    indices = indices[:max_points]
    
    points_2d = np.array([x_coords[indices], y_coords[indices]]).T
    disparity_values = disparity[y_coords[indices], x_coords[indices]]
    
    # Filter out invalid disparities
    valid_indices = disparity_values > 0
    points_2d = points_2d[valid_indices]
    disparity_values = disparity_values[valid_indices]
    
    # Reshape for cv2.reprojectImageTo3D
    disp_for_reprojection = np.zeros((h, w), dtype=np.float32)
    for i, (x, y) in enumerate(points_2d):
        disp_for_reprojection[y, x] = disparity_values[i]
    
    # Reproject to 3D
    Q = stereo_calib_result['disparity_to_depth_matrix']
    points_3d = cv2.reprojectImageTo3D(disp_for_reprojection, Q)
    
    # Flatten points and filter
    points_3d = points_3d.reshape(-1, 3)
    valid_indices = np.where(
        (np.abs(points_3d[:, 0]) < 10000) &  # Filter out points too far in X
        (np.abs(points_3d[:, 1]) < 10000) &  # Filter out points too far in Y
        (np.abs(points_3d[:, 2]) < 10000)    # Filter out points too far in Z
    )[0]
    
    points_3d = points_3d[valid_indices]
    
    # Return points without colors for now
    return points_3d, None

def plot_point_cloud(points_3d, colors=None, title="3D Point Cloud"):
    """
    Plot a 3D point cloud.
    
    Args:
        points_3d: 3D point cloud (Nx3 array)
        colors: Colors for each point (Nx3 array)
        title: Title for the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        # Normalize colors to [0, 1] for matplotlib
        colors = colors / 255.0
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors, s=0.5)
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=0.5)
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    
    # Add a grid
    ax.grid(True)
    
    # Set axis limits if necessary
    percentile = 95
    xlim = np.percentile(np.abs(points_3d[:, 0]), percentile)
    ylim = np.percentile(np.abs(points_3d[:, 1]), percentile)
    zlim = np.percentile(np.abs(points_3d[:, 2]), percentile)
    
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_zlim(-zlim, zlim)
    
    return fig, ax

def triangulate_points(points_master, points_slave, stereo_calib_result):
    """
    Triangulate 3D points from corresponding points in stereo images.
    
    Args:
        points_master: Points in master image (Nx2 array)
        points_slave: Corresponding points in slave image (Nx2 array)
        stereo_calib_result: Stereo calibration results
        
    Returns:
        points_3d: Triangulated 3D points (Nx3 array)
    """
    # Get projection matrices
    P1 = stereo_calib_result['projection_master']
    P2 = stereo_calib_result['projection_slave']
    
    # Convert points to homogeneous coordinates
    points_master = np.array(points_master, dtype=np.float32).T
    points_slave = np.array(points_slave, dtype=np.float32).T
    
    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, points_master, points_slave)
    
    # Convert to 3D (divide by w)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    # Return transposed (Nx3)
    return points_3d.T

def main():
    parser = argparse.ArgumentParser(description='Stereo 3D reconstruction')
    parser.add_argument('--calib_file', type=str, default='calibration_results/stereo_calibration_8mm.pkl',
                        help='Path to the calibration file')
    parser.add_argument('--master_img', type=str, default='Chess 8mm/Master/1.jpeg',
                        help='Path to master camera image')
    parser.add_argument('--slave_img', type=str, default='Chess 8mm/Slave/1.jpeg',
                        help='Path to slave camera image')
    parser.add_argument('--output_dir', type=str, default='reconstruction',
                        help='Directory to save reconstruction results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load calibration results
    stereo_calib_result = load_calibration(args.calib_file)
    
    # Read images
    img_master = cv2.imread(args.master_img)
    img_slave = cv2.imread(args.slave_img)
    
    if img_master is None or img_slave is None:
        raise ValueError(f"Failed to read images: {args.master_img}, {args.slave_img}")
    
    # Rectify images
    rectified_master, rectified_slave = rectify_images(img_master, img_slave, stereo_calib_result)
    
    # Save rectified images
    cv2.imwrite(os.path.join(args.output_dir, 'rectified_master.jpg'), rectified_master)
    cv2.imwrite(os.path.join(args.output_dir, 'rectified_slave.jpg'), rectified_slave)
    
    # Compute disparity map
    disparity = compute_disparity_map(rectified_master, rectified_slave)
    
    # Normalize and save disparity map
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output_dir, 'disparity_map.jpg'), disparity_color)
    
    # Generate point cloud
    points_3d, _ = generate_point_cloud(disparity, stereo_calib_result)
    
    # Save point cloud
    np.save(os.path.join(args.output_dir, 'point_cloud.npy'), points_3d)
    
    # Plot and save point cloud visualization
    fig, ax = plot_point_cloud(points_3d, title="3D Reconstruction from Stereo Images")
    plt.savefig(os.path.join(args.output_dir, 'point_cloud.png'), dpi=300)
    
    # If the input image is a chessboard, also demonstrate triangulation of chessboard corners
    if args.master_img and args.slave_img:
        # Convert to grayscale
        gray_master = cv2.cvtColor(rectified_master, cv2.COLOR_BGR2GRAY)
        gray_slave = cv2.cvtColor(rectified_slave, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        pattern_size = (4, 4)  # 4x4 internal corners in a 5x5 chessboard
        ret_master, corners_master = cv2.findChessboardCorners(gray_master, pattern_size, None)
        ret_slave, corners_slave = cv2.findChessboardCorners(gray_slave, pattern_size, None)
        
        if ret_master and ret_slave:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_master = cv2.cornerSubPix(gray_master, corners_master, (11, 11), (-1, -1), criteria)
            corners_slave = cv2.cornerSubPix(gray_slave, corners_slave, (11, 11), (-1, -1), criteria)
            
            # Draw corners on images
            img_corners_master = rectified_master.copy()
            img_corners_slave = rectified_slave.copy()
            cv2.drawChessboardCorners(img_corners_master, pattern_size, corners_master, ret_master)
            cv2.drawChessboardCorners(img_corners_slave, pattern_size, corners_slave, ret_slave)
            
            # Save images with corners
            cv2.imwrite(os.path.join(args.output_dir, 'corners_master.jpg'), img_corners_master)
            cv2.imwrite(os.path.join(args.output_dir, 'corners_slave.jpg'), img_corners_slave)
            
            # Triangulate corners
            corners_master_reshaped = corners_master.reshape(-1, 2)
            corners_slave_reshaped = corners_slave.reshape(-1, 2)
            
            # Triangulate
            corners_3d = triangulate_points(corners_master_reshaped, corners_slave_reshaped, stereo_calib_result)
            
            # Save 3D corners
            np.save(os.path.join(args.output_dir, 'corners_3d.npy'), corners_3d)
            
            # Plot 3D corners
            fig2, ax2 = plot_point_cloud(corners_3d, title="3D Chessboard Corners")
            
            # Make the plot more readable
            ax2.plot(corners_3d[:, 0], corners_3d[:, 1], corners_3d[:, 2], 'r-')
            
            # Connect points in a grid pattern
            for i in range(pattern_size[0]):
                idx = i * pattern_size[1]
                ax2.plot(corners_3d[idx:idx+pattern_size[1], 0], 
                         corners_3d[idx:idx+pattern_size[1], 1], 
                         corners_3d[idx:idx+pattern_size[1], 2], 'g-')
                
            for i in range(pattern_size[1]):
                idx = np.arange(i, pattern_size[0] * pattern_size[1], pattern_size[1])
                ax2.plot(corners_3d[idx, 0], corners_3d[idx, 1], corners_3d[idx, 2], 'b-')
            
            plt.savefig(os.path.join(args.output_dir, 'corners_3d.png'), dpi=300)
    
    print(f"Reconstruction results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 