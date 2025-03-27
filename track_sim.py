import cv2
import os
import argparse
import pickle
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Track objects in stereo videos and perform 3D triangulation.')
    parser.add_argument('--master_video', type=str, required=True, help='Path to the master camera video file')
    parser.add_argument('--slave_video', type=str, required=True, help='Path to the slave camera video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--model', type=str, default="best.pt", help='Path to the YOLO model weights')
    parser.add_argument('--calib_file', type=str, default="stereo_calibration/calibration_results/stereo_calibration_8mm.pkl", 
                        help='Path to the stereo calibration file')
    parser.add_argument('--debug', action='store_true', help='Save debug images of circle detection')
    parser.add_argument('--debug_dir', type=str, default='debug_circles', help='Directory to save debug images')
    parser.add_argument('--trim_width', type=int, default=400, help='Width to trim image to after rectification')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence for ball detection refinement')
    parser.add_argument('--frame_skip', type=int, default=0, help='Process every Nth frame (0 = process all frames)')
    return parser.parse_args()

def refine_ball_center(frame, bbox, debug=False, frame_idx=None, debug_dir=None, camera=""):
    """
    Refine the center of the golf ball using circle detection.
    
    Args:
        frame: The input frame
        bbox: Bounding box from YOLO [x1, y1, x2, y2]
        debug: Whether to save debug images
        frame_idx: Frame index for debug image naming
        debug_dir: Directory to save debug images
        camera: Camera identifier for debug images ('master' or 'slave')
        
    Returns:
        center: Refined center coordinates (x, y)
        radius: Radius of the detected circle
        success: Whether circle detection was successful
        method: Method used for refinement ('hough', 'contour', or 'bbox')
    """
    # Extract the region of interest (ROI)
    x1, y1, x2, y2 = map(int, bbox)
    
    # Add some padding to the bounding box (20% on each side)
    padding_x = int((x2 - x1) * 0.2)
    padding_y = int((y2 - y1) * 0.2)
    
    # Ensure the padded box stays within the frame
    height, width = frame.shape[:2]
    x1_pad = max(0, x1 - padding_x)
    y1_pad = max(0, y1 - padding_y)
    x2_pad = min(width, x2 + padding_x)
    y2_pad = min(height, y2 + padding_y)
    
    # Extract the ROI
    roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Skip if ROI is too small
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        radius = min((x2 - x1), (y2 - y1)) / 2
        return (center_x, center_y), radius, False, 'bbox'
    
    # Convert to grayscale for processing
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # Calculate expected radius range based on bounding box
    expected_radius = min((x2 - x1), (y2 - y1)) / 2
    min_radius = max(3, int(expected_radius * 0.5))  # At least 3 pixels
    max_radius = int(expected_radius * 1.2)  # Allow slightly larger than bbox suggests
    
    # Try to detect circles using Hough transform with optimized parameters
    circles = cv2.HoughCircles(
        blurred_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Higher dp for better accuracy
        minDist=roi.shape[0],  # Only detect one circle
        param1=100,  # Upper threshold for Canny edge detector
        param2=20,   # Lower threshold for center detection (more sensitive)
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # If circles found, process them
    if circles is not None:
        # Get the best circle (usually only one is detected)
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
        
        # Convert back to original frame coordinates
        center_x = x + x1_pad
        center_y = y + y1_pad
        
        # Save debug image if requested
        if debug and frame_idx is not None and debug_dir is not None:
            debug_img = frame.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # YOLO bbox
            cv2.rectangle(debug_img, (x1_pad, y1_pad), (x2_pad, y2_pad), (255, 0, 0), 2)  # Padded bbox
            cv2.circle(debug_img, (int(center_x), int(center_y)), r, (0, 0, 255), 2)  # Detected circle
            cv2.circle(debug_img, (int(center_x), int(center_y)), 3, (0, 255, 255), -1)  # Center point
            cv2.putText(debug_img, f"Hough: ({center_x:.1f}, {center_y:.1f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{camera}_frame_{frame_idx}_hough.jpg"), debug_img)
        
        return (center_x, center_y), r, True, 'hough'
    
    # If no circles found, fall back to contour detection
    # Apply adaptive threshold to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check circularity to ensure it's actually a ball
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if circularity > 0.7:  # Threshold for circularity (circle has circularity of 1)
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            
            # Calculate centroid as an alternative center
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # Use centroid for better accuracy
                x, y = cx, cy
            
            # Convert back to original frame coordinates
            center_x = x + x1_pad
            center_y = y + y1_pad
            
            # Save debug image if requested
            if debug and frame_idx is not None and debug_dir is not None:
                debug_img = frame.copy()
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # YOLO bbox
                cv2.rectangle(debug_img, (x1_pad, y1_pad), (x2_pad, y2_pad), (255, 0, 0), 2)  # Padded bbox
                cv2.circle(debug_img, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)  # Detected circle
                cv2.circle(debug_img, (int(center_x), int(center_y)), 3, (0, 255, 255), -1)  # Center point
                cv2.putText(debug_img, f"Contour: ({center_x:.1f}, {center_y:.1f})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(debug_img, f"Circularity: {circularity:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{camera}_frame_{frame_idx}_contour.jpg"), debug_img)
            
            return (center_x, center_y), radius, True, 'contour'
    
    # If all methods fail, use the center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    radius = min((x2 - x1), (y2 - y1)) / 2
    
    # Save debug image for failed detection
    if debug and frame_idx is not None and debug_dir is not None:
        debug_img = frame.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # YOLO bbox
        cv2.circle(debug_img, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 2)  # Fallback circle
        cv2.putText(debug_img, f"Fallback: ({center_x:.1f}, {center_y:.1f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{camera}_frame_{frame_idx}_fallback.jpg"), debug_img)
    
    return (center_x, center_y), radius, False, 'bbox'

def triangulate_point(stereo_calib_result, point_master, point_slave):
    """
    Triangulate a 3D point from corresponding points in master and slave views.
    
    Args:
        stereo_calib_result: Stereo calibration results
        point_master: (x, y) coordinates in the master camera view
        point_slave: (x, y) coordinates in the slave camera view
        
    Returns:
        point_3d: (x, y, z) coordinates in 3D space (mm)
    """
    # Get projection matrices from calibration
    P1 = stereo_calib_result['projection_master']
    P2 = stereo_calib_result['projection_slave']
    
    # Convert points to the format needed by triangulatePoints
    points_master = np.array([[point_master[0]], [point_master[1]]], dtype=np.float32)
    points_slave = np.array([[point_slave[0]], [point_slave[1]]], dtype=np.float32)
    
    # Triangulate
    point_4d = cv2.triangulatePoints(P1, P2, points_master, points_slave)
    
    # Convert to 3D (homogeneous to Cartesian)
    point_3d = point_4d[:3, 0] / point_4d[3, 0]
    
    return point_3d

def create_visualization(frame_master, frame_slave, center_master, center_slave, point_3d, radius_master, radius_slave):
    """
    Create a visualization frame showing both camera views and the 3D position.
    
    Args:
        frame_master: Master camera frame
        frame_slave: Slave camera frame
        center_master: Ball center in master frame
        center_slave: Ball center in slave frame
        point_3d: 3D coordinates of the ball
        radius_master: Ball radius in master frame
        radius_slave: Ball radius in slave frame
        
    Returns:
        visualization: Combined visualization frame
    """
    # Make sure frames have the same height
    height = max(frame_master.shape[0], frame_slave.shape[0])
    
    # Resize frames if needed
    if frame_master.shape[0] != height:
        scale = height / frame_master.shape[0]
        frame_master = cv2.resize(frame_master, (int(frame_master.shape[1] * scale), height))
        
    if frame_slave.shape[0] != height:
        scale = height / frame_slave.shape[0]
        frame_slave = cv2.resize(frame_slave, (int(frame_slave.shape[1] * scale), height))
    
    # Draw ball center and circle on master frame
    if center_master is not None:
        cv2.circle(frame_master, (int(center_master[0]), int(center_master[1])), int(radius_master), (0, 0, 255), 2)
        cv2.circle(frame_master, (int(center_master[0]), int(center_master[1])), 5, (0, 255, 255), -1)
    
    # Draw ball center and circle on slave frame
    if center_slave is not None:
        cv2.circle(frame_slave, (int(center_slave[0]), int(center_slave[1])), int(radius_slave), (0, 0, 255), 2)
        cv2.circle(frame_slave, (int(center_slave[0]), int(center_slave[1])), 5, (0, 255, 255), -1)
    
    # Add camera labels
    cv2.putText(frame_master, "Master", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_slave, "Slave", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add 3D coordinates to both frames
    if point_3d is not None:
        text = f"3D: ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}) mm"
        cv2.putText(frame_master, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_slave, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine frames horizontally
    visualization = np.hstack((frame_master, frame_slave))
    
    return visualization

def calculate_motion_parameters(positions_3d, timestamps, fps, output_csv=None):
    """
    Calculate motion parameters (velocity, launch angle) from 3D positions over time.
    
    Args:
        positions_3d: List of 3D positions [(x1,y1,z1), (x2,y2,z2), ...]
        timestamps: List of frame timestamps or frame indices
        fps: Frames per second of the video
        output_csv: Path to save motion parameters to CSV
        
    Returns:
        Dictionary containing motion parameters
    """
    # Convert to numpy arrays for easier calculations
    positions = np.array(positions_3d)
    
    # Ensure we have enough points
    if len(positions) < 2:
        return {
            'initial_velocity': None,
            'launch_angle_xz': None,
            'launch_angle_yz': None,
            'velocities': [],
            'positions': positions.tolist() if len(positions) > 0 else []
        }
    
    # Calculate time differences between frames
    if isinstance(timestamps[0], (int, np.integer)):
        # If timestamps are frame indices, convert to seconds
        time_diffs = np.diff(timestamps) / fps
    else:
        # If timestamps are already in seconds
        time_diffs = np.diff(timestamps)
    
    # Calculate displacements between consecutive positions
    displacements = np.diff(positions, axis=0)
    
    # Calculate velocities (mm/s)
    velocities = displacements / time_diffs[:, np.newaxis]
    
    # Calculate speed (magnitude of velocity)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Calculate initial velocity (using first few frames for stability)
    # Use up to 5 frames or as many as available
    n_frames_for_initial = min(5, len(velocities))
    initial_velocity = np.mean(velocities[:n_frames_for_initial], axis=0)
    initial_speed = np.linalg.norm(initial_velocity)
    
    # Calculate launch angles
    # XZ plane (horizontal angle)
    launch_angle_xz = np.degrees(np.arctan2(initial_velocity[2], initial_velocity[0]))
    
    # YZ plane (vertical angle - launch angle)
    horizontal_component = np.sqrt(initial_velocity[0]**2 + initial_velocity[2]**2)
    launch_angle_yz = np.degrees(np.arctan2(initial_velocity[1], horizontal_component))
    
    # Prepare results
    results = {
        'initial_velocity': initial_velocity.tolist(),
        'initial_speed': float(initial_speed),
        'launch_angle_xz': float(launch_angle_xz),  # Horizontal angle (degrees)
        'launch_angle_yz': float(launch_angle_yz),  # Vertical angle (degrees)
        'velocities': velocities.tolist(),
        'speeds': speeds.tolist(),
        'positions': positions.tolist()
    }
    
    # Save to CSV if requested
    if output_csv:
        with open(output_csv, 'w') as f:
            f.write('frame,time,x,y,z,vx,vy,vz,speed\n')
            
            for i in range(len(positions)):
                # Write position data
                f.write(f'{timestamps[i]},{timestamps[i]/fps:.4f},{positions[i][0]:.2f},{positions[i][1]:.2f},{positions[i][2]:.2f}')
                
                # Write velocity data (if available)
                if i < len(velocities):
                    f.write(f',{velocities[i][0]:.2f},{velocities[i][1]:.2f},{velocities[i][2]:.2f},{speeds[i]:.2f}\n')
                else:
                    f.write(',,,\n')
    
    return results

def visualize_trajectory(positions_3d, output_path, motion_params=None):
    """
    Create a visualization of the 3D trajectory with motion parameters.
    
    Args:
        positions_3d: List of 3D positions
        output_path: Path to save the visualization
        motion_params: Dictionary of motion parameters
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert to numpy array
    positions = np.array(positions_3d)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], 'b-', linewidth=2)
    ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1], c='r', s=50)
    
    # Mark start point
    ax.scatter(positions[0, 0], positions[0, 2], positions[0, 1], c='g', s=100, label='Start')
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_zlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('3D Ball Trajectory')
    
    # Add motion parameters as text if available
    if motion_params:
        info_text = (
            f"Initial Speed: {motion_params['initial_speed']:.1f} mm/s\n"
            f"Launch Angle (vertical): {motion_params['launch_angle_yz']:.1f}°\n"
            f"Launch Angle (horizontal): {motion_params['launch_angle_xz']:.1f}°"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Equal aspect ratio
    max_range = np.max([
        np.max(positions[:, 0]) - np.min(positions[:, 0]),
        np.max(positions[:, 1]) - np.min(positions[:, 1]),
        np.max(positions[:, 2]) - np.min(positions[:, 2])
    ])
    mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) / 2
    mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) / 2
    mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_z - max_range/2, mid_z + max_range/2)
    ax.set_zlim(mid_y - max_range/2, mid_y + max_range/2)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load the YOLO model
    model = YOLO(args.model)

    # Load calibration results
    try:
        with open(args.calib_file, 'rb') as f:
            stereo_calib_result = pickle.load(f)
        print(f"Loaded calibration from {args.calib_file}")
    except FileNotFoundError:
        print(f"Error: Calibration file {args.calib_file} not found. Cannot perform triangulation.")
        return

    # Open the video files
    cap_master = cv2.VideoCapture(args.master_video)
    cap_slave = cv2.VideoCapture(args.slave_video)
    
    if not cap_master.isOpened():
        print(f"Error: Could not open master video: {args.master_video}")
        return
        
    if not cap_slave.isOpened():
        print(f"Error: Could not open slave video: {args.slave_video}")
        return
    
    # Get video properties
    fps_master = cap_master.get(cv2.CAP_PROP_FPS)
    fps_slave = cap_slave.get(cv2.CAP_PROP_FPS)
    width_master = int(cap_master.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_master = int(cap_master.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_slave = int(cap_slave.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_slave = int(cap_slave.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_master = int(cap_master.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_slave = int(cap_slave.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check that videos have similar properties
    if abs(fps_master - fps_slave) > 1.0:
        print(f"Warning: FPS mismatch between videos (master: {fps_master}, slave: {fps_slave})")
    
    print(f"Master video: {width_master}x{height_master} @ {fps_master} fps, {total_frames_master} frames")
    print(f"Slave video: {width_slave}x{height_slave} @ {fps_slave} fps, {total_frames_slave} frames")
    
    # Determine output dimensions
    if args.trim_width > 0:
        output_width_master = min(args.trim_width, width_master)
        output_width_slave = min(args.trim_width, width_slave)
    else:
        output_width_master = width_master
        output_width_slave = width_slave
    
    total_width = output_width_master + output_width_slave
    output_height = max(height_master, height_slave)
    
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(args.output))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps_master, (total_width, output_height))
    
    # Create CSV files to store tracking results
    base_path = os.path.splitext(args.output)[0]
    master_csv_path = f"{base_path}_master.csv"
    slave_csv_path = f"{base_path}_slave.csv"
    combined_csv_path = f"{base_path}_3d.csv"
    
    # Initialize CSV files
    with open(master_csv_path, 'w') as f:
        f.write('frame,x,y,radius,confidence,refined,method\n')
    
    with open(slave_csv_path, 'w') as f:
        f.write('frame,x,y,radius,confidence,refined,method\n')
    
    with open(combined_csv_path, 'w') as f:
        f.write('frame,x_master,y_master,x_slave,y_slave,x_3d,y_3d,z_3d\n')
    
    # Performance tracking
    start_time = time.time()
    processed_frames = 0
    positions_3d = []
    frame_indices = []

    # Frame processing loop
    frame_idx = 0
    while True:
        # Apply frame skip if specified
        if args.frame_skip > 0 and frame_idx % (args.frame_skip + 1) != 0:
            success_master = cap_master.grab()
            success_slave = cap_slave.grab()
            if not success_master or not success_slave:
                break
            frame_idx += 1
            continue
        
        # Read frames from both cameras
        success_master, frame_master = cap_master.read()
        success_slave, frame_slave = cap_slave.read()
        
        # Check if we've reached the end of either video
        if not success_master or not success_slave:
            break
        
        # Apply undistortion and rectification to both frames
        rect_start = time.time()
        frame_master_rect = cv2.remap(
            frame_master,
            stereo_calib_result['rectification_map_x_master'],
            stereo_calib_result['rectification_map_y_master'],
            cv2.INTER_LINEAR
        )
        
        frame_slave_rect = cv2.remap(
            frame_slave,
            stereo_calib_result['rectification_map_x_slave'],
            stereo_calib_result['rectification_map_y_slave'],
            cv2.INTER_LINEAR
        )
        rect_time = time.time() - rect_start
        
        # Trim frames if specified
        if args.trim_width > 0:
            if frame_master_rect.shape[1] > args.trim_width:
                frame_master_rect = frame_master_rect[:, :args.trim_width]
            if frame_slave_rect.shape[1] > args.trim_width:
                frame_slave_rect = frame_slave_rect[:, :args.trim_width]
        
        # Run ball detection on both frames
        detect_start = time.time()
        results_master = model.track(frame_master_rect, persist=True, tracker="bytetrack.yaml", verbose=False)
        results_slave = model.track(frame_slave_rect, persist=True, tracker="bytetrack.yaml", verbose=False)
        detect_time = time.time() - detect_start
        
        # Create annotated frames
        master_annotated = results_master[0].plot() if len(results_master) > 0 else frame_master_rect.copy()
        slave_annotated = results_slave[0].plot() if len(results_slave) > 0 else frame_slave_rect.copy()
        
        # Variables to store detected centers and 3D point
        center_master = None
        center_slave = None
        radius_master = 0
        radius_slave = 0
        point_3d = None
        
        # Process master frame detections
        if len(results_master[0].boxes) > 0:
            boxes_master = results_master[0].boxes.xyxy.cpu().numpy()
            confidences_master = results_master[0].boxes.conf.cpu().numpy()
            
            # Get the highest confidence detection
            best_idx = np.argmax(confidences_master)
            if confidences_master[best_idx] >= args.min_confidence:
                # Refine the ball center
                center_master, radius_master, refined_master, method_master = refine_ball_center(
                    frame_master_rect, boxes_master[best_idx], args.debug, frame_idx, args.debug_dir, "master"
                )
                
                # Save master camera detection to CSV
                with open(master_csv_path, 'a') as f:
                    f.write(f'{frame_idx},{center_master[0]},{center_master[1]},{radius_master},'
                            f'{confidences_master[best_idx]},{refined_master},{method_master}\n')
        
        # Process slave frame detections
        if len(results_slave[0].boxes) > 0:
            boxes_slave = results_slave[0].boxes.xyxy.cpu().numpy()
            confidences_slave = results_slave[0].boxes.conf.cpu().numpy()
            
            # Get the highest confidence detection
            best_idx = np.argmax(confidences_slave)
            if confidences_slave[best_idx] >= args.min_confidence:
                # Refine the ball center
                center_slave, radius_slave, refined_slave, method_slave = refine_ball_center(
                    frame_slave_rect, boxes_slave[best_idx], args.debug, frame_idx, args.debug_dir, "slave"
                )
                
                # Save slave camera detection to CSV
                with open(slave_csv_path, 'a') as f:
                    f.write(f'{frame_idx},{center_slave[0]},{center_slave[1]},{radius_slave},'
                            f'{confidences_slave[best_idx]},{refined_slave},{method_slave}\n')
        
        # If we have detected the ball in both views, triangulate
        if center_master is not None and center_slave is not None:
            triangulate_start = time.time()
            point_3d = triangulate_point(stereo_calib_result, center_master, center_slave)
            triangulate_time = time.time() - triangulate_start

            positions_3d.append(point_3d)
            frame_indices.append(frame_idx)
            
            # Save the 3D coordinates
            with open(combined_csv_path, 'a') as f:
                f.write(f'{frame_idx},{center_master[0]},{center_master[1]},{center_slave[0]},{center_slave[1]},'
                        f'{point_3d[0]},{point_3d[1]},{point_3d[2]}\n')
            
            # Print timing info occasionally
            if frame_idx % 100 == 0:
                print(f"Frame {frame_idx} - Rect: {rect_time*1000:.1f}ms, Detect: {detect_time*1000:.1f}ms, "
                      f"Triangulate: {triangulate_time*1000:.1f}ms")
                print(f"3D Position: ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}) mm")
        
        # Create visualization
        visualization = create_visualization(
            master_annotated, slave_annotated, center_master, center_slave, 
            point_3d, radius_master, radius_slave
        )
        
        # Write to output video and display
        out.write(visualization)
        cv2.imshow("Stereo Tracking with 3D Triangulation", visualization)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # Update counters
        frame_idx += 1
        processed_frames += 1
        
        # Print progress every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_avg = processed_frames / elapsed if elapsed > 0 else 0
            min_frames = min(total_frames_master, total_frames_slave)
            print(f"Progress: {frame_idx}/{min_frames} frames ({frame_idx/min_frames*100:.1f}%) - "
                  f"Average speed: {fps_avg:.1f} fps")
    
    # After processing all frames, calculate motion parameters
    if len(positions_3d) > 1:
        motion_csv_path = f"{base_path}_motion.csv"
        motion_params = calculate_motion_parameters(positions_3d, frame_indices, fps_master, motion_csv_path)
        
        # Print motion parameters
        print("\nMotion Analysis Results:")
        print(f"Initial Speed: {motion_params['initial_speed']:.2f} mm/s")
        print(f"Launch Angle (vertical): {motion_params['launch_angle_yz']:.2f} degrees")
        print(f"Launch Angle (horizontal): {motion_params['launch_angle_xz']:.2f} degrees")
        print(f"Motion data saved to: {motion_csv_path}")
        
        # Optionally, create a visualization of the trajectory
        if len(positions_3d) >= 3:
            trajectory_path = f"{base_path}_trajectory.jpg"
            visualize_trajectory(positions_3d, trajectory_path, motion_params)
            print(f"Trajectory visualization saved to: {trajectory_path}")

    # Release resources
    cap_master.release()
    cap_slave.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate and print performance statistics
    total_time = time.time() - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0
    print(f"Video processing complete. Output saved to: {args.output}")
    print(f"Master positions saved to: {master_csv_path}")
    print(f"Slave positions saved to: {slave_csv_path}")
    print(f"3D positions saved to: {combined_csv_path}")
    print(f"Processed {processed_frames} frames in {total_time:.1f} seconds ({avg_fps:.1f} fps)")

if __name__ == "__main__":
    main() 