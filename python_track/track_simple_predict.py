import cv2
import os
import argparse
import pickle
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO

# Global debug log file
g_debug_log = None

def log_message(message, to_console=True):
    """Function to log messages to both console and file"""
    if to_console:
        print(message)
    if g_debug_log is not None:
        g_debug_log.write(message + "\n")
        g_debug_log.flush()  # Ensure log is written immediately

def get_current_time_string():
    """Function to get the current time as a string for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process objects in a video using YOLO (simple prediction).')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output')
    parser.add_argument('--model', type=str, default="best.pt", help='Path to the YOLO model weights')
    parser.add_argument('--calib_file', type=str, default="stereo_calibration/calibration_results/stereo_calibration_8mm.pkl", 
                        help='Path to the stereo calibration file')
    parser.add_argument('--camera', type=str, choices=['master', 'slave'], default='master',
                        help='Which camera the video is from (master or slave)')
    parser.add_argument('--log_file', type=str, default="python_simple_predict.log", 
                        help='Path to save the debug log file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize debug log file
    global g_debug_log
    g_debug_log = open(args.log_file, "w")
    if g_debug_log is None:
        print(f"Warning: Could not open {args.log_file} for writing")
    
    # Log start information
    log_message("=== Python YOLO Object Detector (Simple Prediction) ===")
    log_message(f"Time: {get_current_time_string()}")
    log_message(f"Video: {args.video}")
    log_message(f"Model: {args.model}")
    log_message(f"Camera: {args.camera}")
    log_message(f"Calibration: {args.calib_file}")
    
    # Load the best model weights into the YOLO model
    model = YOLO(args.model)
    log_message(f"Loaded YOLO model: {args.model}")
    
    # Log model information
    log_message(f"Model task: {model.task}")
    log_message(f"Model stride: {model.stride}")
    log_message(f"Model names: {model.names}")

    # Load calibration results
    stereo_calib_result = None
    try:
        if args.calib_file.endswith('.pkl'):
            with open(args.calib_file, 'rb') as f:
                stereo_calib_result = pickle.load(f)
            log_message(f"Loaded calibration from {args.calib_file}")
        elif args.calib_file.endswith('.npz'):
            stereo_calib_result = np.load(args.calib_file)
            log_message(f"Loaded NPZ calibration from {args.calib_file}")
            log_message(f"Available keys in calibration file: {list(stereo_calib_result.keys())}")
            
            # Convert to format compatible with cv2.remap
            if 'camera_matrix' in stereo_calib_result and 'dist_coeffs' in stereo_calib_result:
                camera_matrix = stereo_calib_result['camera_matrix']
                dist_coeffs = stereo_calib_result['dist_coeffs']
                log_message(f"Loaded camera matrix: {camera_matrix}")
                log_message(f"Loaded distortion coefficients: {dist_coeffs}")
                
                # We'll use these to create maps dynamically since they're not pre-computed
                log_message("Will create undistortion maps dynamically")
                
    except Exception as e:
        log_message(f"Warning: Error loading calibration file {args.calib_file}: {str(e)}")
        log_message("Running without undistortion.")
        stereo_calib_result = None

    # Open the video file
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log_message(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialize maps for undistortion
    undistort_maps = None
    if stereo_calib_result is not None and 'camera_matrix' in stereo_calib_result and 'dist_coeffs' in stereo_calib_result:
        camera_matrix = stereo_calib_result['camera_matrix']
        dist_coeffs = stereo_calib_result['dist_coeffs']
        # Create maps once to avoid repeating this calculation
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1
        )
        undistort_maps = (map1, map2)
        log_message("Created undistortion maps")

    # Frame counter
    frame_count = 0
    
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            frame_header = f"\n================= FRAME {frame_count} =================="
            log_message(frame_header)
            log_message(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
            
            # Create a copy of the original frame
            original_frame = frame.copy()
            
            # Apply undistortion if calibration is available
            if stereo_calib_result is not None:
                if undistort_maps is not None:
                    # Use pre-computed maps
                    map1, map2 = undistort_maps
                    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                    log_message("Applied undistortion using maps")
                elif args.camera == 'master' and 'rectification_map_x_master' in stereo_calib_result:
                    # Use stereo rectification maps if available
                    frame = cv2.remap(
                        frame,
                        stereo_calib_result['rectification_map_x_master'],
                        stereo_calib_result['rectification_map_y_master'],
                        cv2.INTER_LINEAR
                    )
                    log_message("Applied undistortion using master rectification maps")
                elif args.camera == 'slave' and 'rectification_map_x_slave' in stereo_calib_result:
                    # Use stereo rectification maps if available
                    frame = cv2.remap(
                        frame,
                        stereo_calib_result['rectification_map_x_slave'],
                        stereo_calib_result['rectification_map_y_slave'],
                        cv2.INTER_LINEAR
                    )
                    log_message("Applied undistortion using slave rectification maps")
            
            # Measure inference time
            start_time = time.time()
            
            # Use simple prediction instead of tracking
            # model.predict returns a list of Results objects
            results = model(frame, verbose=False)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            log_message(f"Inference time: {inference_time*1000:.2f} ms")
            
            # Log detection results
            log_message("---------------- DETECTION DETAILS ----------------")
            if results and len(results) > 0:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    log_message(f"Total detections: {len(boxes)}")
                    for i, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Calculate width, height, center
                        box_width = x2 - x1
                        box_height = y2 - y1
                        center_x = x1 + box_width / 2
                        center_y = y1 + box_height / 2
                        
                        # Calculate ratios
                        width_ratio = box_width / frame.shape[1]
                        height_ratio = box_height / frame.shape[0]
                        center_x_ratio = center_x / frame.shape[1]
                        center_y_ratio = center_y / frame.shape[0]
                        
                        # Get class ID and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get class name
                        cls_name = results[0].names[cls_id]
                        
                        # Log in similar format to C++ detector
                        log_message(f"Detection #{i} ({cls_name}, conf={conf:.6f}):")
                        log_message(f"  Box coords (x,y,w,h): {x1:.1f},{y1:.1f},{box_width:.1f},{box_height:.1f}")
                        log_message(f"  Center point: ({center_x:.6f},{center_y:.6f})")
                        log_message(f"  Normalized center: ({center_x_ratio:.6f},{center_y_ratio:.6f})")
                        log_message(f"  Box size ratios (w,h): {width_ratio:.6f},{height_ratio:.6f}")
                        log_message(f"  Box area: {box_width * box_height:.1f} pixels ({(box_width * box_height * 100) / (frame.shape[1] * frame.shape[0]):.6f}% of image)")
                else:
                    log_message("No objects detected in this frame")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Write the frame to the output video
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLO Simple Prediction", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
            # Increment frame counter
            frame_count += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, writer, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    log_message(f"Video processing complete. Output saved to: {args.output}")
    
    # Close the debug log file
    if g_debug_log is not None:
        g_debug_log.close()

if __name__ == "__main__":
    main() 