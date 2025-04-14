import cv2
import os
import argparse
import pickle
import numpy as np
import time
from datetime import datetime
import onnxruntime as ort

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
    parser = argparse.ArgumentParser(description='Process objects in a video using ONNX Runtime.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output')
    parser.add_argument('--model', type=str, default="best.onnx", help='Path to the ONNX model file')
    parser.add_argument('--calib_file', type=str, default="stereo_calibration/calibration_results/stereo_calibration_8mm.pkl", 
                        help='Path to the stereo calibration file')
    parser.add_argument('--camera', type=str, choices=['master', 'slave'], default='master',
                        help='Which camera the video is from (master or slave)')
    parser.add_argument('--log_file', type=str, default="python_ort_test.log", 
                        help='Path to save the debug log file')
    return parser.parse_args()

# Load class names (simplified)
def load_class_names():
    # Our custom classes for this specific model
    return {0: 'ball', 1: 'club'}

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize debug log file
    global g_debug_log
    g_debug_log = open(args.log_file, "w")
    if g_debug_log is None:
        print(f"Warning: Could not open {args.log_file} for writing")
    
    # Log start information
    log_message("=== Python ONNX Runtime Object Detector ===")
    log_message(f"Time: {get_current_time_string()}")
    log_message(f"Video: {args.video}")
    log_message(f"Model: {args.model}")
    log_message(f"Camera: {args.camera}")
    log_message(f"Calibration: {args.calib_file}")
    
    # Get ONNX Runtime providers information
    providers = ort.get_available_providers()
    log_message(f"ONNX Runtime available providers: {providers}")
    
    # Try to use GPU if available
    if 'CUDAExecutionProvider' in providers:
        sess_options = ort.SessionOptions()
        session = ort.InferenceSession(args.model, sess_options=sess_options, providers=['CUDAExecutionProvider'])
        log_message("Using CUDA Execution Provider")
    else:
        session = ort.InferenceSession(args.model)
        log_message("Using CPU Execution Provider")
    
    # Get model input and output details
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    
    # Get model outputs
    model_outputs = session.get_outputs()
    output_names = [output.name for output in model_outputs]
    
    log_message(f"Model input name: {input_name}, shape: {input_shape}")
    log_message(f"Model output names: {output_names}")
    
    # Load class names
    class_names = load_class_names()
    log_message(f"Loaded {len(class_names)} class names")

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
            
            # Prepare input for the ONNX model - using minimal preprocessing
            img = frame.copy()
            
            # Get required input dimensions from the model
            if len(input_shape) == 4:
                if input_shape[1] == 3:  # NCHW format
                    # Check if dimensions are dynamic (strings)
                    if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
                        input_height, input_width = 640, 640  # Default values for dynamic dimensions
                    else:
                        input_height, input_width = input_shape[2], input_shape[3]
                else:  # NHWC format
                    # Check if dimensions are dynamic (strings)
                    if isinstance(input_shape[1], str) or isinstance(input_shape[2], str):
                        input_height, input_width = 640, 640  # Default values for dynamic dimensions
                    else:
                        input_height, input_width = input_shape[1], input_shape[2]
            else:
                # Default to common YOLO dimensions if shape is not 4D
                input_height, input_width = 640, 640
                
            log_message(f"Resizing image to model input dimensions: {input_width}x{input_height}")
            
            # Resize the image to the model's input dimensions
            img_resized = cv2.resize(img, (input_width, input_height))
            
            # Basic preprocessing - convert to RGB and normalize
            img_input = img_resized[:, :, ::-1]  # BGR to RGB
            img_input = img_input.astype(np.float32) / 255.0  # Normalize to 0-1
            
            # Reshape to model input shape if needed
            if len(input_shape) == 4:  # NCHW format
                # Check if we need to transpose channels
                if input_shape[1] == 3:  # NCHW format
                    img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
                    
                # Add batch dimension
                img_input = np.expand_dims(img_input, 0)  # Add batch dimension
            
            # Measure inference time
            start_time = time.time()
            
            # Run inference
            outputs = session.run(output_names, {input_name: img_input})
            
            # Calculate inference time
            inference_time = time.time() - start_time
            log_message(f"Inference time: {inference_time*1000:.2f} ms")
            
            # Log output information
            log_message(f"Output shapes: {[output.shape for output in outputs]}")
            
            # Get detections from the first output (assuming YOLO format)
            # For simplicity, we're extracting boxes directly without NMS
            detections = outputs[0]
            
            log_message("---------------- DETECTION DETAILS ----------------")
            
            # Process detections
            if len(detections) > 0:
                log_message(f"Raw detection shape: {detections.shape}")
                
                # Extract detections
                if len(detections.shape) == 3:  # If shape is [batch, num_dets, values]
                    dets = detections[0]  # Get detections for the first batch
                else:
                    dets = detections
                
                # Count valid detections (confidence > threshold)
                valid_detections = 0
                confidence_threshold = 0.5
                
                for i in range(dets.shape[0]):
                    # Get confidence
                    confidence = dets[i][4]
                    
                    if confidence > confidence_threshold:
                        valid_detections += 1
                
                log_message(f"Total detections: {valid_detections}")
                
                # Process and log valid detections
                detection_count = 0
                for i in range(dets.shape[0]):
                    # Assuming format: [x1, y1, x2, y2, confidence, class]
                    det = dets[i]
                    
                    # Check confidence threshold
                    confidence = det[4]
                    if confidence > confidence_threshold:
                        # Extract coordinates
                        x1, y1, x2, y2 = det[0:4]
                        
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
                        
                        # Get class information
                        class_id = int(det[5]) if len(det) > 5 else 0
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        
                        # Log detection details in the same format as track_simple_predict.py
                        log_message(f"Detection #{detection_count} ({class_name}, conf={confidence:.6f}):")
                        log_message(f"  Box coords (x,y,w,h): {x1:.1f},{y1:.1f},{box_width:.1f},{box_height:.1f}")
                        log_message(f"  Center point: ({center_x:.6f},{center_y:.6f})")
                        log_message(f"  Normalized center: ({center_x_ratio:.6f},{center_y_ratio:.6f})")
                        log_message(f"  Box size ratios (w,h): {width_ratio:.6f},{height_ratio:.6f}")
                        log_message(f"  Box area: {box_width * box_height:.1f} pixels ({(box_width * box_height * 100) / (frame.shape[1] * frame.shape[0]):.6f}% of image)")
                        
                        detection_count += 1
            else:
                log_message("No detections in output")
            
            # Create a copy of the original frame for drawing on
            result_frame = original_frame.copy()
            
            # Draw bounding boxes for detections with confidence above threshold
            confidence_threshold = 0.5
            
            # Process and draw detections on the result frame
            if len(detections) > 0 and len(detections.shape) >= 2:
                detection_count = 0
                
                # Extract detections
                if len(detections.shape) == 3:  # If shape is [batch, num_dets, values]
                    dets = detections[0]  # Get detections for the first batch
                else:
                    dets = detections
                
                # Iterate through all detections
                for i in range(dets.shape[0]):
                    # Assuming format: [x1, y1, x2, y2, confidence, class]
                    det = dets[i]
                    
                    # Check confidence threshold
                    confidence = det[4]
                    if confidence > confidence_threshold:
                        detection_count += 1
                        
                        # Extract coordinates and scale to original image dimensions
                        x1, y1, x2, y2 = det[0:4]
                        
                        # Scale bounding box from model input size to original image size
                        x_scale = width / input_width
                        y_scale = height / input_height
                        
                        x1 = int(x1 * x_scale)
                        y1 = int(y1 * y_scale)
                        x2 = int(x2 * x_scale)
                        y2 = int(y2 * y_scale)
                        
                        # Get class index and name
                        class_id = int(det[5]) if len(det) > 5 else 0
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        
                        # Create label with class name and confidence
                        label = f"{class_name}: {confidence:.2f}"
                        
                        # Draw rectangle
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add text with background for better visibility
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                log_message(f"Drew {detection_count} detections with confidence > {confidence_threshold}")
            
            cv2.putText(result_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write the frame to the output video
            out.write(result_frame)

            # Display the result frame
            cv2.imshow("ONNX Runtime Inference", result_frame)

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