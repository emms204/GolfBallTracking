import cv2
import os
import argparse
import pickle
import numpy as np
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Track objects in a video using YOLO.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output')
    parser.add_argument('--model', type=str, default="best.pt", help='Path to the YOLO model weights')
    parser.add_argument('--calib_file', type=str, default="stereo_calibration/calibration_results/stereo_calibration_8mm.pkl", 
                        help='Path to the stereo calibration file')
    parser.add_argument('--camera', type=str, choices=['master', 'slave'], default='master',
                        help='Which camera the video is from (master or slave)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load the best model weights into the YOLO model
    model = YOLO(args.model)

    # Load calibration results
    try:
        with open(args.calib_file, 'rb') as f:
            stereo_calib_result = pickle.load(f)
        print(f"Loaded calibration from {args.calib_file}")
    except FileNotFoundError:
        print(f"Warning: Calibration file {args.calib_file} not found. Running without undistortion.")
        stereo_calib_result = None

    # Open the video file
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Apply undistortion and rectification if calibration is available
            if stereo_calib_result is not None:
                if args.camera == 'master':
                    frame = cv2.remap(
                        frame,
                        stereo_calib_result['rectification_map_x_master'],
                        stereo_calib_result['rectification_map_y_master'],
                        cv2.INTER_LINEAR
                    )
                else:  # slave camera
                    frame = cv2.remap(
                        frame,
                        stereo_calib_result['rectification_map_x_slave'],
                        stereo_calib_result['rectification_map_y_slave'],
                        cv2.INTER_LINEAR
                    )
            
            # Run tracking on the frame with ByteTrack, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            # Print original video dimensions for debugging
            print(f"Original video dimensions: Width={width}, Height={height}")
            
            # Print detection results
            if results and len(results) > 0:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    print(f"Detected {len(boxes)} objects in this frame")
                    for i, box in enumerate(boxes):
                        # Get box coordinates (convert to int for cleaner output)
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                        # Get track ID if available
                        track_id = int(box.id[0]) if box.id is not None else None
                        # Get class ID and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Get class name
                        cls_name = results[0].names[cls_id]
                        
                        print(f"  Object {i+1}: ID={track_id}, Class={cls_name} ({cls_id}), Conf={conf:.2f}")
                        print(f"    Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, width={x2-x1}, height={y2-y1}")
                else:
                    print("No objects detected in this frame")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Write the frame to the output video
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, writer, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to: {args.output}")

if __name__ == "__main__":
    main()