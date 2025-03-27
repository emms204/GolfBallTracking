from collections import defaultdict
import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Track objects in a video using YOLO and save the output.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--model', type=str, default="best.pt", help='Path to the YOLO model weights')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load your trained model
    model = YOLO(args.model)

    # Open the video file
    video_path = args.video
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v' for MP4
    output_path = args.output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run tracking on the frame with ByteTrack, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Write the frame to the output video instead of displaying it
            out.write(annotated_frame)
            
            # Optional: Print progress
            print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}", end="\r")
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print(f"\nProcessing complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()