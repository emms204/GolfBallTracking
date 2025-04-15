#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// ByteTrack includes
#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/Object.h"
#include "ByteTrack/Rect.h"
#include "ByteTrack/STrack.h"

// Include OnnxDetector class
#include "../src/onnx_detector.h"

// Function to convert ONNX detections to ByteTrack Objects
std::vector<byte_track::Object> convertDetectionsToObjects(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& confidences,
    const std::vector<int>& classIds)
{
    std::vector<byte_track::Object> objects;
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        const cv::Rect& box = boxes[i];
        float confidence = confidences[i];
        int classId = classIds[i];
        
        // Convert OpenCV Rect to ByteTrack Rect
        byte_track::Rect<float> rect(
            static_cast<float>(box.x),
            static_cast<float>(box.y),
            static_cast<float>(box.width),
            static_cast<float>(box.height)
        );
        
        // Create ByteTrack Object (rect, label, confidence)
        byte_track::Object object(rect, classId, confidence);
        objects.push_back(object);
    }
    
    return objects;
}

// Function to draw tracked objects
void drawTrackedObjects(cv::Mat& frame, 
                       const std::vector<std::shared_ptr<byte_track::STrack>>& tracked_objects,
                       const OnnxDetector& detector) 
{
    // Define colors for different classes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),     // Blue
        cv::Scalar(0, 255, 0),     // Green
        cv::Scalar(0, 0, 255),     // Red
        cv::Scalar(255, 255, 0),   // Cyan
        cv::Scalar(255, 0, 255),   // Magenta
        cv::Scalar(0, 255, 255),   // Yellow
    };
    
    for (const auto& track : tracked_objects) {
        // Get the track information
        int track_id = track->getTrackId();
        const byte_track::Rect<float>& rect = track->getRect();
        
        // Convert ByteTrack rect to OpenCV rect
        cv::Rect cv_rect(
            static_cast<int>(rect.x()),
            static_cast<int>(rect.y()),
            static_cast<int>(rect.width()),
            static_cast<int>(rect.height())
        );
        
        // Use track_id to select color (cycling through the color array)
        cv::Scalar color = colors[track_id % colors.size()];
        
        // Draw bounding box
        cv::rectangle(frame, cv_rect, color, 2);
        
        // Create label with track ID
        std::string label = "ID: " + std::to_string(track_id);
        
        // Draw label background
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        cv::rectangle(frame, 
                     cv::Point(cv_rect.x, cv_rect.y - labelSize.height - baseLine - 5),
                     cv::Point(cv_rect.x + labelSize.width, cv_rect.y),
                     color, cv::FILLED);
        
        // Draw label text
        cv::putText(frame, label, cv::Point(cv_rect.x, cv_rect.y - baseLine - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?  |      | Print this message}"
        "{video v         |      | Path to input video file}"
        "{camera c        |      | Use camera as input (specify device ID, e.g. 0)}"
        "{model m         |best.onnx| Path to ONNX model}"
        "{classes         |TrainResults1/classes.txt| Path to class names file (optional)}"
        "{conf            |0.25  | Confidence threshold}"
        "{nms             |0.45  | NMS threshold}"
        "{fps             |30    | Frames per second for tracking}"
        "{track_buffer    |30    | Track buffer for ByteTrack}"
        "{track_thresh    |0.5   | Track threshold for ByteTrack}"
        "{high_thresh     |0.6   | High detection threshold for ByteTrack}"
        "{match_thresh    |0.8   | Match threshold for ByteTrack}");
    
    if (parser.has("help") || (!parser.has("video") && !parser.has("camera"))) {
        parser.printMessage();
        return 1;
    }
    
    // Initialize video capture
    cv::VideoCapture cap;
    std::string inputSource;
    
    if (parser.has("video")) {
        std::string videoPath = parser.get<std::string>("video");
        cap.open(videoPath);
        inputSource = "Video: " + videoPath;
    } else if (parser.has("camera")) {
        int cameraId = parser.get<int>("camera");
        cap.open(cameraId);
        inputSource = "Camera: " + std::to_string(cameraId);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video source" << std::endl;
        return -1;
    }
    
    std::string modelPath = parser.get<std::string>("model");
    std::string classesPath = parser.get<std::string>("classes");
    float confThreshold = parser.get<float>("conf");
    float nmsThreshold = parser.get<float>("nms");
    
    // ByteTrack parameters
    int fps = parser.get<int>("fps");
    int trackBuffer = parser.get<int>("track_buffer");
    float trackThresh = parser.get<float>("track_thresh");
    float highThresh = parser.get<float>("high_thresh");
    float matchThresh = parser.get<float>("match_thresh");
    
    std::cout << "=== YOLO+ByteTrack Integration ===" << std::endl;
    std::cout << "Input: " << inputSource << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Classes: " << classesPath << std::endl;
    std::cout << "Detection threshold: " << confThreshold << std::endl;
    std::cout << "Tracking parameters: FPS=" << fps 
              << ", Track buffer=" << trackBuffer
              << ", Track threshold=" << trackThresh
              << ", High threshold=" << highThresh
              << ", Match threshold=" << matchThresh << std::endl;
    
    // Initialize the detector
    OnnxDetector detector(modelPath, classesPath, confThreshold, nmsThreshold);
    
    // Initialize the tracker
    byte_track::BYTETracker tracker(fps, trackBuffer, trackThresh, highThresh, matchThresh);
    
    // Create window for display
    const std::string windowName = "YOLO+ByteTrack";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::resizeWindow(windowName, width, height);
    
    // Process video frames
    cv::Mat frame;
    int frameCount = 0;
    double totalFps = 0.0;
    
    while (true) {
        auto frameStartTime = std::chrono::high_resolution_clock::now();
        
        bool success = cap.read(frame);
        if (!success) {
            std::cout << "\nEnd of video or camera disconnected" << std::endl;
            break;
        }
        
        frameCount++;
        
        // Step 1: Detect objects using ONNX YOLO model
        std::vector<float> confidences;
        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detector.detect(frame, confidences, classIds);
        
        // Step 2: Convert detections to ByteTrack objects
        std::vector<byte_track::Object> objects = convertDetectionsToObjects(boxes, confidences, classIds);
        
        // Step 3: Update tracker with new detections
        std::vector<std::shared_ptr<byte_track::STrack>> tracked_objects = tracker.update(objects);
        
        // Step 4: Draw tracking results
        drawTrackedObjects(frame, tracked_objects, detector);
        
        // Calculate and display FPS
        auto frameEndTime = std::chrono::high_resolution_clock::now();
        auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            frameEndTime - frameStartTime).count();
        
        double currentFps = frameDuration > 0 ? 1000.0 / frameDuration : 0.0;
        totalFps += currentFps;
        
        std::string fpsText = "FPS: " + cv::format("%.1f", currentFps);
        cv::putText(frame, fpsText, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        
        // Display count of tracked objects
        std::string countText = "Tracked Objects: " + std::to_string(tracked_objects.size());
        cv::putText(frame, countText, cv::Point(10, 70), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        
        // Display frame
        cv::imshow(windowName, frame);
        
        // Exit on ESC key
        int key = cv::waitKey(1);
        if (key == 27) { // ESC key
            std::cout << "\nExiting on user request" << std::endl;
            break;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    if (frameCount > 0) {
        double avgFps = totalFps / frameCount;
        std::cout << "Average FPS: " << avgFps << std::endl;
    }
    
    std::cout << "Processing complete!" << std::endl;
    
    return 0;
} 