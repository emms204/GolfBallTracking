#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnx_detector.h"
#include "camera_params.h"

void printUsage() {
    std::cout << "Test Detector Application" << std::endl;
    std::cout << "Usage: test_detector [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --camera=<id>          Camera device ID (default: 0)" << std::endl;
    std::cout << "  --video=<path>         Path to video file to process" << std::endl;
    std::cout << "  --model=<path>         Path to ONNX model file (default: best.onnx)" << std::endl;
    std::cout << "  --classes=<path>       Path to class names file (default: classes.txt)" << std::endl;
    std::cout << "  --params=<path>        Path to camera parameters file" << std::endl;
    std::cout << "  --conf=<threshold>     Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  --nms=<threshold>      NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    int camera_id = 0;
    std::string video_path = "";
    std::string model_path = "best.onnx";
    std::string classes_path = "classes.txt";
    std::string params_path = "";
    float conf_thresh = 0.25f;
    float nms_thresh = 0.45f;
    bool use_camera = true;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--camera=") == 0) {
            camera_id = std::stoi(arg.substr(9));
            use_camera = true;
        } else if (arg.find("--video=") == 0) {
            video_path = arg.substr(8);
            use_camera = false;
        } else if (arg.find("--model=") == 0) {
            model_path = arg.substr(8);
        } else if (arg.find("--classes=") == 0) {
            classes_path = arg.substr(10);
        } else if (arg.find("--params=") == 0) {
            params_path = arg.substr(9);
        } else if (arg.find("--conf=") == 0) {
            conf_thresh = std::stof(arg.substr(7));
        } else if (arg.find("--nms=") == 0) {
            nms_thresh = std::stof(arg.substr(6));
        } else if (arg == "--help") {
            printUsage();
            return 0;
        }
    }
    
    // Load camera parameters if provided
    common::CameraParams camera_params;
    bool has_camera_params = false;
    
    if (!params_path.empty()) {
        has_camera_params = camera_params.load(params_path);
        if (has_camera_params) {
            std::cout << "Camera parameters loaded from " << params_path << std::endl;
        } else {
            std::cerr << "Warning: Could not load camera parameters from " << params_path << std::endl;
        }
    }
    
    // TEST MODE: Process just one frame to isolate detection issues
    std::cout << "TEST MODE: Processing just one frame with detection" << std::endl;
    
    // Create and initialize detector
    detector::ONNXDetector detector;
    
    std::cout << "Loading model: " << model_path << std::endl;
    if (!detector.loadModel(model_path, classes_path)) {
        std::cerr << "Error: Could not load model" << std::endl;
        return -1;
    }
    
    std::cout << "Model loaded successfully" << std::endl;
    
    // Set detector parameters
    detector.setConfidenceThreshold(conf_thresh);
    detector.setNMSThreshold(nms_thresh);
    
    if (has_camera_params) {
        detector.setCameraParams(camera_params);
        std::cout << "Camera parameters set in detector" << std::endl;
    }
    
    // Open video source
    cv::VideoCapture cap;
    if (use_camera) {
        cap.open(camera_id);
        std::cout << "Opening camera device " << camera_id << std::endl;
    } else {
        cap.open(video_path);
        std::cout << "Opening video file: " << video_path << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source" << std::endl;
        return -1;
    }
    
    // Display video metadata
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    std::cout << "Video metadata:" << std::endl;
    std::cout << "  Width: " << width << std::endl;
    std::cout << "  Height: " << height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Frame count: " << frame_count << std::endl;
    
    // Read just one frame for testing
    cv::Mat frame;
    if (!cap.read(frame)) {
        std::cerr << "Error: Could not read frame from video source" << std::endl;
        return -1;
    }
    
    std::cout << "Read frame successfully, size: " << frame.cols << "x" << frame.rows << std::endl;
    std::cout << "Frame type: " << frame.type() << " (8UC3=" << CV_8UC3 << ")" << std::endl;
    
    try {
        // Try to detect objects in the frame
        std::cout << "Starting detection..." << std::endl;
        cv::Mat result = detector.detect(frame, has_camera_params);
        std::cout << "Detection completed successfully!" << std::endl;
        
        // Display the result
        cv::imshow("Detection Result", result);
        cv::waitKey(0);  // Wait for a key press
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during detection: " << e.what() << std::endl;
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
} 