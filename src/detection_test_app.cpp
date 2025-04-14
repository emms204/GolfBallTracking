#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include "onnx_detector.h"
#include "preprocessor.h"
#include "camera_params.h"

// Calibration utilities
bool loadCalibration(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Size& imageSize) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open calibration file " << filename << std::endl;
            return false;
        }

        // Read calibration parameters
        int width, height;
        fs["image_width"] >> width;
        fs["image_height"] >> height;
        imageSize = cv::Size(width, height);
        
        fs["camera_matrix"] >> cameraMatrix;
        fs["distortion_coefficients"] >> distCoeffs;
        
        double reprojectionError;
        fs["avg_reprojection_error"] >> reprojectionError;
        
        std::cout << "Calibration loaded successfully from: " << filename << std::endl;
        std::cout << "Reprojection error: " << reprojectionError << " pixels" << std::endl;
        
        fs.release();
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading calibration parameters: " << e.what() << std::endl;
        return false;
    }
}

std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void printHelp() {
    std::cout << "Usage: detection_test_app [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                Show this help message" << std::endl;
    std::cout << "  --camera <id>         Camera ID (default: 0)" << std::endl;
    std::cout << "  --video <path>        Path to video file (overrides camera input)" << std::endl;
    std::cout << "  --calibration <file>  Path to calibration file (yaml/xml)" << std::endl;
    std::cout << "  --no-calibration      Disable calibration even if calibration file is provided" << std::endl;
    std::cout << "  --model <file>        Path to ONNX model file (default: best.onnx)" << std::endl;
    std::cout << "  --classes <file>      Path to classes file (default: classes.txt)" << std::endl;
    std::cout << "  --conf <threshold>    Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  --nms <threshold>     NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  --width <pixels>      Camera width (default: 640)" << std::endl;
    std::cout << "  --height <pixels>     Camera height (default: 480)" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    int cameraId = 0;
    std::string videoPath = "";
    std::string calibrationFile = "";
    std::string modelPath = "best.onnx";
    std::string classesPath = "classes.txt";
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    int width = 640;
    int height = 480;
    bool forceNoCalibration = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printHelp();
            return 0;
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraId = std::stoi(argv[++i]);
        } else if (arg == "--video" && i + 1 < argc) {
            videoPath = argv[++i];
        } else if (arg == "--calibration" && i + 1 < argc) {
            calibrationFile = argv[++i];
        } else if (arg == "--no-calibration") {
            forceNoCalibration = true;
        } else if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--classes" && i + 1 < argc) {
            classesPath = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            confThreshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            nmsThreshold = std::stof(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        }
    }
    
    // Initialize video capture - either from camera or file
    cv::VideoCapture cap;
    if (!videoPath.empty()) {
        cap.open(videoPath);
        std::cout << "Opening video file: " << videoPath << std::endl;
    } else {
        cap.open(cameraId);
        std::cout << "Opening camera ID: " << cameraId << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source" << std::endl;
        return -1;
    }
    
    // Set camera resolution (only applicable for camera, not video file)
    if (videoPath.empty()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    
    // Get frame dimensions
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame resolution: " << width << "x" << height << std::endl;
    
    // Load calibration parameters if provided and not explicitly disabled
    cv::Mat cameraMatrix, distCoeffs;
    cv::Size imageSize(width, height);
    bool useCalibration = false;
    
    // Create camera parameters object for detector
    common::CameraParams cameraParams;
    
    if (!calibrationFile.empty() && !forceNoCalibration) {
        if (loadCalibration(calibrationFile, cameraMatrix, distCoeffs, imageSize)) {
            // Set the camera parameters
            cameraParams = common::CameraParams(cameraMatrix, distCoeffs, imageSize);
            useCalibration = true;
        }
    }
    
    if (forceNoCalibration) {
        std::cout << "Calibration explicitly disabled via --no-calibration flag" << std::endl;
        useCalibration = false;
    }
    
    // Load ONNX detector (using the enhanced detector::ONNXDetector class)
    detector::ONNXDetector detector;
    if (!detector.loadModel(modelPath, classesPath)) {
        std::cerr << "Failed to load ONNX model" << std::endl;
        return -1;
    }
    
    // Set detector parameters
    detector.setConfidenceThreshold(confThreshold);
    detector.setNMSThreshold(nmsThreshold);
    
    // Set camera parameters if calibration was loaded
    if (useCalibration) {
        detector.setCameraParams(cameraParams);
    }
    
    std::cout << "Detector loaded successfully" << std::endl;
    std::cout << "Starting detection..." << std::endl;
    std::cout << "Press 'q' to quit, 'c' to toggle calibration, 'p' to pause/resume" << std::endl;
    
    cv::Mat frame, processedFrame;
    bool paused = false;
    
    // Main loop
    while (true) {
        // Capture frame
        if (!paused) {
            cap >> frame;
            if (frame.empty()) {
                if (!videoPath.empty()) {
                    std::cout << "End of video file reached" << std::endl;
                } else {
                    std::cerr << "Error: Could not read frame from camera" << std::endl;
                }
                break;
            }
        }
        
        // Perform detection
        auto start = std::chrono::high_resolution_clock::now();
        processedFrame = detector.detect(frame, useCalibration);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Display FPS
        std::string fpsText = "FPS: " + std::to_string(1000 / (duration > 0 ? duration : 1));
        cv::putText(processedFrame, fpsText, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Display calibration status
        std::string calibText = useCalibration ? "Calibration: ON" : "Calibration: OFF";
        cv::putText(processedFrame, calibText, cv::Point(10, height - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        // Display paused status if applicable
        if (paused) {
            cv::putText(processedFrame, "PAUSED", cv::Point(width - 120, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
        
        // Show the frame
        cv::imshow("Detection with Calibration", processedFrame);
        
        // Handle key press
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 'c') { // Toggle calibration
            useCalibration = !useCalibration;
        } else if (key == 'p') { // Toggle pause
            paused = !paused;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
} 