#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <iomanip>
#include "simple_onnx_detector.h" // Include the header file

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

// The getCurrentTimeString function is already defined in simple_onnx_detector.h

void printHelp() {
    std::cout << "Usage: detection_test_app [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                Show this help message" << std::endl;
    std::cout << "  --camera=<id>         Camera ID (default: 0)" << std::endl;
    std::cout << "  --video=<path>        Path to video file (overrides camera input)" << std::endl;
    std::cout << "  --calibration=<file>  Path to calibration file (yaml/xml)" << std::endl;
    std::cout << "  --no-calibration      Disable calibration even if calibration file is provided" << std::endl;
    std::cout << "  --model=<file>        Path to ONNX model file (default: best.onnx)" << std::endl;
    std::cout << "  --classes=<file>      Path to classes file (default: classes.txt)" << std::endl;
    std::cout << "  --conf=<threshold>    Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  --nms=<threshold>     NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  --width=<pixels>      Camera width (default: 640)" << std::endl;
    std::cout << "  --height=<pixels>     Camera height (default: 480)" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    int camera_id = 0;
    std::string video_path = "";
    std::string calibration_file = "";
    std::string model_path = "best.onnx";
    std::string classes_path = "classes.txt";
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int width = 640;
    int height = 480;
    bool force_no_calibration = false;
    
    // Parse command-line arguments using equals format
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printHelp();
            return 0;
        } else if (arg == "--no-calibration") {
            force_no_calibration = true;
        } else if (arg.find("--camera=") == 0) {
            camera_id = std::stoi(arg.substr(9));
        } else if (arg.find("--video=") == 0) {
            video_path = arg.substr(8);
        } else if (arg.find("--calibration=") == 0) {
            calibration_file = arg.substr(14);
        } else if (arg.find("--model=") == 0) {
            model_path = arg.substr(8);
        } else if (arg.find("--classes=") == 0) {
            classes_path = arg.substr(10);
        } else if (arg.find("--conf=") == 0) {
            conf_threshold = std::stof(arg.substr(7));
        } else if (arg.find("--nms=") == 0) {
            nms_threshold = std::stof(arg.substr(6));
        } else if (arg.find("--width=") == 0) {
            width = std::stoi(arg.substr(8));
        } else if (arg.find("--height=") == 0) {
            height = std::stoi(arg.substr(9));
        }
    }
    
    // Initialize video capture - either from camera or file
    cv::VideoCapture cap;
    if (!video_path.empty()) {
        cap.open(video_path);
        std::cout << "Opening video file: " << video_path << std::endl;
    } else {
        cap.open(camera_id);
        std::cout << "Opening camera ID: " << camera_id << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source" << std::endl;
        return -1;
    }
    
    // Set camera resolution (only applicable for camera, not video file)
    if (video_path.empty()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    
    // Get frame dimensions
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame resolution: " << width << "x" << height << std::endl;
    
    // Load calibration parameters if provided and not explicitly disabled
    cv::Mat camera_matrix, dist_coeffs;
    cv::Size image_size(width, height);
    bool use_calibration = false;
    
    if (!calibration_file.empty() && !force_no_calibration) {
        if (loadCalibration(calibration_file, camera_matrix, dist_coeffs, image_size)) {
            use_calibration = true;
        }
    }
    
    if (force_no_calibration) {
        std::cout << "Calibration explicitly disabled via --no-calibration flag" << std::endl;
        use_calibration = false;
    }
    
    // Load ONNX detector (using the simple OnnxDetector class)
    OnnxDetector detector(model_path, classes_path, conf_threshold, nms_threshold);
    std::cout << "Model loaded successfully" << std::endl;
    
    // Set calibration if it was loaded
    if (use_calibration) {
        detector.loadCalibration(calibration_file);
        std::cout << "Calibration loaded for detector" << std::endl;
    }
    
    std::cout << "Detector loaded successfully" << std::endl;
    std::cout << "Starting detection..." << std::endl;
    std::cout << "Press 'q' to quit, 'c' to toggle calibration, 'p' to pause/resume" << std::endl;
    
    cv::Mat frame;
    bool paused = false;
    
    // Main loop
    while (true) {
        // Capture frame
        if (!paused) {
            cap >> frame;
            if (frame.empty()) {
                if (!video_path.empty()) {
                    std::cout << "End of video file reached" << std::endl;
                } else {
                    std::cerr << "Error: Could not read frame from camera" << std::endl;
                }
                break;
            }
        }
        
        // Perform detection
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process with the detector
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<cv::Rect> boxes = detector.detect(frame, confidences, class_ids, use_calibration);
        
        // Clone the frame to draw on
        cv::Mat processed_frame = frame.clone();
        
        // Draw detection results
        for (size_t i = 0; i < boxes.size(); ++i) {
            // Get detection data
            cv::Rect box = boxes[i];
            float confidence = confidences[i];
            int class_id = class_ids[i];
            std::string class_name = detector.getClassName(class_id);
            
            // Define color (can be class-specific)
            cv::Scalar color(0, 255, 0); // Green for all classes
            
            // Draw bounding box
            cv::rectangle(processed_frame, box, color, 2);
            
            // Create label text
            std::string label = class_name + ": " + std::to_string(confidence).substr(0, 4);
            
            // Draw label background
            int baseline;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(processed_frame, 
                        cv::Point(box.x, box.y - label_size.height - 5),
                        cv::Point(box.x + label_size.width, box.y),
                        color, cv::FILLED);
            
            // Draw label text
            cv::putText(processed_frame, label, cv::Point(box.x, box.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Display FPS
        std::string fps_text = "FPS: " + std::to_string(1000 / (duration > 0 ? duration : 1));
        cv::putText(processed_frame, fps_text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Display calibration status
        std::string calib_text = use_calibration ? "Calibration: ON" : "Calibration: OFF";
        cv::putText(processed_frame, calib_text, cv::Point(10, height - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        // Display paused status if applicable
        if (paused) {
            cv::putText(processed_frame, "PAUSED", cv::Point(width - 120, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
        
        // Show the frame
        cv::imshow("Detection with Calibration", processed_frame);
        
        // Handle key press
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 'c') { // Toggle calibration
            use_calibration = !use_calibration;
        } else if (key == 'p') { // Toggle pause
            paused = !paused;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
} 