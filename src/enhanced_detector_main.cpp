#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

// Include library headers
#include "onnx_detector.h"
#include "camera_params.h"

// Forward declarations
namespace detector {
    class ONNXDetector;
}

namespace common {
    class CameraParams;
}

void printUsage() {
    std::cout << "Enhanced Detector Application" << std::endl;
    std::cout << "Usage: enhanced_detector [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --camera=<id>          Camera device ID (default: 0)" << std::endl;
    std::cout << "  --video=<path>         Path to video file to process" << std::endl;
    std::cout << "  --model=<path>         Path to ONNX model file (default: best.onnx)" << std::endl;
    std::cout << "  --classes=<path>       Path to class names file (default: classes.txt)" << std::endl;
    std::cout << "  --params=<path>        Path to camera parameters file" << std::endl;
    std::cout << "  --conf=<threshold>     Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  --nms=<threshold>      NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  --undistort            Apply camera undistortion" << std::endl;
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
    bool apply_undistortion = false;
    
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
        } else if (arg == "--undistort") {
            apply_undistortion = true;
        } else if (arg == "--help") {
            printUsage();
            return 0;
        }
    }
    
    // Initialize detector
    detector::ONNXDetector detector;
    
    // Load model
    std::cout << "Loading model: " << model_path << std::endl;
    if (!detector.loadModel(model_path, classes_path)) {
        std::cerr << "Error: Failed to load model!" << std::endl;
        return -1;
    }
    
    // Set detector parameters
    detector.setConfidenceThreshold(conf_thresh);
    detector.setNMSThreshold(nms_thresh);
    
    // Load camera parameters if provided
    bool has_camera_params = false;
    if (!params_path.empty()) {
        std::cout << "Loading camera parameters: " << params_path << std::endl;
        
        // Load parameters from file
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            cv::Mat camera_matrix, dist_coeffs;
            int width, height;
            
            fs["camera_matrix"] >> camera_matrix;
            fs["distortion_coefficients"] >> dist_coeffs;
            fs["image_width"] >> width;
            fs["image_height"] >> height;
            
            fs.release();
            
            if (!camera_matrix.empty() && !dist_coeffs.empty() && width > 0 && height > 0) {
                common::CameraParams camera_params(camera_matrix, dist_coeffs, cv::Size(width, height));
                detector.setCameraParams(camera_params);
                has_camera_params = true;
                std::cout << "Camera parameters loaded successfully" << std::endl;
            } else {
                std::cerr << "Error: Invalid camera parameters in file" << std::endl;
            }
        } else {
            std::cerr << "Error: Could not open camera parameters file" << std::endl;
        }
    }
    
    std::cout << "Enhanced detector application" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Classes: " << classes_path << std::endl;
    std::cout << "Confidence threshold: " << conf_thresh << std::endl;
    std::cout << "NMS threshold: " << nms_thresh << std::endl;
    std::cout << "Undistortion: " << (apply_undistortion && has_camera_params ? "ON" : "OFF") << std::endl;
    
    // Open video source if specified
    cv::VideoCapture cap;
    if (use_camera) {
        cap.open(camera_id);
        std::cout << "Opening camera device " << camera_id << std::endl;
    } else if (!video_path.empty()) {
        cap.open(video_path);
        std::cout << "Opening video file: " << video_path << std::endl;
    }
    
    if (cap.isOpened()) {
        // Process frames
        std::cout << "Camera/video opened successfully" << std::endl;
        std::cout << "Press 'q' to quit, 'u' to toggle undistortion" << std::endl;
        
        // FPS calculation variables
        int frame_count = 0;
        float fps = 0.0f;
        auto fps_start_time = std::chrono::high_resolution_clock::now();
        
        while (true) {
            // Read frame
            cv::Mat frame;
            if (!cap.read(frame)) {
                if (!use_camera) {
                    // End of video file
                    std::cout << "End of video file" << std::endl;
                } else {
                    std::cerr << "Error: Could not read frame from camera" << std::endl;
                }
                break;
            }
            
            // Start timing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Apply detection with the detector
            cv::Mat result = detector.detect(frame, apply_undistortion && has_camera_params);
            
            // Calculate processing time
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            // Update FPS calculation
            frame_count++;
            if (frame_count >= 10) {
                auto fps_end_time = std::chrono::high_resolution_clock::now();
                auto fps_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fps_end_time - fps_start_time).count();
                fps = frame_count * 1000.0f / fps_duration;
                
                frame_count = 0;
                fps_start_time = std::chrono::high_resolution_clock::now();
            }
            
            // Display FPS and processing time
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + 
                                  " | Process time: " + std::to_string(duration) + "ms";
            cv::putText(result, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(0, 255, 0), 1);
            
            // Add undistortion status
            std::string undistort_text = "Undistortion: " + std::string(apply_undistortion && has_camera_params ? "ON" : "OFF");
            cv::putText(result, undistort_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(0, 255, 0), 1);
            
            // Display result
            cv::imshow("Enhanced Detector", result);
            
            // Check for key press
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 'q' or ESC
                break;
            } else if (key == 'u') {  // Toggle undistortion
                apply_undistortion = !apply_undistortion;
                std::cout << "Undistortion: " << (apply_undistortion && has_camera_params ? "ON" : "OFF") << std::endl;
            }
        }
        
        // Release resources
        cap.release();
        cv::destroyAllWindows();
    } else if (!video_path.empty() || use_camera) {
        std::cerr << "Error: Could not open video source" << std::endl;
        return -1;
    }
    
    return 0;
}