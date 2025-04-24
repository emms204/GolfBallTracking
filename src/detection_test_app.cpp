#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <iomanip>
#include <fstream>
#include "simple_onnx_detector.h" // Include the header file
#include "logging.h" // Include logging header

// Calibration utilities
bool loadCalibration(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Size& imageSize) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            logMessage("Error: Could not open calibration file " + filename, true);
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
        
        logMessage("Calibration loaded successfully from: " + filename);
        logMessage("Reprojection error: " + std::to_string(reprojectionError) + " pixels");
        
        fs.release();
        return true;
    } catch (const cv::Exception& e) {
        logMessage("Error loading calibration parameters: " + std::string(e.what()), true);
        return false;
    }
}

// The getCurrentTimeString function is already defined in simple_onnx_detector.h
// Implement it here to avoid linking issues
std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void printHelp() {
    logMessage("Usage: detection_test_app [options]");
    logMessage("Options:");
    logMessage("  --help                Show this help message");
    logMessage("  --camera=<id>         Camera ID (default: 0)");
    logMessage("  --video=<path>        Path to video file (overrides camera input)");
    logMessage("  --calibration=<file>  Path to calibration file (yaml/xml)");
    logMessage("  --no-calibration      Disable calibration even if calibration file is provided");
    logMessage("  --model=<file>        Path to ONNX model file (default: best.onnx)");
    logMessage("  --classes=<file>      Path to classes file (default: classes.txt)");
    logMessage("  --conf=<threshold>    Confidence threshold (default: 0.25)");
    logMessage("  --nms=<threshold>     NMS threshold (default: 0.45)");
    logMessage("  --width=<pixels>      Camera width (default: 640)");
    logMessage("  --height=<pixels>     Camera height (default: 480)");
    logMessage("  --log=<path>          Path to log file (default: debug.log)");
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
    std::string log_file = "debug.log";
    
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
        } else if (arg.find("--log=") == 0) {
            log_file = arg.substr(6);
        }
    }
    
    // Initialize debug log file
    g_debugLogFile.open(log_file, std::ios::out);
    if (!g_debugLogFile.is_open()) {
        std::cerr << "Warning: Could not open " << log_file << " for writing" << std::endl;
    }
    
    logMessage("=== Detection Test Application ===");
    logMessage("Time: " + getCurrentTimeString());
    
    // Initialize video capture - either from camera or file
    cv::VideoCapture cap;
    if (!video_path.empty()) {
        cap.open(video_path);
        logMessage("Opening video file: " + video_path);
    } else {
        cap.open(camera_id);
        logMessage("Opening camera ID: " + std::to_string(camera_id));
    }
    
    if (!cap.isOpened()) {
        logMessage("Error: Could not open video source", true);
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
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
    logMessage("Frame resolution: " + std::to_string(width) + "x" + std::to_string(height));
    
    // Log parameters
    logMessage("Model: " + model_path);
    logMessage("Classes: " + classes_path);
    logMessage("Confidence threshold: " + std::to_string(conf_threshold));
    logMessage("NMS threshold: " + std::to_string(nms_threshold));
    
    // Load calibration parameters if provided and not explicitly disabled
    cv::Mat camera_matrix, dist_coeffs;
    cv::Size image_size(width, height);
    bool use_calibration = false;
    
    if (!calibration_file.empty() && !force_no_calibration) {
        logMessage("Loading calibration from: " + calibration_file);
        if (loadCalibration(calibration_file, camera_matrix, dist_coeffs, image_size)) {
            use_calibration = true;
        }
    }
    
    if (force_no_calibration) {
        logMessage("Calibration explicitly disabled via --no-calibration flag");
        use_calibration = false;
    }
    
    // Load ONNX detector (using the simple OnnxDetector class)
    try {
        OnnxDetector detector(model_path, classes_path, conf_threshold, nms_threshold);
        logMessage("Model loaded successfully");
        
        // Set calibration if it was loaded
        if (use_calibration) {
            if (detector.loadCalibration(calibration_file)) {
                logMessage("Calibration loaded for detector");
            } else {
                logMessage("Failed to apply calibration to detector", true);
                use_calibration = false;
            }
        }
        
        logMessage("Detector loaded successfully");
        logMessage("Starting detection...");
        logMessage("Press 'q' to quit, 'c' to toggle calibration, 'p' to pause/resume");
        
        cv::Mat frame;
        bool paused = false;
        int frame_count = 0;
        double total_fps = 0.0;
        
        // Main loop
        while (true) {
            // Start time for FPS calculation
            auto start = std::chrono::high_resolution_clock::now();
            
            // Capture frame
            if (!paused) {
                cap >> frame;
                if (frame.empty()) {
                    if (!video_path.empty()) {
                        logMessage("End of video file reached");
                    } else {
                        logMessage("Error: Could not read frame from camera", true);
                    }
                    break;
                }
                frame_count++;
            }
            
            // Print image dimensions
            std::string frameHeader = "\n================= FRAME " + std::to_string(frame_count) + " =================";
            logMessage(frameHeader);
            logMessage("Image dimensions: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows));
            
            if (use_calibration) {
                logMessage("Using calibration: YES");
            }
            
            // Process with the detector
            std::vector<float> confidences;
            std::vector<int> class_ids;
            std::vector<cv::Rect> boxes = detector.detect(frame, confidences, class_ids, use_calibration, false);
            
            // Print detailed information about detections
            logMessage("---------------- DETECTION DETAILS ----------------");
            logMessage("Total detections: " + std::to_string(boxes.size()));
            
            // Clone the frame to draw on
            cv::Mat processed_frame = frame.clone();
            
            // Draw detection results
            for (size_t i = 0; i < boxes.size(); ++i) {
                // Get detection data
                cv::Rect box = boxes[i];
                float confidence = confidences[i];
                int class_id = class_ids[i];
                std::string class_name = detector.getClassName(class_id);
                
                // Calculate ratios of bounding box to image dimensions
                float boxWidth = static_cast<float>(box.width);
                float boxHeight = static_cast<float>(box.height);
                float widthRatio = boxWidth / processed_frame.cols;
                float heightRatio = boxHeight / processed_frame.rows;
                float centerX = box.x + boxWidth / 2;
                float centerY = box.y + boxHeight / 2;
                float centerXRatio = centerX / processed_frame.cols;
                float centerYRatio = centerY / processed_frame.rows;
                
                logMessage("Detection #" + std::to_string(i) + " (" + class_name + ", conf=" + std::to_string(confidence) + "):");
                logMessage("  Box coords (x,y,w,h): " + std::to_string(box.x) + "," + std::to_string(box.y) + "," 
                          + std::to_string(box.width) + "," + std::to_string(box.height));
                logMessage("  Center point: (" + std::to_string(centerX) + "," + std::to_string(centerY) + ")");
                logMessage("  Normalized center: (" + std::to_string(centerXRatio) + "," + std::to_string(centerYRatio) + ")");
                logMessage("  Box size ratios (w,h): " + std::to_string(widthRatio) + "," + std::to_string(heightRatio));
                logMessage("  Box area: " + std::to_string(box.area()) + " pixels (" 
                          + std::to_string((box.area() * 100.0f) / (processed_frame.cols * processed_frame.rows)) + "% of image)");
                
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
            
            // Calculate and display FPS
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            double fps = 1000.0 / (duration > 0 ? duration : 1);
            total_fps += fps;
            
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
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
                logMessage("Exiting on user request");
                break;
            } else if (key == 'c') { // Toggle calibration
                use_calibration = !use_calibration;
                logMessage("Calibration " + std::string(use_calibration ? "enabled" : "disabled"));
            } else if (key == 'p') { // Toggle pause
                paused = !paused;
                logMessage("Playback " + std::string(paused ? "paused" : "resumed"));
            }
        }
        
        // Release resources
        cap.release();
        cv::destroyAllWindows();
        
        // Print statistics
        if (frame_count > 0) {
            double avg_fps = total_fps / frame_count;
            logMessage("\nProcessed " + std::to_string(frame_count) + " frames");
            logMessage("Average FPS: " + std::to_string(avg_fps));
        }
        
        logMessage("Detection complete!");
        
    } catch (const std::exception& e) {
        logMessage("Error: " + std::string(e.what()), true);
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
        return -1;
    }
    
    // Close the debug log file
    if (g_debugLogFile.is_open()) {
        g_debugLogFile.close();
    }
    
    return 0;
} 