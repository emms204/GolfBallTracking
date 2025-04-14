#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include "calibrator.h"

namespace calibration {

// Print help message
void printHelp() {
    std::cout << "Usage: calibrate_camera [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h                  Print this help message" << std::endl;
    std::cout << "  --input <source>            Camera index or directory path" << std::endl;
    std::cout << "  --pattern_size <w>x<h>      Size of chessboard (internal corners), e.g. 9x6" << std::endl;
    std::cout << "  --square_size <mm>          Size of square in millimeters" << std::endl;
    std::cout << "  --output <file>             Output file path (YAML)" << std::endl;
    std::cout << "  --min_images <num>          Minimum number of images for calibration (default: 10)" << std::endl;
    std::cout << "  --skip_frames <num>         Number of frames to skip between captures (default: 20)" << std::endl;
    std::cout << "Camera Mode Controls:" << std::endl;
    std::cout << "  Space                       Capture current frame" << std::endl;
    std::cout << "  C                           Capture current frame (alternate)" << std::endl;
    std::cout << "  A                           Toggle auto-capture" << std::endl;
    std::cout << "  ESC                         Exit application" << std::endl;
    std::cout << "  Enter                       Start calibration" << std::endl;
}

// Parse pattern size string (e.g., "9x6")
bool parsePatternSize(const std::string& str, cv::Size& size) {
    size_t xPos = str.find('x');
    if (xPos == std::string::npos) {
        return false;
    }
    
    try {
        size.width = std::stoi(str.substr(0, xPos));
        size.height = std::stoi(str.substr(xPos + 1));
        return (size.width > 0 && size.height > 0);
    } catch (...) {
        return false;
    }
}

// Draw progress bar
void drawProgressBar(cv::Mat& img, int progress, int total, bool success) {
    int barWidth = img.cols - 40;
    int barHeight = 30;
    cv::Point topLeft(20, img.rows - 50);
    cv::Point bottomRight(topLeft.x + barWidth, topLeft.y + barHeight);
    
    // Draw background
    cv::rectangle(img, topLeft, bottomRight, cv::Scalar(50, 50, 50), -1);
    
    // Draw progress
    int progressWidth = static_cast<int>((float)progress / total * barWidth);
    cv::Scalar color = success ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 128, 255);
    cv::rectangle(img, topLeft, cv::Point(topLeft.x + progressWidth, bottomRight.y), color, -1);
    
    // Draw text
    std::string text = std::to_string(progress) + " / " + std::to_string(total);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    cv::Point textOrg(topLeft.x + (barWidth - textSize.width) / 2, 
                    topLeft.y + (barHeight + textSize.height) / 2);
    cv::putText(img, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
}

// Process directory of images
bool processImageDirectory(const std::string& dirPath, Calibrator& calibrator, 
                         int minImages, const std::string& outputFile) {
    std::vector<std::string> imageFiles;
    
    // Check if directory exists
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        std::cerr << "Error: Directory " << dirPath << " does not exist or is not a directory." << std::endl;
        return false;
    }
    
    // Collect all image files
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (imageFiles.empty()) {
        std::cerr << "Error: No image files found in directory " << dirPath << std::endl;
        return false;
    }
    
    std::cout << "Found " << imageFiles.size() << " images in directory." << std::endl;
    
    // Process each image
    size_t successCount = 0;
    cv::Mat display;
    
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        cv::Mat img = cv::imread(imageFiles[i]);
        if (img.empty()) {
            std::cerr << "Warning: Could not read image " << imageFiles[i] << std::endl;
            continue;
        }
        
        bool found = calibrator.detectChessboard(img, true);
        
        if (found) {
            calibrator.addCurrentPoints();
            successCount++;
            std::cout << "Found chessboard in image " << i+1 << "/" << imageFiles.size() 
                      << " (" << successCount << " total)" << std::endl;
        } else {
            std::cout << "No chessboard found in image " << i+1 << "/" << imageFiles.size() << std::endl;
        }
        
        // Create display image
        if (img.cols > 1024 || img.rows > 768) {
            double scale = std::min(1024.0 / img.cols, 768.0 / img.rows);
            cv::resize(img, display, cv::Size(), scale, scale);
        } else {
            display = img.clone();
        }
        
        // Add progress information
        drawProgressBar(display, i+1, imageFiles.size(), found);
        std::string statusText = found ? "FOUND" : "NOT FOUND";
        cv::putText(display, "Chessboard: " + statusText, cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(display, "Images with corners: " + std::to_string(successCount), 
                    cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Camera Calibration", display);
        int key = cv::waitKey(100);
        if (key == 27) { // ESC
            break;
        }
    }
    
    // Check if we have enough images
    if (successCount < minImages) {
        std::cerr << "Warning: Only found " << successCount << " images with chessboard. "
                  << "Minimum required is " << minImages << "." << std::endl;
        std::cout << "Do you want to continue with calibration anyway? [y/N]: ";
        std::string response;
        std::getline(std::cin, response);
        if (response != "y" && response != "Y") {
            return false;
        }
    }
    
    // Perform calibration
    std::cout << "Performing calibration with " << successCount << " images..." << std::endl;
    if (!calibrator.calibrate()) {
        std::cerr << "Error: Calibration failed." << std::endl;
        return false;
    }
    
    // Show calibration results
    std::cout << calibrator.getQualityAssessment() << std::endl;
    
    // Save calibration parameters
    std::cout << "Saving calibration parameters to " << outputFile << "..." << std::endl;
    if (!calibrator.saveCalibration(outputFile)) {
        std::cerr << "Error: Failed to save calibration parameters." << std::endl;
        return false;
    }
    
    std::cout << "Calibration completed successfully!" << std::endl;
    return true;
}

// Process live camera feed
bool processCamera(int cameraIndex, Calibrator& calibrator, int minImages, int skipFrames,
                 const std::string& outputFile) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
        return false;
    }
    
    std::cout << "Camera opened successfully." << std::endl;
    std::cout << "Press 'Space' or 'c' to capture frame, 'a' to toggle auto-capture, 'Enter' to calibrate, ESC to exit." << std::endl;
    
    cv::Mat frame, displayFrame;
    bool autoCapture = false;
    int frameCount = 0;
    size_t successCount = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame from camera." << std::endl;
            break;
        }
        
        // Create a copy of the frame for display
        displayFrame = frame.clone();
        
        // Detect chessboard (but don't add to calibration yet)
        bool found = calibrator.detectChessboard(displayFrame, true);
        
        // Draw status and instructions
        std::string statusText = found ? "FOUND - Press SPACE to capture" : "NOT FOUND";
        cv::putText(displayFrame, "Chessboard: " + statusText, cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(displayFrame, "Images captured: " + std::to_string(successCount) + "/" + std::to_string(minImages), 
                    cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
        
        std::string modeText = autoCapture ? "AUTO-CAPTURE (press 'a' to disable)" : "MANUAL (press 'a' for auto)";
        cv::putText(displayFrame, modeText, cv::Point(20, displayFrame.rows - 70), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, autoCapture ? cv::Scalar(0, 255, 255) : cv::Scalar(200, 200, 200), 2);
        
        // Draw progress bar
        drawProgressBar(displayFrame, successCount, minImages, found);
        
        // Show frame
        cv::imshow("Camera Calibration", displayFrame);
        
        // Handle auto-capture
        bool shouldCapture = false;
        if (autoCapture && found && (frameCount % skipFrames == 0)) {
            shouldCapture = true;
        }
        
        // Handle key presses
        int key = cv::waitKey(30);
        if (key == 27) { // ESC key
            break;
        } else if (key == ' ' || key == 'c' || key == 'C' || shouldCapture) {
            if (found) {
                // Using the original frame (without drawn corners) for calibration
                calibrator.detectChessboard(frame, false);
                calibrator.addCurrentPoints();
                successCount++;
                std::cout << "Captured frame " << successCount << "/" << minImages << std::endl;
                
                // Flash screen to indicate capture
                cv::Mat flash = cv::Mat::ones(displayFrame.size(), displayFrame.type()) * 255;
                cv::addWeighted(displayFrame, 0.7, flash, 0.3, 0, displayFrame);
                cv::imshow("Camera Calibration", displayFrame);
                cv::waitKey(100);
                
                // Exit auto-capture if we have enough images
                if (successCount >= minImages) {
                    autoCapture = false;
                }
            }
        } else if (key == 'a' || key == 'A') {
            autoCapture = !autoCapture;
            std::cout << (autoCapture ? "Auto-capture enabled" : "Auto-capture disabled") << std::endl;
        } else if (key == 13) { // Enter key
            if (successCount < minImages) {
                std::cerr << "Warning: Only captured " << successCount << " frames. "
                          << "Minimum required is " << minImages << "." << std::endl;
                std::cout << "Do you want to continue with calibration anyway? [y/N]: ";
                std::string response;
                std::getline(std::cin, response);
                if (response != "y" && response != "Y") {
                    continue;
                }
            }
            break;
        }
        
        frameCount++;
    }
    
    // Release camera
    cap.release();
    
    // Check if we have any images
    if (successCount == 0) {
        std::cerr << "Error: No images captured for calibration." << std::endl;
        return false;
    }
    
    // Perform calibration
    std::cout << "Performing calibration with " << successCount << " images..." << std::endl;
    if (!calibrator.calibrate()) {
        std::cerr << "Error: Calibration failed." << std::endl;
        return false;
    }
    
    // Show calibration results
    std::cout << calibrator.getQualityAssessment() << std::endl;
    
    // Save calibration parameters
    std::cout << "Saving calibration parameters to " << outputFile << "..." << std::endl;
    if (!calibrator.saveCalibration(outputFile)) {
        std::cerr << "Error: Failed to save calibration parameters." << std::endl;
        return false;
    }
    
    std::cout << "Calibration completed successfully!" << std::endl;
    
    // Show undistorted view
    std::cout << "Press any key to see undistorted view, ESC to exit..." << std::endl;
    cv::waitKey(0);
    
    cap.open(cameraIndex);
    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            cv::Mat undistorted;
            calibrator.undistortWithMaps(frame, undistorted);
            
            // Show original and undistorted side by side
            cv::Mat sideBySide;
            cv::hconcat(frame, undistorted, sideBySide);
            
            cv::putText(sideBySide, "Original", cv::Point(20, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
            cv::putText(sideBySide, "Undistorted", cv::Point(frame.cols + 20, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("Before/After Undistortion", sideBySide);
            if (cv::waitKey(30) == 27) break;
        }
        cap.release();
    }
    
    return true;
}

} // namespace calibration

int main(int argc, char** argv) {
    // Default parameters
    std::string inputSource;
    cv::Size patternSize(9, 6);
    float squareSize = 20.0f;
    std::string outputFile = "camera_calibration.yaml";
    int minImages = 10;
    int skipFrames = 20;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            calibration::printHelp();
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            inputSource = argv[++i];
        } else if (arg == "--pattern_size" && i + 1 < argc) {
            if (!calibration::parsePatternSize(argv[++i], patternSize)) {
                std::cerr << "Error: Invalid pattern size format. Use WxH (e.g., 9x6)." << std::endl;
                return 1;
            }
        } else if (arg == "--square_size" && i + 1 < argc) {
            try {
                squareSize = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid square size." << std::endl;
                return 1;
            }
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--min_images" && i + 1 < argc) {
            try {
                minImages = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid min_images value." << std::endl;
                return 1;
            }
        } else if (arg == "--skip_frames" && i + 1 < argc) {
            try {
                skipFrames = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid skip_frames value." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            calibration::printHelp();
            return 1;
        }
    }
    
    // Validate parameters
    if (inputSource.empty()) {
        std::cerr << "Error: No input source specified. Use --input option." << std::endl;
        calibration::printHelp();
        return 1;
    }
    
    // Initialize calibrator
    calibration::Calibrator calibrator(patternSize, squareSize);
    
    // Process input based on type
    bool isNumeric = true;
    for (char c : inputSource) {
        if (!std::isdigit(c)) {
            isNumeric = false;
            break;
        }
    }
    
    bool success;
    if (isNumeric) {
        // Input is a camera index
        int cameraIndex = std::stoi(inputSource);
        std::cout << "Using camera " << cameraIndex << " as input source." << std::endl;
        std::cout << "Pattern size: " << patternSize.width << "x" << patternSize.height << std::endl;
        std::cout << "Square size: " << squareSize << " mm" << std::endl;
        success = calibration::processCamera(cameraIndex, calibrator, minImages, skipFrames, outputFile);
    } else {
        // Input is a directory path
        std::cout << "Using directory " << inputSource << " as input source." << std::endl;
        std::cout << "Pattern size: " << patternSize.width << "x" << patternSize.height << std::endl;
        std::cout << "Square size: " << squareSize << " mm" << std::endl;
        success = calibration::processImageDirectory(inputSource, calibrator, minImages, outputFile);
    }
    
    // Cleanup
    cv::destroyAllWindows();
    
    return success ? 0 : 1;
} 