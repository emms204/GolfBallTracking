#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "preprocessor.h"

void printUsage() {
    std::cout << "Test Calibration Image" << std::endl;
    std::cout << "Usage: test_calibration_image <input_image> <calibration_file> <output_image>" << std::endl;
    std::cout << "Example: test_calibration_image image.jpg calibration.yaml undistorted.jpg" << std::endl;
}

bool loadCalibration(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open calibration file " << filename << std::endl;
            return false;
        }

        // Read calibration parameters
        fs["camera_matrix"] >> cameraMatrix;
        fs["distortion_coefficients"] >> distCoeffs;
        
        fs.release();
        
        if (cameraMatrix.empty() || distCoeffs.empty()) {
            std::cerr << "Error: Failed to read valid calibration parameters from " << filename << std::endl;
            return false;
        }
        
        std::cout << "Calibration loaded successfully from: " << filename << std::endl;
        std::cout << "Camera Matrix: " << std::endl << cameraMatrix << std::endl;
        std::cout << "Distortion Coefficients: " << std::endl << distCoeffs << std::endl;
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading calibration parameters: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printUsage();
        return 1;
    }
    
    std::string inputImage = argv[1];
    std::string calibrationFile = argv[2];
    std::string outputImage = argv[3];
    
    // Load the input image
    cv::Mat image = cv::imread(inputImage);
    if (image.empty()) {
        std::cerr << "Error: Could not read input image " << inputImage << std::endl;
        return 1;
    }
    
    // Load calibration parameters
    cv::Mat cameraMatrix, distCoeffs;
    if (!loadCalibration(calibrationFile, cameraMatrix, distCoeffs)) {
        return 1;
    }
    
    // Create preprocessor and load calibration
    detector::Preprocessor preprocessor(image.size());
    preprocessor.loadCalibration(cameraMatrix, distCoeffs);
    
    // Create an image with side-by-side comparison
    cv::Mat original = image.clone();
    
    // Process image with undistortion
    cv::Mat undistorted = preprocessor.process(image, true);
    if (undistorted.empty()) {
        std::cerr << "Error: Undistortion failed" << std::endl;
        return 1;
    }
    
    // Convert back to BGR for visualization
    if (undistorted.channels() == 3) {
        cv::cvtColor(undistorted, undistorted, cv::COLOR_RGB2BGR);
    }
    
    // Create debug images
    cv::Mat undistorted_debug, letterboxed_debug;
    preprocessor.createDebugImages(original, undistorted_debug, letterboxed_debug);
    
    // Combine original and undistorted images side by side
    cv::Mat combined;
    cv::hconcat(original, undistorted_debug, combined);
    
    // Add headers to indicate which is which
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    cv::Scalar color(0, 0, 255); // Red color
    
    cv::putText(combined, "Original", cv::Point(original.cols/2 - 100, 30), 
                fontFace, fontScale, color, thickness);
    cv::putText(combined, "Undistorted", cv::Point(original.cols + original.cols/2 - 100, 30), 
                fontFace, fontScale, color, thickness);
    
    // Add grid lines to better visualize distortion correction
    int gridSpacing = 50;
    cv::Scalar gridColor(0, 255, 0); // Green color
    
    for (int i = gridSpacing; i < combined.rows; i += gridSpacing) {
        cv::line(combined, cv::Point(0, i), cv::Point(combined.cols, i), gridColor, 1);
    }
    
    for (int i = gridSpacing; i < original.cols; i += gridSpacing) {
        // Vertical line in the original image
        cv::line(combined, cv::Point(i, 0), cv::Point(i, combined.rows), gridColor, 1);
        // Vertical line in the undistorted image
        cv::line(combined, cv::Point(i + original.cols, 0), cv::Point(i + original.cols, combined.rows), gridColor, 1);
    }
    
    // Save the combined image
    if (!cv::imwrite(outputImage, combined)) {
        std::cerr << "Error: Could not write output image " << outputImage << std::endl;
        return 1;
    }
    
    std::cout << "Calibration test completed. Output image saved to: " << outputImage << std::endl;
    
    return 0;
} 