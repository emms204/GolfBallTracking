#include "camera_params.h"
#include <fstream>
#include <opencv2/calib3d.hpp>

namespace common {

CameraParams::CameraParams() : maps_initialized_(false) {
    // Initialize with default values
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
}

CameraParams::CameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs, const cv::Size& image_size) 
    : maps_initialized_(false), image_size_(image_size) {
    // Copy camera parameters
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
    
    // Initialize undistortion maps if valid image size
    if (image_size.width > 0 && image_size.height > 0) {
        initUndistortMaps();
    }
}

bool CameraParams::load(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs["camera_matrix"] >> camera_matrix_;
    fs["dist_coeffs"] >> dist_coeffs_;
    
    // Try to load image size if available
    int width = 0, height = 0;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    if (width > 0 && height > 0) {
        image_size_ = cv::Size(width, height);
    }
    
    fs.release();
    
    maps_initialized_ = false;
    return true;
}

bool CameraParams::save(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs << "camera_matrix" << camera_matrix_;
    fs << "dist_coeffs" << dist_coeffs_;
    
    // Save image size if valid
    if (image_size_.width > 0 && image_size_.height > 0) {
        fs << "image_width" << image_size_.width;
        fs << "image_height" << image_size_.height;
    }
    
    fs.release();
    
    return true;
}

cv::Mat CameraParams::undistort(const cv::Mat& distorted) const {
    cv::Mat undistorted;
    
    if (!maps_initialized_) {
        // Initialize undistortion maps on first use
        // This is declared mutable in the implementation to allow const method
        cv::Mat& map1 = const_cast<cv::Mat&>(map1_);
        cv::Mat& map2 = const_cast<cv::Mat&>(map2_);
        bool& initialized = const_cast<bool&>(maps_initialized_);
        
        cv::initUndistortRectifyMap(
            camera_matrix_, 
            dist_coeffs_, 
            cv::Mat(), 
            camera_matrix_, 
            distorted.size(), 
            CV_32FC1, 
            map1, 
            map2
        );
        
        initialized = true;
    }
    
    cv::remap(distorted, undistorted, map1_, map2_, cv::INTER_LINEAR);
    return undistorted;
}

void CameraParams::setCameraMatrix(const cv::Mat& camera_matrix) {
    camera_matrix_ = camera_matrix.clone();
    maps_initialized_ = false;
}

void CameraParams::setDistCoeffs(const cv::Mat& dist_coeffs) {
    dist_coeffs_ = dist_coeffs.clone();
    maps_initialized_ = false;
}

cv::Mat CameraParams::getCameraMatrix() const {
    return camera_matrix_.clone();
}

cv::Mat CameraParams::getDistCoeffs() const {
    return dist_coeffs_.clone();
}

void CameraParams::initUndistortMaps() {
    if (image_size_.width <= 0 || image_size_.height <= 0) {
        return;  // Invalid image size
    }
    
    cv::initUndistortRectifyMap(
        camera_matrix_,
        dist_coeffs_,
        cv::Mat(),  // No rectification
        camera_matrix_,  // Use same camera matrix
        image_size_,
        CV_32FC1,
        map1_,
        map2_
    );
    
    maps_initialized_ = true;
}

} // namespace common 