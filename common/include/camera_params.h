#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace common {

/**
 * @brief Class for managing camera calibration parameters
 */
class CameraParams {
public:
    /**
     * @brief Default constructor
     */
    CameraParams();
    
    /**
     * @brief Constructor with camera parameters
     * 
     * @param camera_matrix Camera matrix (3x3)
     * @param dist_coeffs Distortion coefficients
     * @param image_size Image size
     */
    CameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs, const cv::Size& image_size);
    
    /**
     * @brief Load camera parameters from a file
     * 
     * @param filename Path to the calibration file
     * @return bool True if loaded successfully
     */
    bool load(const std::string& filename);
    
    /**
     * @brief Save camera parameters to a file
     * 
     * @param filename Path to the calibration file
     * @return bool True if saved successfully
     */
    bool save(const std::string& filename) const;
    
    /**
     * @brief Undistort an image using the camera parameters
     * 
     * @param distorted Distorted input image
     * @return cv::Mat Undistorted output image
     */
    cv::Mat undistort(const cv::Mat& distorted) const;
    
    /**
     * @brief Set the camera matrix
     * 
     * @param camera_matrix Camera matrix (3x3)
     */
    void setCameraMatrix(const cv::Mat& camera_matrix);
    
    /**
     * @brief Set the distortion coefficients
     * 
     * @param dist_coeffs Distortion coefficients
     */
    void setDistCoeffs(const cv::Mat& dist_coeffs);
    
    /**
     * @brief Get the camera matrix
     * 
     * @return cv::Mat Camera matrix
     */
    cv::Mat getCameraMatrix() const;
    
    /**
     * @brief Get the distortion coefficients
     * 
     * @return cv::Mat Distortion coefficients
     */
    cv::Mat getDistCoeffs() const;

private:
    cv::Mat camera_matrix_;    // 3x3 camera matrix
    cv::Mat dist_coeffs_;      // Distortion coefficients
    cv::Mat map1_;             // Undistortion map 1
    cv::Mat map2_;             // Undistortion map 2
    cv::Size image_size_;      // Image size
    bool maps_initialized_;    // Flag indicating if maps are initialized
    
    /**
     * @brief Initialize undistortion maps
     */
    void initUndistortMaps();
};

} // namespace common 