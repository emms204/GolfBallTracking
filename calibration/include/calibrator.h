#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace calibration {

/**
 * @brief Camera calibration component that handles chessboard detection,
 * calibration and parameter storage.
 */
class Calibrator {
public:
    /**
     * @brief Constructor
     * @param boardSize The size of the chessboard (internal corners)
     * @param squareSize The size of a square in millimeters
     */
    Calibrator(cv::Size boardSize, float squareSize);

    /**
     * @brief Detect chessboard corners in an image
     * @param image Input image
     * @param drawCorners Whether to draw the detected corners on the image
     * @return True if corners were detected, false otherwise
     */
    bool detectChessboard(cv::Mat& image, bool drawCorners = false);

    /**
     * @brief Add current detected corners to calibration points
     * @return Number of successful corner detections so far
     */
    size_t addCurrentPoints();

    /**
     * @brief Perform camera calibration using collected points
     * @return True if calibration was successful, false otherwise
     */
    bool calibrate();

    /**
     * @brief Save calibration parameters to a file
     * @param filename Output filename (YAML format)
     * @return True if parameters were saved successfully, false otherwise
     */
    bool saveCalibration(const std::string& filename);

    /**
     * @brief Load calibration parameters from a file
     * @param filename Input filename (YAML format)
     * @return True if parameters were loaded successfully, false otherwise
     */
    bool loadCalibration(const std::string& filename);

    /**
     * @brief Get the reprojection error of the calibration
     * @return Reprojection error in pixels
     */
    double getReprojectionError() const;

    /**
     * @brief Get a detailed assessment of calibration quality
     * @return String with quality assessment details
     */
    std::string getQualityAssessment() const;

    /**
     * @brief Get the number of images with detected corners
     * @return Number of successful chessboard detections
     */
    size_t getDetectionCount() const;

    /**
     * @brief Clear all collected calibration points
     */
    void clearPoints();

    /**
     * @brief Set Filtered Points
     * @param objectPoints Object points
     * @param imagePoints Image points
     */
    void setFilteredPoints(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                           const std::vector<std::vector<cv::Point2f>>& imagePoints);
    
    /**
     * @brief Get the camera matrix
     * @return Camera intrinsic matrix
     */
    cv::Mat getCameraMatrix() const;

    /**
     * @brief Get the distortion coefficients
     * @return Distortion coefficients
     */
    cv::Mat getDistortionCoefficients() const;

    /**
     * @brief Apply undistortion to an image
     * @param image Input/output image
     * @return True if undistortion was applied, false otherwise
     */
    bool undistortImage(cv::Mat& image);

    /**
     * @brief Create undistortion maps for efficient undistortion
     * @param imageSize Size of the images to be undistorted
     */
    void initUndistortMaps(const cv::Size& imageSize);

    /**
     * @brief Apply undistortion using precomputed maps
     * @param input Input image
     * @param output Output image
     * @return True if undistortion was applied, false otherwise
     */
    bool undistortWithMaps(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief Get the object points used for calibration
     * @return Vector of object point vectors
     */
    const std::vector<std::vector<cv::Point3f>>& getObjectPoints() const;

    /**
     * @brief Get the image points used for calibration
     * @return Vector of image point vectors
     */
    const std::vector<std::vector<cv::Point2f>>& getImagePoints() const;

    /**
     * @brief Get the board size (chessboard internal corners)
     * @return Board size
     */
    cv::Size getBoardSize() const;

    /**
     * @brief Get the square size in mm
     * @return Square size
     */
    float getSquareSize() const;

    /**
     * @brief Get the image size used for calibration
     * @return Image size
     */
    cv::Size getImageSize() const;

private:
    /**
     * @brief Calculate reprojection error for calibration quality assessment
     */
    void calculateReprojectionError();

    /**
     * @brief Assess calibration quality based on various metrics
     */
    void assessCalibrationQuality();

    // Chessboard parameters
    cv::Size mBoardSize;
    float mSquareSize;
    
    // Calibration results
    cv::Mat mCameraMatrix;
    cv::Mat mDistCoeffs;
    cv::Size mImageSize;
    double mReprojectionError;
    std::string mQualityAssessment;
    bool mCalibrated;
    
    // Collected points
    std::vector<std::vector<cv::Point3f>> mObjectPoints;
    std::vector<std::vector<cv::Point2f>> mImagePoints;
    std::vector<float> mPerViewErrors;
    
    // Last detected corners
    std::vector<cv::Point2f> mCurrentCorners;
    bool mCornersDetected;
    
    // Undistortion maps
    cv::Mat mMapX, mMapY;
    bool mMapsInitialized;
    
    // Rotation and translation vectors
    std::vector<cv::Mat> mRVecs;
    std::vector<cv::Mat> mTVecs;
};

} // namespace calibration 