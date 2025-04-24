#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "calibrator.h"

namespace calibration {

/**
 * @brief Stereo camera calibration component that handles calibration and rectification
 * of a stereo camera pair.
 */
class StereoCalibrator {
public:
    /**
     * @brief Constructor
     * @param masterCalibrator Calibrator for the master camera
     * @param slaveCalibrator Calibrator for the slave camera
     */
    StereoCalibrator(Calibrator& masterCalibrator, Calibrator& slaveCalibrator);

    /**
     * @brief Perform stereo calibration using individual camera calibration results
     * @return True if calibration was successful, false otherwise
     */
    bool calibrateStereo();

    /**
     * @brief Save stereo calibration parameters to a file
     * @param filename Output filename (YAML format)
     * @return True if parameters were saved successfully, false otherwise
     */
    bool saveCalibration(const std::string& filename);

    /**
     * @brief Load stereo calibration parameters from a file
     * @param filename Input filename (YAML format)
     * @return True if parameters were loaded successfully, false otherwise
     */
    bool loadCalibration(const std::string& filename);

    /**
     * @brief Rectify a pair of stereo images
     * @param masterImage Input master camera image
     * @param slaveImage Input slave camera image
     * @param rectifiedMaster Output rectified master image
     * @param rectifiedSlave Output rectified slave image
     * @return True if rectification was applied, false otherwise
     */
    bool rectifyImages(const cv::Mat& masterImage, const cv::Mat& slaveImage, 
                      cv::Mat& rectifiedMaster, cv::Mat& rectifiedSlave);

    /**
     * @brief Get the reprojection error of the stereo calibration
     * @return Reprojection error in pixels
     */
    double getReprojectionError() const;

    /**
     * @brief Get the rotation matrix between cameras
     * @return Rotation matrix
     */
    cv::Mat getRotationMatrix() const;

    /**
     * @brief Get the translation vector between cameras
     * @return Translation vector
     */
    cv::Mat getTranslationVector() const;

    /**
     * @brief Get the essential matrix
     * @return Essential matrix
     */
    cv::Mat getEssentialMatrix() const;

    /**
     * @brief Get the fundamental matrix
     * @return Fundamental matrix
     */
    cv::Mat getFundamentalMatrix() const;

    /**
     * @brief Get the disparity-to-depth mapping matrix
     * @return Q matrix
     */
    cv::Mat getDisparityToDepthMatrix() const;

    /**
     * @brief Verify stereo calibration by showing rectified images
     * @param masterImage Input master camera image
     * @param slaveImage Input slave camera image
     * @param outputImage Output image showing rectification
     * @return True if verification was successful, false otherwise
     */
    bool verifyCalibration(const cv::Mat& masterImage, const cv::Mat& slaveImage, 
                          cv::Mat& outputImage);

private:
    /**
     * @brief Initialize rectification maps for efficient undistortion and rectification
     */
    void initRectificationMaps();

    // Reference to individual camera calibrators
    Calibrator& mMasterCalibrator;
    Calibrator& mSlaveCalibrator;
    
    // Image size
    cv::Size mImageSize;
    
    // Stereo calibration results
    double mStereoCalibrateError;
    cv::Mat mRotationMatrix;    // R
    cv::Mat mTranslationVector; // T
    cv::Mat mEssentialMatrix;   // E
    cv::Mat mFundamentalMatrix; // F
    
    // Rectification parameters
    cv::Mat mRectificationMaster;   // R1
    cv::Mat mRectificationSlave;    // R2
    cv::Mat mProjectionMaster;      // P1
    cv::Mat mProjectionSlave;       // P2
    cv::Mat mDisparityToDepthMatrix; // Q
    cv::Rect mValidRoiMaster;
    cv::Rect mValidRoiSlave;
    
    // Rectification maps
    cv::Mat mMapXMaster, mMapYMaster;
    cv::Mat mMapXSlave, mMapYSlave;
    bool mMapsInitialized;
    bool mCalibrated;
};

} // namespace calibration 