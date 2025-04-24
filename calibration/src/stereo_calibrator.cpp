#include "stereo_calibrator.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace calibration {

StereoCalibrator::StereoCalibrator(Calibrator& masterCalibrator, Calibrator& slaveCalibrator)
    : mMasterCalibrator(masterCalibrator)
    , mSlaveCalibrator(slaveCalibrator)
    , mStereoCalibrateError(0.0)
    , mMapsInitialized(false)
    , mCalibrated(false)
{
    // Initialize matrices
    mRotationMatrix = cv::Mat::eye(3, 3, CV_64F);
    mTranslationVector = cv::Mat::zeros(3, 1, CV_64F);
    mEssentialMatrix = cv::Mat::zeros(3, 3, CV_64F);
    mFundamentalMatrix = cv::Mat::zeros(3, 3, CV_64F);
    mRectificationMaster = cv::Mat::eye(3, 3, CV_64F);
    mRectificationSlave = cv::Mat::eye(3, 3, CV_64F);
    mProjectionMaster = cv::Mat::zeros(3, 4, CV_64F);
    mProjectionSlave = cv::Mat::zeros(3, 4, CV_64F);
    mDisparityToDepthMatrix = cv::Mat::zeros(4, 4, CV_64F);
}


bool StereoCalibrator::calibrateStereo()
{
    // Check if both calibrators have calibrated cameras
    if (!mMasterCalibrator.getDetectionCount() || !mSlaveCalibrator.getDetectionCount()) {
        std::cerr << "Error: Both cameras must be calibrated individually before stereo calibration" << std::endl;
        return false;
    }
    
    // Get intrinsic parameters from individual calibrators
    cv::Mat cameraMatrixMaster = mMasterCalibrator.getCameraMatrix();
    cv::Mat distCoeffsMaster = mMasterCalibrator.getDistortionCoefficients();
    cv::Mat cameraMatrixSlave = mSlaveCalibrator.getCameraMatrix();
    cv::Mat distCoeffsSlave = mSlaveCalibrator.getDistortionCoefficients();
    
    // Get image size from master camera
    mImageSize = mMasterCalibrator.getImageSize();
    
    // Get object points and image points
    const std::vector<std::vector<cv::Point3f>>& objectPointsMaster = mMasterCalibrator.getObjectPoints();
    const std::vector<std::vector<cv::Point2f>>& imagePointsMaster = mMasterCalibrator.getImagePoints();
    const std::vector<std::vector<cv::Point2f>>& imagePointsSlave = mSlaveCalibrator.getImagePoints();
    
    // Validate that we have the same number of points
    if (objectPointsMaster.empty() || imagePointsMaster.empty() || imagePointsSlave.empty()) {
        std::cerr << "Error: No calibration points available" << std::endl;
        return false;
    }
    
    if (imagePointsMaster.size() != imagePointsSlave.size()) {
        std::cerr << "Error: Master and slave cameras have different number of calibration images" << std::endl;
        std::cerr << "Master: " << imagePointsMaster.size() << ", Slave: " << imagePointsSlave.size() << std::endl;
        return false;
    }
    std::cout << "Performing stereo calibration with " << imagePointsMaster.size() << " image pairs..." << std::endl;
    
    // Perform stereo calibration
    int flags = cv::CALIB_FIX_INTRINSIC; // Use the intrinsic parameters we already found
    
    try {
        mStereoCalibrateError = cv::stereoCalibrate(
            objectPointsMaster, imagePointsMaster, imagePointsSlave,
            cameraMatrixMaster, distCoeffsMaster,
            cameraMatrixSlave, distCoeffsSlave,
            mImageSize, mRotationMatrix, mTranslationVector,
            mEssentialMatrix, mFundamentalMatrix,
            flags
        );
        
        std::cout << "Stereo calibration successful. RMS error: " << mStereoCalibrateError << std::endl;
        
        // Compute rectification parameters
        cv::stereoRectify(
            cameraMatrixMaster, distCoeffsMaster,
            cameraMatrixSlave, distCoeffsSlave,
            mImageSize, mRotationMatrix, mTranslationVector,
            mRectificationMaster, mRectificationSlave,
            mProjectionMaster, mProjectionSlave,
            mDisparityToDepthMatrix,
            cv::CALIB_ZERO_DISPARITY, 1, mImageSize,
            &mValidRoiMaster, &mValidRoiSlave
        );
        
        std::cout << "Rectification parameters computed successfully" << std::endl;
        
        // Initialize rectification maps
        mCalibrated = true;
        initRectificationMaps();
        
        // Display calibration results
        std::cout << "Rotation matrix: " << mRotationMatrix << std::endl;
        std::cout << "Translation vector: " << mTranslationVector << std::endl;
        std::cout << "Baseline: " << cv::norm(mTranslationVector) << " mm" << std::endl;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in stereo calibration: " << e.what() << std::endl;
        return false;
    }
}

void StereoCalibrator::initRectificationMaps()
{
    if (!mCalibrated) {
        std::cerr << "Error: Stereo calibration must be performed before initializing maps" << std::endl;
        return;
    }

    std::cout << "Initializing rectification maps" << std::endl;

    try {         
        // Get intrinsic parameters from individual calibrators
        cv::Mat cameraMatrixMaster = mMasterCalibrator.getCameraMatrix();
        cv::Mat distCoeffsMaster = mMasterCalibrator.getDistortionCoefficients();
        cv::Mat cameraMatrixSlave = mSlaveCalibrator.getCameraMatrix();
        cv::Mat distCoeffsSlave = mSlaveCalibrator.getDistortionCoefficients();
        
        // Initialize maps for master camera
        cv::initUndistortRectifyMap(
            cameraMatrixMaster, distCoeffsMaster,
            mRectificationMaster, mProjectionMaster,
            mImageSize, CV_32FC1,
            mMapXMaster, mMapYMaster
        );
        
        // Initialize maps for slave camera
        cv::initUndistortRectifyMap(
            cameraMatrixSlave, distCoeffsSlave,
            mRectificationSlave, mProjectionSlave,
            mImageSize, CV_32FC1,
            mMapXSlave, mMapYSlave
        );
        
        mMapsInitialized = true;

        std::cout << "Rectification maps initialized successfully" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in rectification map initialization: " << e.what() << std::endl;
    }
}

bool StereoCalibrator::rectifyImages(const cv::Mat& masterImage, const cv::Mat& slaveImage, 
                                    cv::Mat& rectifiedMaster, cv::Mat& rectifiedSlave)
{
    if (!mMapsInitialized) {
        std::cerr << "Error: Rectification maps not initialized" << std::endl;
        return false;
    }
    
    if (masterImage.empty() || slaveImage.empty()) {
        std::cerr << "Error: Input images must not be empty" << std::endl;
        return false;
    }
    
    // Check if image size matches calibration
    if (masterImage.size() != mImageSize || slaveImage.size() != mImageSize) {
        std::cerr << "Warning: Input image size doesn't match calibration size" << std::endl;
        std::cerr << "Expected: " << mImageSize << ", Master: " << masterImage.size() 
                 << ", Slave: " << slaveImage.size() << std::endl;
        // We'll continue anyway, but the results might be incorrect
    }
    
    // Rectify images
    cv::remap(masterImage, rectifiedMaster, mMapXMaster, mMapYMaster, cv::INTER_LINEAR);
    cv::remap(slaveImage, rectifiedSlave, mMapXSlave, mMapYSlave, cv::INTER_LINEAR);
    
    return true;
}

bool StereoCalibrator::verifyCalibration(const cv::Mat& masterImage, const cv::Mat& slaveImage, 
                                        cv::Mat& outputImage)
{
    cv::Mat rectifiedMaster, rectifiedSlave;
    
    if (!rectifyImages(masterImage, slaveImage, rectifiedMaster, rectifiedSlave)) {
        return false;
    }
    
    // Draw horizontal lines every 50 pixels to visualize rectification
    const int lineStep = 50;
    cv::Mat masterWithLines = rectifiedMaster.clone();
    cv::Mat slaveWithLines = rectifiedSlave.clone();
    
    for (int y = 0; y < masterWithLines.rows; y += lineStep) {
        cv::line(masterWithLines, cv::Point(0, y), cv::Point(masterWithLines.cols, y), cv::Scalar(0, 255, 0), 1);
        cv::line(slaveWithLines, cv::Point(0, y), cv::Point(slaveWithLines.cols, y), cv::Scalar(0, 255, 0), 1);
    }
    
    // Combine rectified images side by side
    cv::hconcat(masterWithLines, slaveWithLines, outputImage);
    
    // Add calibration info to the output image
    std::stringstream ss;
    ss << "Stereo Calibration | RMS Error: " << std::fixed << std::setprecision(3) << mStereoCalibrateError;
    cv::putText(outputImage, ss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    ss.str("");
    ss << "Baseline: " << std::fixed << std::setprecision(2) << cv::norm(mTranslationVector) << " mm";
    cv::putText(outputImage, ss.str(), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    return true;
}

bool StereoCalibrator::saveCalibration(const std::string& filename)
{
    if (!mCalibrated) {
        std::cerr << "Error: Stereo calibration not performed yet" << std::endl;
        return false;
    }
    
    try {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        
        // Get current time
        time_t now = time(0);
        char timeStr[100];
        strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", localtime(&now));
        
        // Write calibration parameters
        fs << "calibration_time" << timeStr;
        fs << "image_width" << mImageSize.width;
        fs << "image_height" << mImageSize.height;
        
        fs << "rms_error" << mStereoCalibrateError;
        
        // Intrinsic parameters
        fs << "camera_matrix_master" << mMasterCalibrator.getCameraMatrix();
        fs << "dist_coeffs_master" << mMasterCalibrator.getDistortionCoefficients();
        fs << "camera_matrix_slave" << mSlaveCalibrator.getCameraMatrix();
        fs << "dist_coeffs_slave" << mSlaveCalibrator.getDistortionCoefficients();
        
        // Extrinsic parameters
        fs << "rotation_matrix" << mRotationMatrix;
        fs << "translation_vector" << mTranslationVector;
        fs << "essential_matrix" << mEssentialMatrix;
        fs << "fundamental_matrix" << mFundamentalMatrix;
        
        // Rectification parameters
        fs << "rectification_master" << mRectificationMaster;
        fs << "rectification_slave" << mRectificationSlave;
        fs << "projection_master" << mProjectionMaster;
        fs << "projection_slave" << mProjectionSlave;
        fs << "disparity_to_depth_matrix" << mDisparityToDepthMatrix;
        
        // Valid ROIs
        fs << "valid_roi_master" << mValidRoiMaster;
        fs << "valid_roi_slave" << mValidRoiSlave;
        
        fs.release();
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error saving stereo calibration parameters: " << e.what() << std::endl;
        return false;
    }
}

bool StereoCalibrator::loadCalibration(const std::string& filename)
{
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open stereo calibration file " << filename << std::endl;
            return false;
        }
        
        // Read image size
        fs["image_width"] >> mImageSize.width;
        fs["image_height"] >> mImageSize.height;
        
        // Read stereo calibration error
        fs["rms_error"] >> mStereoCalibrateError;
        
        // Read extrinsic parameters
        fs["rotation_matrix"] >> mRotationMatrix;
        fs["translation_vector"] >> mTranslationVector;
        fs["essential_matrix"] >> mEssentialMatrix;
        fs["fundamental_matrix"] >> mFundamentalMatrix;
        
        // Read rectification parameters
        fs["rectification_master"] >> mRectificationMaster;
        fs["rectification_slave"] >> mRectificationSlave;
        fs["projection_master"] >> mProjectionMaster;
        fs["projection_slave"] >> mProjectionSlave;
        fs["disparity_to_depth_matrix"] >> mDisparityToDepthMatrix;
        
        // Read valid ROIs
        fs["valid_roi_master"] >> mValidRoiMaster;
        fs["valid_roi_slave"] >> mValidRoiSlave;
        
        fs.release();
        
        // Initialize rectification maps
        initRectificationMaps();
        
        mCalibrated = true;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error loading stereo calibration parameters: " << e.what() << std::endl;
        return false;
    }
}

double StereoCalibrator::getReprojectionError() const
{
    return mStereoCalibrateError;
}

cv::Mat StereoCalibrator::getRotationMatrix() const
{
    return mRotationMatrix;
}

cv::Mat StereoCalibrator::getTranslationVector() const
{
    return mTranslationVector;
}

cv::Mat StereoCalibrator::getEssentialMatrix() const
{
    return mEssentialMatrix;
}

cv::Mat StereoCalibrator::getFundamentalMatrix() const
{
    return mFundamentalMatrix;
}

cv::Mat StereoCalibrator::getDisparityToDepthMatrix() const
{
    return mDisparityToDepthMatrix;
}

} // namespace calibration 