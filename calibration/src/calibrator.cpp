#include "calibrator.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace calibration {

Calibrator::Calibrator(cv::Size boardSize, float squareSize)
    : mBoardSize(boardSize)
    , mSquareSize(squareSize)
    , mReprojectionError(0.0)
    , mCalibrated(false)
    , mCornersDetected(false)
    , mMapsInitialized(false)
{
    // Initialize camera matrix with default values
    mCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    mDistCoeffs = cv::Mat::zeros(5, 1, CV_64F);
}

bool Calibrator::detectChessboard(cv::Mat& image, bool drawCorners)
{
    if (image.empty()) {
        return false;
    }

    // Store image size for later use
    mImageSize = image.size();
    
    // Try multiple methods to detect chessboard
    
    // Method 1: Standard OpenCV detection with grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    mCornersDetected = cv::findChessboardCorners(gray, mBoardSize, mCurrentCorners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        
    if (mCornersDetected) {
        // Refine corner locations
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray, mCurrentCorners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
        
        // Draw corners if requested
        if (drawCorners) {
            cv::drawChessboardCorners(image, mBoardSize, mCurrentCorners, mCornersDetected);
        }
        
        return true;
    }
    
    // Method 2: HSV filtering approach from the Python script
    // Define a list of HSV thresholds to try
    std::vector<cv::Scalar> hsvLowerThresholds = {
        cv::Scalar(0, 0, 90),
        cv::Scalar(0, 0, 135),
        cv::Scalar(0, 0, 235)
    };
    cv::Scalar hsvUpperThreshold(179, 255, 255);
    
    for (const auto& lowerThreshold : hsvLowerThresholds) {
        cv::Mat hsv, mask, dilated, result;
        
        // Convert to HSV
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        
        // Create mask
        cv::inRange(hsv, lowerThreshold, hsvUpperThreshold, mask);
        
        // Dilate
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 30));
        cv::dilate(mask, dilated, kernel, cv::Point(-1, -1), 5);
        
        // Create result (255 - bitwise_and)
        cv::bitwise_and(dilated, mask, result);
        result = 255 - result;
        
        // Find chessboard corners
        mCornersDetected = cv::findChessboardCorners(result, mBoardSize, mCurrentCorners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
            
        if (mCornersDetected) {
            // Refine corner locations
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(result, mCurrentCorners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            
            // Draw corners if requested
            if (drawCorners) {
                cv::drawChessboardCorners(image, mBoardSize, mCurrentCorners, mCornersDetected);
            }
            
            return true;
        }
    }
    
    return false;  // Chessboard not found with any method
}

size_t Calibrator::addCurrentPoints()
{
    if (!mCornersDetected) {
        return mImagePoints.size();
    }

    // Prepare object points (0,0,0), (1,0,0), (2,0,0) ... etc
    std::vector<cv::Point3f> objPoints;
    for (int i = 0; i < mBoardSize.height; i++) {
        for (int j = 0; j < mBoardSize.width; j++) {
            objPoints.push_back(cv::Point3f(j * mSquareSize, i * mSquareSize, 0));
        }
    }

    mObjectPoints.push_back(objPoints);
    mImagePoints.push_back(mCurrentCorners);
    mCornersDetected = false;  // Reset for next detection

    return mImagePoints.size();
}

bool Calibrator::calibrate()
{
    if (mImagePoints.size() < 3) {
        std::cerr << "Error: At least 3 valid chessboard images are required for calibration." << std::endl;
        return false;
    }
    
    mCalibrated = false;
    mRVecs.clear();
    mTVecs.clear();

    // Perform camera calibration
    mReprojectionError = cv::calibrateCamera(
        mObjectPoints, mImagePoints, mImageSize,
        mCameraMatrix, mDistCoeffs, mRVecs, mTVecs
    );

    mCalibrated = true;

    // Assess calibration quality
    calculateReprojectionError();
    assessCalibrationQuality();

    return mCalibrated;
}

void Calibrator::calculateReprojectionError()
{
    mPerViewErrors.clear();
    mPerViewErrors.resize(mObjectPoints.size());

    double totalError = 0.0;
    int totalPoints = 0;

    for (size_t i = 0; i < mObjectPoints.size(); i++) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(mObjectPoints[i], mRVecs[i], mTVecs[i], 
                        mCameraMatrix, mDistCoeffs, projectedPoints);
    
        double err = 0.0;
        for (size_t j = 0; j < projectedPoints.size(); j++) {
            err += cv::norm(mImagePoints[i][j] - projectedPoints[j]);
        }
        
        mPerViewErrors[i] = (err / projectedPoints.size());
        totalError += err;
        totalPoints += projectedPoints.size();
    }

    mReprojectionError = totalError / totalPoints;
}

void Calibrator::assessCalibrationQuality()
{
    std::ostringstream oss;
    oss << "Calibration Quality Assessment:" << std::endl;
    oss << "  Average reprojection error: " << std::fixed << std::setprecision(3) 
        << mReprojectionError << " pixels" << std::endl;
    
    // Find the worst views (highest reprojection error)
    std::vector<std::pair<float, size_t>> sortedErrors;
    for (size_t i = 0; i < mPerViewErrors.size(); i++) {
        sortedErrors.push_back(std::make_pair(mPerViewErrors[i], i));
    }
    std::sort(sortedErrors.begin(), sortedErrors.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Report worst views
    oss << "  Views with highest error:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), sortedErrors.size()); i++) {
        oss << "    View #" << sortedErrors[i].second + 1 << ": " 
            << std::fixed << std::setprecision(3) 
            << sortedErrors[i].first << " pixels" << std::endl;
    }

    // Quality suggestions
    if (mReprojectionError > 1.0) {
        oss << "  WARNING: Error above 1 pixel suggests poor calibration." << std::endl;
        oss << "  Recommendation: Try recapturing images with better coverage." << std::endl;
    } else if (mReprojectionError > 0.5) {
        oss << "  Acceptable calibration quality. Could be improved with more views." << std::endl;
    } else {
        oss << "  Good calibration quality (error < 0.5 pixel)." << std::endl;
    }
    
    // Coverage assessment
    if (mImagePoints.size() < 10) {
        oss << "  Only " << mImagePoints.size() << " views used. Consider using more views for better accuracy." << std::endl;
    }

    mQualityAssessment = oss.str();
}

bool Calibrator::saveCalibration(const std::string& filename)
{
    if (!mCalibrated) {
        std::cerr << "Error: Camera not calibrated yet. Cannot save parameters." << std::endl;
        return false;
    }

    try {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
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
        fs << "board_width" << mBoardSize.width;
        fs << "board_height" << mBoardSize.height;
        fs << "square_size" << mSquareSize;
        
        fs << "camera_matrix" << mCameraMatrix;
        fs << "distortion_coefficients" << mDistCoeffs;
        fs << "avg_reprojection_error" << mReprojectionError;
        fs << "per_view_reprojection_errors" << mPerViewErrors;
        fs << "num_images" << (int)mImagePoints.size();

        fs.release();
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error saving calibration parameters: " << e.what() << std::endl;
        return false;
    }
}

bool Calibrator::loadCalibration(const std::string& filename)
{
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
        mImageSize = cv::Size(width, height);
        
        fs["board_width"] >> mBoardSize.width;
        fs["board_height"] >> mBoardSize.height;
        fs["square_size"] >> mSquareSize;
        
        fs["camera_matrix"] >> mCameraMatrix;
        fs["distortion_coefficients"] >> mDistCoeffs;
        fs["avg_reprojection_error"] >> mReprojectionError;

        // Optional values
        if (fs["per_view_reprojection_errors"].isNone()) {
            mPerViewErrors.clear();
    } else {
            fs["per_view_reprojection_errors"] >> mPerViewErrors;
        }

        mCalibrated = true;
        fs.release();
        
        // Initialize undistortion maps
        initUndistortMaps(mImageSize);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading calibration parameters: " << e.what() << std::endl;
        return false;
    }
}

double Calibrator::getReprojectionError() const
{
    return mReprojectionError;
}

std::string Calibrator::getQualityAssessment() const
{
    return mQualityAssessment;
}

size_t Calibrator::getDetectionCount() const
{
    return mImagePoints.size();
}

void Calibrator::clearPoints()
{
    mObjectPoints.clear();
    mImagePoints.clear();
    mPerViewErrors.clear();
    mCalibrated = false;
}

void Calibrator::setFilteredPoints(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                                   const std::vector<std::vector<cv::Point2f>>& imagePoints)
{
    mObjectPoints = objectPoints;
    mImagePoints = imagePoints;
}

cv::Mat Calibrator::getCameraMatrix() const
{
    return mCameraMatrix;
}

cv::Mat Calibrator::getDistortionCoefficients() const
{
    return mDistCoeffs;
}

bool Calibrator::undistortImage(cv::Mat& image)
{
    if (!mCalibrated || image.empty()) {
        return false;
    }

    cv::Mat temp = image.clone();
    cv::undistort(temp, image, mCameraMatrix, mDistCoeffs);
    return true;
}

void Calibrator::initUndistortMaps(const cv::Size& imageSize)
{
    if (!mCalibrated) {
        return;
    }

    cv::initUndistortRectifyMap(
        mCameraMatrix, mDistCoeffs, cv::Mat(),
        cv::getOptimalNewCameraMatrix(mCameraMatrix, mDistCoeffs, imageSize, 1, imageSize, 0),
        imageSize, CV_32FC1, mMapX, mMapY
    );
    
    mMapsInitialized = true;
}

bool Calibrator::undistortWithMaps(const cv::Mat& input, cv::Mat& output)
{
    if (!mCalibrated || !mMapsInitialized || input.empty()) {
        return false;
    }

    // Initialize maps if needed or if image size changed
    if (input.size() != mImageSize) {
        mImageSize = input.size();
        initUndistortMaps(mImageSize);
                }

    cv::remap(input, output, mMapX, mMapY, cv::INTER_LINEAR);
    return true;
}

const std::vector<std::vector<cv::Point3f>>& Calibrator::getObjectPoints() const 
{
    return mObjectPoints;
}

const std::vector<std::vector<cv::Point2f>>& Calibrator::getImagePoints() const 
{
    return mImagePoints;
}

cv::Size Calibrator::getBoardSize() const 
{
    return mBoardSize;
}

float Calibrator::getSquareSize() const 
{
    return mSquareSize;
}

cv::Size Calibrator::getImageSize() const 
{
    return mImageSize;
}

} // namespace calibration 