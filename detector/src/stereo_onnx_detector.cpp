#define _USE_MATH_DEFINES // For M_PI constant
#include "stereo_onnx_detector.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include "logging.h" // Include logging header

StereoOnnxDetector::StereoOnnxDetector(const std::string& modelPath, const std::string& classesPath,
                                     const std::string& stereoCalibFile,
                                     float confThreshold, float nmsThreshold)
    : mConfThreshold(confThreshold)
    , mNmsThreshold(nmsThreshold)
    , mFrameCounter(0)
    , mCalibrationLoaded(false)
    , mLastMasterCenter(cv::Point2f(-1, -1))
    , mLastSlaveCenter(cv::Point2f(-1, -1))
    , mFrameRate(30.0f) // Default to 30 FPS if not set explicitly
{
    // Create individual detectors for master and slave cameras
    mMasterDetector = std::make_unique<OnnxDetector>(modelPath, classesPath, confThreshold, nmsThreshold);
    mSlaveDetector = std::make_unique<OnnxDetector>(modelPath, classesPath, confThreshold, nmsThreshold);
    
    // Load stereo calibration if provided
    if (!stereoCalibFile.empty()) {
        if (loadStereoCalibration(stereoCalibFile)) {
            logMessage("Stereo calibration loaded successfully from: " + stereoCalibFile, true);
        } else {
            logMessage("Failed to load stereo calibration from: " + stereoCalibFile, true);
        }
    }
}

StereoOnnxDetector::~StereoOnnxDetector() {
    // Unique_ptr will handle cleanup of detectors
}

bool StereoOnnxDetector::loadStereoCalibration(const std::string& filename) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            logMessage("Error: Could not open stereo calibration file: " + filename, true);
            return false;
        }
        
        // Read image size
        fs["image_width"] >> mImageSize.width;
        fs["image_height"] >> mImageSize.height;
        
        if (mImageSize.width <= 0 || mImageSize.height <= 0) {
            logMessage("Error: Invalid image size in calibration file", true);
            return false;
        }
        
        // Read intrinsic parameters
        fs["camera_matrix_master"] >> mCameraMatrixMaster;
        fs["dist_coeffs_master"] >> mDistCoeffsMaster;
        fs["camera_matrix_slave"] >> mCameraMatrixSlave;
        fs["dist_coeffs_slave"] >> mDistCoeffsSlave;
        
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
        
        // Read ROI information if available
        if (!fs["valid_roi_master"].empty() && !fs["valid_roi_slave"].empty()) {
            fs["valid_roi_master"] >> mValidRoiMaster;
            fs["valid_roi_slave"] >> mValidRoiSlave;
        }
        
        fs.release();
        
        // Initialize rectification maps
        initRectificationMaps();
        
        // Set calibration in individual detectors
        mMasterDetector->setCalibration(mCameraMatrixMaster, mDistCoeffsMaster);
        mSlaveDetector->setCalibration(mCameraMatrixSlave, mDistCoeffsSlave);
        
        mCalibrationLoaded = true;
        return true;
    }
    catch (const cv::Exception& e) {
        logMessage("Error loading stereo calibration: " + std::string(e.what()), true);
        return false;
    }
}

void StereoOnnxDetector::initRectificationMaps() {
    if (mImageSize.width <= 0 || mImageSize.height <= 0) {
        logMessage("Error: Cannot initialize rectification maps with invalid image size", true);
        return;
    }
    
    // Initialize maps for master camera
    cv::initUndistortRectifyMap(
        mCameraMatrixMaster, mDistCoeffsMaster,
        mRectificationMaster, mProjectionMaster,
        mImageSize, CV_32FC1,
        mMapXMaster, mMapYMaster
    );
    
    // Initialize maps for slave camera
    cv::initUndistortRectifyMap(
        mCameraMatrixSlave, mDistCoeffsSlave,
        mRectificationSlave, mProjectionSlave,
        mImageSize, CV_32FC1,
        mMapXSlave, mMapYSlave
    );
}

cv::Mat StereoOnnxDetector::processFrames(const cv::Mat& frameMaster, const cv::Mat& frameSlave, bool showDetections) {
    if (frameMaster.empty() || frameSlave.empty()) {
        logMessage("Error: Empty input frames", true);
        return cv::Mat();
    }
    
    // Increment frame counter
    mFrameCounter++;
    
    // Create copies for processing
    cv::Mat masterFrame = frameMaster.clone();
    cv::Mat slaveFrame = frameSlave.clone();
    
    // Apply rectification if calibration is loaded
    cv::Mat rectifiedMaster, rectifiedSlave;
    if (mCalibrationLoaded) {
        cv::remap(masterFrame, rectifiedMaster, mMapXMaster, mMapYMaster, cv::INTER_LINEAR);
        cv::remap(slaveFrame, rectifiedSlave, mMapXSlave, mMapYSlave, cv::INTER_LINEAR);
    } else {
        rectifiedMaster = masterFrame;
        rectifiedSlave = slaveFrame;
    }
    
    // Perform detection on both frames
    std::vector<float> masterConfidences, slaveConfidences;
    std::vector<int> masterClassIds, slaveClassIds;
    std::vector<cv::Rect> masterBoxes = mMasterDetector->detect(rectifiedMaster, masterConfidences, masterClassIds, false, false);
    std::vector<cv::Rect> slaveBoxes = mSlaveDetector->detect(rectifiedSlave, slaveConfidences, slaveClassIds, false, false);
    
    // Tracking flag to know if we detected ball in current frame
    bool ballDetectedMaster = false;
    bool ballDetectedSlave = false;
    
    // Find the best detection in each frame (highest confidence)
    cv::Point2f centerMaster(-1, -1);
    cv::Point2f centerSlave(-1, -1);
    float bestMasterConf = 0.0f;
    float bestSlaveConf = 0.0f;
    
    // Process master frame ball detections
    for (size_t i = 0; i < masterBoxes.size(); i++) {
        int masterClassId = masterClassIds[i];
        float confidence = masterConfidences[i];
        
        // Check if this is a ball detection
        bool isBall = (masterClassId == 1 || mMasterDetector->getClassName(masterClassId) == "ball");
        
        if (isBall) {
            // If we find a better ball detection (higher confidence), update
            if (confidence > bestMasterConf) {
                centerMaster = findObjectCenter(rectifiedMaster, masterBoxes[i]);
                if (centerMaster.x >= 0 && centerMaster.y >= 0) {
                    centerMaster = refineBallCenter(rectifiedMaster, masterBoxes[i]);
                    bestMasterConf = confidence;
                    ballDetectedMaster = true;
                }
            }
        }
    }
    
    // Process slave frame ball detections
    for (size_t i = 0; i < slaveBoxes.size(); i++) {
        int slaveClassId = slaveClassIds[i];
        float confidence = slaveConfidences[i];
        
        // Check if this is a ball detection
        bool isBall = (slaveClassId == 1 || mSlaveDetector->getClassName(slaveClassId) == "ball");
        
        if (isBall) {
            // If we find a better ball detection (higher confidence), update
            if (confidence > bestSlaveConf) {
                centerSlave = findObjectCenter(rectifiedSlave, slaveBoxes[i]);
                if (centerSlave.x >= 0 && centerSlave.y >= 0) {
                    centerSlave = refineBallCenter(rectifiedSlave, slaveBoxes[i]);
                    bestSlaveConf = confidence;
                    ballDetectedSlave = true;
                }
            }
        }
    }
    
    // Only update the stored centers if we actually detected a ball in this frame
    if (ballDetectedMaster) {
        mLastMasterCenter = centerMaster;
    } else {
        // If no ball detected in this frame, invalidate the center
        centerMaster = cv::Point2f(-1, -1);
    }
    
    if (ballDetectedSlave) {
        mLastSlaveCenter = centerSlave;
    } else {
        // If no ball detected in this frame, invalidate the center
        centerSlave = cv::Point2f(-1, -1);
    }
    
    // Triangulate 3D position if both detections are valid (we detected ball in BOTH cameras)
    Point3D point3D;
    if (ballDetectedMaster && ballDetectedSlave && mCalibrationLoaded) {
        point3D = triangulatePoint(centerMaster, centerSlave);
        
        // Store 3D position
        mPositions3D.push_back(point3D);
        mFrameIndices.push_back(mFrameCounter);
        
        // Update motion parameters if we have enough points
        if (mPositions3D.size() >= 2) {
            calculateMotionParameters(mFrameRate, true);
        }
    }
    
    // Create visualization
    cv::Mat visualization = createVisualization(rectifiedMaster, rectifiedSlave, centerMaster, centerSlave, point3D);
    
    return visualization;
}

Point3D StereoOnnxDetector::triangulatePoint(const cv::Point2f& pointMaster, const cv::Point2f& pointSlave) const {
    if (!mCalibrationLoaded) {
        logMessage("Error: Cannot triangulate without calibration", true);
        return Point3D();
    }
    
    // Convert points to the format needed by triangulatePoints
    cv::Mat pointsMaster(2, 1, CV_32F);
    cv::Mat pointsSlave(2, 1, CV_32F);
    
    pointsMaster.at<float>(0, 0) = pointMaster.x;
    pointsMaster.at<float>(1, 0) = pointMaster.y;
    pointsSlave.at<float>(0, 0) = pointSlave.x;
    pointsSlave.at<float>(1, 0) = pointSlave.y;
    
    // Triangulate
    cv::Mat point4D;
    cv::triangulatePoints(mProjectionMaster, mProjectionSlave, pointsMaster, pointsSlave, point4D);
    
    // Convert to 3D (homogeneous to Cartesian)
    point4D = point4D / point4D.at<float>(3, 0);
    
    return Point3D(
        point4D.at<float>(0, 0),
        point4D.at<float>(1, 0),
        point4D.at<float>(2, 0)
    );
}

cv::Point2f StereoOnnxDetector::findObjectCenter(const cv::Mat& frame, const cv::Rect& box) const {
    // Simple center calculation (can be enhanced for specific object types)
    return cv::Point2f(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
}

cv::Point2f StereoOnnxDetector::refineBallCenter(const cv::Mat& frame, const cv::Rect& box) const {

    // Extract the bounding box coordinates
    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.x + box.width;
    int y2 = box.y + box.height;

    // Add padding to the bounding box (20% on each side)
    int padding_x = static_cast<int>((x2 - x1) * 0.2);
    int padding_y = static_cast<int>((y2 - y1) * 0.2);

    // Ensure the padded box stays within the frame
    int x1_pad = std::max(0, x1 - padding_x);
    int y1_pad = std::max(0, y1 - padding_y);
    int x2_pad = std::min(frame.cols, x2 + padding_x);
    int y2_pad = std::min(frame.rows, y2 + padding_y);

    // Create the ROI rectangle
    cv::Rect roi(x1_pad, y1_pad, x2_pad - x1_pad, y2_pad - y1_pad);
    
    // Extract the ROI from the frame
    cv::Mat roiImg = frame(roi);
    
    // Convert to grayscale
    cv::Mat grayImg;
    cv::cvtColor(roiImg, grayImg, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(grayImg, grayImg, cv::Size(5, 5), 0);

    // Calculate expected radius range based on bounding box
    float expected_radius = std::min((x2 - x1), (y2 - y1)) / 2;
    int min_radius = std::max(30, static_cast<int>(expected_radius * 0.5));
    int max_radius = static_cast<int>(expected_radius * 1.2);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(
        grayImg,                   // Input image (must be grayscale)
        circles,                   // Output vector of found circles
        cv::HOUGH_GRADIENT,        // Detection method
        1.2,                       // dp parameter
        grayImg.rows,              // minDist (use ROI height to detect only one circle)
        100,                       // param1 (Canny edge detector upper threshold)
        20,                        // param2 (center detection threshold)
        min_radius,                // Minimum radius
        max_radius                 // Maximum radius
    );
    
    // If circles found, process them
    if (!circles.empty()) {
        // Get the best circle (usually only one is detected)
        float x = circles[0][0];
        float y = circles[0][1];
        float r = circles[0][2];
        
        // Convert back to original frame coordinates
        float center_x = x + x1_pad;
        float center_y = y + y1_pad;
        
        return cv::Point2f(center_x, center_y);
    }
    
    // Threshold the image
    cv::Mat threshImg;
    cv::threshold(grayImg, threshImg, 100, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(threshImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        
        // Check circularity (a perfect circle has circularity of 1.0)
        double perimeter = cv::arcLength(contours[i], true);
        double circularity = perimeter > 0 ? 4 * M_PI * area / (perimeter * perimeter) : 0;
        
        // Find the largest contour with good circularity
        if (circularity > 0.7) {
            // Get moments of the largest contour
            cv::Moments moments = cv::moments(contours[i]);
            
            // Ensure moments are valid
            if (moments.m00 > 0) {
        // Calculate center from moments
        double cX = moments.m10 / moments.m00;
        double cY = moments.m01 / moments.m00;
        
        // Adjust for ROI offset
                return cv::Point2f(cX + x1_pad, cY + y1_pad);
            }
        }
    }
    
    float center_x = (x1 + x2) / 2.0f;
    float center_y = (y1 + y2) / 2.0f;

    return cv::Point2f(center_x, center_y);
}

cv::Mat StereoOnnxDetector::createVisualization(const cv::Mat& rectifiedMaster, const cv::Mat& rectifiedSlave,
                                              const cv::Point2f& centerMaster, const cv::Point2f& centerSlave,
                                              const Point3D& point3D) const {
    // Create copies for drawing
    cv::Mat masterVis = rectifiedMaster.clone();
    cv::Mat slaveVis = rectifiedSlave.clone();
    
    // Draw horizontal lines to show rectification
    if (mCalibrationLoaded) {
        for (int y = 0; y < masterVis.rows; y += 100) {
            cv::line(masterVis, cv::Point(0, y), cv::Point(masterVis.cols, y), cv::Scalar(0, 255, 0), 1);
            cv::line(slaveVis, cv::Point(0, y), cv::Point(slaveVis.cols, y), cv::Scalar(0, 255, 0), 1);
        }
    }
    
    // Draw detection centers
    if (centerMaster.x >= 0 && centerMaster.y >= 0) {
        //cv::circle(masterVis, centerMaster, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(masterVis, centerMaster, 10, cv::Scalar(0, 0, 255), 2);
    }
    
    if (centerSlave.x >= 0 && centerSlave.y >= 0) {
        //cv::circle(slaveVis, centerSlave, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(slaveVis, centerSlave, 10, cv::Scalar(0, 0, 255), 2);
    }
    
    // Combine images side by side
    cv::Mat combined;
    cv::hconcat(masterVis, slaveVis, combined);
    
    // Add labels for master and slave
    cv::putText(combined, "Master Camera", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(combined, "Slave Camera", cv::Point(masterVis.cols + 20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    // Add 3D position information if available
    if (point3D.x != 0 || point3D.y != 0 || point3D.z != 0) {
        draw3DPosition(combined, point3D);
    }
    
    return combined;
}

void StereoOnnxDetector::draw3DPosition(cv::Mat& image, const Point3D& point3D) const {
    // Format 3D coordinates with 2 decimal places
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "3D Position: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ") mm";
    
    // Draw text in bottom area of the image
    cv::putText(image, ss.str(), cv::Point(20, image.rows - 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    // Display motion parameters if we have enough points
    if (mPositions3D.size() >= 2) {
        std::stringstream ssVel;
        ssVel << std::fixed << std::setprecision(2);
        ssVel << "Velocity: (" << mMotionParams.initialVelocity[0] << ", " << mMotionParams.initialVelocity[1] << ", " 
              << mMotionParams.initialVelocity[2] << ") mm/s, Magnitude: " << mMotionParams.initialSpeed << " mm/s";
        
        cv::putText(image, ssVel.str(), cv::Point(20, image.rows - 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        // Also display current/latest velocity if available
        if (!mMotionParams.velocities.empty()) {
            std::stringstream ssCurrent;
            ssCurrent << std::fixed << std::setprecision(2);
            
            // Get the last velocity
            size_t lastIdx = mMotionParams.velocities.size() - 1;
            float lastSpeed = mMotionParams.speeds[lastIdx];
            
            ssCurrent << "Current Velocity: (" 
                    << mMotionParams.velocities[lastIdx][0] << ", " 
                    << mMotionParams.velocities[lastIdx][1] << ", " 
                    << mMotionParams.velocities[lastIdx][2] << ") mm/s, Magnitude: " 
                    << lastSpeed << " mm/s";
            
            cv::putText(image, ssCurrent.str(), cv::Point(20, image.rows - 100), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        }
    }
}

MotionParameters StereoOnnxDetector::calculateMotionParameters(float frameRate, bool updateMotionParams) const {
    MotionParameters params;
    
    // Check if we have enough positions
    if (mPositions3D.size() < 2 || mFrameIndices.size() < 2) {
        return params;
    }
    
    // Prepare vectors for calculations
    size_t n = mPositions3D.size();
    std::vector<std::vector<float>> positions(n, std::vector<float>(3, 0.0f));
    std::vector<float> timestamps(n, 0.0f);
    
    // Fill positions and timestamps arrays
    for (size_t i = 0; i < n; i++) {
        positions[i][0] = mPositions3D[i].x;
        positions[i][1] = mPositions3D[i].y;
        positions[i][2] = mPositions3D[i].z;
        timestamps[i] = static_cast<float>(mFrameIndices[i]) / frameRate;  // Convert frame indices to seconds
    }
    
    // Calculate time differences between consecutive frames
    std::vector<float> timeDiffs(n - 1, 0.0f);
    for (size_t i = 0; i < n - 1; i++) {
        timeDiffs[i] = timestamps[i + 1] - timestamps[i];
        if (timeDiffs[i] <= 0) {
            timeDiffs[i] = 1.0f / frameRate;  // Avoid division by zero
        }
    }
    
    // Calculate displacements between consecutive positions
    std::vector<std::vector<float>> displacements(n - 1, std::vector<float>(3, 0.0f));
    for (size_t i = 0; i < n - 1; i++) {
        displacements[i][0] = positions[i + 1][0] - positions[i][0];  // dx
        displacements[i][1] = positions[i + 1][1] - positions[i][1];  // dy
        displacements[i][2] = positions[i + 1][2] - positions[i][2];  // dz
    }
    
    // Calculate velocities (mm/s)
    std::vector<std::vector<float>> velocities(n - 1, std::vector<float>(3, 0.0f));
    std::vector<float> speeds(n - 1, 0.0f);
    
    // Define a displacement threshold to filter out noise (in mm)
    // Movements smaller than this are likely just measurement noise
    const float displacementThreshold = 75.0f; // 5mm threshold, adjust as needed
    
    for (size_t i = 0; i < n - 1; i++) {
        // Calculate total displacement magnitude
        float totalDisplacement = std::sqrt(
            displacements[i][0] * displacements[i][0] +
            displacements[i][1] * displacements[i][1] +
            0.5f * displacements[i][2] * displacements[i][2]
        );
        
        // Only calculate velocity if displacement is above threshold
        if (totalDisplacement > displacementThreshold) {
            velocities[i][0] = displacements[i][0] / timeDiffs[i];  // vx
            velocities[i][1] = displacements[i][1] / timeDiffs[i];  // vy
            velocities[i][2] = displacements[i][2] / timeDiffs[i];  // vz
            
            // Calculate speed (magnitude of velocity)
            speeds[i] = std::sqrt(
                velocities[i][0] * velocities[i][0] +
                velocities[i][1] * velocities[i][1] +
                velocities[i][2] * velocities[i][2]
            );
        } else {
            // If displacement is below threshold, set velocity and speed to zero
            velocities[i][0] = 0.0f;
            velocities[i][1] = 0.0f;
            velocities[i][2] = 0.0f;
            speeds[i] = 0.0f;
        }
    }
    
    // Calculate initial velocity (using first few frames for stability)
    // Use up to 5 frames or as many as available
    size_t nFramesForInitial = std::min(static_cast<size_t>(5), velocities.size());
    std::vector<float> initialVelocity(3, 0.0f);
    size_t nonZeroFrames = 0;
    
    for (size_t i = 0; i < nFramesForInitial; i++) {
        // Only count frames with non-zero velocity
        if (speeds[i] > 0.0f) {
            initialVelocity[0] += velocities[i][0];
            initialVelocity[1] += velocities[i][1];
            initialVelocity[2] += velocities[i][2];
            nonZeroFrames++;
        }
    }
    
    if (nonZeroFrames > 0) {
        initialVelocity[0] /= nonZeroFrames;
        initialVelocity[1] /= nonZeroFrames;
        initialVelocity[2] /= nonZeroFrames;
    }
    
    // Calculate initial speed (magnitude of initial velocity)
    float initialSpeed = std::sqrt(
        initialVelocity[0] * initialVelocity[0] +
        initialVelocity[1] * initialVelocity[1] +
        initialVelocity[2] * initialVelocity[2]
    );
    
    // Calculate launch angles
    float launchAngleXZ = 0.0f;
    float launchAngleYZ = 0.0f;
    
    if (initialSpeed > 0) {
        // XZ plane (horizontal angle)
        launchAngleXZ = std::atan2(initialVelocity[0], initialVelocity[2]) * 180.0f / M_PI;
        
        // YZ plane (vertical angle - launch angle)
        float horizontalComponent = std::sqrt(
            initialVelocity[0] * initialVelocity[0] +
            initialVelocity[2] * initialVelocity[2]
        );
        launchAngleYZ = std::atan2(initialVelocity[1], horizontalComponent) * 180.0f / M_PI;
    }
    
    // Set the results in the parameters struct
    params.initialVelocity = initialVelocity;
    params.initialSpeed = initialSpeed;
    params.launchAngleXZ = launchAngleXZ;
    params.launchAngleYZ = launchAngleYZ;
    params.velocities = velocities;
    params.speeds = speeds;
    params.positions = positions;
    
    // Update the class member if requested
    if (updateMotionParams) {
        const_cast<StereoOnnxDetector*>(this)->mMotionParams = params;
    }
    
    return params;
}

MotionParameters StereoOnnxDetector::getMotionParameters() const {
    return mMotionParams;
}

bool StereoOnnxDetector::saveTrajectoryToCSV(const std::string& filename, bool withMotionParams) const {
    if (mPositions3D.empty() || mFrameIndices.empty()) {
        logMessage("No trajectory data to save", true);
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        logMessage("Error: Could not open file for writing: " + filename, true);
        return false;
    }
    
    // Write header
    file << "frame,time,x,y,z";
    if (withMotionParams) {
        file << ",vx,vy,vz,speed";
    }
    file << std::endl;
    
    // Write data for each frame
    for (size_t i = 0; i < mPositions3D.size(); i++) {
        // Write frame number and timestamp in seconds
        float timeInSeconds = static_cast<float>(mFrameIndices[i]) / mFrameRate;
        
        // Format with fixed precision
        file << std::fixed << std::setprecision(0) << mFrameIndices[i] << ","
             << std::fixed << std::setprecision(4) << timeInSeconds << ","
             << std::fixed << std::setprecision(2) 
             << mPositions3D[i].x << ","
             << mPositions3D[i].y << ","
             << mPositions3D[i].z;
        
        // Add velocity data if we have it and it's requested
        if (withMotionParams && !mMotionParams.velocities.empty()) {
            // Check if velocity data exists for this position
            if (i < mMotionParams.velocities.size()) {
                file << "," 
                     << std::fixed << std::setprecision(2)
                     << mMotionParams.velocities[i][0] << ","
                     << mMotionParams.velocities[i][1] << ","
                     << mMotionParams.velocities[i][2] << ","
                     << mMotionParams.speeds[i];
            } else {
                // No velocity data for this position
                file << ",,,,";
            }
        }
        
        file << std::endl;
    }
    
    file.close();
    logMessage("Trajectory saved to: " + filename, true);
    return true;
}

void StereoOnnxDetector::resetTracking() {
    mPositions3D.clear();
    mFrameIndices.clear();
    mFrameCounter = 0;
    
    // Reset motion parameters
    mMotionParams = MotionParameters();
}

std::vector<Point3D> StereoOnnxDetector::getAllPositions() const {
    return mPositions3D;
}

cv::Point2f StereoOnnxDetector::getMasterCenter() const {
    return mLastMasterCenter;
}

cv::Point2f StereoOnnxDetector::getSlaveCenter() const {
    return mLastSlaveCenter;
}

bool StereoOnnxDetector::isCalibrated() const {
    return mCalibrationLoaded;
}

void StereoOnnxDetector::setFrameRate(float frameRate) {
    if (frameRate <= 0.0f) {
        logMessage("Warning: Invalid frame rate value, must be positive. Using default of 30 FPS instead.", true);
        mFrameRate = 30.0f;
    } else {
        mFrameRate = frameRate;
        logMessage("Frame rate set to: " + std::to_string(mFrameRate) + " FPS", false);
    }
}

float StereoOnnxDetector::getFrameRate() const {
    return mFrameRate;
}

// Utility function to draw trajectory on a 3D plot
cv::Mat StereoOnnxDetector::createTrajectoryVisualization() const {
    if (mPositions3D.size() < 2) {
        return cv::Mat();
    }
    
    // Create an image to visualize trajectory
    cv::Mat visualization(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Find trajectory bounds
    Point3D minValues(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Point3D maxValues(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    
    for (const auto& point : mPositions3D) {
        minValues.x = std::min(minValues.x, point.x);
        minValues.y = std::min(minValues.y, point.y);
        minValues.z = std::min(minValues.z, point.z);
        
        maxValues.x = std::max(maxValues.x, point.x);
        maxValues.y = std::max(maxValues.y, point.y);
        maxValues.z = std::max(maxValues.z, point.z);
    }
    
    // Add margin
    Point3D range;
    range.x = maxValues.x - minValues.x;
    range.y = maxValues.y - minValues.y;
    range.z = maxValues.z - minValues.z;
    
    // Ensure minimum range to avoid division by zero
    range.x = std::max(range.x, 0.001f);
    range.y = std::max(range.y, 0.001f);
    range.z = std::max(range.z, 0.001f);
    
    // Scale factors for visualization
    // We'll create a top-down view (X-Z plane) and a side view (Y-Z plane)
    int margin = 50;
    int topDownWidth = 350;
    int topDownHeight = 500;
    int sideViewWidth = 350;
    int sideViewHeight = 500;
    
    // Draw top-down view (X-Z plane)
    cv::Rect topDownRegion(margin, margin, topDownWidth, topDownHeight);
    cv::rectangle(visualization, topDownRegion, cv::Scalar(50, 50, 50), -1);
    cv::rectangle(visualization, topDownRegion, cv::Scalar(100, 100, 100), 1);
    
    // Draw side view (Y-Z plane)
    cv::Rect sideViewRegion(margin + topDownWidth + margin, margin, sideViewWidth, sideViewHeight);
    cv::rectangle(visualization, sideViewRegion, cv::Scalar(50, 50, 50), -1);
    cv::rectangle(visualization, sideViewRegion, cv::Scalar(100, 100, 100), 1);
    
    // Add labels
    cv::putText(visualization, "Top View (X-Z)", cv::Point(margin, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    cv::putText(visualization, "Side View (Y-Z)", cv::Point(margin + topDownWidth + margin, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    
    // Draw axes for top-down view
    int axisLength = 50;
    cv::Point origin(margin + 50, margin + topDownHeight - 50);
    
    // X-axis
    cv::arrowedLine(visualization, origin, 
                   cv::Point(origin.x + axisLength, origin.y), 
                   cv::Scalar(0, 0, 255), 2);
    cv::putText(visualization, "X", 
                cv::Point(origin.x + axisLength + 5, origin.y + 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    
    // Z-axis
    cv::arrowedLine(visualization, origin, 
                   cv::Point(origin.x, origin.y - axisLength), 
                   cv::Scalar(255, 0, 0), 2);
    cv::putText(visualization, "Z", 
                cv::Point(origin.x - 15, origin.y - axisLength - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    
    // Draw axes for side view
    cv::Point sideOrigin(margin + topDownWidth + margin + 50, margin + sideViewHeight - 50);
    
    // Y-axis
    cv::arrowedLine(visualization, sideOrigin, 
                   cv::Point(sideOrigin.x + axisLength, sideOrigin.y), 
                   cv::Scalar(0, 255, 0), 2);
    cv::putText(visualization, "Y", 
                cv::Point(sideOrigin.x + axisLength + 5, sideOrigin.y + 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    
    // Z-axis (same direction as top-down view)
    cv::arrowedLine(visualization, sideOrigin, 
                   cv::Point(sideOrigin.x, sideOrigin.y - axisLength), 
                   cv::Scalar(255, 0, 0), 2);
    cv::putText(visualization, "Z", 
                cv::Point(sideOrigin.x - 15, sideOrigin.y - axisLength - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    
    // Draw trajectory in top-down view
    std::vector<cv::Point> topDownPoints;
    for (const auto& point : mPositions3D) {
        int x = origin.x + int((point.x - minValues.x) / range.x * (topDownWidth - 100));
        int z = origin.y - int((point.z - minValues.z) / range.z * (topDownHeight - 100));
        topDownPoints.push_back(cv::Point(x, z));
    }
    
    // Draw trajectory in side view
    std::vector<cv::Point> sideViewPoints;
    for (const auto& point : mPositions3D) {
        int y = sideOrigin.x + int((point.y - minValues.y) / range.y * (sideViewWidth - 100));
        int z = sideOrigin.y - int((point.z - minValues.z) / range.z * (sideViewHeight - 100));
        sideViewPoints.push_back(cv::Point(y, z));
    }
    
    // Draw connecting lines and points
    for (size_t i = 1; i < topDownPoints.size(); i++) {
        // Draw lines
        cv::line(visualization, topDownPoints[i-1], topDownPoints[i], cv::Scalar(0, 255, 255), 2);
        cv::line(visualization, sideViewPoints[i-1], sideViewPoints[i], cv::Scalar(0, 255, 255), 2);
        
        // Draw points
        cv::circle(visualization, topDownPoints[i-1], 3, cv::Scalar(255, 255, 0), -1);
        cv::circle(visualization, sideViewPoints[i-1], 3, cv::Scalar(255, 255, 0), -1);
    }
    
    // Draw the last point
    if (!topDownPoints.empty()) {
        cv::circle(visualization, topDownPoints.back(), 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(visualization, sideViewPoints.back(), 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // Add trajectory information
    if (mPositions3D.size() >= 2) {
        std::stringstream ssInfo;
        ssInfo << std::fixed << std::setprecision(2);
        ssInfo << "Trajectory Points: " << mPositions3D.size() << std::endl;
        ssInfo << "Velocity: " << mMotionParams.initialSpeed << " m/s";
        
        cv::putText(visualization, ssInfo.str(), cv::Point(margin, visualization.rows - 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    }
    
    return visualization;
} 