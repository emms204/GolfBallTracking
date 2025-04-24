#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include "simple_onnx_detector.h"

struct Point3D {
    float x;
    float y;
    float z;
    
    Point3D() : x(0), y(0), z(0) {}
    Point3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

struct MotionParameters {
    std::vector<float> initialVelocity; // [vx, vy, vz]
    float initialSpeed;
    float launchAngleXZ;  // Horizontal angle (degrees)
    float launchAngleYZ;  // Vertical angle (degrees)
    std::vector<std::vector<float>> velocities;
    std::vector<float> speeds;
    std::vector<std::vector<float>> positions;
    
    MotionParameters() : 
        initialVelocity({0, 0, 0}), 
        initialSpeed(0), 
        launchAngleXZ(0), 
        launchAngleYZ(0) {}
};

/**
 * @brief Stereo ONNX Runtime based object detector with triangulation
 */
class StereoOnnxDetector {
public:
    /**
     * @brief Construct a new StereoOnnxDetector object
     * 
     * @param modelPath Path to the ONNX model file
     * @param classesPath Path to the class names file
     * @param stereoCalibFile Path to the stereo calibration file (YAML)
     * @param confThreshold Confidence threshold for detection (default: 0.25)
     * @param nmsThreshold NMS threshold for detection (default: 0.45)
     */
    StereoOnnxDetector(const std::string& modelPath, const std::string& classesPath, 
                     const std::string& stereoCalibFile, 
                     float confThreshold = 0.25, float nmsThreshold = 0.45);
    
    /**
     * @brief Destroy the StereoOnnxDetector object
     */
    ~StereoOnnxDetector();
    
    /**
     * @brief Process a pair of stereo frames
     * 
     * @param frameMaster Master camera frame
     * @param frameSlave Slave camera frame
     * @param showDetections Whether to draw detections on the output frames
     * @return cv::Mat Combined visualization frame
     */
    cv::Mat processFrames(const cv::Mat& frameMaster, const cv::Mat& frameSlave, bool showDetections = true);
    
    /**
     * @brief Get the latest 3D position
     * 
     * @return Point3D Last triangulated 3D position
     */
    Point3D getLastPosition() const;
    
    /**
     * @brief Get all 3D positions
     * 
     * @return std::vector<Point3D> All triangulated 3D positions
     */
    std::vector<Point3D> getAllPositions() const;
    
    /**
     * @brief Calculate motion parameters from tracked positions
     * 
     * @param frameRate Frame rate of the video
     * @param updateMotionParams Whether to update the stored motion parameters
     * @return MotionParameters Calculated motion parameters
     */
    MotionParameters calculateMotionParameters(float frameRate, bool updateMotionParams = true) const;
    
    /**
     * @brief Save 3D trajectory to CSV file
     * 
     * @param filename Output CSV file
     * @param withMotionParams Whether to include motion parameters
     * @return bool Success/failure
     */

    /** 
     * @brief Get the motion parameters
     * @return MotionParameters
     */
    MotionParameters getMotionParameters() const;
    
    bool saveTrajectoryToCSV(const std::string& filename, bool withMotionParams = true) const;
    
    /**
     * @brief Generate a 3D visualization of the trajectory
     * 
     * @param filename Output image file
     * @param withMotionParams Whether to include motion parameters
     * @return bool Success/failure
     */
    bool saveTrajectoryVisualization(const std::string& filename, bool withMotionParams = true) const;
    
    /**
     * @brief Reset tracking (clear trajectory data)
     */
    void resetTracking();
    
    /**
     * @brief Get latest master detection center point
     * @return cv::Point2f Center point (x, y) in master camera
     */
    cv::Point2f getMasterCenter() const;
    
    /**
     * @brief Get latest slave detection center point
     * @return cv::Point2f Center point (x, y) in slave camera
     */
    cv::Point2f getSlaveCenter() const;

    /**
     * @brief Set parameters for tracking
     * 
     * @param maxTrackingFrames Maximum number of frames to keep in tracking history
     * @param minConfidence Minimum confidence for a detection to be tracked
     */
    void setTrackingParameters(int maxTrackingFrames, float minConfidence);
    
    /**
     * @brief Get the current confidence threshold
     * 
     * @return float Current confidence threshold
     */
    float getConfidenceThreshold() const;
    
    /**
     * @brief Set the confidence threshold
     * 
     * @param threshold New confidence threshold
     */
    void setConfidenceThreshold(float threshold);
    
    /**
     * @brief Get the number of tracked positions
     * 
     * @return size_t Number of 3D positions tracked
     */
    size_t getTrackedPositionsCount() const;
    
    /**
     * @brief Export current trajectory data to JSON format
     * 
     * @param filename Output JSON filename
     * @param includeMotionParams Whether to include calculated motion parameters
     * @return bool Success/failure
     */
    bool exportTrajectoryToJSON(const std::string& filename, bool includeMotionParams = true) const;
    
    /**
     * @brief Check if stereo calibration is loaded
     * 
     * @return bool Whether calibration is loaded
     */
    bool isCalibrated() const;
    
    /**
     * @brief Create a visualization of the trajectory
     * 
     * @return cv::Mat Visualization image
     */
    cv::Mat createTrajectoryVisualization() const;

    /**
     * @brief Set the frame rate for motion calculations
     * 
     * @param frameRate Frame rate in frames per second
     */
    void setFrameRate(float frameRate);
    
    /**
     * @brief Get the current frame rate used for motion calculations
     * 
     * @return float Current frame rate
     */
    float getFrameRate() const;

private:
    // Detector objects for master and slave cameras
    std::unique_ptr<OnnxDetector> mMasterDetector;
    std::unique_ptr<OnnxDetector> mSlaveDetector;
    
    // Stereo calibration parameters
    cv::Mat mCameraMatrixMaster;
    cv::Mat mDistCoeffsMaster;
    cv::Mat mCameraMatrixSlave;
    cv::Mat mDistCoeffsSlave;
    cv::Mat mRotationMatrix;
    cv::Mat mTranslationVector;
    cv::Mat mEssentialMatrix;
    cv::Mat mFundamentalMatrix;
    cv::Mat mRectificationMaster;
    cv::Mat mRectificationSlave;
    cv::Mat mProjectionMaster;
    cv::Mat mProjectionSlave;
    cv::Mat mDisparityToDepthMatrix;
    cv::Rect mValidRoiMaster;
    cv::Rect mValidRoiSlave;
    
    // Rectification maps
    cv::Mat mMapXMaster, mMapYMaster;
    cv::Mat mMapXSlave, mMapYSlave;
    
    // Image size
    cv::Size mImageSize;
    
    // Tracking data
    std::vector<Point3D> mPositions3D;
    std::vector<int> mFrameIndices;
    cv::Point2f mLastMasterCenter;
    cv::Point2f mLastSlaveCenter;
    int mFrameCounter;
    
    // Motion parameters
    MotionParameters mMotionParams;
    
    // Detection parameters
    float mConfThreshold;
    float mNmsThreshold;
    
    // Whether calibration is loaded
    bool mCalibrationLoaded;
    
    // Frame rate for motion calculations
    float mFrameRate;
    
    /**
     * @brief Load stereo calibration parameters from a file
     * 
     * @param filename Path to the calibration file
     * @return bool Success/failure
     */
    bool loadStereoCalibration(const std::string& filename);
    
    /**
     * @brief Initialize rectification maps
     */
    void initRectificationMaps();
    
    /**
     * @brief Triangulate a 3D point from corresponding 2D points
     * 
     * @param pointMaster Point in master camera view
     * @param pointSlave Point in slave camera view
     * @return Point3D 3D point
     */
    Point3D triangulatePoint(const cv::Point2f& pointMaster, const cv::Point2f& pointSlave) const;
    
    /**
     * @brief Create visualization of stereo frames and detections
     * 
     * @param rectifiedMaster Rectified master frame
     * @param rectifiedSlave Rectified slave frame
     * @param centerMaster Center of detection in master frame
     * @param centerSlave Center of detection in slave frame
     * @param point3D Triangulated 3D point
     * @return cv::Mat Combined visualization
     */
    cv::Mat createVisualization(const cv::Mat& rectifiedMaster, const cv::Mat& rectifiedSlave,
                              const cv::Point2f& centerMaster, const cv::Point2f& centerSlave,
                              const Point3D& point3D) const;
    
    /**
     * @brief Find the center of a detected object
     * 
     * @param frame Input frame
     * @param box Bounding box
     * @return cv::Point2f Center point
     */
    cv::Point2f findObjectCenter(const cv::Mat& frame, const cv::Rect& box) const;
    
    /**
     * @brief Refine the center of a ball detection
     * 
     * @param frame Input frame
     * @param center Initial center point
     * @return cv::Point2f Refined center point
     */
    cv::Point2f refineBallCenter(const cv::Mat& frame, const cv::Rect& box) const;

    /**
     * @brief Draw 3D position information on image
     * 
     * @param image Image to draw on
     * @param point3D 3D point to display
     */
    void draw3DPosition(cv::Mat& image, const Point3D& point3D) const;
};

/**
 * @brief Draw 3D position information on an image
 * 
 * @param img Image to draw on
 * @param position 3D position
 * @param offset Offset from top-left corner
 */
void draw3DPosition(cv::Mat& img, const Point3D& position, const cv::Point& offset = cv::Point(20, 60)); 