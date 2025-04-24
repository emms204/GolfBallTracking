#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <onnxruntime_c_api.h>

// Forward declaration for ONNX Runtime types
struct OrtApi;
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtAllocator;
struct OrtMemoryInfo;
struct OrtValue;
struct OrtTensorTypeAndShapeInfo;
struct OrtStatus;

/**
 * @brief Simple ONNX Runtime based object detector
 */
class OnnxDetector {
private:
    const OrtApi* ort;
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* sessionOptions;
    OrtAllocator* allocator;
    
    std::vector<std::string> classes;
    float confThreshold;
    float nmsThreshold;
    int inputWidth;
    int inputHeight;
    
    std::string inputName;
    std::string outputName;
    bool hasBuiltInNms; // Flag to check if model has built-in NMS
    
    // Camera calibration parameters
    cv::Mat cameraMatrix;        // Camera intrinsic matrix (3x3)
    cv::Mat distCoeffs;          // Distortion coefficients
    cv::Mat undistortMap1;       // Undistortion map 1 (for cv::remap)
    cv::Mat undistortMap2;       // Undistortion map 2 (for cv::remap)
    bool calibrationLoaded;      // Flag indicating if calibration is loaded
    bool mapsInitialized;        // Flag indicating if undistortion maps are initialized
    
    // Frame counter for logging
    int frameCount;
    
    /**
     * @brief Initialize undistortion maps for efficient image correction
     * 
     * @param imageSize Size of the images to be corrected
     */
    void initUndistortMaps(const cv::Size& imageSize);

public:
    /**
     * @brief Construct a new OnnxDetector object
     * 
     * @param modelPath Path to the ONNX model file
     * @param classesPath Path to the class names file
     * @param confThreshold Confidence threshold for detection (default: 0.25)
     * @param nmsThreshold NMS threshold for detection (default: 0.45)
     */
    OnnxDetector(const std::string& modelPath, const std::string& classesPath, 
                float confThreshold = 0.25, float nmsThreshold = 0.45);
    
    /**
     * @brief Destroy the OnnxDetector object
     */
    ~OnnxDetector();
    
    /**
     * @brief Apply undistortion to an input image using precomputed maps
     * 
     * @param input Input image
     * @return cv::Mat Undistorted image
     */
    cv::Mat undistortImage(const cv::Mat& input);
    
    /**
     * @brief Load camera calibration parameters from a YAML/XML file
     * 
     * @param calibrationFile Path to the calibration file
     * @return bool True if loading was successful
     */
    bool loadCalibration(const std::string& calibrationFile);
    
    /**
     * @brief Set camera calibration parameters directly
     * 
     * @param cameraMatrix Camera matrix (3x3)
     * @param distCoeffs Distortion coefficients
     * @return bool True if parameters are valid
     */
    bool setCalibration(const cv::Mat& newCameraMatrix, const cv::Mat& newDistCoeffs);
    
    /**
     * @brief Check if calibration is loaded
     * 
     * @return bool True if calibration is loaded
     */
    bool hasCalibration() const;
    
    /**
     * @brief Detect objects in an image
     * 
     * @param frame Input image
     * @param confidences Output vector for detection confidences
     * @param classIds Output vector for detection class IDs
     * @param applyUndistortion Whether to apply camera undistortion (default: false)
     * @param enableLogging Whether to log detection details (default: true)
     * @return std::vector<cv::Rect> Detected bounding boxes
     */
    std::vector<cv::Rect> detect(cv::Mat& frame, std::vector<float>& confidences, 
                               std::vector<int>& classIds, bool applyUndistortion = false, 
                               bool enableLogging = true);
    
    /**
     * @brief Get class name from class ID
     * 
     * @param classId Class ID
     * @return std::string Class name
     */
    std::string getClassName(int classId) const;
};

void drawPredictions(cv::Mat& frame, const std::vector<cv::Rect>& boxes, 
                     const std::vector<float>& confidences, 
                     const std::vector<int>& classIds,
                     const OnnxDetector& detector);

std::string getCurrentTimeString(); 