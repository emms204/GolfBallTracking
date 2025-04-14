#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "preprocessor.h"
#include "camera_params.h"

namespace detector {

enum class OutputFormat {
    FORMAT_NMS,       // Output shape [batch, num_detections, 6]
    FORMAT_ALT,       // Output shape [num_detections, 6]
    FORMAT_LEGACY     // Output shape [1, num_classes+5, num_boxes]
};

/**
 * @brief Enhanced ONNX object detector with preprocessing
 */
class ONNXDetector {
public:
    /**
     * @brief Constructor
     */
    ONNXDetector();
    
    /**
     * @brief Destructor
     */
    ~ONNXDetector();
    
    /**
     * @brief Load model from file
     * 
     * @param model_path Path to ONNX model file
     * @param class_names_path Path to class names file
     * @return bool True if model loaded successfully
     */
    bool loadModel(const std::string& model_path, const std::string& class_names_path);
    
    /**
     * @brief Set camera parameters for distortion correction
     * 
     * @param camera_params Camera parameters
     */
    void setCameraParams(const common::CameraParams& camera_params);
    
    /**
     * @brief Set confidence threshold
     * 
     * @param conf_thresh Confidence threshold (0.0-1.0)
     */
    void setConfidenceThreshold(float conf_thresh);
    
    /**
     * @brief Set NMS threshold
     * 
     * @param nms_thresh NMS threshold (0.0-1.0)
     */
    void setNMSThreshold(float nms_thresh);
    
    /**
     * @brief Detect objects in an image
     * 
     * @param img Input image
     * @param apply_camera_correction Whether to apply camera distortion correction
     * @return cv::Mat Image with detection results drawn
     */
    cv::Mat detect(const cv::Mat& img, bool apply_camera_correction = false);
    
    /**
     * @brief Get detected boxes
     * 
     * @return std::vector<cv::Rect> Bounding boxes
     */
    std::vector<cv::Rect> getDetectedBoxes() const;
    
    /**
     * @brief Get detected scores
     * 
     * @return std::vector<float> Confidence scores
     */
    std::vector<float> getDetectedScores() const;
    
    /**
     * @brief Get detected class IDs
     * 
     * @return std::vector<int> Class IDs
     */
    std::vector<int> getDetectedClassIDs() const;
    
    /**
     * @brief Get class names
     * 
     * @return std::vector<std::string> Class names
     */
    std::vector<std::string> getClassNames() const;

private:
    // ONNX Runtime
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    // Model information
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_char_;
    std::vector<const char*> output_names_char_;
    std::vector<int64_t> input_dims_;
    OutputFormat output_format_;
    
    // Preprocessor
    Preprocessor preprocessor_;
    
    // Camera parameters
    common::CameraParams camera_params_;
    bool camera_params_has_data_ = false;
    
    // Detection parameters
    float conf_thresh_;
    float nms_thresh_;
    
    // Class information
    std::vector<std::string> class_names_;
    
    // Detection results
    std::vector<cv::Rect> detected_boxes_;
    std::vector<float> detected_scores_;
    std::vector<int> detected_class_ids_;
    
    /**
     * @brief Process model output
     * 
     * @param output_tensor Output tensor from ONNX Runtime
     * @param img_width Original image width
     * @param img_height Original image height
     */
    void processOutput(const Ort::Value& output_tensor, int img_width, int img_height);
    
    /**
     * @brief Apply non-maximum suppression
     * 
     * @param indices Indices of boxes to consider
     * @param confidences Confidence scores
     * @param num_values_per_box Number of values per detection in the output tensor
     * @param output_data Pointer to output tensor data
     * @param threshold NMS threshold
     * @param nms_indices Output vector for indices of kept boxes
     * @return std::vector<size_t> Indices of kept boxes (deprecated, use nms_indices)
     */
    std::vector<size_t> applyNMS(
        const std::vector<size_t>& indices,
        const std::vector<float>& confidences,
        size_t num_values_per_box,
        const float* output_data,
        float threshold,
        std::vector<int>& nms_indices
    );
};

} // namespace detector 