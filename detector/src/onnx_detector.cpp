#include "onnx_detector.h"
#include "image_utils.h"
#include <fstream>
#include <algorithm>

namespace detector {

ONNXDetector::ONNXDetector()
    : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXDetector"),
      preprocessor_(cv::Size(640, 640)),  // Default size, will be updated when model is loaded
      camera_params_has_data_(false),
      conf_thresh_(0.25f),
      nms_thresh_(0.45f),
      output_format_(OutputFormat::FORMAT_NMS) {
    
    // Configure session options
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
}

ONNXDetector::~ONNXDetector() {
    // Clean up char* vectors
    for (auto& name : input_names_char_) {
        delete[] name;
    }
    
    for (auto& name : output_names_char_) {
        delete[] name;
    }
}

bool ONNXDetector::loadModel(const std::string& model_path, const std::string& class_names_path) {
    try {
        // Load the model
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Get model information
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input information
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.resize(num_input_nodes);
        input_names_char_.resize(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            // Get input name
            input_names_[i] = session_->GetInputNameAllocated(i, allocator).get();
            
            // Allocate and copy name for API calls
            input_names_char_[i] = new char[input_names_[i].length() + 1];
            strcpy(const_cast<char*>(input_names_char_[i]), input_names_[i].c_str());
            
            // Get input shape
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_dims_ = tensor_info.GetShape();
            
            // Update preprocessor with correct input size
            if (input_dims_.size() >= 3) {
                int height = input_dims_[2] > 0 ? input_dims_[2] : 640;
                int width = input_dims_[3] > 0 ? input_dims_[3] : 640;
                preprocessor_ = Preprocessor(cv::Size(width, height));
            }
        }
        
        // Get output information
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.resize(num_output_nodes);
        output_names_char_.resize(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            // Get output name
            output_names_[i] = session_->GetOutputNameAllocated(i, allocator).get();
            
            // Allocate and copy name for API calls
            output_names_char_[i] = new char[output_names_[i].length() + 1];
            strcpy(const_cast<char*>(output_names_char_[i]), output_names_[i].c_str());
            
            // Determine output format from shape
            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto output_shape = tensor_info.GetShape();
            
            // Detect output format based on shape
            if (output_shape.size() == 3 && output_shape[2] == 6) {
                output_format_ = OutputFormat::FORMAT_NMS;
            } else if (output_shape.size() == 2 && output_shape[1] == 6) {
                output_format_ = OutputFormat::FORMAT_ALT;
            } else {
                output_format_ = OutputFormat::FORMAT_LEGACY;
            }
        }
        
        // Load class names
        std::ifstream file(class_names_path);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        class_names_.clear();
        while (std::getline(file, line)) {
            if (!line.empty()) {
                class_names_.push_back(line);
            }
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        // Handle ORT exception
        return false;
    } catch (const std::exception& e) {
        // Handle other exceptions
        return false;
    }
}

void ONNXDetector::setCameraParams(const common::CameraParams& camera_params) {
    camera_params_ = camera_params;
    camera_params_has_data_ = true;
    
    // If we have camera parameters, pass the calibration to preprocessor
    if (camera_params_has_data_) {
        // Get calibration matrix and distortion coefficients from camera_params
        cv::Mat camera_matrix = camera_params_.getCameraMatrix();
        cv::Mat dist_coeffs = camera_params_.getDistCoeffs();
        
        // Set these directly in the preprocessor
        // Assuming preprocessor has a method to set calibration directly
        if (!camera_matrix.empty() && !dist_coeffs.empty()) {
            preprocessor_.loadCalibration(camera_matrix, dist_coeffs);
        }
    }
}

void ONNXDetector::setConfidenceThreshold(float conf_thresh) {
    conf_thresh_ = conf_thresh;
}

void ONNXDetector::setNMSThreshold(float nms_thresh) {
    nms_thresh_ = nms_thresh;
}

cv::Mat ONNXDetector::detect(const cv::Mat& img, bool apply_camera_correction) {
    // Check if session is initialized
    if (!session_) {
        std::cerr << "ERROR: ONNX session is not initialized. Did you call loadModel()?" << std::endl;
        return img.clone();  // Return original image if model not loaded
    }
    
    // Check if input image is valid
    if (img.empty()) {
        std::cerr << "ERROR: Input image is empty" << std::endl;
        return img.clone();
    }
    
    std::cout << "DEBUG: Starting detection on image " << img.cols << "x" << img.rows 
              << " channels=" << img.channels() << std::endl;
    std::cout << "DEBUG: Using calibration: " << (apply_camera_correction ? "YES" : "NO") << std::endl;
    
    try {
        // Clear previous detections
        detected_boxes_.clear();
        detected_scores_.clear();
        detected_class_ids_.clear();
        
        // Preprocess the image using the Preprocessor
        std::cout << "DEBUG: Starting preprocessing..." << std::endl;
        cv::Mat processed;
        try {
            processed = preprocessor_.process(img, apply_camera_correction);
            if (processed.empty()) {
                std::cerr << "ERROR: Preprocessor returned empty image" << std::endl;
                return img.clone();
            }
            std::cout << "DEBUG: Preprocessing completed. Result: " << processed.cols << "x" 
                      << processed.rows << " channels=" << processed.channels() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception during preprocessing: " << e.what() << std::endl;
            return img.clone();
        }
        
        // Check if processed image has the expected format
        if (processed.channels() != 3) {
            std::cerr << "ERROR: Processed image should have 3 channels, but has " 
                      << processed.channels() << std::endl;
            return img.clone();
        }
        
        // Create input tensor
        std::cout << "DEBUG: Preparing input tensor..." << std::endl;
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(processed.cols * processed.rows * processed.channels());
        
        // HWC to CHW and copy to flat array using correct data type
        try {
            // Access data properly using Vec3f (3-channel float) for each pixel
            for (int c = 0; c < processed.channels(); c++) {
                for (int h = 0; h < processed.rows; h++) {
                    for (int w = 0; w < processed.cols; w++) {
                        // Access the whole pixel as Vec3f and then get the specific channel
                        input_tensor_values.push_back(processed.at<cv::Vec3f>(h, w)[c]);
                    }
                }
            }
            std::cout << "DEBUG: Input tensor prepared with " << input_tensor_values.size() 
                      << " values" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception during tensor preparation: " << e.what() << std::endl;
            return img.clone();
        }
        
        // Create input tensor object
        std::cout << "DEBUG: Creating ONNX tensor..." << std::endl;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Adjust input dimensions if needed
        std::vector<int64_t> input_tensor_shape = {1, 3, processed.rows, processed.cols};
        std::cout << "DEBUG: Input tensor shape: [" 
                  << input_tensor_shape[0] << ", " 
                  << input_tensor_shape[1] << ", " 
                  << input_tensor_shape[2] << ", " 
                  << input_tensor_shape[3] << "]" << std::endl;
        
        // Validate input tensor values
        if (input_tensor_values.empty()) {
            std::cerr << "ERROR: Input tensor values vector is empty" << std::endl;
            return img.clone();
        }
        
        // Check for NaN or Inf values in the tensor
        for (size_t i = 0; i < std::min(size_t(100), input_tensor_values.size()); i++) {
            if (std::isnan(input_tensor_values[i]) || std::isinf(input_tensor_values[i])) {
                std::cerr << "ERROR: Input tensor contains NaN or Inf values at index " << i 
                          << ": " << input_tensor_values[i] << std::endl;
                return img.clone();
            }
        }
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_tensor_shape.data(),
            input_tensor_shape.size()
        );
        
        // Check if ONNX names arrays are valid
        if (input_names_char_.empty() || output_names_char_.empty()) {
            std::cerr << "ERROR: Input or output names arrays are empty" << std::endl;
            return img.clone();
        }
        
        if (input_names_char_[0] == nullptr) {
            std::cerr << "ERROR: Input name is null" << std::endl;
            return img.clone();
        }
        
        for (size_t i = 0; i < output_names_char_.size(); i++) {
            if (output_names_char_[i] == nullptr) {
                std::cerr << "ERROR: Output name at index " << i << " is null" << std::endl;
                return img.clone();
            }
        }
        
        // Run inference
        std::cout << "DEBUG: Running ONNX inference..." << std::endl;
        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_char_.data(),
                &input_tensor,
                1,
                output_names_char_.data(),
                output_names_char_.size()
            );
            std::cout << "DEBUG: Inference completed successfully with " 
                      << output_tensors.size() << " output tensors" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "ERROR: ONNX Runtime exception during inference: " << e.what() << std::endl;
            return img.clone();
        }
        
        // Check if we got valid output
        if (output_tensors.empty()) {
            std::cerr << "ERROR: No output tensors returned from inference" << std::endl;
            return img.clone();
        }
        
        // Process output
        std::cout << "DEBUG: Processing output tensors..." << std::endl;
        try {
            processOutput(output_tensors[0], img.cols, img.rows);
            std::cout << "DEBUG: Output processing completed with " 
                      << detected_boxes_.size() << " detections" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception during output processing: " << e.what() << std::endl;
            return img.clone();
        }
        
        // Draw results on a copy of the original image
        std::cout << "DEBUG: Drawing detections on result image..." << std::endl;
        cv::Mat result = img.clone();
        try {
            common::ImageUtils::drawDetections(
                result,
                detected_boxes_,
                detected_scores_,
                detected_class_ids_,
                class_names_,
                conf_thresh_
            );
            std::cout << "DEBUG: Drawing completed successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception while drawing detections: " << e.what() << std::endl;
            return img.clone();
        }
        
        std::cout << "DEBUG: Detection process completed successfully" << std::endl;
        return result;
    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR: ONNX exception in detect(): " << e.what() << std::endl;
        return img.clone();
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: OpenCV exception in detect(): " << e.what() << std::endl;
        return img.clone();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Standard exception in detect(): " << e.what() << std::endl;
        return img.clone();
    } catch (...) {
        std::cerr << "ERROR: Unknown exception in detect()" << std::endl;
        return img.clone();
    }
}

std::vector<cv::Rect> ONNXDetector::getDetectedBoxes() const {
    return detected_boxes_;
}

std::vector<float> ONNXDetector::getDetectedScores() const {
    return detected_scores_;
}

std::vector<int> ONNXDetector::getDetectedClassIDs() const {
    return detected_class_ids_;
}

std::vector<std::string> ONNXDetector::getClassNames() const {
    return class_names_;
}

void ONNXDetector::processOutput(const Ort::Value& output_tensor, int img_width, int img_height) {
    // Clear previous detections
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    try {
        // Get tensor data
        auto tensor_type_shape = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_type_shape.GetShape();
        
        std::cout << "DEBUG: Processing output with shape dimensions: " << shape.size() << std::endl;
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << "DEBUG: Shape[" << i << "]: " << shape[i] << std::endl;
        }
        
        // Process based on detected format
        std::cout << "DEBUG: Using output format: " << 
            (output_format_ == OutputFormat::FORMAT_NMS ? "NMS" : 
             (output_format_ == OutputFormat::FORMAT_ALT ? "ALT" : "LEGACY")) << std::endl;
        
        if (output_format_ == OutputFormat::FORMAT_NMS || output_format_ == OutputFormat::FORMAT_ALT) {
            // Format with built-in NMS
            const float* output_data = output_tensor.GetTensorData<float>();
            if (!output_data) {
                std::cerr << "ERROR: Output tensor data is null" << std::endl;
                return;
            }
            
            size_t num_detections = (output_format_ == OutputFormat::FORMAT_NMS) ? shape[1] : shape[0];
            std::cout << "DEBUG: Processing " << num_detections << " detections" << std::endl;
            
            for (size_t i = 0; i < num_detections; i++) {
                size_t detection_idx = (output_format_ == OutputFormat::FORMAT_NMS) ? i * 6 : i * 6;
                
                // Bounds check before accessing data
                if (detection_idx + 5 >= tensor_type_shape.GetElementCount()) {
                    std::cerr << "ERROR: Detection index out of bounds: " << (detection_idx + 5) 
                              << " >= " << tensor_type_shape.GetElementCount() << std::endl;
                    break;
                }
                
                // Get detection data (format: [x, y, width, height, confidence, class_id])
                float confidence = output_data[detection_idx + 4];
                
                // Skip low confidence detections
                if (confidence < conf_thresh_) {
                    continue;
                }
                
                float x_center_lb = output_data[detection_idx];
                float y_center_lb = output_data[detection_idx + 1];
                float w_lb = output_data[detection_idx + 2];
                float h_lb = output_data[detection_idx + 3];
                int class_id = static_cast<int>(output_data[detection_idx + 5]);
                
                std::cout << "DEBUG: Raw detection " << i << ": [" 
                          << x_center_lb << ", " << y_center_lb << ", " 
                          << w_lb << ", " << h_lb << "], class=" << class_id 
                          << ", conf=" << confidence << std::endl;
                
                // Sanity check on values
                if (std::isnan(x_center_lb) || std::isnan(y_center_lb) || 
                    std::isnan(w_lb) || std::isnan(h_lb)) {
                    std::cerr << "ERROR: NaN values in detection" << std::endl;
                    continue;
                }
                
                // Check if class ID is in range
                if (class_id < 0 || class_id >= static_cast<int>(class_names_.size())) {
                    std::cerr << "ERROR: Class ID out of range: " << class_id 
                              << " (max: " << class_names_.size() - 1 << ")" << std::endl;
                    continue;
                }
                
                // Use preprocessor to scale box to original image coordinates
                try {
                    cv::Rect final_box = preprocessor_.scaleBoxToOriginal(x_center_lb, y_center_lb, w_lb, h_lb);
                    
                    // Sanity check on box
                    if (final_box.width <= 0 || final_box.height <= 0) {
                        std::cerr << "ERROR: Invalid box dimensions: " << final_box.width 
                                  << "x" << final_box.height << std::endl;
                        continue;
                    }
                    
                    std::cout << "DEBUG: Scaled box " << i << ": [" 
                              << final_box.x << ", " << final_box.y << ", " 
                              << final_box.width << ", " << final_box.height << "]" << std::endl;
                    
                    // Store detection
                    boxes.push_back(final_box);
                    scores.push_back(confidence);
                    class_ids.push_back(class_id);
                } catch (const std::exception& e) {
                    std::cerr << "ERROR: Exception scaling box: " << e.what() << std::endl;
                    continue;
                }
            }
        } else {
            // Legacy format
            const float* output_data = output_tensor.GetTensorData<float>();
            if (!output_data) {
                std::cerr << "ERROR: Output tensor data is null" << std::endl;
                return;
            }
            
            if (shape.size() < 3) {
                std::cerr << "ERROR: Unexpected output shape for legacy format: dimensions=" 
                          << shape.size() << std::endl;
                return;
            }
            
            size_t num_boxes = shape[1];
            size_t num_values_per_box = shape[2];
            
            std::cout << "DEBUG: Legacy format with " << num_boxes << " boxes, " 
                      << num_values_per_box << " values per box" << std::endl;
            
            // Verify we have enough values for class scores (at least x,y,w,h,obj_score + 1 class)
            if (num_values_per_box < 6) {
                std::cerr << "ERROR: Not enough values per box for class scores" << std::endl;
                return;
            }
            
            std::vector<float> confidences(num_boxes);
            std::vector<size_t> indices;
            
            // First pass: collect all confidence scores
            std::cout << "DEBUG: First pass - collecting confidence scores" << std::endl;
            for (size_t i = 0; i < num_boxes; ++i) {
                // Bounds check
                if ((i + 1) * num_values_per_box > tensor_type_shape.GetElementCount()) {
                    std::cerr << "ERROR: Box index out of bounds: " << (i + 1) * num_values_per_box 
                              << " > " << tensor_type_shape.GetElementCount() << std::endl;
                    break;
                }
                
                // Extract highest class score for each box
                // YOLO format: [x, y, w, h, obj_score, class1_score, class2_score, ...]
                float max_class_score = 0.0f;
                int max_class_id = 0;
                
                // Skip first 5 values (x, y, w, h, obj_score) and find highest class score
                for (size_t j = 5; j < num_values_per_box; ++j) {
                    size_t idx = i * num_values_per_box + j;
                    if (idx >= tensor_type_shape.GetElementCount()) {
                        std::cerr << "ERROR: Class score index out of bounds: " << idx 
                                  << " >= " << tensor_type_shape.GetElementCount() << std::endl;
                        break;
                    }
                    
                    float score = output_data[idx];
                    if (score > max_class_score) {
                        max_class_score = score;
                        max_class_id = static_cast<int>(j - 5);
                    }
                }
                
                // Confidence is obj_score * class_score
                float obj_score = output_data[i * num_values_per_box + 4];
                confidences[i] = obj_score * max_class_score;
                
                // Pre-filter to reduce NMS input size
                if (confidences[i] >= conf_thresh_) {
                    indices.push_back(i);
                }
            }
            
            std::cout << "DEBUG: Collected " << indices.size() 
                      << " boxes above confidence threshold" << std::endl;
            
            // Apply NMS
            std::vector<int> nms_indices;
            try {
                applyNMS(indices, confidences, num_values_per_box, output_data, nms_thresh_, nms_indices);
                std::cout << "DEBUG: After NMS, kept " << nms_indices.size() << " boxes" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Exception in NMS: " << e.what() << std::endl;
                return;
            }
            
            // Second pass: collect final detections after NMS
            std::cout << "DEBUG: Second pass - collecting final detections" << std::endl;
            for (int idx : nms_indices) {
                if (idx < 0 || idx >= static_cast<int>(num_boxes)) {
                    std::cerr << "ERROR: NMS index out of range: " << idx << std::endl;
                    continue;
                }
                
                float x_center_lb = output_data[idx * num_values_per_box];     // center x
                float y_center_lb = output_data[idx * num_values_per_box + 1]; // center y
                float w_lb = output_data[idx * num_values_per_box + 2];        // width
                float h_lb = output_data[idx * num_values_per_box + 3];        // height
                
                // Sanity check values
                if (std::isnan(x_center_lb) || std::isnan(y_center_lb) || 
                    std::isnan(w_lb) || std::isnan(h_lb)) {
                    std::cerr << "ERROR: NaN values in box coordinates" << std::endl;
                    continue;
                }
                
                // Find class with highest score
                float max_class_score = 0.0f;
                int max_class_id = 0;
                for (size_t j = 5; j < num_values_per_box; ++j) {
                    float score = output_data[idx * num_values_per_box + j];
                    if (score > max_class_score) {
                        max_class_score = score;
                        max_class_id = static_cast<int>(j - 5);
                    }
                }
                
                // Calculate final confidence
                float obj_score = output_data[idx * num_values_per_box + 4];
                float confidence = obj_score * max_class_score;
                
                // Check if class ID is in range
                if (max_class_id < 0 || max_class_id >= static_cast<int>(class_names_.size())) {
                    std::cerr << "ERROR: Class ID out of range: " << max_class_id << std::endl;
                    continue;
                }
                
                // Use preprocessor to scale box to original image coordinates
                try {
                    cv::Rect final_box = preprocessor_.scaleBoxToOriginal(x_center_lb, y_center_lb, w_lb, h_lb);
                    
                    // Sanity check on box
                    if (final_box.width <= 0 || final_box.height <= 0) {
                        std::cerr << "ERROR: Invalid box dimensions: " << final_box.width 
                                  << "x" << final_box.height << std::endl;
                        continue;
                    }
                    
                    // Store detection
                    boxes.push_back(final_box);
                    scores.push_back(confidence);
                    class_ids.push_back(max_class_id);
                } catch (const std::exception& e) {
                    std::cerr << "ERROR: Exception scaling box: " << e.what() << std::endl;
                    continue;
                }
            }
        }
        
        std::cout << "DEBUG: Processed " << boxes.size() << " final detections" << std::endl;
        
        // Update member variables with results
        detected_boxes_ = boxes;
        detected_scores_ = scores;
        detected_class_ids_ = class_ids;
    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR: ONNX exception in processOutput(): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in processOutput(): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception in processOutput()" << std::endl;
    }
}

std::vector<size_t> ONNXDetector::applyNMS(
    const std::vector<size_t>& indices,
    const std::vector<float>& confidences,
    size_t num_values_per_box,
    const float* output_data,
    float threshold,
    std::vector<int>& nms_indices
) {
    // Create list of indices for internal processing
    std::vector<size_t> remaining_indices = indices;
    
    // Sort indices by score
    std::sort(remaining_indices.begin(), remaining_indices.end(), [&confidences, &indices](size_t i1, size_t i2) {
        return confidences[indices[i1]] > confidences[indices[i2]];
    });
    
    std::vector<size_t> keep;
    
    while (!remaining_indices.empty()) {
        size_t current_idx = remaining_indices[0];
        size_t current = indices[current_idx];
        keep.push_back(current);
        
        std::vector<size_t> remaining;
        
        for (size_t i = 1; i < remaining_indices.size(); i++) {
            size_t idx_pos = remaining_indices[i];
            size_t idx = indices[idx_pos];
            
            // Skip boxes with different class IDs
            int current_class = static_cast<int>(output_data[indices[current_idx] * num_values_per_box + 5]);
            int idx_class = static_cast<int>(output_data[idx * num_values_per_box + 5]);
            if (current_class != idx_class) {
                remaining.push_back(idx_pos);
                continue;
            }
            
            // Extract coordinates for current box
            float x1_current = output_data[current * num_values_per_box] - output_data[current * num_values_per_box + 2] / 2;
            float y1_current = output_data[current * num_values_per_box + 1] - output_data[current * num_values_per_box + 3] / 2;
            float x2_current = x1_current + output_data[current * num_values_per_box + 2];
            float y2_current = y1_current + output_data[current * num_values_per_box + 3];
            
            // Extract coordinates for candidate box
            float x1_idx = output_data[idx * num_values_per_box] - output_data[idx * num_values_per_box + 2] / 2;
            float y1_idx = output_data[idx * num_values_per_box + 1] - output_data[idx * num_values_per_box + 3] / 2;
            float x2_idx = x1_idx + output_data[idx * num_values_per_box + 2];
            float y2_idx = y1_idx + output_data[idx * num_values_per_box + 3];
            
            // Calculate intersection coordinates
            float x1_inter = std::max(x1_current, x1_idx);
            float y1_inter = std::max(y1_current, y1_idx);
            float x2_inter = std::min(x2_current, x2_idx);
            float y2_inter = std::min(y2_current, y2_idx);
            
            // Calculate areas
            float width_inter = std::max(0.0f, x2_inter - x1_inter);
            float height_inter = std::max(0.0f, y2_inter - y1_inter);
            float area_inter = width_inter * height_inter;
            
            float area_current = (x2_current - x1_current) * (y2_current - y1_current);
            float area_idx = (x2_idx - x1_idx) * (y2_idx - y1_idx);
            float area_union = area_current + area_idx - area_inter;
            
            float iou = area_inter / area_union;
            
            if (iou <= threshold) {
                remaining.push_back(idx_pos);
            }
        }
        
        remaining_indices = remaining;
    }
    
    // Convert kept indices to output format
    nms_indices.clear();
    for (size_t i : keep) {
        nms_indices.push_back(static_cast<int>(i));
    }
    
    // Return the indices for backward compatibility
    return keep;
}

} // namespace detector 