#include "preprocessor.h"
// #include "image_utils.h" // Assuming common::ImageUtils exists if needed later
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> // For cv::FileStorage
#include <iostream>
#include <cmath> // For std::round

namespace detector {

Preprocessor::Preprocessor(const cv::Size& target_size)
    : target_size_(target_size),
      calibration_loaded_(false),
      maps_initialized_(false) {
    // Ensure target size is valid
    if (target_size_.width <= 0 || target_size_.height <= 0) {
        std::cerr << "Warning: Invalid target size provided to Preprocessor. Using 640x640." << std::endl;
        target_size_ = cv::Size(640, 640);
    }
}

bool Preprocessor::loadCalibration(const std::string& filename) {
    cv::FileStorage fs;
    try {
         fs.open(filename, cv::FileStorage::READ);
         if (!fs.isOpened()) {
            std::cerr << "Error: Could not open calibration file: " << filename << std::endl;
            calibration_loaded_ = false;
            maps_initialized_ = false;
            return false;
         }

         fs["camera_matrix"] >> camera_matrix_;
         // Use standard name "distortion_coefficients" for compatibility
         fs["distortion_coefficients"] >> dist_coeffs_;

         fs.release(); // Close the file

         if (camera_matrix_.empty() || camera_matrix_.size() != cv::Size(3, 3)) {
             std::cerr << "Error: Failed to read valid camera matrix (must be 3x3) from " << filename << std::endl;
             calibration_loaded_ = false;
             maps_initialized_ = false;
             return false;
         }
         if (dist_coeffs_.empty()) {
             std::cerr << "Warning: Failed to read distortion coefficients from " << filename << ". Assuming zero distortion." << std::endl;
             // Create zero distortion coefficients if none are found
             dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
         }

         std::cout << "Calibration loaded successfully from: " << filename << std::endl;
         calibration_loaded_ = true;
         maps_initialized_ = false; // Force reinitialization of maps on next use
         return true;

    } catch (const cv::Exception& e) {
        std::cerr << "Error reading calibration file '" << filename << "': " << e.what() << std::endl;
        if(fs.isOpened()) fs.release();
        calibration_loaded_ = false;
        maps_initialized_ = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error processing calibration file '" << filename << "': " << e.what() << std::endl;
         if(fs.isOpened()) fs.release();
        calibration_loaded_ = false;
        maps_initialized_ = false;
        return false;
    }
}

bool Preprocessor::loadCalibration(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    try {
        // Validate camera matrix
        if (camera_matrix.empty() || camera_matrix.size() != cv::Size(3, 3)) {
            std::cerr << "Error: Invalid camera matrix (must be 3x3)" << std::endl;
            calibration_loaded_ = false;
            maps_initialized_ = false;
            return false;
        }
        
        // Store the matrices
        camera_matrix_ = camera_matrix.clone();
        
        // Handle dist_coeffs
        if (dist_coeffs.empty()) {
            // Create zero distortion coefficients if none are provided
            dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        } else {
            dist_coeffs_ = dist_coeffs.clone();
        }
        
        std::cout << "Calibration loaded successfully from matrices" << std::endl;
        calibration_loaded_ = true;
        maps_initialized_ = false; // Force reinitialization of maps on next use
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error processing calibration matrices: " << e.what() << std::endl;
        calibration_loaded_ = false;
        maps_initialized_ = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error processing calibration matrices: " << e.what() << std::endl;
        calibration_loaded_ = false;
        maps_initialized_ = false;
        return false;
    }
}

cv::Mat Preprocessor::process(const cv::Mat& input, bool apply_undistortion) {
    if (input.empty()) {
        std::cerr << "ERROR: Input image to Preprocessor::process is empty." << std::endl;
        return cv::Mat();
    }
    
    std::cout << "DEBUG: Preprocessor::process - Input image: " << input.cols << "x" 
              << input.rows << ", channels=" << input.channels() 
              << ", type=" << input.type() << std::endl;
    std::cout << "DEBUG: Apply undistortion: " << (apply_undistortion ? "YES" : "NO") 
              << ", Calibration loaded: " << (calibration_loaded_ ? "YES" : "NO") << std::endl;

    cv::Mat current_image = input.clone();

    // 1. Undistortion (optional)
    if (apply_undistortion && calibration_loaded_) {
        std::cout << "DEBUG: Applying undistortion..." << std::endl;
        current_image = undistort(current_image);
        if (current_image.empty()) {
             std::cerr << "ERROR: Undistortion failed." << std::endl;
             return cv::Mat();
        }
        std::cout << "DEBUG: Undistortion completed successfully: " << current_image.cols 
                  << "x" << current_image.rows << std::endl;
    }

    // 2. Letterbox Resizing
    std::cout << "DEBUG: Applying letterbox resizing..." << std::endl;
    current_image = letterbox(current_image);
     if (current_image.empty()) {
         std::cerr << "ERROR: Letterboxing failed." << std::endl;
         return cv::Mat();
     }
    std::cout << "DEBUG: Letterboxing completed successfully: " << current_image.cols 
              << "x" << current_image.rows << std::endl;
    
    // 3. Color Conversion (BGR to RGB)
    std::cout << "DEBUG: Converting color space..." << std::endl;
    if (current_image.channels() == 3) {
         cv::cvtColor(current_image, current_image, cv::COLOR_BGR2RGB);
         std::cout << "DEBUG: Converted BGR to RGB" << std::endl;
    } else if (current_image.channels() == 1) {
        // Handle grayscale: convert to 3 channels RGB
        cv::cvtColor(current_image, current_image, cv::COLOR_GRAY2RGB);
        std::cout << "DEBUG: Converted grayscale to RGB" << std::endl;
    } else if (current_image.channels() != 3) {
         std::cerr << "ERROR: Input image must have 1 or 3 channels for processing. Found: " 
                   << current_image.channels() << std::endl;
         return cv::Mat();
    }

    // 4. Normalization (to float32, 0-1 range)
    std::cout << "DEBUG: Normalizing pixel values to 0-1 range..." << std::endl;
    cv::Mat normalized;
    // Ensure the image is CV_8U before converting to float and normalizing
    if (current_image.depth() != CV_8U) {
        std::cout << "DEBUG: Converting image from depth " << current_image.depth() 
                  << " to CV_8U before normalization" << std::endl;
        current_image.convertTo(current_image, CV_8U); // Example conversion if needed
    }
    
    // Convert to floating point with 3 channels (CV_32FC3)
    current_image.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    
    std::cout << "DEBUG: Normalized image type: " << normalized.type() 
              << " (should be " << CV_32FC3 << " for 3-channel float)" << std::endl;
    
    current_image = normalized;
    std::cout << "DEBUG: Normalization completed. Output type: " << current_image.type() << std::endl;

    return current_image;
}

const Preprocessor::LetterboxInfo& Preprocessor::getLetterboxInfo() const {
    return last_letterbox_info_;
}

cv::Rect Preprocessor::scaleBoxToOriginal(const cv::Rect2f& box_letterboxed) const {
    // Check if scale is valid
    if (last_letterbox_info_.scale <= 1e-6) { // Use a small epsilon for float comparison
        std::cerr << "Warning: Invalid or zero scale factor in LetterboxInfo. Returning original box." << std::endl;
        return cv::Rect(static_cast<int>(std::round(box_letterboxed.x)),
                        static_cast<int>(std::round(box_letterboxed.y)),
                        static_cast<int>(std::round(box_letterboxed.width)),
                        static_cast<int>(std::round(box_letterboxed.height)));
    }

    float x_lb = box_letterboxed.x;
    float y_lb = box_letterboxed.y;
    float w_lb = box_letterboxed.width;
    float h_lb = box_letterboxed.height;

    // Check if coordinates are normalized (0-1 range) relative to the letterboxed image
    // A strict check (<= 1.0) might fail due to floating point inaccuracies
    // A more robust check would be needed if both normalized and absolute coordinates are expected.
    // Assuming coordinates are always relative to the letterboxed image (either normalized 0-1 or pixels).
    bool normalized = (x_lb <= 1.0f && y_lb <= 1.0f && w_lb <= 1.0f && h_lb <= 1.0f &&
                       (x_lb + w_lb) <= 1.01f && (y_lb + h_lb) <= 1.01f); // Allow slight tolerance

    if (normalized && last_letterbox_info_.target_width > 0 && last_letterbox_info_.target_height > 0) {
        // Convert normalized coordinates relative to letterboxed image to absolute pixel coordinates
        x_lb *= last_letterbox_info_.target_width;
        y_lb *= last_letterbox_info_.target_height;
        w_lb *= last_letterbox_info_.target_width;
        h_lb *= last_letterbox_info_.target_height;
    }
     // else: assume coordinates are already absolute pixel coordinates within the letterboxed image

    // Adjust for padding
    float x_unpadded = x_lb - last_letterbox_info_.pad_left;
    float y_unpadded = y_lb - last_letterbox_info_.pad_top;

    // Scale back to original image size
    float x_original = x_unpadded / last_letterbox_info_.scale;
    float y_original = y_unpadded / last_letterbox_info_.scale;
    float w_original = w_lb / last_letterbox_info_.scale;
    float h_original = h_lb / last_letterbox_info_.scale;

    // Clamp coordinates to original image boundaries
    // Ensure clamping uses the original dimensions stored in the info struct
    int orig_w = last_letterbox_info_.original_width;
    int orig_h = last_letterbox_info_.original_height;

    float x1_orig = std::max(0.0f, x_original);
    float y1_orig = std::max(0.0f, y_original);
    float x2_orig = std::min((float)orig_w, x_original + w_original);
    float y2_orig = std::min((float)orig_h, y_original + h_original);

    // Handle cases where clamping makes width/height zero or negative
    if (x1_orig >= x2_orig || y1_orig >= y2_orig) {
        return cv::Rect(0, 0, 0, 0); // Return an empty rectangle
    }

    // Round to nearest integer for the final cv::Rect
    return cv::Rect(static_cast<int>(std::round(x1_orig)),
                    static_cast<int>(std::round(y1_orig)),
                    static_cast<int>(std::round(x2_orig - x1_orig)),
                    static_cast<int>(std::round(y2_orig - y1_orig)));
}


cv::Rect Preprocessor::scaleBoxToOriginal(float x_center_lb, float y_center_lb, float width_lb, float height_lb) const {
    // Convert center coordinates relative to letterboxed image to top-left format
    std::cout << "DEBUG: scaleBoxToOriginal - Input (center format): x=" << x_center_lb 
              << ", y=" << y_center_lb << ", w=" << width_lb << ", h=" << height_lb << std::endl;
    
    float x_tl_lb = x_center_lb - width_lb / 2.0f;
    float y_tl_lb = y_center_lb - height_lb / 2.0f;
    
    std::cout << "DEBUG: Converted to top-left format: x=" << x_tl_lb 
              << ", y=" << y_tl_lb << ", w=" << width_lb << ", h=" << height_lb << std::endl;
    
    cv::Rect result = scaleBoxToOriginal(cv::Rect2f(x_tl_lb, y_tl_lb, width_lb, height_lb));
    
    std::cout << "DEBUG: Final scaled box: x=" << result.x << ", y=" << result.y 
              << ", w=" << result.width << ", h=" << result.height << std::endl;
    
    return result;
}


void Preprocessor::createDebugImages(const cv::Mat& original,
                                   cv::Mat& undistorted_debug,
                                   cv::Mat& letterboxed_debug) const {

    // Create undistorted debug image (if applicable)
    if (calibration_loaded_ && !original.empty()) {
         // Need to run undistort again if the input size potentially changed since last process()
         // Or rely on maps being initialized correctly for the size used in last process()
         if (!maps_initialized_ || undistort_map1_.size() != original.size()) {
             std::cerr << "Warning: Cannot create accurate undistorted debug image. Maps not initialized for original size." << std::endl;
             undistorted_debug = original.clone(); // Fallback
             cv::putText(undistorted_debug, "Undistortion Failed (Map Size Mismatch)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
         } else {
            cv::remap(original, undistorted_debug, undistort_map1_, undistort_map2_, cv::INTER_LINEAR);
         }
    } else {
        undistorted_debug = original.clone(); // No calibration loaded or empty input
        if (!calibration_loaded_) {
            cv::putText(undistorted_debug, "No Calibration Loaded", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
    }

    // Create letterboxed debug image (based on the state *after* last process call)
    // This requires the undistorted image (or original if no undistortion)
    cv::Mat source_for_letterbox = undistorted_debug;
    if (source_for_letterbox.empty()) {
        letterboxed_debug = cv::Mat::zeros(target_size_, CV_8UC3); // Empty gray image
        cv::putText(letterboxed_debug, "Input Empty", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        return;
    }

    // Reuse the letterbox function logic for visualization
    cv::Mat resized_for_debug;
    cv::resize(source_for_letterbox, resized_for_debug,
               cv::Size(last_letterbox_info_.resized_width, last_letterbox_info_.resized_height),
               0, 0, cv::INTER_LINEAR);

    // Ensure the debug image has 3 channels for drawing colors
    if (source_for_letterbox.channels() == 1) {
        cv::cvtColor(source_for_letterbox, source_for_letterbox, cv::COLOR_GRAY2BGR);
        cv::cvtColor(resized_for_debug, resized_for_debug, cv::COLOR_GRAY2BGR);
    }

    letterboxed_debug = cv::Mat(target_size_, source_for_letterbox.type(), cv::Scalar(114, 114, 114)); // Gray padding
    resized_for_debug.copyTo(letterboxed_debug(cv::Rect(last_letterbox_info_.pad_left,
                                                      last_letterbox_info_.pad_top,
                                                      last_letterbox_info_.resized_width,
                                                      last_letterbox_info_.resized_height)));

    // Draw padding area or scale info on the images
    cv::rectangle(letterboxed_debug,
                  cv::Rect(last_letterbox_info_.pad_left, last_letterbox_info_.pad_top,
                           last_letterbox_info_.resized_width, last_letterbox_info_.resized_height),
                  cv::Scalar(0, 255, 0), 2); // Green box showing scaled image area

    std::string scale_text = "Scale: " + std::to_string(last_letterbox_info_.scale);
    std::string pad_text = "Pad L/T: " + std::to_string(last_letterbox_info_.pad_left) + "," + std::to_string(last_letterbox_info_.pad_top);
    cv::putText(letterboxed_debug, scale_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(letterboxed_debug, pad_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

}

// --- Private Helper Functions ---

cv::Mat Preprocessor::letterbox(const cv::Mat& input) {
    std::cout << "DEBUG: Starting letterbox operation on image " << input.cols << "x" << input.rows << std::endl;
    
    last_letterbox_info_ = {}; // Reset info for the current operation
    last_letterbox_info_.original_width = input.cols;
    last_letterbox_info_.original_height = input.rows;
    last_letterbox_info_.target_width = target_size_.width;
    last_letterbox_info_.target_height = target_size_.height;

    if (input.empty() || target_size_.width <= 0 || target_size_.height <= 0) {
        std::cerr << "ERROR: Invalid input or target size for letterboxing." << std::endl;
        return cv::Mat();
    }

    // Calculate scale ratio (fit inside target dimensions)
    double ratio = std::min(static_cast<double>(target_size_.width) / input.cols,
                          static_cast<double>(target_size_.height) / input.rows);

    last_letterbox_info_.scale = ratio;
    // Use std::round for potentially better accuracy with integer casting
    last_letterbox_info_.resized_width = static_cast<int>(std::round(input.cols * ratio));
    last_letterbox_info_.resized_height = static_cast<int>(std::round(input.rows * ratio));
    
    std::cout << "DEBUG: Letterbox scale factor: " << ratio 
              << ", Resized dimensions: " << last_letterbox_info_.resized_width 
              << "x" << last_letterbox_info_.resized_height << std::endl;

    // Ensure resized dimensions are at least 1x1 if scale is very small but positive
    if (last_letterbox_info_.resized_width == 0) {
        std::cout << "DEBUG: Resized width was 0, setting to 1" << std::endl;
        last_letterbox_info_.resized_width = 1;
    }
    if (last_letterbox_info_.resized_height == 0) {
        std::cout << "DEBUG: Resized height was 0, setting to 1" << std::endl;
        last_letterbox_info_.resized_height = 1;
    }

    cv::Mat resized_img;
    try{
        cv::resize(input, resized_img, cv::Size(last_letterbox_info_.resized_width, last_letterbox_info_.resized_height), 0, 0, cv::INTER_LINEAR);
        std::cout << "DEBUG: Resized image to " << resized_img.cols << "x" << resized_img.rows << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: Exception during resize in letterbox: " << e.what() << std::endl;
        return cv::Mat();
    }

    // Calculate padding
    last_letterbox_info_.pad_left = (target_size_.width - last_letterbox_info_.resized_width) / 2;
    last_letterbox_info_.pad_top = (target_size_.height - last_letterbox_info_.resized_height) / 2;
    
    std::cout << "DEBUG: Padding: left=" << last_letterbox_info_.pad_left 
              << ", top=" << last_letterbox_info_.pad_top << std::endl;
    
    // Calculate remaining padding precisely to handle odd target dimensions
    int pad_right = target_size_.width - last_letterbox_info_.resized_width - last_letterbox_info_.pad_left;
    int pad_bottom = target_size_.height - last_letterbox_info_.resized_height - last_letterbox_info_.pad_top;

    // Apply padding
    cv::Mat letterboxed_img;
     try {
        cv::copyMakeBorder(resized_img, letterboxed_img, last_letterbox_info_.pad_top, pad_bottom,
                           last_letterbox_info_.pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)); // Standard gray padding
    } catch (const cv::Exception& e) {
        std::cerr << "Error during copyMakeBorder in letterbox: " << e.what() << std::endl;
        return cv::Mat();
    }

    // Sanity check: Ensure final image size matches target size
    if (letterboxed_img.cols != target_size_.width || letterboxed_img.rows != target_size_.height) {
        std::cerr << "Warning: Letterboxed image size (" << letterboxed_img.cols << "x" << letterboxed_img.rows
                  << ") does not match target size (" << target_size_.width << "x" << target_size_.height << "). Resizing forcefully." << std::endl;
         try {
            cv::resize(letterboxed_img, letterboxed_img, target_size_);
             // Note: Forceful resize invalidates the precise padding calculation.
             // Recalculating exact padding/scale info after forceful resize is non-trivial.
             // Update info struct to reflect the reality, though it might not be perfectly reversible.
             last_letterbox_info_.pad_left = 0;
             last_letterbox_info_.pad_top = 0;
             last_letterbox_info_.resized_width = target_size_.width;
             last_letterbox_info_.resized_height = target_size_.height;
             // Scale becomes less meaningful here, maybe set to average ratio?
             last_letterbox_info_.scale = (static_cast<double>(target_size_.width) / input.cols +
                                          static_cast<double>(target_size_.height) / input.rows) / 2.0;

         } catch (const cv::Exception& e) {
             std::cerr << "Error during forceful resize in letterbox: " << e.what() << std::endl;
             return cv::Mat();
         }
    }

    return letterboxed_img;
}

cv::Mat Preprocessor::undistort(const cv::Mat& input) {
    if (input.empty()) {
        std::cerr << "ERROR: Input image to undistort is empty" << std::endl;
        return cv::Mat();
    }

    if (!calibration_loaded_) {
        std::cerr << "ERROR: Cannot undistort image - calibration not loaded" << std::endl;
        return input.clone(); // Return input image unmodified
    }

    try {
        // Initialize undistortion maps if not already done
        if (!maps_initialized_ || undistort_map1_.empty() || undistort_map2_.empty()) {
            std::cout << "DEBUG: Initializing undistortion maps..." << std::endl;
            
            // Get optimal new camera matrix
            cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
                camera_matrix_, dist_coeffs_, cv::Size(input.cols, input.rows), 1.0);
            
            // Initialize undistortion maps
            cv::initUndistortRectifyMap(
                camera_matrix_, dist_coeffs_, cv::Mat(), new_camera_matrix,
                cv::Size(input.cols, input.rows), CV_32FC1, undistort_map1_, undistort_map2_);
            
            maps_initialized_ = true;
            std::cout << "DEBUG: Undistortion maps initialized for size " 
                      << input.cols << "x" << input.rows << std::endl;
        }

        // Apply undistortion
        cv::Mat undistorted;
        cv::remap(input, undistorted, undistort_map1_, undistort_map2_, cv::INTER_LINEAR);
        
        std::cout << "DEBUG: Undistortion applied successfully" << std::endl;
        return undistorted;
    }
    catch (const cv::Exception& e) {
        std::cerr << "ERROR: OpenCV exception during undistortion: " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: Exception during undistortion: " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (...) {
        std::cerr << "ERROR: Unknown exception during undistortion" << std::endl;
        return cv::Mat();
    }
}


} // namespace detector