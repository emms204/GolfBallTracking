#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace common {

/**
 * @brief Utility functions for image processing
 */
class ImageUtils {
public:
    /**
     * @brief Resize an image with letterboxing to maintain aspect ratio
     * 
     * @param img Input image
     * @param target_size Target size (width, height)
     * @return cv::Mat Resized image
     */
    static cv::Mat letterboxResize(const cv::Mat& img, const cv::Size& target_size);
    
    /**
     * @brief Convert an image from BGR to RGB format
     * 
     * @param img Input image (BGR)
     * @return cv::Mat Output image (RGB)
     */
    static cv::Mat bgrToRgb(const cv::Mat& img);
    
    /**
     * @brief Normalize pixel values to 0-1 range
     * 
     * @param img Input image
     * @return cv::Mat Normalized image
     */
    static cv::Mat normalize(const cv::Mat& img);
    
    /**
     * @brief Draw detection results on an image
     * 
     * @param img Image to draw on
     * @param boxes Bounding boxes
     * @param scores Confidence scores
     * @param class_ids Class IDs
     * @param class_names Class names
     * @param conf_threshold Confidence threshold
     */
    static void drawDetections(
        cv::Mat& img,
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores,
        const std::vector<int>& class_ids,
        const std::vector<std::string>& class_names,
        float conf_threshold = 0.25
    );
};

} // namespace common 