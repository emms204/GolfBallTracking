#include "image_utils.h"
#include <random>

namespace common {

cv::Mat ImageUtils::letterboxResize(const cv::Mat& img, const cv::Size& target_size) {
    // Calculate scaling factor to maintain aspect ratio
    float scale = std::min(
        static_cast<float>(target_size.width) / static_cast<float>(img.cols),
        static_cast<float>(target_size.height) / static_cast<float>(img.rows)
    );
    
    // Calculate new dimensions
    int scaled_width = static_cast<int>(img.cols * scale);
    int scaled_height = static_cast<int>(img.rows * scale);
    
    // Resize the image
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(scaled_width, scaled_height), 0, 0, cv::INTER_LINEAR);
    
    // Create a black canvas of the target size
    cv::Mat canvas = cv::Mat::zeros(target_size, img.type());
    
    // Calculate padding
    int dx = (target_size.width - scaled_width) / 2;
    int dy = (target_size.height - scaled_height) / 2;
    
    // Copy the resized image to the canvas with padding
    resized.copyTo(canvas(cv::Rect(dx, dy, scaled_width, scaled_height)));
    
    return canvas;
}

cv::Mat ImageUtils::bgrToRgb(const cv::Mat& img) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat ImageUtils::normalize(const cv::Mat& img) {
    cv::Mat normalized;
    img.convertTo(normalized, CV_32F, 1.0/255.0);
    return normalized;
}

void ImageUtils::drawDetections(
    cv::Mat& img,
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    const std::vector<std::string>& class_names,
    float conf_threshold
) {
    static std::vector<cv::Scalar> colors;
    
    // Initialize colors if needed
    if (colors.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        
        for (size_t i = 0; i < class_names.size(); ++i) {
            colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
        }
    }
    
    // Draw bounding boxes and labels
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (scores[i] >= conf_threshold) {
            int class_id = class_ids[i];
            cv::Scalar color = colors[class_id % colors.size()];
            
            // Draw bounding box
            cv::rectangle(img, boxes[i], color, 2);
            
            // Create label
            std::string label = class_names[class_id] + ": " + 
                                std::to_string(static_cast<int>(scores[i] * 100)) + "%";
            
            // Draw label background
            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(img, 
                         cv::Point(boxes[i].x, boxes[i].y - label_size.height - baseline - 5), 
                         cv::Point(boxes[i].x + label_size.width, boxes[i].y), 
                         color, 
                         cv::FILLED);
            
            // Draw label text
            cv::putText(img, 
                       label, 
                       cv::Point(boxes[i].x, boxes[i].y - baseline - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       cv::Scalar(255, 255, 255), 
                       1);
        }
    }
}

} // namespace common 