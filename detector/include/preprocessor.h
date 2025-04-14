#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>

namespace common {
    // Forward declaration to avoid circular dependency if needed
    class CameraParams;
}

namespace detector {

/**
 * @brief Reusable preprocessor for object detection models
 *
 * Handles:
 * - Optional camera distortion correction
 * - Letterbox resizing (preserving aspect ratio with padding)
 * - Color conversion (BGR to RGB)
 * - Normalization (pixel values to 0-1 range)
 * - Tracking transformation parameters for coordinate scaling
 */
class Preprocessor {
public:
    /**
     * @brief Information about the letterbox transformation applied.
     */
    struct LetterboxInfo {
        double scale = 1.0;       ///< Scaling factor applied to the original image.
        int pad_left = 0;         ///< Padding added to the left side.
        int pad_top = 0;          ///< Padding added to the top side.
        int original_width = 0;   ///< Original image width.
        int original_height = 0;  ///< Original image height.
        int target_width = 0;     ///< Target width after letterboxing.
        int target_height = 0;    ///< Target height after letterboxing.
        int resized_width = 0;    ///< Width after scaling, before padding.
        int resized_height = 0;   ///< Height after scaling, before padding.
    };

    /**
     * @brief Constructor.
     *
     * @param target_size The target square size (width and height) for the model input.
     */
    explicit Preprocessor(const cv::Size& target_size);

    /**
     * @brief Load camera calibration parameters from a YAML/XML file.
     *
     * @param filename Path to the calibration file.
     * @return bool True if loading was successful, false otherwise.
     */
    bool loadCalibration(const std::string& filename);

    /**
     * @brief Load camera calibration parameters directly from matrices.
     *
     * @param camera_matrix Camera matrix (3x3).
     * @param dist_coeffs Distortion coefficients.
     * @return bool True if loading was successful, false otherwise.
     */
    bool loadCalibration(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

    /**
     * @brief Preprocess an image for model inference.
     *
     * Applies undistortion (if calibration loaded) and letterboxing.
     * Converts to RGB and normalizes.
     *
     * @param input Input image (BGR).
     * @param apply_undistortion If true, applies undistortion using loaded parameters.
     * @return cv::Mat Processed image (RGB, float32, normalized 0-1) ready for inference.
     */
    cv::Mat process(const cv::Mat& input, bool apply_undistortion = true);

    /**
     * @brief Get the letterbox transformation info from the last `process` call.
     *
     * @return const LetterboxInfo& Information about the last applied transformation.
     */
    const LetterboxInfo& getLetterboxInfo() const;

    /**
     * @brief Scale a bounding box from the model's output coordinates (relative to
     *        the letterboxed input) back to the original image coordinates.
     *
     * @param box_letterboxed Bounding box (x, y, w, h) in the letterboxed image coordinates.
     *                        Coordinates can be normalized (0-1) or absolute pixels.
     * @return cv::Rect Bounding box in the original image coordinate system.
     */
    cv::Rect scaleBoxToOriginal(const cv::Rect2f& box_letterboxed) const;
    cv::Rect scaleBoxToOriginal(float x_center, float y_center, float width, float height) const;

    /**
     * @brief Creates debug images showing the preprocessing steps.
     *
     * @param original Original input image.
     * @param undistorted Image after undistortion (if applied).
     * @param letterboxed Final letterboxed image.
     */
    void createDebugImages(const cv::Mat& original,
                           cv::Mat& undistorted_debug,
                           cv::Mat& letterboxed_debug) const;

private:
    /**
     * @brief Apply letterbox resizing to an image.
     *
     * @param input Input image.
     * @return cv::Mat Letterboxed image.
     */
    cv::Mat letterbox(const cv::Mat& input);

    /**
     * @brief Apply camera undistortion to an image.
     *
     * @param input Distorted input image.
     * @return cv::Mat Undistorted image.
     */
    cv::Mat undistort(const cv::Mat& input);

    // Member variables
    cv::Size target_size_;            ///< Target input size for the model.
    cv::Mat camera_matrix_;           ///< Intrinsic camera matrix (3x3).
    cv::Mat dist_coeffs_;             ///< Distortion coefficients.
    cv::Mat undistort_map1_;          ///< Undistortion map 1 (for cv::remap).
    cv::Mat undistort_map2_;          ///< Undistortion map 2 (for cv::remap).
    bool calibration_loaded_ = false; ///< Flag indicating if calibration is loaded.
    bool maps_initialized_ = false;   ///< Flag indicating if undistortion maps are ready.
    LetterboxInfo last_letterbox_info_; ///< Stores info from the last letterbox operation.
};

} // namespace detector 