#include "simple_onnx_detector.h"
#include "logging.h" // Include logging header
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <onnxruntime_c_api.h>


// Function to get the current time as a string (for logging)
inline std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void OnnxDetector::initUndistortMaps(const cv::Size& imageSize) {
    // Create undistortion maps if calibration is loaded and image size is valid
    if (calibrationLoaded && !cameraMatrix.empty() && !distCoeffs.empty() && 
        imageSize.width > 0 && imageSize.height > 0) {
        
        cv::initUndistortRectifyMap(
            cameraMatrix,
            distCoeffs,
            cv::Mat(),  // No rectification
            cameraMatrix,  // Use same camera matrix
            imageSize,
            CV_32FC1,
            undistortMap1,
            undistortMap2
        );
        
        mapsInitialized = true;
        logMessage("Undistortion maps initialized for image size " 
                  + std::to_string(imageSize.width) + "x" + std::to_string(imageSize.height));
    }
}

OnnxDetector::OnnxDetector(const std::string& modelPath, const std::string& classesPath, float confThreshold, float nmsThreshold) 
    : confThreshold(confThreshold), nmsThreshold(nmsThreshold), inputWidth(640), inputHeight(640), 
      hasBuiltInNms(true), calibrationLoaded(false), mapsInitialized(false), frameCount(0) {
    
    // Initialize ONNX Runtime
    const OrtApiBase* apiBase = OrtGetApiBase();
    ort = apiBase->GetApi(ORT_API_VERSION);
    
    // Initialize allocator
    OrtStatus* status = ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting allocator: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        throw std::runtime_error("Failed to get allocator");
    }
    
    // Create environment
    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OnnxDetector", &env);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error creating environment: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        throw std::runtime_error("Failed to create environment");
    }
    
    // Create session options
    status = ort->CreateSessionOptions(&sessionOptions);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error creating session options: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseEnv(env);
        throw std::runtime_error("Failed to create session options");
    }
    
    // Create session
#ifdef PLATFORM_WINDOWS
    // Windows requires UTF-16 (wchar_t*) strings for model paths
    // Convert narrow string to wide string
    std::wstring wideModelPath(modelPath.begin(), modelPath.end());
    status = ort->CreateSession(env, wideModelPath.c_str(), sessionOptions, &session);
#else
    // Linux/Mac use standard UTF-8 strings
    status = ort->CreateSession(env, modelPath.c_str(), sessionOptions, &session);
#endif

    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error creating session: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);
        throw std::runtime_error("Failed to create session");
    }
    
    logMessage("Successfully loaded model from: " + modelPath);
    
    // Get input and output names
    size_t numInputNodes;
    status = ort->SessionGetInputCount(session, &numInputNodes);
    if (status != nullptr || numInputNodes != 1) {
        logMessage("Error getting input count or unexpected number of inputs", true);
        if (status != nullptr) {
            ort->ReleaseStatus(status);
        }
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);
        throw std::runtime_error("Failed to get input count");
    }
    
    char* inputNameRaw;
    status = ort->SessionGetInputName(session, 0, allocator, &inputNameRaw);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting input name: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);
        throw std::runtime_error("Failed to get input name");
    }
    inputName = inputNameRaw;
    ort->AllocatorFree(allocator, inputNameRaw);
    
    size_t numOutputNodes;
    status = ort->SessionGetOutputCount(session, &numOutputNodes);
    if (status != nullptr || numOutputNodes != 1) {
        logMessage("Error getting output count or unexpected number of outputs", true);
        if (status != nullptr) {
            ort->ReleaseStatus(status);
        }
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);
        throw std::runtime_error("Failed to get output count");
    }
    
    char* outputNameRaw;
    status = ort->SessionGetOutputName(session, 0, allocator, &outputNameRaw);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting output name: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);
        throw std::runtime_error("Failed to get output name");
    }
    outputName = outputNameRaw;
    ort->AllocatorFree(allocator, outputNameRaw);
    
    logMessage("Model input name: " + inputName);
    logMessage("Model output name: " + outputName);
    
    // Load class names
    if (!classesPath.empty()) {
        std::ifstream ifs(classesPath);
        if (!ifs.is_open()) {
            logMessage("Failed to open classes file: " + classesPath, true);
            classes.push_back("object"); // Default class
        } else {
            std::string line;
            while (getline(ifs, line)) {
                classes.push_back(line);
            }
            logMessage("Loaded " + std::to_string(classes.size()) + " classes from: " + classesPath);
        }
    } else {
        classes.push_back("object"); // Default class
    }
    
    // Initialize camera calibration variables
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
}

OnnxDetector::~OnnxDetector() {
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(sessionOptions);
    ort->ReleaseEnv(env);
}

cv::Mat OnnxDetector::undistortImage(const cv::Mat& input) {
    cv::Mat undistorted;
    
    // Check if we have valid maps
    if (!calibrationLoaded || input.empty()) {
        return input.clone();  // Return original image
    }
    
    // Initialize maps if needed
    if (!mapsInitialized) {
        initUndistortMaps(input.size());
    }
    
    // Apply undistortion using maps
    if (mapsInitialized) {
        cv::remap(input, undistorted, undistortMap1, undistortMap2, cv::INTER_LINEAR);
        return undistorted;
    } else {
        // Fallback to standard undistort if maps aren't available
        cv::undistort(input, undistorted, cameraMatrix, distCoeffs);
        return undistorted;
    }
}

bool OnnxDetector::loadCalibration(const std::string& calibrationFile) {
    try {
        cv::FileStorage fs(calibrationFile, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            logMessage("Error: Could not open calibration file " + calibrationFile, true);
            return false;
        }
        
        // Read camera matrix and distortion coefficients
        fs["camera_matrix"] >> cameraMatrix;
        if (fs["distortion_coefficients"].isNone()) {
            fs["dist_coeffs"] >> distCoeffs; // Try alternative name
        } else {
            fs["distortion_coefficients"] >> distCoeffs;
        }
        
        // Check if we got valid data
        if (cameraMatrix.empty() || distCoeffs.empty()) {
            logMessage("Error: Invalid calibration data in " + calibrationFile, true);
            return false;
        }
        
        // Reset undistortion maps (they will be initialized on demand)
        undistortMap1 = cv::Mat();
        undistortMap2 = cv::Mat();
        mapsInitialized = false;
        calibrationLoaded = true;
        
        logMessage("Calibration loaded successfully from " + calibrationFile);
        
        return true;
    } catch (const cv::Exception& e) {
        logMessage("Error loading calibration: " + std::string(e.what()), true);
        return false;
    }
}

bool OnnxDetector::setCalibration(const cv::Mat& newCameraMatrix, const cv::Mat& newDistCoeffs) {
    if (newCameraMatrix.empty() || newDistCoeffs.empty()) {
        logMessage("Error: Invalid calibration parameters", true);
        return false;
    }
    
    // Copy parameters
    this->cameraMatrix = newCameraMatrix.clone();
    this->distCoeffs = newDistCoeffs.clone();
    
    // Reset maps so they'll be regenerated with correct size
    mapsInitialized = false;
    calibrationLoaded = true;
    
    logMessage("Camera calibration parameters set directly");
    return true;
}

bool OnnxDetector::hasCalibration() const {
    return calibrationLoaded;
}

std::vector<cv::Rect> OnnxDetector::detect(cv::Mat& frame, std::vector<float>& confidences, std::vector<int>& classIds, bool applyUndistortion, bool enableLogging) {
    std::vector<cv::Rect> boxes;
    
    // Increment frame counter
    frameCount++;
    
    // Apply undistortion if requested and calibration is available
    cv::Mat processedFrame = frame.clone();

    // Print image dimensions if logging is enabled
    if (enableLogging) {
        logMessage("\n================= FRAME " + std::to_string(frameCount) + " =================");
        logMessage("Image dimensions: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows));
    }

    if (applyUndistortion && calibrationLoaded) {
        processedFrame = undistortImage(frame);
        if (enableLogging) {
            logMessage("Applied undistortion...");
        }
    }
    
    // Prepare input tensor
    cv::Mat blob;
    cv::resize(processedFrame, blob, cv::Size(inputWidth, inputHeight));
    cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
    blob.convertTo(blob, CV_32F, 1.0/255.0);
    
    // NHWC to NCHW
    std::vector<float> inputTensorValues;
    inputTensorValues.reserve(inputWidth * inputHeight * 3);
    
    // Organize the data in NCHW format
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < inputHeight; h++) {
            for (int w = 0; w < inputWidth; w++) {
                inputTensorValues.push_back(blob.at<cv::Vec3f>(h, w)[c]);
            }
        }
    }
    
    // Create input tensor
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    OrtMemoryInfo* memoryInfo;
    OrtStatus* status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error creating memory info: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        return boxes;
    }
    
    OrtValue* inputTensor = nullptr;
    status = ort->CreateTensorWithDataAsOrtValue(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size() * sizeof(float),
        inputShape.data(), inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);
    
    ort->ReleaseMemoryInfo(memoryInfo);
    
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error creating input tensor: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        return boxes;
    }
    
    // Create output tensor
    OrtValue* outputTensor = nullptr;
    const char* inputNames[] = {inputName.c_str()};
    const char* outputNames[] = {outputName.c_str()};
    
    // Run inference
    status = ort->Run(session, nullptr, inputNames, (const OrtValue* const*)&inputTensor, 1, outputNames, 1, &outputTensor);
    ort->ReleaseValue(inputTensor);
    
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error running model: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        return boxes;
    }
    
    // Get output data
    float* outputData;
    status = ort->GetTensorMutableData(outputTensor, (void**)&outputData);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting output tensor data: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseValue(outputTensor);
        return boxes;
    }
    
    // Get output tensor shape
    OrtTensorTypeAndShapeInfo* outputInfo;
    status = ort->GetTensorTypeAndShape(outputTensor, &outputInfo);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting output tensor info: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseValue(outputTensor);
        return boxes;
    }
    
    size_t numDims;
    status = ort->GetDimensionsCount(outputInfo, &numDims);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting dimensions count: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
        ort->ReleaseValue(outputTensor);
        return boxes;
    }
    
    std::vector<int64_t> outputDims(numDims);
    status = ort->GetDimensions(outputInfo, outputDims.data(), numDims);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting dimensions: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
        ort->ReleaseValue(outputTensor);
        return boxes;
    }
    
    size_t outputSize;
    status = ort->GetTensorShapeElementCount(outputInfo, &outputSize);
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        logMessage("Error getting tensor shape element count: " + std::string(msg), true);
        ort->ReleaseStatus(status);
        ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
        ort->ReleaseValue(outputTensor);
        return boxes;
    }
    
    ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
    
    // Print output tensor shape for debugging if logging is enabled
    if (enableLogging) {
        logMessage("Output shape: ");
        for (size_t i = 0; i < numDims; i++) {
            logMessage(std::to_string(outputDims[i]) + " ");
        }
        logMessage("");
    }
    
    // Process detection output in the format matching YOLOv8's ONNX export
    // YOLOv8 ONNX output format is typically [batch, num_detections, 6]
    // where each detection is [x1, y1, x2, y2, confidence, class_id]
    
    int numClasses = classes.size();
    int numDetections = 0;
    
    if (numDims == 3 && outputDims[2] == 6) {
        // YOLOv8 format with NMS: [batch, num_detections, 6]
        numDetections = outputDims[1];
        
        if (enableLogging) {
            logMessage("Processing " + std::to_string(numDetections) + " detections (YOLOv8 with NMS)");
        }
        
        // Extract detections from batch 0
        float* batchData = outputData;
        
        // Count valid detections for logging
        if (enableLogging) {
            int validDetections = 0;
            for (int i = 0; i < numDetections; i++) {
                float confidence = batchData[i * 6 + 4];
                if (confidence > confThreshold) {
                    validDetections++;
                }
            }
            logMessage("Total detections: " + std::to_string(validDetections));
        }
        
        // Process each detection
        int detectionCount = 0;
        for (int i = 0; i < numDetections; i++) {
            // Get detection data - YOLOv8 ONNX format: [x1, y1, x2, y2, confidence, class_id]
            float x1 = batchData[i * 6 + 0];
            float y1 = batchData[i * 6 + 1];
            float x2 = batchData[i * 6 + 2];
            float y2 = batchData[i * 6 + 3];
            float confidence = batchData[i * 6 + 4];
            int classId = static_cast<int>(batchData[i * 6 + 5]);
            
            if (confidence > confThreshold && classId < numClasses) {
                // Calculate width and height
                float boxWidth = x2 - x1;
                float boxHeight = y2 - y1;
                
                // Calculate center point
                float centerX = x1 + boxWidth / 2;
                float centerY = y1 + boxHeight / 2;
                
                // Calculate ratios relative to frame dimensions
                float widthRatio = boxWidth / processedFrame.cols;
                float heightRatio = boxHeight / processedFrame.rows;
                float centerXRatio = centerX / processedFrame.cols;
                float centerYRatio = centerY / processedFrame.rows;
                
                // Log detection details if logging is enabled
                if (enableLogging) {
                    logMessage("Detection #" + std::to_string(detectionCount) + " (" + getClassName(classId) 
                              + ", conf=" + std::to_string(confidence) + "):");
                    logMessage("  Box coords (x,y,w,h): " + std::to_string(x1) + "," + std::to_string(y1) + "," 
                              + std::to_string(boxWidth) + "," + std::to_string(boxHeight));
                    logMessage("  Center point: (" + std::to_string(centerX) + "," + std::to_string(centerY) + ")");
                    logMessage("  Normalized center: (" + std::to_string(centerXRatio) + "," + std::to_string(centerYRatio) + ")");
                    logMessage("  Box size ratios (w,h): " + std::to_string(widthRatio) + "," + std::to_string(heightRatio));
                    logMessage("  Box area: " + std::to_string(boxWidth * boxHeight) + " pixels (" 
                              + std::to_string((boxWidth * boxHeight * 100.0f) / (processedFrame.cols * processedFrame.rows)) 
                              + "% of image)");
                }
                
                detectionCount++;
                
                // Scale coordinates back to original image size
                float scaleX = float(processedFrame.cols) / inputWidth;
                float scaleY = float(processedFrame.rows) / inputHeight;
                
                // Apply scaling to get coordinates in original image space
                int left = int(x1 * scaleX);
                int top = int(y1 * scaleY);
                int width = int(boxWidth * scaleX);
                int height = int(boxHeight * scaleY);
                
                // Debug the scaled values too if logging is enabled
                if (enableLogging) {
                    logMessage("SCALED BOX " + std::to_string(i) + ": (" + std::to_string(left) + "," + std::to_string(top) + ","
                              + std::to_string(width) + "," + std::to_string(height) + ")");
                }
                
                // Add the detection to results
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                classIds.push_back(classId);
            }
        }
    }
    else {
        logMessage("Unsupported output format. Expected 3D tensor with shape [batch, detections, 6]");
    }
    
    ort->ReleaseValue(outputTensor);
    
    // Debug output if logging is enabled
    if (enableLogging) {
        logMessage("========== DETECTIONS ==========");
        logMessage("Total detections: " + std::to_string(boxes.size()));
        
        // Print detection details
        for (size_t i = 0; i < std::min(boxes.size(), size_t(10)); ++i) {
            logMessage("Box " + std::to_string(i) + ": (" + std::to_string(boxes[i].x) + "," + std::to_string(boxes[i].y) + "," 
                      + std::to_string(boxes[i].width) + "," + std::to_string(boxes[i].height) + "), Class: " 
                      + std::to_string(classIds[i]) + " (" + getClassName(classIds[i]) + "), Conf: " 
                      + std::to_string(confidences[i]));
        }
        if (boxes.size() > 10) logMessage("... (showing first 10 only)");
        
        logMessage("Detected " + std::to_string(boxes.size()) + " objects");
    }
    
    return boxes;
}

std::string OnnxDetector::getClassName(int classId) const {
    if (classId >= 0 && classId < static_cast<int>(classes.size())) {
        return classes[classId];
    }
    return "unknown";
}

void drawPredictions(cv::Mat& frame, const std::vector<cv::Rect>& boxes, 
                     const std::vector<float>& confidences, 
                     const std::vector<int>& classIds,
                     const OnnxDetector& detector) {
    // Define colors for different classes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),     // Blue
        cv::Scalar(0, 255, 0),     // Green
        cv::Scalar(0, 0, 255),     // Red
        cv::Scalar(255, 255, 0),   // Cyan
        cv::Scalar(255, 0, 255),   // Magenta
        cv::Scalar(0, 255, 255),   // Yellow
    };
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::Rect box = boxes[i];
        
        // Ensure box is within frame boundaries
        box.x = std::max(box.x, 0);
        box.y = std::max(box.y, 0);
        box.width = std::min(box.width, frame.cols - box.x);
        box.height = std::min(box.height, frame.rows - box.y);
        
        // Get color for this class
        cv::Scalar color = colors[classIds[i] % colors.size()];
        
        // Draw bounding box
        cv::rectangle(frame, box, color, 2);
        
        // Draw label
        std::string className = detector.getClassName(classIds[i]);
        std::string label = className + ": " + cv::format("%.2f", confidences[i]);
        
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        cv::rectangle(frame, 
                     cv::Point(box.x, box.y - labelSize.height - baseLine - 5),
                     cv::Point(box.x + labelSize.width, box.y),
                     color, cv::FILLED);
        
        cv::putText(frame, label, cv::Point(box.x, box.y - baseLine - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

#ifdef INCLUDE_MAIN_FUNCTION
int main(int argc, char** argv) {
    // Initialize debug log file
    g_debugLogFile.open("debug.log", std::ios::out);
    if (!g_debugLogFile.is_open()) {
        logMessage("Warning: Could not open debug.log for writing");
    }
    
    // Parse command line arguments
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?  |      | Print this message}"
        "{video v         |      | Path to input video file}"
        "{camera c        |      | Use camera as input (specify device ID, e.g. 0)}"
        "{model m         |best.onnx| Path to ONNX model}"
        "{classes         |TrainResults1/classes.txt| Path to class names file (optional)}"
        "{conf            |0.25  | Confidence threshold}"
        "{nms             |0.45  | NMS threshold}"
        "{calibration     |      | Path to camera calibration file}"
        "{undistort       |true  | Apply undistortion correction}");
    
    if (parser.has("help") || (!parser.has("video") && !parser.has("camera"))) {
        parser.printMessage();
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
        return 1;
    }
    
    // Initialize video capture
    cv::VideoCapture cap;
    std::string inputSource;
    
    if (parser.has("video")) {
        std::string videoPath = parser.get<std::string>("video");
        cap.open(videoPath);
        inputSource = "Video: " + videoPath;
    } else if (parser.has("camera")) {
        int cameraId = parser.get<int>("camera");
        cap.open(cameraId);
        inputSource = "Camera: " + std::to_string(cameraId);
    }
    
    if (!cap.isOpened()) {
        logMessage("Error opening video source");
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
        return -1;
    }
    
    std::string modelPath = parser.get<std::string>("model");
    std::string classesPath = parser.get<std::string>("classes");
    float confThreshold = parser.get<float>("conf");
    float nmsThreshold = parser.get<float>("nms");
    std::string calibrationPath = parser.get<std::string>("calibration");
    bool applyUndistortion = parser.get<bool>("undistort");
    
    logMessage("=== Real-time ONNX Object Detector ===");
    logMessage("Time: " + getCurrentTimeString());
    logMessage("Input: " + inputSource);
    logMessage("Model: " + modelPath);
    logMessage("Classes: " + classesPath);
    logMessage("Confidence threshold: " + std::to_string(confThreshold));
    logMessage("NMS threshold: " + std::to_string(nmsThreshold));
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    logMessage("Video info: " + std::to_string(width) + "x" + std::to_string(height) + ", " 
              + std::to_string(fps) + " fps");
    
    // Create window for display
    const std::string windowName = "ONNX Object Detection";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, width, height);
    
    // Load the detector
    try {
        OnnxDetector detector(modelPath, classesPath, confThreshold, nmsThreshold);
        
        // Load calibration if provided
        bool calibrationLoaded = false;
        if (!calibrationPath.empty()) {
            logMessage("Loading camera calibration from: " + calibrationPath);
            calibrationLoaded = detector.loadCalibration(calibrationPath);
            
            if (calibrationLoaded) {
                logMessage("Camera calibration loaded successfully");
            } else {
                logMessage("Failed to load camera calibration", true);
                // We'll continue without calibration
                applyUndistortion = false;
            }
        }
        
        logMessage("Undistortion: " + std::string(applyUndistortion && calibrationLoaded ? "ON" : "OFF"));
        
        // Process video frames
        cv::Mat frame;
        int frameCount = 0;
        double totalFps = 0.0;
        
        while (true) {
            auto frameStartTime = std::chrono::high_resolution_clock::now();
            
            bool success = cap.read(frame);
            if (!success) {
                logMessage("\nEnd of video or camera disconnected");
                break;
            }
            
            // Process the frame (undistort if needed)
            cv::Mat processedFrame = frame.clone();
            
            // Print image dimensions
            logMessage("\n================= FRAME " + std::to_string(frameCount) + " =================");
            logMessage("Image dimensions: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows));
            
            if (applyUndistortion && calibrationLoaded) {
                processedFrame = detector.undistortImage(frame);
                logMessage("Applied undistortion");
            }
            
            // Detect objects
            std::vector<float> confidences;
            std::vector<int> classIds;
            std::vector<cv::Rect> boxes;
            
            boxes = detector.detect(processedFrame, confidences, classIds, false, true);
            
            // Print detailed information about detections and their relationship to image dimensions
            logMessage("---------------- DETECTION DETAILS ----------------");
            logMessage("Total detections: " + std::to_string(boxes.size()));
            
            for (size_t i = 0; i < boxes.size(); i++) {
                cv::Rect box = boxes[i];
                std::string className = detector.getClassName(classIds[i]);
                float confidence = confidences[i];
                
                // Calculate ratios of bounding box to image dimensions
                float boxWidth = static_cast<float>(box.width);
                float boxHeight = static_cast<float>(box.height);
                float widthRatio = boxWidth / processedFrame.cols;
                float heightRatio = boxHeight / processedFrame.rows;
                float centerX = box.x + boxWidth / 2;
                float centerY = box.y + boxHeight / 2;
                float centerXRatio = centerX / processedFrame.cols;
                float centerYRatio = centerY / processedFrame.rows;
                
                logMessage("Detection #" + std::to_string(i) + " (" + className + ", conf=" + std::to_string(confidence) + "):");
                logMessage("  Box coords (x,y,w,h): " + std::to_string(box.x) + "," + std::to_string(box.y) + "," 
                          + std::to_string(box.width) + "," + std::to_string(box.height));
                logMessage("  Center point: (" + std::to_string(centerX) + "," + std::to_string(centerY) + ")");
                logMessage("  Normalized center: (" + std::to_string(centerXRatio) + "," + std::to_string(centerYRatio) + ")");
                logMessage("  Box size ratios (w,h): " + std::to_string(widthRatio) + "," + std::to_string(heightRatio));
                logMessage("  Box area: " + std::to_string(box.area()) + " pixels (" 
                          + std::to_string((box.area() * 100.0f) / (processedFrame.cols * processedFrame.rows)) + "% of image)");
            }
            
            // Draw predictions on the processed frame
            drawPredictions(processedFrame, boxes, confidences, classIds, detector);
            
            // Calculate and display FPS
            auto frameEndTime = std::chrono::high_resolution_clock::now();
            auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                frameEndTime - frameStartTime).count();
            
            double currentFps = frameDuration > 0 ? 1000.0 / frameDuration : 0.0;
            totalFps += currentFps;
            frameCount++;
            
            std::string fpsText = "FPS: " + cv::format("%.1f", currentFps);
            cv::putText(processedFrame, fpsText, cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            
            // Add undistortion status to the processed frame
            if (calibrationLoaded) {
                std::string distText = "Undistort: " + std::string(applyUndistortion ? "ON" : "OFF");
                cv::putText(processedFrame, distText, cv::Point(10, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            
            // Display the processed frame
            cv::imshow(windowName, processedFrame);
            
            // Toggle undistortion on 'u' key (only if calibration is loaded)
            int key = cv::waitKey(1);
            if (key == 27) { // ESC key
                logMessage("\nExiting on user request");
                break;
            } else if (key == 'u' && calibrationLoaded) {
                // Toggle undistortion
                applyUndistortion = !applyUndistortion;
                logMessage("Undistortion " + std::string(applyUndistortion ? "enabled" : "disabled"));
            }
        }
        
        // Release resources
        cap.release();
        cv::destroyAllWindows();
        
        if (frameCount > 0) {
            double avgFps = totalFps / frameCount;
            logMessage("Average FPS: " + std::to_string(avgFps));
        }
        
        logMessage("Detection complete!");
        
    } catch (const std::exception& e) {
        logMessage("Error: " + std::string(e.what()), true);
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
        return -1;
    }
    
    // Close the debug log file
    if (g_debugLogFile.is_open()) {
        g_debugLogFile.close();
    }
    return 0;
}
#endif // INCLUDE_MAIN_FUNCTION