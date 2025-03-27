#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <onnxruntime_c_api.h>

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

public:
    OnnxDetector(const std::string& modelPath, const std::string& classesPath, float confThreshold = 0.25, float nmsThreshold = 0.45) 
        : confThreshold(confThreshold), nmsThreshold(nmsThreshold), inputWidth(640), inputHeight(480), hasBuiltInNms(true) {
        
        // Initialize ONNX Runtime
        const OrtApiBase* apiBase = OrtGetApiBase();
        ort = apiBase->GetApi(ORT_API_VERSION);
        
        // Initialize allocator
        OrtStatus* status = ort->GetAllocatorWithDefaultOptions(&allocator);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error getting allocator: " << msg << std::endl;
            ort->ReleaseStatus(status);
            throw std::runtime_error("Failed to get allocator");
        }
        
        // Create environment
        status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OnnxDetector", &env);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error creating environment: " << msg << std::endl;
            ort->ReleaseStatus(status);
            throw std::runtime_error("Failed to create environment");
        }
        
        // Create session options
        status = ort->CreateSessionOptions(&sessionOptions);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error creating session options: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseEnv(env);
            throw std::runtime_error("Failed to create session options");
        }
        
        // Create session
        status = ort->CreateSession(env, modelPath.c_str(), sessionOptions, &session);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error creating session: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseSessionOptions(sessionOptions);
            ort->ReleaseEnv(env);
            throw std::runtime_error("Failed to create session");
        }
        
        std::cout << "Successfully loaded model from: " << modelPath << std::endl;
        
        // Get input and output names
        size_t numInputNodes;
        status = ort->SessionGetInputCount(session, &numInputNodes);
        if (status != nullptr || numInputNodes != 1) {
            std::cerr << "Error getting input count or unexpected number of inputs" << std::endl;
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
            std::cerr << "Error getting input name: " << msg << std::endl;
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
            std::cerr << "Error getting output count or unexpected number of outputs" << std::endl;
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
            std::cerr << "Error getting output name: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseSession(session);
            ort->ReleaseSessionOptions(sessionOptions);
            ort->ReleaseEnv(env);
            throw std::runtime_error("Failed to get output name");
        }
        outputName = outputNameRaw;
        ort->AllocatorFree(allocator, outputNameRaw);
        
        std::cout << "Model input name: " << inputName << std::endl;
        std::cout << "Model output name: " << outputName << std::endl;
        
        // Load class names
        if (!classesPath.empty()) {
            std::ifstream ifs(classesPath);
            if (!ifs.is_open()) {
                std::cerr << "Failed to open classes file: " << classesPath << std::endl;
                classes.push_back("object"); // Default class
            } else {
                std::string line;
                while (getline(ifs, line)) {
                    classes.push_back(line);
                }
                std::cout << "Loaded " << classes.size() << " classes from: " << classesPath << std::endl;
            }
        } else {
            classes.push_back("object"); // Default class
        }
    }
    
    ~OnnxDetector() {
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);
    }
    
    std::vector<cv::Rect> detect(cv::Mat& frame, std::vector<float>& confidences, std::vector<int>& classIds) {
        std::vector<cv::Rect> boxes;
        
        // Prepare input tensor
        cv::Mat blob;
        cv::resize(frame, blob, cv::Size(inputWidth, inputHeight));
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
            std::cerr << "Error creating memory info: " << msg << std::endl;
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
            std::cerr << "Error creating input tensor: " << msg << std::endl;
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
            std::cerr << "Error running model: " << msg << std::endl;
            ort->ReleaseStatus(status);
            return boxes;
        }
        
        // Get output data
        float* outputData;
        status = ort->GetTensorMutableData(outputTensor, (void**)&outputData);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error getting output tensor data: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseValue(outputTensor);
            return boxes;
        }
        
        // Get output tensor shape
        OrtTensorTypeAndShapeInfo* outputInfo;
        status = ort->GetTensorTypeAndShape(outputTensor, &outputInfo);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error getting output tensor info: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseValue(outputTensor);
            return boxes;
        }
        
        size_t numDims;
        status = ort->GetDimensionsCount(outputInfo, &numDims);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error getting dimensions count: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
            ort->ReleaseValue(outputTensor);
            return boxes;
        }
        
        std::vector<int64_t> outputDims(numDims);
        status = ort->GetDimensions(outputInfo, outputDims.data(), numDims);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error getting dimensions: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
            ort->ReleaseValue(outputTensor);
            return boxes;
        }
        
        size_t outputSize;
        status = ort->GetTensorShapeElementCount(outputInfo, &outputSize);
        if (status != nullptr) {
            const char* msg = ort->GetErrorMessage(status);
            std::cerr << "Error getting tensor shape element count: " << msg << std::endl;
            ort->ReleaseStatus(status);
            ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
            ort->ReleaseValue(outputTensor);
            return boxes;
        }
        
        ort->ReleaseTensorTypeAndShapeInfo(outputInfo);
        
        // Print output tensor shape for debugging
        std::cout << "Output shape: ";
        for (size_t i = 0; i < numDims; i++) {
            std::cout << outputDims[i] << " ";
        }
        std::cout << std::endl;
        
        // Process output data based on YOLOv11 format with built-in NMS
        // The output from YOLOv11 with NMS enabled is typically [batch, num_detections, 6]
        // where each detection is [x, y, w, h, confidence, class_id]
        
        int numClasses = classes.size();
        int numDetections = 0;
        
        if (numDims == 3 && outputDims[2] == 6) {
            // YOLOv11 format with NMS: [batch, num_detections, 6]
            numDetections = outputDims[1];
            
            std::cout << "Processing " << numDetections << " detections (YOLOv11 with NMS)" << std::endl;
            
            for (int i = 0; i < numDetections; i++) {
                float x = outputData[i * 6 + 0];
                float y = outputData[i * 6 + 1];
                float w = outputData[i * 6 + 2];
                float h = outputData[i * 6 + 3];
                float confidence = outputData[i * 6 + 4];
                int classId = static_cast<int>(outputData[i * 6 + 5]);
                
                if (confidence > confThreshold && classId < numClasses) {
                    // Add detailed debugging - print raw values
                    std::cout << "RAW BOX " << i << ": center=(" << x << "," << y << "), dims=(" 
                              << w << "," << h << "), conf=" << confidence 
                              << ", class=" << classId << std::endl;
                    
                    // Convert normalized coordinates to pixel coordinates
                    float scaleX = float(frame.cols) / inputWidth;
                    float scaleY = float(frame.rows) / inputHeight;
                    
                    int left = int((x - w/2) * scaleX);
                    int top = int((y - h/2) * scaleY);
                    int width = int(w * scaleX);
                    int height = int(h * scaleY);
                    
                    // Debug the scaled values too
                    std::cout << "SCALED BOX " << i << ": (" << left << "," << top << ","
                              << width << "," << height << ")" << std::endl;
                    
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                    classIds.push_back(classId);
                }
            }
        }
        else if (numDims == 2 && outputDims[1] == 6) {
            // Alternative format: [num_detections, 6]
            numDetections = outputDims[0];
            
            std::cout << "Processing " << numDetections << " detections (flattened output)" << std::endl;
            
            for (int i = 0; i < numDetections; i++) {
                float x = outputData[i * 6 + 0];
                float y = outputData[i * 6 + 1];
                float w = outputData[i * 6 + 2];
                float h = outputData[i * 6 + 3];
                float confidence = outputData[i * 6 + 4];
                int classId = static_cast<int>(outputData[i * 6 + 5]);
                
                if (confidence > confThreshold && classId < numClasses) {
                    // Add detailed debugging - print raw values
                    std::cout << "RAW BOX " << i << ": center=(" << x << "," << y << "), dims=(" 
                              << w << "," << h << "), conf=" << confidence 
                              << ", class=" << classId << std::endl;
                    
                    // Convert normalized coordinates to pixel coordinates
                    float scaleX = float(frame.cols) / inputWidth;
                    float scaleY = float(frame.rows) / inputHeight;
                    
                    int left = int((x - w/2) * scaleX);
                    int top = int((y - h/2) * scaleY);
                    int width = int(w * scaleX);
                    int height = int(h * scaleY);
                    
                    // Debug the scaled values too
                    std::cout << "SCALED BOX " << i << ": (" << left << "," << top << ","
                              << width << "," << height << ")" << std::endl;
                    
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                    classIds.push_back(classId);
                }
            }
        }
        else {
            // Fall back to original code for other formats
            // Assuming the format is [1, num_classes+5, num_boxes] or similar
            if (numDims == 3) {
                // Typically [1, 84, num_boxes]
                numDetections = outputDims[2];
                int stride = outputDims[1];
                
                std::cout << "Processing " << numDetections << " detections (legacy format)" << std::endl;
                
                for (int i = 0; i < numDetections; i++) {
                    int baseIdx = i * stride;
                    
                    float x = outputData[baseIdx + 0];
                    float y = outputData[baseIdx + 1];
                    float w = outputData[baseIdx + 2];
                    float h = outputData[baseIdx + 3];
                    float objectness = outputData[baseIdx + 4];
                    
                    if (objectness > confThreshold) {
                        // Find best class
                        int classId = 0;
                        float maxScore = 0;
                        
                        for (int j = 0; j < numClasses && (j + 5) < stride; j++) {
                            float score = outputData[baseIdx + 5 + j];
                            if (score > maxScore) {
                                maxScore = score;
                                classId = j;
                            }
                        }
                        
                        // Final confidence
                        float confidence = objectness * maxScore;
                        
                        if (confidence > confThreshold) {
                            // Add detailed debugging - print raw values
                            std::cout << "RAW BOX " << i << ": center=(" << x << "," << y << "), dims=(" 
                                      << w << "," << h << "), conf=" << confidence 
                                      << ", class=" << classId << std::endl;
                            
                            // Convert normalized coordinates to pixel coordinates
                            float scaleX = float(frame.cols) / inputWidth;
                            float scaleY = float(frame.rows) / inputHeight;
                            
                            int left = int((x - w/2) * scaleX);
                            int top = int((y - h/2) * scaleY);
                            int width = int(w * scaleX);
                            int height = int(h * scaleY);
                            
                            // Debug the scaled values too
                            std::cout << "SCALED BOX " << i << ": (" << left << "," << top << ","
                                      << width << "," << height << ")" << std::endl;
                            
                            boxes.push_back(cv::Rect(left, top, width, height));
                            confidences.push_back(confidence);
                            classIds.push_back(classId);
                        }
                    }
                }
            } else if (numDims == 2) {
                // Alternative format: each row is a detection [x, y, w, h, confidence, class_scores...]
                numDetections = outputDims[0];
                int stride = outputDims[1];
                
                std::cout << "Processing " << numDetections << " detections (alternative format)" << std::endl;
                
                for (int i = 0; i < numDetections; i++) {
                    int baseIdx = i * stride;
                    
                    float x = outputData[baseIdx + 0];
                    float y = outputData[baseIdx + 1];
                    float w = outputData[baseIdx + 2];
                    float h = outputData[baseIdx + 3];
                    float objectness = outputData[baseIdx + 4];
                    
                    if (objectness > confThreshold) {
                        // Find best class
                        int classId = 0;
                        float maxScore = 0;
                        
                        for (int j = 0; j < numClasses && (j + 5) < stride; j++) {
                            float score = outputData[baseIdx + 5 + j];
                            if (score > maxScore) {
                                maxScore = score;
                                classId = j;
                            }
                        }
                        
                        // Final confidence
                        float confidence = objectness * maxScore;
                        
                        if (confidence > confThreshold) {
                            // Add detailed debugging - print raw values
                            std::cout << "RAW BOX " << i << ": center=(" << x << "," << y << "), dims=(" 
                                      << w << "," << h << "), conf=" << confidence 
                                      << ", class=" << classId << std::endl;
                            
                            // Convert normalized coordinates to pixel coordinates
                            float scaleX = float(frame.cols) / inputWidth;
                            float scaleY = float(frame.rows) / inputHeight;
                            
                            int left = int((x - w/2) * scaleX);
                            int top = int((y - h/2) * scaleY);
                            int width = int(w * scaleX);
                            int height = int(h * scaleY);
                            
                            // Debug the scaled values too
                            std::cout << "SCALED BOX " << i << ": (" << left << "," << top << ","
                                      << width << "," << height << ")" << std::endl;
                            
                            boxes.push_back(cv::Rect(left, top, width, height));
                            confidences.push_back(confidence);
                            classIds.push_back(classId);
                        }
                    }
                }
            } else {
                std::cerr << "Unexpected output tensor shape" << std::endl;
            }
        }
        
        ort->ReleaseValue(outputTensor);
        
        // Debug output
        std::cout << "========== DETECTIONS ==========" << std::endl;
        std::cout << "Total detections: " << boxes.size() << std::endl;
        
        // Since we're using a model with built-in NMS, we can skip the OpenCV NMS
        // unless hasBuiltInNms is false
        if (!hasBuiltInNms) {
            // Apply non-maximum suppression to remove overlapping boxes
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
            
            std::vector<cv::Rect> filteredBoxes;
            std::vector<float> filteredConfidences;
            std::vector<int> filteredClassIds;
            
            for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                filteredBoxes.push_back(boxes[idx]);
                filteredConfidences.push_back(confidences[idx]);
                filteredClassIds.push_back(classIds[idx]);
            }
            
            // Update the original vectors
            boxes = filteredBoxes;
            confidences = filteredConfidences;
            classIds = filteredClassIds;
        }
        
        // Print detection details
        for (size_t i = 0; i < std::min(boxes.size(), size_t(10)); ++i) {
            std::cout << "Box " << i << ": (" << boxes[i].x << "," << boxes[i].y << "," 
                      << boxes[i].width << "," << boxes[i].height << "), Class: " 
                      << classIds[i] << " (" << getClassName(classIds[i]) << "), Conf: " 
                      << confidences[i] << std::endl;
        }
        if (boxes.size() > 10) std::cout << "... (showing first 10 only)" << std::endl;
        
        std::cout << "Detected " << boxes.size() << " objects" << std::endl;
        
        return boxes;
    }
    
    std::string getClassName(int classId) const {
        if (classId >= 0 && classId < static_cast<int>(classes.size())) {
            return classes[classId];
        }
        return "unknown";
    }
};

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

// Function to get the current time as a string (for logging)
std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

int main(int argc, char** argv) {
    // Parse command line arguments
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?  |      | Print this message}"
        "{video v         |      | Path to input video file}"
        "{camera c        |      | Use camera as input (specify device ID, e.g. 0)}"
        "{model m         |best.onnx| Path to ONNX model}"
        "{classes         |TrainResults1/classes.txt| Path to class names file (optional)}"
        "{conf            |0.25  | Confidence threshold}"
        "{nms             |0.45  | NMS threshold}");
    
    if (parser.has("help") || (!parser.has("video") && !parser.has("camera"))) {
        parser.printMessage();
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
        std::cerr << "Error opening video source" << std::endl;
        return -1;
    }
    
    std::string modelPath = parser.get<std::string>("model");
    std::string classesPath = parser.get<std::string>("classes");
    float confThreshold = parser.get<float>("conf");
    float nmsThreshold = parser.get<float>("nms");
    
    std::cout << "=== Real-time ONNX Object Detector ===" << std::endl;
    std::cout << "Time: " << getCurrentTimeString() << std::endl;
    std::cout << "Input: " << inputSource << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Classes: " << classesPath << std::endl;
    std::cout << "Confidence threshold: " << confThreshold << std::endl;
    std::cout << "NMS threshold: " << nmsThreshold << std::endl;
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video info: " << width << "x" << height << ", " 
              << fps << " fps" << std::endl;
    
    // Create window for display
    const std::string windowName = "ONNX Object Detection";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, width, height);
    
    // Load the detector
    try {
        OnnxDetector detector(modelPath, classesPath, confThreshold, nmsThreshold);
        
        // Process video frames
        cv::Mat frame;
        int frameCount = 0;
        double totalFps = 0.0;
        
        while (true) {
            auto frameStartTime = std::chrono::high_resolution_clock::now();
            
            bool success = cap.read(frame);
            if (!success) {
                std::cout << "\nEnd of video or camera disconnected" << std::endl;
                break;
            }
            
            // Detect objects
            std::vector<float> confidences;
            std::vector<int> classIds;
            std::vector<cv::Rect> boxes = detector.detect(frame, confidences, classIds);
            
            // Draw predictions on the frame
            drawPredictions(frame, boxes, confidences, classIds, detector);
            
            // Calculate and display FPS
            auto frameEndTime = std::chrono::high_resolution_clock::now();
            auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                frameEndTime - frameStartTime).count();
            
            double currentFps = frameDuration > 0 ? 1000.0 / frameDuration : 0.0;
            totalFps += currentFps;
            frameCount++;
            
            std::string fpsText = "FPS: " + cv::format("%.1f", currentFps);
            cv::putText(frame, fpsText, cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            
            // Display frame
            cv::imshow(windowName, frame);
            
            // Exit on ESC key
            int key = cv::waitKey(1);
            if (key == 27) { // ESC key
                std::cout << "\nExiting on user request" << std::endl;
                break;
            }
        }
        
        // Release resources
        cap.release();
        cv::destroyAllWindows();
        
        if (frameCount > 0) {
            double avgFps = totalFps / frameCount;
            std::cout << "Average FPS: " << avgFps << std::endl;
        }
        
        std::cout << "Detection complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}