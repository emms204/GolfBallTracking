#pragma once

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
    OnnxDetector(const std::string& modelPath, const std::string& classesPath, float confThreshold = 0.25, float nmsThreshold = 0.45);
    ~OnnxDetector();
    
    std::vector<cv::Rect> detect(cv::Mat& frame, std::vector<float>& confidences, std::vector<int>& classIds);
    std::string getClassName(int classId) const;
};

void drawPredictions(cv::Mat& frame, const std::vector<cv::Rect>& boxes, 
                     const std::vector<float>& confidences, 
                     const std::vector<int>& classIds,
                     const OnnxDetector& detector);

std::string getCurrentTimeString(); 