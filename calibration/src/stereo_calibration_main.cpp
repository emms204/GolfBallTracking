#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include "calibrator.h"
#include "stereo_calibrator.h"
#include "path_utils.h"

void printHelp() {
    std::cout << "Usage: stereo_calibration [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h                    Print this help message" << std::endl;
    std::cout << "  --output <file>               Output stereo calibration file (YAML)" << std::endl;
    std::cout << "  --master_images <dir>         Directory with master camera images" << std::endl;
    std::cout << "  --slave_images <dir>          Directory with slave camera images" << std::endl;
    std::cout << "  --pattern_size <w>x<h>        Size of chessboard pattern (default: 9x6)" << std::endl;
    std::cout << "  --square_size <mm>            Size of square in mm (default: 20)" << std::endl;
    std::cout << "  --verify                      Show rectified images to verify calibration" << std::endl;
    std::cout << "  --master <file>               Optional: Master camera output file (YAML)" << std::endl;
    std::cout << "                                If not provided, will be stored as master_params.yaml" << std::endl;
    std::cout << "                                in the same directory as the stereo output file" << std::endl;
    std::cout << "  --slave <file>                Optional: Slave camera output file (YAML)" << std::endl;
    std::cout << "                                If not provided, will be stored as slave_params.yaml" << std::endl;
    std::cout << "                                in the same directory as the stereo output file" << std::endl;
}

// Parse pattern size string (e.g., "9x6")
bool parsePatternSize(const std::string& str, cv::Size& size) {
    size_t xPos = str.find('x');
    if (xPos == std::string::npos) {
        return false;
    }
    
    try {
        size.width = std::stoi(str.substr(0, xPos));
        size.height = std::stoi(str.substr(xPos + 1));
        return (size.width > 0 && size.height > 0);
    } catch (...) {
        return false;
    }
}

std::vector<int> checkCornerOrderingConsistency(
    const std::vector<std::vector<cv::Point2f>>& imagePointsMaster,
    const std::vector<std::vector<cv::Point2f>>& imagePointsSlave,
    const cv::Size& boardSize)
{
    std::vector<int> badIndices;

    for (size_t i = 0; i < imagePointsMaster.size(); ++i) {
        const auto& masterCorners = imagePointsMaster[i];
        const auto& slaveCorners = imagePointsSlave[i];

        // Get first row in both images
        std::vector<cv::Point2f> firstRowMaster(boardSize.width);
        std::vector<cv::Point2f> firstRowSlave(boardSize.width);

        for (int col = 0; col < boardSize.width; col++) {
            firstRowMaster[col] = masterCorners[col];
            firstRowSlave[col] = slaveCorners[col];
        }

        //Calculate direction vectors from first to last point in first row
        cv::Point2f masterDir = firstRowMaster.back() - firstRowMaster.front();
        cv::Point2f slaveDir = firstRowSlave.back() - firstRowSlave.front();

        // Normalize vectors for dot product comparison
        double masterNorm = cv::norm(masterDir);
        double slaveNorm = cv::norm(slaveDir);

        if (masterNorm > 0 && slaveNorm > 0) {
            cv::Point2f masterDirNorm = masterDir / masterNorm;
            cv::Point2f slaveDirNorm = slaveDir / slaveNorm;

            // Compute dot product to check if directions are similar
            double dotProduct = masterDirNorm.x * slaveDirNorm.x + masterDirNorm.y * slaveDirNorm.y;

            if (dotProduct < 0.8) {
                std::cout << "Inconsistent corner ordering detected in image pair " << i << std::endl;
                std::cout << "Dot product of row directions: " << dotProduct << std::endl;
                badIndices.push_back(i);
                continue;
            }
        }

        // Check column directions as well
        bool badColumns = false;
        for (int col = 0; col < boardSize.width; col++) {
            std::vector<cv::Point2f> colMaster(boardSize.height);
            std::vector<cv::Point2f> colSlave(boardSize.height);

            for (int row = 0; row < boardSize.height; row++) {
                int idx = row * boardSize.width + col;
                colMaster[row] = masterCorners[idx];
                colSlave[row] = slaveCorners[idx];
            }

            cv::Point2f masterColDir = colMaster.back() - colMaster.front();
            cv::Point2f slaveColDir = colSlave.back() - colSlave.front();

            double masterColNorm = cv::norm(masterColDir);
            double slaveColNorm = cv::norm(slaveColDir);

            if (masterColNorm > 0 && slaveColNorm > 0) {
                cv::Point2f masterColDirNorm = masterColDir / masterColNorm;
                cv::Point2f slaveColDirNorm = slaveColDir / slaveColNorm;

                double colDotProduct = masterColDirNorm.x * slaveColDirNorm.x + masterColDirNorm.y * slaveColDirNorm.y;

                if (colDotProduct < 0.8) {
                    if (std::find(badIndices.begin(), badIndices.end(), i) == badIndices.end()) {
                        std::cout << "Inconsistent column ordering detected in image pair " << i << std::endl;
                        std::cout << "Dot product of column directions: " << colDotProduct << std::endl;
                        badIndices.push_back(i);
                    }
                    badColumns = true;
                    break;
                }
            }
        }

        if (badColumns) continue;
    }

    return badIndices;      
}

// Add a helper function to handle individual camera calibration
bool calibrateAndSaveCamera(calibration::Calibrator& calibrator, 
                           const std::string& outputFile,
                           const std::string& cameraName) {
    // Perform calibration
    std::cout << "Calibrating " << cameraName << " camera..." << std::endl;
    if (!calibrator.calibrate()) {
        std::cerr << "Error: " << cameraName << " camera calibration failed." << std::endl;
        return false;
    }
    
    // Show calibration results
    std::cout << calibrator.getQualityAssessment() << std::endl;
    
    // Normalize the output file path
    std::string normalizedOutputPath = common::PathUtils::normalizePath(outputFile);
    
    // Ensure the output directory exists
    std::string outputDir = common::PathUtils::getDirectoryName(normalizedOutputPath);
    if (!outputDir.empty() && !common::PathUtils::exists(outputDir)) {
        std::cout << "Creating output directory: " << outputDir << std::endl;
        if (!common::PathUtils::createDirectory(outputDir)) {
            std::cerr << "Error: Failed to create output directory " << outputDir << std::endl;
            return false;
        }
    }
    
    // Save calibration parameters
    std::cout << "Saving " << cameraName << " calibration parameters to " << normalizedOutputPath << "..." << std::endl;
    if (!calibrator.saveCalibration(normalizedOutputPath)) {
        std::cerr << "Error: Failed to save " << cameraName << " calibration parameters." << std::endl;
        return false;
    }

    std::cout << cameraName << " calibration saved successfully!" << std::endl;
    return true;
}

// Collect image pairs from directories that have the same filename in both directories
std::vector<std::pair<std::string, std::string>> collectImagePairs(const std::string& masterDir, const std::string& slaveDir) {
    std::vector<std::pair<std::string, std::string>> imagePairs;
    
    // Check if directories exist
    if (!std::filesystem::exists(masterDir) || !std::filesystem::is_directory(masterDir)) {
        std::cerr << "Error: Master images directory not found: " << masterDir << std::endl;
        return imagePairs;
    }
    
    if (!std::filesystem::exists(slaveDir) || !std::filesystem::is_directory(slaveDir)) {
        std::cerr << "Error: Slave images directory not found: " << slaveDir << std::endl;
        return imagePairs;
    }
    
    // Get list of files in master directory
    std::vector<std::string> masterFiles;
    for (const auto& entry : std::filesystem::directory_iterator(masterDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                masterFiles.push_back(entry.path().filename().string());
            }
        }
    }
    // Sort master files to ensure consistent ordering
    // std::sort(masterFiles.begin(), masterFiles.end());
    
    // Check for matching files in slave directory
    for (const auto& masterFile : masterFiles) {
        std::string slavePath = slaveDir + "/" + masterFile;
        if (std::filesystem::exists(slavePath)) {
            imagePairs.push_back(std::make_pair(masterDir + "/" + masterFile, slavePath));
        }
    }
    
    std::cout << "Found " << imagePairs.size() << " image pairs" << std::endl;
    return imagePairs;
}

bool processImagePairs(const std::vector<std::pair<std::string, std::string>>& imagePairs,
                      calibration::Calibrator& masterCalibrator,
                      calibration::Calibrator& slaveCalibrator,
                      const std::string& masterOutputFile, 
                      const std::string& slaveOutputFile) {
    size_t successCount = 0;
    
    for (size_t i = 0; i < imagePairs.size(); ++i) {
        cv::Mat masterImg = cv::imread(imagePairs[i].first);
        cv::Mat slaveImg = cv::imread(imagePairs[i].second);
        
        if (masterImg.empty() || slaveImg.empty()) {
            std::cerr << "Warning: Could not read image pair: " << imagePairs[i].first << ", " << imagePairs[i].second << std::endl;
            continue;
        }
        
        bool masterFound = masterCalibrator.detectChessboard(masterImg, true);
        bool slaveFound = slaveCalibrator.detectChessboard(slaveImg, true);
        
        if (masterFound && slaveFound) {
            masterCalibrator.addCurrentPoints();
            slaveCalibrator.addCurrentPoints();
            successCount++;
            
            std::cout << "Found chessboard in image pair " << i+1 << "/" << imagePairs.size() 
                     << " (" << successCount << " total)" << std::endl;
                     
            // Display the images with detected corners
            cv::Mat combinedImg;
            cv::hconcat(masterImg, slaveImg, combinedImg);
            cv::imshow("Chessboard Detection", combinedImg);
            cv::waitKey(100);
        } else {
            std::cout << "No chessboard found in one or both images of pair " << i+1 << "/" << imagePairs.size() << std::endl;
        }
    }
    
    std::cout << "Successfully detected chessboard in " << successCount << " image pairs" << std::endl;
    // Check corner ordering consistency
    std::vector<int> badIndices = checkCornerOrderingConsistency(
        masterCalibrator.getImagePoints(),
        slaveCalibrator.getImagePoints(),
        masterCalibrator.getBoardSize()
    );

    if (!badIndices.empty()){
        std::cout << "Found " <<badIndices.size() << "image pairs with inconsistent corner ordering." << std::endl;
        std::cout << "These pairs will be excluded from stereo calibration: " << std::endl;
        for (int idx : badIndices) {
            std::cout << "Image pair " << idx <<std::endl;
        }

        //Create filtered point collections
        std::vector<std::vector<cv::Point3f>> filteredObjectPointsMaster;
        std::vector<std::vector<cv::Point3f>> filteredObjectPointsSlave;
        std::vector<std::vector<cv::Point2f>> filteredImagePointsMaster;
        std::vector<std::vector<cv::Point2f>> filteredImagePointsSlave;

         // Copy points that aren't in badIndices
        const auto& masterObjectPoints = masterCalibrator.getObjectPoints();
        const auto& masterImagePoints = masterCalibrator.getImagePoints();
        const auto& slaveObjectPoints = slaveCalibrator.getObjectPoints();
        const auto& slaveImagePoints = slaveCalibrator.getImagePoints();
    

        for (size_t i = 0; i<masterObjectPoints.size(); i++){
            if (std::find(badIndices.begin(), badIndices.end(), i) == badIndices.end()) {
                filteredObjectPointsMaster.push_back(masterObjectPoints[i]);
                filteredImagePointsMaster.push_back(masterImagePoints[i]);
                filteredObjectPointsSlave.push_back(slaveObjectPoints[i]);
                filteredImagePointsSlave.push_back(slaveImagePoints[i]);
            }
        }

        std::cout << "Using " << filteredObjectPointsMaster.size() << " out of " 
                    << masterObjectPoints.size() << " image pairs for master calibration." << std::endl;
        std::cout << "Using " << filteredObjectPointsSlave.size() << " out of " 
                    << slaveObjectPoints.size() << " image pairs for slave calibration." << std::endl;

        // Set filtered points for calibration
        masterCalibrator.setFilteredPoints(filteredObjectPointsMaster, filteredImagePointsMaster);
        slaveCalibrator.setFilteredPoints(filteredObjectPointsSlave, filteredImagePointsSlave);
    
    }
    else {
        std::cout << "No inconsistent corner ordering detected. Proceeding with full calibration." << std::endl;
    }

    // Calibrate and save master camera
    if (!calibrateAndSaveCamera(masterCalibrator, masterOutputFile, "master")) {
        return false;
    }
    
    // Calibrate and save slave camera
    if (!calibrateAndSaveCamera(slaveCalibrator, slaveOutputFile, "slave")) {
        return false;
    }
    std::cout << " Master and Slave camera calibration completed successfully!" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string masterOutputFile;
    std::string slaveOutputFile;
    std::string outputFile = "stereo_calibration.yaml";
    std::string masterImagesDir;
    std::string slaveImagesDir;
    cv::Size patternSize(9, 6);
    float squareSize = 20.0f;
    bool verifyCalibration = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printHelp();
            return 0;
        } else if (arg == "--master" && i + 1 < argc) {
            masterOutputFile = argv[++i];
        } else if (arg == "--slave" && i + 1 < argc) {
            slaveOutputFile = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--master_images" && i + 1 < argc) {
            masterImagesDir = argv[++i];
        } else if (arg == "--slave_images" && i + 1 < argc) {
            slaveImagesDir = argv[++i];
        } else if (arg == "--pattern_size" && i + 1 < argc) {
            if (!parsePatternSize(argv[++i], patternSize)) {
                std::cerr << "Error: Invalid pattern size format. Use WxH (e.g., 9x6)." << std::endl;
                return 1;
            }
        } else if (arg == "--square_size" && i + 1 < argc) {
            try {
                squareSize = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid square size." << std::endl;
                return 1;
            }
        } else if (arg == "--verify") {
            verifyCalibration = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printHelp();
            return 1;
        }
    }

    if (outputFile.empty()) {
        std::cerr << "Error: Output file not specified" << std::endl;
        printHelp();
        return 1;
    }

    if (masterImagesDir.empty() || slaveImagesDir.empty()) {
        std::cerr << "Error: Master and slave image directories must be specified" << std::endl;
        printHelp();
        return 1;
    }

    if (masterOutputFile.empty() || slaveOutputFile.empty()) {
        // Get the directory of the stereo output file
        std::string outputDir = common::PathUtils::getDirectoryName(outputFile);
        // If the output file is just a filename, use the current working directory
        if (outputDir.empty()) {
            outputDir = ".";
        }

        // Set default filenames if not provided
        if (masterOutputFile.empty()) {
            masterOutputFile = outputDir + "/master_params.yaml";
        }
        if (slaveOutputFile.empty()) {
            slaveOutputFile = outputDir + "/slave_params.yaml";
        }  
    }

    std::cout << "Using output files:" << std::endl;
    std::cout << "  Stereo: " << outputFile << std::endl;
    std::cout << "  Master: " << masterOutputFile << std::endl;
    std::cout << "  Slave: " << slaveOutputFile << std::endl;

    // Create calibrators
    calibration::Calibrator masterCalibrator(patternSize, squareSize);
    calibration::Calibrator slaveCalibrator(patternSize, squareSize);
    
    // Create stereo calibrator
    calibration::StereoCalibrator stereoCalibrator(masterCalibrator, slaveCalibrator);
    
    // Check if we need to process image pairs
    if (!masterImagesDir.empty() && !slaveImagesDir.empty()) {
        std::cout << "Processing image pairs from directories:" << std::endl;
        std::cout << "  Master: " << masterImagesDir << std::endl;
        std::cout << "  Slave: " << slaveImagesDir << std::endl;
        
        // Collect image pairs
        auto imagePairs = collectImagePairs(masterImagesDir, slaveImagesDir);
        if (imagePairs.empty()) {
            std::cerr << "Error: No matching image pairs found" << std::endl;
            return 1;
        }
        
        // Process image pairs
        if (!processImagePairs(imagePairs, masterCalibrator, slaveCalibrator, masterOutputFile, slaveOutputFile)) {
            std::cerr << "Error: Failed to process image pairs and calibrate cameras" << std::endl;
            return 1;
        }
    }
    
    // Perform stereo calibration
    std::cout << "Performing stereo calibration..." << std::endl;
    if (!stereoCalibrator.calibrateStereo()) {
        std::cerr << "Error: Stereo calibration failed" << std::endl;
        return 1;
    }
    
    // Save calibration
    std::cout << "Saving stereo calibration to " << outputFile << std::endl;
    if (!stereoCalibrator.saveCalibration(outputFile)) {
        std::cerr << "Error: Failed to save stereo calibration" << std::endl;
        return 1;
    }
    
    // Verify calibration if requested
    if (verifyCalibration && !masterImagesDir.empty() && !slaveImagesDir.empty()) {
        auto imagePairs = collectImagePairs(masterImagesDir, slaveImagesDir);
        if (!imagePairs.empty()) {
            cv::Mat masterImg = cv::imread(imagePairs[0].first);
            cv::Mat slaveImg = cv::imread(imagePairs[0].second);
            cv::Mat rectifiedImg;
            
            if (!masterImg.empty() && !slaveImg.empty()) {
                std::cout << "Verifying calibration with image pair: " << std::endl;
                std::cout << "  Master: " << imagePairs[0].first << std::endl;
                std::cout << "  Slave: " << imagePairs[0].second << std::endl;
                
                if (stereoCalibrator.verifyCalibration(masterImg, slaveImg, rectifiedImg)) {
                    cv::imshow("Stereo Rectification", rectifiedImg);
                    std::cout << "Press any key to continue..." << std::endl;
                    cv::waitKey(0);
                }
            }
        }
    }
    
    std::cout << "Stereo calibration completed successfully!" << std::endl;
    return 0;
} 