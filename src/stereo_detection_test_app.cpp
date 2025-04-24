#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <iomanip>
#include <fstream>
#include "stereo_onnx_detector.h" // Include the stereo detector header
#include "logging.h" // Include logging header

// Get current time as string (for logging)
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void printHelp() {
    logMessage("Usage: stereo_detection_test_app [options]");
    logMessage("Options:");
    logMessage("  --help                      Show this help message");
    logMessage("  --master_camera=<id>        Master camera ID (default: 0)");
    logMessage("  --slave_camera=<id>         Slave camera ID (default: 1)");
    logMessage("  --master_video=<path>       Path to master video file (overrides camera input)");
    logMessage("  --slave_video=<path>        Path to slave video file (overrides camera input)");
    logMessage("  --stereo_calibration=<file> Path to stereo calibration file (yaml/xml)");
    logMessage("  --model=<file>              Path to ONNX model file (default: best.onnx)");
    logMessage("  --classes=<file>            Path to classes file (default: classes.txt)");
    logMessage("  --conf=<threshold>          Confidence threshold (default: 0.25)");
    logMessage("  --nms=<threshold>           NMS threshold (default: 0.45)");
    logMessage("  --width=<pixels>            Camera width (default: 640)");
    logMessage("  --height=<pixels>           Camera height (default: 480)");
    logMessage("  --visualize=<0|1>           Enable or disable visualization (default: 1)");
    logMessage("  --save_trajectory=<path>    Save trajectory to CSV file");
    logMessage("  --display_stereo=<0|1>      Display stereo rectification (default: 1)");
    logMessage("  --log=<path>                Path to log file (default: stereo_debug.log)");
}

int main(int argc, char** argv) {
    // Default parameters
    int master_camera_id = 0;
    int slave_camera_id = 1;
    std::string master_video_path = "";
    std::string slave_video_path = "";
    std::string stereo_calibration_file = "";
    std::string model_path = "best.onnx";
    std::string classes_path = "classes.txt";
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int width = 640;
    int height = 480;
    bool visualize = true;
    std::string trajectory_file = "";
    bool display_stereo = true;
    std::string log_file = "stereo_debug.log";
    
    // Parse command-line arguments using equals format
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printHelp();
            return 0;
        } else if (arg.find("--master_camera=") == 0) {
            master_camera_id = std::stoi(arg.substr(16));
        } else if (arg.find("--slave_camera=") == 0) {
            slave_camera_id = std::stoi(arg.substr(15));
        } else if (arg.find("--master_video=") == 0) {
            master_video_path = arg.substr(15);
        } else if (arg.find("--slave_video=") == 0) {
            slave_video_path = arg.substr(14);
        } else if (arg.find("--stereo_calibration=") == 0) {
            stereo_calibration_file = arg.substr(21);
        } else if (arg.find("--model=") == 0) {
            model_path = arg.substr(8);
        } else if (arg.find("--classes=") == 0) {
            classes_path = arg.substr(10);
        } else if (arg.find("--conf=") == 0) {
            conf_threshold = std::stof(arg.substr(7));
        } else if (arg.find("--nms=") == 0) {
            nms_threshold = std::stof(arg.substr(6));
        } else if (arg.find("--width=") == 0) {
            width = std::stoi(arg.substr(8));
        } else if (arg.find("--height=") == 0) {
            height = std::stoi(arg.substr(9));
        } else if (arg.find("--visualize=") == 0) {
            visualize = std::stoi(arg.substr(12)) != 0;
        } else if (arg.find("--save_trajectory=") == 0) {
            trajectory_file = arg.substr(18);
        } else if (arg.find("--display_stereo=") == 0) {
            display_stereo = std::stoi(arg.substr(17)) != 0;
        } else if (arg.find("--log=") == 0) {
            log_file = arg.substr(6);
        }
    }
    
    // Initialize debug log file
    g_debugLogFile.open(log_file, std::ios::out);
    if (!g_debugLogFile.is_open()) {
        std::cerr << "Warning: Could not open " << log_file << " for writing" << std::endl;
    }
    
    logMessage("=== Stereo Detection Test Application ===");
    logMessage("Time: " + getTimestamp());
    
    // Validate inputs - need either two cameras or two video files
    bool use_video = !master_video_path.empty() && !slave_video_path.empty();
    bool use_camera = master_video_path.empty() && slave_video_path.empty();
    
    if (!use_video && !use_camera) {
        logMessage("Error: You must provide either two camera IDs or two video files", true);
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
        return -1;
    }
    
    // Initialize video captures for master and slave
    cv::VideoCapture master_cap, slave_cap;
    
    if (use_video) {
        master_cap.open(master_video_path);
        slave_cap.open(slave_video_path);
        logMessage("Opening video files:");
        logMessage("  Master: " + master_video_path);
        logMessage("  Slave: " + slave_video_path);
    } else {
        master_cap.open(master_camera_id);
        slave_cap.open(slave_camera_id);
        logMessage("Opening cameras:");
        logMessage("  Master: ID " + std::to_string(master_camera_id));
        logMessage("  Slave: ID " + std::to_string(slave_camera_id));
    }
    
    if (!master_cap.isOpened() || !slave_cap.isOpened()) {
        logMessage("Error: Could not open one or both video sources", true);
        if (g_debugLogFile.is_open()) {
            g_debugLogFile.close();
        }
        return -1;
    }
    
    // Set camera resolution (only applicable for camera, not video file)
    if (use_camera) {
        master_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        master_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        slave_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        slave_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    
    // Get actual frame dimensions
    width = static_cast<int>(master_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(master_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    logMessage("Frame resolution: " + std::to_string(width) + "x" + std::to_string(height));
    
    // Log parameters
    logMessage("Model: " + model_path);
    logMessage("Classes: " + classes_path);
    logMessage("Confidence threshold: " + std::to_string(conf_threshold));
    logMessage("NMS threshold: " + std::to_string(nms_threshold));
    logMessage("Visualization: " + std::string(visualize ? "ON" : "OFF"));
    logMessage("Display stereo rectification: " + std::string(display_stereo ? "ON" : "OFF"));
    
    try {
        // Load the stereo detector
        StereoOnnxDetector detector(model_path, classes_path, stereo_calibration_file, conf_threshold, nms_threshold);
        
        logMessage("Model loaded successfully from: " + model_path);
        logMessage("Classes loaded from: " + classes_path);
        
        if (stereo_calibration_file.empty()) {
            logMessage("Warning: No stereo calibration file provided. 3D tracking will not be available.");
        } else if (detector.isCalibrated()) {
            logMessage("Stereo calibration loaded successfully from: " + stereo_calibration_file);
        } else {
            logMessage("Warning: Failed to load stereo calibration from: " + stereo_calibration_file);
        }
        
        logMessage("Starting stereo detection...");
        logMessage("Press 'q' to quit, 's' to save trajectory, 'r' to reset tracking, 'p' to pause/resume");
        
        cv::Mat master_frame, slave_frame;
        bool paused = false;
        int frame_count = 0;
        double total_fps = 0.0;
        
        // Create window for visualization
        if (visualize) {
            cv::namedWindow("Stereo Detection", cv::WINDOW_NORMAL);
            cv::resizeWindow("Stereo Detection", width * 2, height);  // Side-by-side view
        }
        
        // Main loop
        while (true) {
            // Start time for FPS calculation
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Capture frames
            if (!paused) {
                master_cap >> master_frame;
                slave_cap >> slave_frame;
                
                if (master_frame.empty() || slave_frame.empty()) {
                    if (use_video) {
                        logMessage("End of video files reached");
                    } else {
                        logMessage("Error: Could not read frames from cameras", true);
                    }
                    break;
                }
                
                frame_count++;
            }
            
            // Log frame information
            std::string frameHeader = "\n================= FRAME " + std::to_string(frame_count) + " =================";
            logMessage(frameHeader);
            logMessage("Master frame dimensions: " + std::to_string(master_frame.cols) + "x" + std::to_string(master_frame.rows));
            logMessage("Slave frame dimensions: " + std::to_string(slave_frame.cols) + "x" + std::to_string(slave_frame.rows));
            
            // Process frames with the stereo detector
            cv::Mat visualization = detector.processFrames(master_frame, slave_frame, display_stereo);
            
            // Get detection results from stereo detector
            cv::Point2f masterCenter = detector.getMasterCenter();
            cv::Point2f slaveCenter = detector.getSlaveCenter();
            std::vector<Point3D> positions = detector.getAllPositions();
            
            // Log detection details
            logMessage("---------------- DETECTION DETAILS ----------------");
            if (masterCenter.x >= 0 && masterCenter.y >= 0) {
                logMessage("Master Ball Detection: (" + std::to_string(masterCenter.x) + ", " + std::to_string(masterCenter.y) + ")");
                logMessage("  Normalized center: (" + 
                           std::to_string(masterCenter.x / master_frame.cols) + ", " + 
                           std::to_string(masterCenter.y / master_frame.rows) + ")");
            } else {
                logMessage("No Ball Detection in master camera");
            }
            
            if (slaveCenter.x >= 0 && slaveCenter.y >= 0) {
                logMessage("Slave Ball Detection: (" + std::to_string(slaveCenter.x) + ", " + std::to_string(slaveCenter.y) + ")");
                logMessage("  Normalized center: (" + 
                           std::to_string(slaveCenter.x / slave_frame.cols) + ", " + 
                           std::to_string(slaveCenter.y / slave_frame.rows) + ")");
            } else {
                logMessage("No Ball Detection in slave camera");
            }
            
            // Log 3D position if available
            if (!positions.empty() && positions.size() > frame_count - 1) {
                Point3D currentPos = positions.back();
                logMessage("3D Position: (" + 
                           std::to_string(currentPos.x) + ", " + 
                           std::to_string(currentPos.y) + ", " + 
                           std::to_string(currentPos.z) + ") mm");
                
                // If we have enough positions, log motion parameters
                if (positions.size() >= 2) {
                    MotionParameters params = detector.getMotionParameters();
                    size_t lastIdx = params.velocities.size() - 1;
                    float lastSpeed = params.speeds[lastIdx];

                    logMessage("Velocity: (" + 
                               std::to_string(params.velocities[lastIdx][0]) + ", " + 
                               std::to_string(params.velocities[lastIdx][1]) + ", " + 
                               std::to_string(params.velocities[lastIdx][2]) + ") mm/s");
                    logMessage("Speed: " + std::to_string(lastSpeed) + " mm/s");
                }
            } else if (!positions.empty()) {
                logMessage("Tracking positions: " + std::to_string(positions.size()));
            } else {
                logMessage("No 3D position available (triangulation failed)");
            }
            
            // Calculate FPS
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            double fps = duration > 0 ? 1000.0 / duration : 0.0;
            total_fps += fps;
            
            logMessage("Current FPS: " + std::to_string(fps));
            
            // Add FPS to visualization
            if (!visualization.empty() && visualize) {
                std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
                cv::putText(visualization, fps_text, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                
                // Add paused status if applicable
                if (paused) {
                    cv::putText(visualization, "PAUSED", cv::Point(visualization.cols - 150, 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }
                
                // Display the visualization
                cv::imshow("Stereo Detection", visualization);
            }
            
            // Handle key presses
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) { // 'q' or ESC
                logMessage("Exiting on user request");
                break;
            } else if (key == 'p') { // Pause/resume
                paused = !paused;
                logMessage("Playback " + std::string(paused ? "paused" : "resumed"));
            } else if (key == 'r') { // Reset tracking
                detector.resetTracking();
                logMessage("Tracking reset");
            } else if (key == 's') { // Save trajectory
                if (!trajectory_file.empty()) {
                    if (detector.saveTrajectoryToCSV(trajectory_file)) {
                        logMessage("Trajectory saved to: " + trajectory_file);
                    } else {
                        logMessage("Failed to save trajectory to: " + trajectory_file, true);
                    }
                } else {
                    // Generate a default filename with timestamp
                    std::string timestamp = getTimestamp();
                    // Replace spaces and colons with underscores for safe filenames
                    std::replace(timestamp.begin(), timestamp.end(), ' ', '_');
                    std::replace(timestamp.begin(), timestamp.end(), ':', '-');
                    std::string default_file = "trajectory_" + timestamp + ".csv";
                    if (detector.saveTrajectoryToCSV(default_file)) {
                        logMessage("Trajectory saved to: " + default_file);
                    } else {
                        logMessage("Failed to save trajectory to: " + default_file, true);
                    }
                }
            } else if (key == 'v') { // Save visualization
                if (!visualization.empty()) {
                    // Save visualization to file
                    std::string timestamp = getTimestamp();
                    std::replace(timestamp.begin(), timestamp.end(), ' ', '_');
                    std::replace(timestamp.begin(), timestamp.end(), ':', '-');
                    std::string viz_file = "trajectory_viz_" + timestamp + ".png";
                    if (cv::imwrite(viz_file, visualization)) {
                        logMessage("Visualization saved to: " + viz_file);
                    } else {
                        logMessage("Failed to save visualization to: " + viz_file, true);
                    }
                    
                    // Create and save 3D trajectory visualization
                    cv::Mat trajectory_viz = detector.createTrajectoryVisualization();
                    if (!trajectory_viz.empty()) {
                        std::string traj_viz_file = "trajectory_3d_" + timestamp + ".png";
                        if (cv::imwrite(traj_viz_file, trajectory_viz)) {
                            logMessage("3D trajectory visualization saved to: " + traj_viz_file);
                        } else {
                            logMessage("Failed to save 3D visualization to: " + traj_viz_file, true);
                        }
                    }
                }
            }
        }
        
        // Release resources
        master_cap.release();
        slave_cap.release();
        if (visualize) {
            cv::destroyAllWindows();
        }
        
        // Print statistics
        if (frame_count > 0) {
            double avg_fps = total_fps / frame_count;
            logMessage("\nProcessed " + std::to_string(frame_count) + " frames");
            logMessage("Average FPS: " + std::to_string(avg_fps));
        }
        
        // Save trajectory if requested and not already saved
        if (!trajectory_file.empty() && detector.getAllPositions().size() > 0) {
            if (detector.saveTrajectoryToCSV(trajectory_file)) {
                logMessage("Final trajectory saved to: " + trajectory_file);
            } else {
                logMessage("Failed to save final trajectory to: " + trajectory_file, true);
            }
        }
        
        logMessage("Stereo detection complete!");
        
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