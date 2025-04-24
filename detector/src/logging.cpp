#include "logging.h" // Include the header with declarations
#include <iostream>
#include <fstream> // Already included via header, but good practice

// Define the global variable (only here!)
std::ofstream g_debugLogFile;

// Define the function (only here!)
void logMessage(const std::string& message, bool toConsole) {
    if (toConsole) {
        std::cout << message << std::endl;
    }
    if (g_debugLogFile.is_open()) {
        g_debugLogFile << message << std::endl;
    }
}