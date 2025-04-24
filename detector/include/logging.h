#ifndef LOGGING_H
#define LOGGING_H

#include <fstream>
#include <string>

// Declare the global variable (using extern)
extern std::ofstream g_debugLogFile;

// Declare the function
void logMessage(const std::string& message, bool toConsole = true);

#endif // LOGGING_H