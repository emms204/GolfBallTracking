#pragma once

#include <string>
#include <filesystem>
#include <algorithm>

namespace common {

/**
 * @brief Utility functions for cross-platform path handling
 */
class PathUtils {
public:
    /**
     * @brief Normalize a path for cross-platform compatibility
     * 
     * @param path Input path
     * @return std::string Normalized path with appropriate platform-specific separators
     */
    static std::string normalizePath(const std::string& path) {
        std::filesystem::path fsPath(path);
        return fsPath.make_preferred().string();
    }
    
    /**
     * @brief Convert path to use forward slashes for storage in configuration files
     * This ensures consistency across platforms when reading/writing paths
     * 
     * @param path Input path
     * @return std::string Path with forward slashes
     */
    static std::string toPortablePath(const std::string& path) {
        std::string result = path;
        std::replace(result.begin(), result.end(), '\\', '/');
        return result;
    }
    
    /**
     * @brief Join paths in a platform-independent way
     * 
     * @param base Base path
     * @param subPath Sub path to append
     * @return std::string Joined path
     */
    static std::string joinPaths(const std::string& base, const std::string& subPath) {
        std::filesystem::path basePath(base);
        return (basePath / subPath).make_preferred().string();
    }
    
    /**
     * @brief Get the file name from a path
     * 
     * @param path Input path
     * @return std::string File name
     */
    static std::string getFileName(const std::string& path) {
        return std::filesystem::path(path).filename().string();
    }
    
    /**
     * @brief Get the directory name from a path
     * 
     * @param path Input path
     * @return std::string Directory path
     */
    static std::string getDirectoryName(const std::string& path) {
        return std::filesystem::path(path).parent_path().string();
    }
    
    /**
     * @brief Check if a path exists
     * 
     * @param path Path to check
     * @return bool True if path exists
     */
    static bool exists(const std::string& path) {
        return std::filesystem::exists(path);
    }
    
    /**
     * @brief Create a directory and any necessary parent directories
     * 
     * @param path Directory path to create
     * @return bool True if successful or directory already exists
     */
    static bool createDirectory(const std::string& path) {
        try {
            return std::filesystem::create_directories(path);
        } catch (const std::filesystem::filesystem_error&) {
            return false;
        }
    }
};

} // namespace common 