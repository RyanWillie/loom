#pragma once

#include <atomic>
#include <condition_variable>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

// ============================================================================
// Log Level Enumeration
// ============================================================================

enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };
enum class LogOutput { CONSOLE, FILE, BOTH };

// ============================================================================
// Log Entry Structure
// ============================================================================

struct LogEntry {
    std::string timestamp;
    std::string log_level_str;  // "[INFO]", "[ERROR]", etc.
    LogLevel log_level_enum;    // For determining color
    std::string scope;          // "[System]", "[MemoryUtils]", etc.
    std::string message;        // The actual message
};

// ============================================================================
// Logger Class
// ============================================================================

/**
 * @brief Logger class for logging messages
 * @details This class provides a multithreaded logger for logging messages to
 *          the console. It supports different log levels and can be used to
 *          log messages from different sources. It uses a singleton pattern to
 *          ensure that there is only one instance of the logger per scope.
 *          Different scopes can be provided to the logger to log messages from
 *          different sources.
 */
class Logger {
  public:
    ~Logger();

    // Delete copy and move operations (singleton pattern)
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    // ------------------------------------------------------------------------
    // Logging Methods
    // ------------------------------------------------------------------------

    /// Log a message with the default scope level
    void log(const std::string& message) const;

    /// Log with explicit levels
    void trace(const std::string& message) const;
    void debug(const std::string& message) const;
    void info(const std::string& message) const;
    void warning(const std::string& message) const;
    void error(const std::string& message) const;
    void fatal(const std::string& message) const;

    /// Formatted logging with std::format (C++20)
    template <typename... Args>
    void trace(std::format_string<Args...> fmt, Args&&... args) const {
        trace(std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) const {
        debug(std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) const {
        info(std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void warning(std::format_string<Args...> fmt, Args&&... args) const {
        warning(std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) const {
        error(std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void fatal(std::format_string<Args...> fmt, Args&&... args) const {
        fatal(std::format(fmt, std::forward<Args>(args)...));
    }

    // ------------------------------------------------------------------------
    // Static Methods
    // ------------------------------------------------------------------------

    /// Get or create a logger instance for a given scope
    static Logger& getInstance(const std::string& scope, const LogLevel& level = LogLevel::INFO);

    /// Set the minimum log level for all loggers
    static void setMinLogLevel(const LogLevel& level);

    /// Set the log file for all loggers
    static void setLogFile(const std::string& file);

    /// Set the log output for all loggers
    static void setLogOutput(const LogOutput& output);

    /// Shutdown the logging thread
    static void shutdown();

    /// Wait for all queued messages to be processed
    static void flush();

  private:
    // ------------------------------------------------------------------------
    // Constructor (Private - Singleton Pattern)
    // ------------------------------------------------------------------------

    Logger(const std::string& scope, LogLevel level = LogLevel::INFO);

    // ------------------------------------------------------------------------
    // Static Members (Required for Singleton Pattern)
    // ------------------------------------------------------------------------

    // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)
    static std::unordered_map<std::string, std::unique_ptr<Logger>> mLoggers;
    static std::mutex sRegistryMutex;
    static std::mutex sFileMutex;  /// Mutex for file operations
    static LogLevel sMinLogLevel;
    static LogOutput sLogOutput;
    static std::string sLogFile;       /// Log file path
    static std::ofstream sFileStream;  /// Persistent file stream
    static bool sFileStreamOpen;       /// Track if file stream is open

    // Logging thread members
    static std::queue<LogEntry> sLogQueue;              /// Queue for log messages
    static std::mutex sLogQueueMutex;                   /// Mutex for log queue
    static std::thread sLoggingThread;                  /// Logging thread
    static std::atomic<bool> sRunning;                  /// Flag to control logging thread
    static std::mutex sRunningMutex;                    /// Mutex for running flag
    static std::condition_variable sLogQueueCondition;  /// Condition variable for log queue
    static std::atomic<bool> sShutdownRequested;  /// Flag to request shutdown of logging thread

    // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    static const std::unordered_map<LogLevel, std::string> sLogLevelToString;

    // ------------------------------------------------------------------------
    // Private Helper Methods
    // ------------------------------------------------------------------------

    void setLogLevel(LogLevel level);
    [[nodiscard]] std::string getCurrentTimestamp() const;
    void logInternal(const std::string& message, const LogLevel& level) const;

    /// Internal helper to open the log file
    /// PRECONDITION: Caller must already hold sFileMutex lock
    /// This is necessary to maintain atomicity across multiple file operations
    static void openLogFileInternal();

    static void runLoggingThread();

    static void closeLogFile();

    // ------------------------------------------------------------------------
    // Instance Members
    // ------------------------------------------------------------------------

    std::string mScope = "System";        ///< Scope of the logger
    LogLevel mLogLevel = LogLevel::INFO;  ///< Log level of the logger
};