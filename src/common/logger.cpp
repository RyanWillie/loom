#include "common/logger.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unistd.h>  // For isatty()

// ============================================================================
// ANSI Color Codes for Terminal Output
// ============================================================================

namespace colors {
// Text attributes
constexpr const char* RESET = "\033[0m";
constexpr const char* BOLD = "\033[1m";
constexpr const char* DIM = "\033[2m";

// Regular colors
constexpr const char* RED = "\033[31m";
constexpr const char* GREEN = "\033[32m";
constexpr const char* YELLOW = "\033[33m";
constexpr const char* BLUE = "\033[34m";
constexpr const char* MAGENTA = "\033[35m";
constexpr const char* CYAN = "\033[36m";

// Bright colors
constexpr const char* BRIGHT_RED = "\033[91m";
constexpr const char* BRIGHT_GREEN = "\033[92m";
constexpr const char* BRIGHT_YELLOW = "\033[93m";
constexpr const char* BRIGHT_CYAN = "\033[96m";
constexpr const char* BRIGHT_WHITE = "\033[97m";
}  // namespace colors

// ============================================================================
// Static Helper Functions
// ============================================================================

namespace {
// Generate a timestamped filename for the log file
std::string generateTimestampedLogFilename() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << "./logs/logs_" << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d_%H-%M-%S")
       << ".txt";

    return ss.str();
}

// Detect if terminal supports colors
bool isTerminalColorSupported() {
#ifndef _WIN32
    // Unix: check if stdout is connected to a terminal
    return isatty(fileno(stdout));
#else
    // Windows: default to false for now
    return false;
#endif
}

// Get color code for a log level
const char* getLogLevelColor(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:
            return colors::DIM;
        case LogLevel::DEBUG:
            return colors::CYAN;
        case LogLevel::INFO:
            return colors::BRIGHT_GREEN;
        case LogLevel::WARNING:
            return colors::BRIGHT_YELLOW;
        case LogLevel::ERROR:
            return colors::BRIGHT_RED;
        case LogLevel::FATAL:
            return colors::BRIGHT_RED;  // We'll add BOLD separately
        default:
            return colors::RESET;
    }
}

// Format a log entry for plain text output (files and non-color terminals)
std::string formatPlain(const LogEntry& entry) {
    return entry.timestamp + " " + entry.log_level_str + " " + entry.scope + " " + entry.message +
           '\n';
}

// Format a log entry with fancy colors (console)
std::string formatColored(const LogEntry& entry) {
    std::stringstream ss;

    // Timestamp in dim
    ss << colors::DIM << entry.timestamp << colors::RESET << " ";

    // Log level in color (bold for FATAL)
    if (entry.log_level_enum == LogLevel::FATAL) {
        ss << colors::BOLD;
    }
    ss << getLogLevelColor(entry.log_level_enum) << entry.log_level_str << colors::RESET << " ";

    // Scope in bright white/bold
    ss << colors::CYAN << entry.scope << colors::RESET << " ";

    // Message in normal color
    ss << entry.message << "\n";

    return ss.str();
}

}  // namespace

// ============================================================================
// Static Member Initialization
// ============================================================================

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<std::string, std::unique_ptr<Logger>> Logger::mLoggers = {};
std::mutex Logger::sRegistryMutex;
std::mutex Logger::sFileMutex;
LogLevel Logger::sMinLogLevel = LogLevel::INFO;
LogOutput Logger::sLogOutput = LogOutput::CONSOLE;
std::string Logger::sLogFile = generateTimestampedLogFilename();
std::ofstream Logger::sFileStream;
bool Logger::sFileStreamOpen = false;
std::queue<LogEntry> Logger::sLogQueue = {};

std::mutex Logger::sLogQueueMutex;
std::mutex Logger::sRunningMutex;
std::thread Logger::sLoggingThread;
std::atomic<bool> Logger::sRunning = false;
std::condition_variable Logger::sLogQueueCondition;
std::atomic<bool> Logger::sShutdownRequested = false;

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

const std::unordered_map<LogLevel, std::string> Logger::sLogLevelToString = {
    {LogLevel::TRACE, "TRACE"},     {LogLevel::DEBUG, "DEBUG"}, {LogLevel::INFO, "INFO"},
    {LogLevel::WARNING, "WARNING"}, {LogLevel::ERROR, "ERROR"}, {LogLevel::FATAL, "FATAL"},
};

// ============================================================================
// Constructor & Destructor
// ============================================================================

// NOLINTNEXTLINE(modernize-pass-by-value)
Logger::Logger(const std::string& scope, LogLevel level) : mScope(scope), mLogLevel(level) {}

Logger::~Logger() = default;

// ============================================================================
// Static Methods
// ============================================================================

Logger& Logger::getInstance(const std::string& scope, const LogLevel& level) {
    // Check if logging thread is running
    if (!sRunning) {
        std::lock_guard<std::mutex> lock(sRunningMutex);
        if (!sLoggingThread.joinable()) {
            sLoggingThread = std::thread(runLoggingThread);
            sRunning = true;
        }
    }

    std::lock_guard<std::mutex> lock(sRegistryMutex);
    auto it = mLoggers.find(scope);

    if (it == mLoggers.end()) {
        it = mLoggers.emplace(scope, std::unique_ptr<Logger>(new Logger(scope, level))).first;
        return *it->second;
    }

    it->second->setLogLevel(level);
    return *it->second;
}

void Logger::setMinLogLevel(const LogLevel& level) {
    sMinLogLevel = level;
}

void Logger::setLogOutput(const LogOutput& output) {
    sLogOutput = output;

    // Auto-open log file if output includes FILE and file isn't already open
    if ((output == LogOutput::FILE || output == LogOutput::BOTH) && !sFileStreamOpen) {
        std::lock_guard<std::mutex> lock(sFileMutex);
        if (!sFileStreamOpen) {  // Double-check after acquiring lock
            openLogFileInternal();
        }
    }
}

void Logger::setLogFile(const std::string& file) {
    std::lock_guard<std::mutex> lock(sFileMutex);

    // Close existing file if open
    if (sFileStreamOpen) {
        sFileStream.close();
        sFileStreamOpen = false;
    }

    // Open new file in append mode
    sLogFile = file;
    openLogFileInternal();

    if (sFileStreamOpen) {
        sLogOutput = LogOutput::BOTH;
    }
}

void Logger::closeLogFile() {
    std::lock_guard<std::mutex> lock(sFileMutex);
    if (sFileStreamOpen) {
        sFileStream.close();
        sFileStreamOpen = false;
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void Logger::openLogFileInternal() {
    // PRECONDITION: sFileMutex must be locked by caller
    //
    // Design rationale: This function does NOT acquire its own lock because:
    // 1. setLogFile() needs atomicity across close+open operations
    // 2. setLogOutput() already performs double-checked locking
    // 3. Acquiring the lock here would cause deadlock in setLogFile()
    //
    // This is a private function and all call sites properly acquire the lock.

    sFileStream.open(sLogFile, std::ios::app);

    if (sFileStream.is_open()) {
        sFileStreamOpen = true;
    } else {
        std::cerr << "Error: Failed to open log file: " << sLogFile << std::endl;
    }
}

// ============================================================================
// Logging Methods
// ============================================================================

void Logger::log(const std::string& message) const {
    if (mLogLevel < sMinLogLevel) {
        return;
    }
    logInternal(message, mLogLevel);
}

void Logger::trace(const std::string& message) const {
    if (LogLevel::TRACE < sMinLogLevel) {
        return;
    }
    logInternal(message, LogLevel::TRACE);
}

void Logger::debug(const std::string& message) const {
    if (LogLevel::DEBUG < sMinLogLevel) {
        return;
    }
    logInternal(message, LogLevel::DEBUG);
}

void Logger::info(const std::string& message) const {
    if (LogLevel::INFO < sMinLogLevel) {
        return;
    }
    logInternal(message, LogLevel::INFO);
}

void Logger::warning(const std::string& message) const {
    if (LogLevel::WARNING < sMinLogLevel) {
        return;
    }
    logInternal(message, LogLevel::WARNING);
}

void Logger::error(const std::string& message) const {
    if (LogLevel::ERROR < sMinLogLevel) {
        return;
    }
    logInternal(message, LogLevel::ERROR);
}

void Logger::fatal(const std::string& message) const {
    if (LogLevel::FATAL < sMinLogLevel) {
        return;
    }
    logInternal(message, LogLevel::FATAL);
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void Logger::logInternal(const std::string& message, const LogLevel& level) const {
    if (sShutdownRequested) {
        return;
    }

    // Create structured log entry
    LogEntry entry;
    entry.timestamp = getCurrentTimestamp();
    entry.log_level_str = "[" + sLogLevelToString.at(level) + "]";
    entry.log_level_enum = level;
    entry.scope = "[" + mScope + "]";
    entry.message = message;

    // Push structured entry to queue
    std::lock_guard<std::mutex> lock(sLogQueueMutex);
    sLogQueue.push(entry);
    sLogQueueCondition.notify_one();
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::string Logger::getCurrentTimestamp() const {
    // NOLINTBEGIN(readability-identifier-naming)
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    // NOLINTEND(readability-identifier-naming)

    std::string timestamp = std::ctime(&now_time);
    timestamp.pop_back();  // Remove trailing newline
    return timestamp;
}

void Logger::setLogLevel(LogLevel level) {
    mLogLevel = level;
}

// ============================================================================
// Logging thread helper methods
// ============================================================================

void Logger::runLoggingThread() {
    while (!sShutdownRequested || !sLogQueue.empty()) {
        std::unique_lock<std::mutex> lock(sLogQueueMutex);
        sLogQueueCondition.wait(lock, [] { return !sLogQueue.empty() || sShutdownRequested; });

        // Drain all log entries from the queue
        std::vector<LogEntry> entries;
        while (!sLogQueue.empty()) {
            entries.emplace_back(sLogQueue.front());
            sLogQueue.pop();
        }
        lock.unlock();

        // Write to console if appropriate (with colors!)
        if (sLogOutput == LogOutput::CONSOLE || sLogOutput == LogOutput::BOTH) {
            static const bool use_color = isTerminalColorSupported();

            for (const auto& entry : entries) {
                if (use_color) {
                    std::cout << formatColored(entry);
                } else {
                    std::cout << formatPlain(entry);
                }
            }
            std::cout.flush();
        }

        // Write to file if appropriate (always plain text, no colors!)
        if (sLogOutput == LogOutput::FILE || sLogOutput == LogOutput::BOTH) {
            std::lock_guard<std::mutex> file_lock(sFileMutex);
            if (sFileStreamOpen && sFileStream.is_open()) {
                for (const auto& entry : entries) {
                    sFileStream << formatPlain(entry);
                }
                sFileStream.flush();  // Ensure immediate write for critical logs
            }
        }
    }
}

void Logger::shutdown() {
    // Set the shutdown requested flag
    sShutdownRequested = true;

    // Notify all waiting threads
    sLogQueueCondition.notify_all();

    // Join the logging thread
    if (sLoggingThread.joinable()) {
        sLoggingThread.join();
    }
    // Close the log file
    Logger::closeLogFile();
    // Set the running flag to false
    sRunning = false;
}

void Logger::flush() {
    // Wait until the queue is empty
    while (true) {
        {
            std::lock_guard<std::mutex> lock(sLogQueueMutex);
            if (sLogQueue.empty()) {
                return;  // Queue is empty, all messages processed
            }
        }
        // Release lock and give worker thread time to process
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}