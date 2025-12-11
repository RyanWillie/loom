#include <sstream>
#include <thread>
#include <vector>

#include "loom/logger.h"
#include <gtest/gtest.h>

// ============================================================================
// Test Fixture
// ============================================================================

class LoggerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Capture cout for output verification
        old_cout_buf = std::cout.rdbuf();
        std::cout.rdbuf(cout_buffer.rdbuf());
    }

    void TearDown() override {
        Logger::shutdown();

        // Restore cout
        std::cout.rdbuf(old_cout_buf);
        cout_buffer.str("");
        cout_buffer.clear();

        // Reset global min log level
        Logger::setMinLogLevel(LogLevel::INFO);
    }

    std::string getOutput() const { return cout_buffer.str(); }

    void clearOutput() {
        cout_buffer.str("");
        cout_buffer.clear();
    }

  private:
    std::stringstream cout_buffer;
    std::streambuf* old_cout_buf = nullptr;
};

// ============================================================================
// Singleton Pattern Tests
// ============================================================================

TEST_F(LoggerTest, SingletonReturnsSameInstanceForSameScope) {
    Logger& logger1 = Logger::getInstance("TestScope");
    Logger& logger2 = Logger::getInstance("TestScope");
    EXPECT_EQ(&logger1, &logger2);
}

TEST_F(LoggerTest, SingletonReturnsDifferentInstancesForDifferentScopes) {
    Logger& logger1 = Logger::getInstance("Scope1");
    Logger& logger2 = Logger::getInstance("Scope2");
    EXPECT_NE(&logger1, &logger2);
}

// ============================================================================
// Log Level Tests
// ============================================================================

TEST_F(LoggerTest, DefaultLogLevelIsInfo) {
    Logger& logger = Logger::getInstance("DefaultLevel", LogLevel::INFO);

    logger.trace("trace");
    logger.debug("debug");
    logger.info("info");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    // Wait for log file
    EXPECT_EQ(output.find("trace"), std::string::npos);
    EXPECT_EQ(output.find("debug"), std::string::npos);
    EXPECT_NE(output.find("info"), std::string::npos);
}

TEST_F(LoggerTest, MinLogLevelFilteringWorks) {
    Logger::setMinLogLevel(LogLevel::WARNING);
    Logger& logger = Logger::getInstance("FilterTest");

    logger.info("info");
    logger.warning("warning");
    logger.error("error");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_EQ(output.find("info"), std::string::npos);
    EXPECT_NE(output.find("warning"), std::string::npos);
    EXPECT_NE(output.find("error"), std::string::npos);
}

TEST_F(LoggerTest, TraceLogLevelLogsAll) {
    Logger::setMinLogLevel(LogLevel::TRACE);
    Logger& logger = Logger::getInstance("TraceTest", LogLevel::TRACE);

    logger.trace("trace_msg");
    logger.debug("debug_msg");
    logger.info("info_msg");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("trace_msg"), std::string::npos);
    EXPECT_NE(output.find("debug_msg"), std::string::npos);
    EXPECT_NE(output.find("info_msg"), std::string::npos);
}

// ============================================================================
// Individual Log Method Tests
// ============================================================================

TEST_F(LoggerTest, TraceMethodWorks) {
    Logger::setMinLogLevel(LogLevel::TRACE);
    Logger& logger = Logger::getInstance("TraceScope", LogLevel::TRACE);

    logger.trace("trace message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[TRACE]"), std::string::npos);
    EXPECT_NE(output.find("[TraceScope]"), std::string::npos);
    EXPECT_NE(output.find("trace message"), std::string::npos);
}

TEST_F(LoggerTest, DebugMethodWorks) {
    Logger::setMinLogLevel(LogLevel::DEBUG);
    Logger& logger = Logger::getInstance("DebugScope", LogLevel::DEBUG);

    logger.debug("debug message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[DEBUG]"), std::string::npos);
    EXPECT_NE(output.find("[DebugScope]"), std::string::npos);
    EXPECT_NE(output.find("debug message"), std::string::npos);
}

TEST_F(LoggerTest, InfoMethodWorks) {
    Logger& logger = Logger::getInstance("InfoScope", LogLevel::INFO);

    logger.info("info message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[INFO]"), std::string::npos);
    EXPECT_NE(output.find("[InfoScope]"), std::string::npos);
    EXPECT_NE(output.find("info message"), std::string::npos);
}

TEST_F(LoggerTest, WarningMethodWorks) {
    Logger& logger = Logger::getInstance("WarnScope", LogLevel::WARNING);

    logger.warning("warning message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[WARNING]"), std::string::npos);
    EXPECT_NE(output.find("[WarnScope]"), std::string::npos);
    EXPECT_NE(output.find("warning message"), std::string::npos);
}

TEST_F(LoggerTest, ErrorMethodWorks) {
    Logger& logger = Logger::getInstance("ErrorScope", LogLevel::ERROR);

    logger.error("error message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[ERROR]"), std::string::npos);
    EXPECT_NE(output.find("[ErrorScope]"), std::string::npos);
    EXPECT_NE(output.find("error message"), std::string::npos);
}

TEST_F(LoggerTest, FatalMethodWorks) {
    Logger& logger = Logger::getInstance("FatalScope", LogLevel::FATAL);

    logger.fatal("fatal message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[FATAL]"), std::string::npos);
    EXPECT_NE(output.find("[FatalScope]"), std::string::npos);
    EXPECT_NE(output.find("fatal message"), std::string::npos);
}

TEST_F(LoggerTest, LogMethodUsesDefaultLevel) {
    Logger& logger = Logger::getInstance("DefaultLog", LogLevel::WARNING);

    logger.log("default log");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[WARNING]"), std::string::npos);
    EXPECT_NE(output.find("default log"), std::string::npos);
}

// ============================================================================
// Scope Management Tests
// ============================================================================

TEST_F(LoggerTest, MultipleScopesWorkIndependently) {
    Logger& logger1 = Logger::getInstance("Service1", LogLevel::INFO);
    Logger& logger2 = Logger::getInstance("Service2", LogLevel::WARNING);

    Logger::setMinLogLevel(LogLevel::TRACE);

    logger1.info("service1 info");
    logger2.info("service2 info");
    logger2.warning("service2 warning");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("[Service1]"), std::string::npos);
    EXPECT_NE(output.find("[Service2]"), std::string::npos);
    EXPECT_NE(output.find("service1 info"), std::string::npos);
    EXPECT_NE(output.find("service2 info"), std::string::npos);
    EXPECT_NE(output.find("service2 warning"), std::string::npos);
}

TEST_F(LoggerTest, GetInstanceUpdatesLogLevel) {
    Logger& logger = Logger::getInstance("UpdateTest", LogLevel::ERROR);

    logger.info("first");
    Logger::flush();  // Wait for async processing
    clearOutput();

    Logger& same_logger = Logger::getInstance("UpdateTest", LogLevel::INFO);
    EXPECT_EQ(&logger, &same_logger);

    same_logger.info("second");
    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    EXPECT_NE(output.find("second"), std::string::npos);
}

// ============================================================================
// Thread Safety Test
// ============================================================================

// Note: Concurrent access test disabled - requires thread-safe output capture
// TEST_F(LoggerTest, ConcurrentAccessIsSafe) {
//     const int num_threads = 10;
//     const int logs_per_thread = 100;
//
//     auto log_func = [](int thread_id) {
//         Logger& logger = Logger::getInstance("ThreadTest", LogLevel::INFO);
//         for (int i = 0; i < logs_per_thread; ++i) {
//             logger.info("Thread " + std::to_string(thread_id) + " msg " + std::to_string(i));
//         }
//     };
//
//     std::vector<std::thread> threads;
//     for (int i = 0; i < num_threads; ++i) {
//         threads.emplace_back(log_func, i);
//     }
//
//     for (auto& thread : threads) {
//         thread.join();
//     }
//
//     // Verify that we got all expected log messages
//     std::string output = getOutput();
//     int count = 0;
//     size_t pos = 0;
//     while ((pos = output.find("[INFO]", pos)) != std::string::npos) {
//         ++count;
//         ++pos;
//     }
//
//     EXPECT_EQ(count, num_threads * logs_per_thread);
// }

// ============================================================================
// Output Format Test
// ============================================================================

TEST_F(LoggerTest, OutputContainsAllRequiredElements) {
    Logger& logger = Logger::getInstance("FormatTest", LogLevel::INFO);

    logger.info("test message");

    Logger::flush();  // Wait for async processing
    std::string output = getOutput();
    // Should contain timestamp (contains day/month), log level, scope, and message
    EXPECT_NE(output.find("[INFO]"), std::string::npos);
    EXPECT_NE(output.find("[FormatTest]"), std::string::npos);
    EXPECT_NE(output.find("test message"), std::string::npos);
    // Basic timestamp check - should contain colon (time separator)
    EXPECT_NE(output.find(':'), std::string::npos);
}
