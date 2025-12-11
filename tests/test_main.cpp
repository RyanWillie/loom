// Main test runner for Google Test
// Individual test suites are in separate files:
// - test_dtypes.cpp: Data type utilities
// - test_device.cpp: Device abstraction
// - test_allocator_registry.cpp: Allocator registry
// - test_storage.cpp: Storage class
// - test_type_traits.cpp: C++20 concepts and type traits
// - test_logger.cpp: Logger functionality

#include "loom/logger.h"
#include <gtest/gtest.h>

// ============================================================================
// Global Test Environment for Logger Management
// ============================================================================
// This environment is shared across ALL test files and ensures:
// 1. Logger is pre-initialized before any tests run (prevents race conditions)
// 2. Logger thread is properly shut down after all tests complete (prevents hanging)
// ============================================================================

class GlobalLoggerTestEnvironment : public ::testing::Environment {
  public:
    ~GlobalLoggerTestEnvironment() override = default;

    void SetUp() override {
        // Pre-initialize the logger before any tests run
        // This prevents race conditions during concurrent logger initialization
        auto& logger = Logger::getInstance("TestSuite");
        logger.info("Global test environment initialized - all tests starting");
    }

    void TearDown() override {
        // Shutdown logger thread to prevent hanging at exit
        auto& logger = Logger::getInstance("TestSuite");
        logger.info("Global test environment shutting down - all tests completed");
        Logger::flush();  // Ensure all messages are written
        Logger::shutdown();
    }
};

// Register the global test environment
// GoogleTest will automatically manage its lifecycle
static ::testing::Environment* const global_logger_env =
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerTestEnvironment);

// Note: We still use Google Test's main() from gtest_main library
// The global environment is registered before main() runs via static initialization
