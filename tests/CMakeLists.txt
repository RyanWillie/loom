# Locate all test files
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(unit_tests ${TEST_SOURCES})

# Link against loom library and GoogleTest
target_link_libraries(unit_tests PRIVATE loom GTest::gtest_main)

# Apply clang-tidy only to our test targets, not dependencies
if(ENABLE_CLANG_TIDY AND CLANG_TIDY_COMMAND)
    set_target_properties(unit_tests PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)

