# Locate all test files
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(unit_tests ${TEST_SOURCES} ${COMMON_SOURCES} ${CPU_SOURCES})

# Link GoogleTest
target_link_libraries(unit_tests GTest::gtest_main)

# If CUDA is enabled, we might want to link against CUDA runtime or include GPU sources 
# if we write GPU specific tests later. For now, we keep it simple.
if(USE_CUDA)
    target_sources(unit_tests PRIVATE ${GPU_SOURCES})
    set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Apply clang-tidy only to our test targets, not dependencies
if(ENABLE_CLANG_TIDY AND CLANG_TIDY_COMMAND)
    set_target_properties(unit_tests PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)

