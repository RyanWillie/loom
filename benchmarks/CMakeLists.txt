# Benchmarks using Google Benchmark
# These are performance tests, not correctness tests

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
# Don't build benchmark tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Collect all benchmark source files
file(GLOB_RECURSE BENCHMARK_SOURCES "*.cpp")

# Create benchmark executable
add_executable(benchmarks ${BENCHMARK_SOURCES})

# Link against loom library and Google Benchmark
target_link_libraries(benchmarks PRIVATE loom benchmark::benchmark)

# Apply clang-tidy if enabled
if(ENABLE_CLANG_TIDY AND CLANG_TIDY_COMMAND)
    set_target_properties(benchmarks PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

