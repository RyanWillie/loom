cmake_minimum_required(VERSION 3.14)
project(mnist_neural_net LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy and IDE integration

# --- Build Options ---
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)
option(ENABLE_CLANG_FORMAT "Enable clang-format checks" OFF)
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)
option(ENABLE_SANITIZER_ADDRESS "Enable AddressSanitizer" OFF)
option(ENABLE_SANITIZER_UNDEFINED "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_SANITIZER_THREAD "Enable ThreadSanitizer" OFF)
option(ENABLE_SANITIZER_MEMORY "Enable MemorySanitizer (Clang only)" OFF)

# --- Compiler Warnings ---
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Debug Symbols ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-g)
endif()

# --- Sanitizers ---
if(ENABLE_SANITIZER_ADDRESS)
    message(STATUS "Building with AddressSanitizer")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_SANITIZER_UNDEFINED)
    message(STATUS "Building with UndefinedBehaviorSanitizer")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

if(ENABLE_SANITIZER_THREAD)
    message(STATUS "Building with ThreadSanitizer")
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

if(ENABLE_SANITIZER_MEMORY)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Building with MemorySanitizer")
        add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
        add_link_options(-fsanitize=memory)
    else()
        message(WARNING "MemorySanitizer only supported with Clang")
    endif()
endif()

# --- Coverage ---
if(ENABLE_COVERAGE)
    message(STATUS "Building with code coverage")

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # GCC coverage flags
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)

        # Find gcov
        find_program(GCOV_PATH gcov)
        if(NOT GCOV_PATH)
            message(WARNING "gcov not found, coverage reporting may not work")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Clang coverage flags
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        add_link_options(-fprofile-instr-generate -fcoverage-mapping)

        # Find llvm-cov
        find_program(LLVM_COV_PATH llvm-cov)
        if(NOT LLVM_COV_PATH)
            message(WARNING "llvm-cov not found, coverage reporting may not work")
        endif()
    else()
        message(WARNING "Code coverage only supported with GCC or Clang")
    endif()

    # Find lcov for HTML report generation
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)

    if(LCOV_PATH AND GENHTML_PATH)
        message(STATUS "Found lcov and genhtml - HTML coverage reports available")

        # Add custom target for coverage report generation
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E echo "Generating coverage report..."

            # Create coverage directory
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage

            # Initialize coverage data (reset counters)
            COMMAND ${LCOV_PATH} --directory . --zerocounters --quiet

            # Run tests to generate coverage data
            COMMAND ${CMAKE_COMMAND} -E echo "Running tests..."
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure

            # Capture coverage data
            COMMAND ${CMAKE_COMMAND} -E echo "Capturing coverage data..."
            COMMAND ${LCOV_PATH}
                --directory .
                --capture
                --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
                --quiet

            # Remove coverage from external dependencies (googletest, etc.)
            COMMAND ${LCOV_PATH}
                --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info
                '*/googletest/*'
                '*/benchmark/*'
                '*/c++/*'
                '*/gcc/*'
                '/usr/*'
                '*/tests/*'
                '*/benchmarks/*'
                --output-file ${CMAKE_BINARY_DIR}/coverage/coverage_filtered.info
                --quiet

            # Generate HTML report
            COMMAND ${CMAKE_COMMAND} -E echo "Generating HTML report..."
            COMMAND ${GENHTML_PATH}
                ${CMAKE_BINARY_DIR}/coverage/coverage_filtered.info
                --output-directory ${CMAKE_BINARY_DIR}/coverage/html
                --title "Loom Coverage Report"
                --legend
                --demangle-cpp
                --quiet

            # Print summary
            COMMAND ${LCOV_PATH}
                --list ${CMAKE_BINARY_DIR}/coverage/coverage_filtered.info

            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated!"
            COMMAND ${CMAKE_COMMAND} -E echo "Open: ${CMAKE_BINARY_DIR}/coverage/html/index.html"

            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
        )
    else()
        message(WARNING "lcov or genhtml not found - coverage target will not be available")
        message(WARNING "Install with: sudo apt-get install lcov (Ubuntu) or brew install lcov (macOS)")
    endif()
endif()

# --- Clang-Tidy Integration ---
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        # Don't set CMAKE_CXX_CLANG_TIDY globally to avoid analyzing dependencies
        # Instead, we'll set it per-target below
        set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endif()

# --- Dependencies ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Includes ---
include_directories(include)

# --- Source Groups ---
# Common sources (always built)
file(GLOB_RECURSE COMMON_SOURCES "src/common/*.cpp")
file(GLOB_RECURSE CPU_SOURCES "src/cpu/*.cpp")

# --- CUDA Detection ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    
    # CUDA specific sources
    file(GLOB_RECURSE GPU_SOURCES "src/gpu/*.cu")
    
    add_definitions(-DUSE_CUDA)
else()
    message(STATUS "CUDA compiler NOT found. Building CPU version only.")
    set(GPU_SOURCES "")
endif()

# --- Loom Library ---
add_library(loom SHARED ${COMMON_SOURCES} ${CPU_SOURCES} ${GPU_SOURCES})

# Public include directory (users will include <loom/...>)
target_include_directories(loom
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(loom PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Apply clang-tidy only to our targets, not dependencies
if(ENABLE_CLANG_TIDY AND CLANG_TIDY_COMMAND)
    set_target_properties(loom PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

# --- Examples ---
add_subdirectory(examples)

# --- Tests ---
enable_testing()
add_subdirectory(tests)

# --- Benchmarks ---
add_subdirectory(benchmarks)

# --- Clang-Format Targets ---
if(ENABLE_CLANG_FORMAT)
    find_program(CLANG_FORMAT_EXE NAMES "clang-format")
    if(CLANG_FORMAT_EXE)
        message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
        
        # Find all source files
        file(GLOB_RECURSE ALL_SOURCE_FILES 
            ${CMAKE_SOURCE_DIR}/src/*.cpp 
            ${CMAKE_SOURCE_DIR}/src/*.cu
            ${CMAKE_SOURCE_DIR}/include/*.h
            ${CMAKE_SOURCE_DIR}/include/*.hpp
        )
        
        # Target to check formatting
        add_custom_target(
            format-check
            COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${ALL_SOURCE_FILES}
            COMMENT "Checking code formatting..."
            VERBATIM
        )
        
        # Target to apply formatting
        add_custom_target(
            format-fix
            COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
            COMMENT "Applying code formatting..."
            VERBATIM
        )
    else()
        message(WARNING "clang-format requested but not found")
    endif()
endif()

