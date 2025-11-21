#!/bin/bash

# Build script for MNIST Neural Network Project
# This script simplifies building the project with different configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_DIR="build"
BUILD_TYPE="Debug"
GENERATOR="Ninja"
CMAKE_ARGS=""
ENABLE_CUDA=ON
JOBS=$(nproc 2>/dev/null || echo 4)

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display help
show_help() {
    cat << EOF
${GREEN}MNIST Neural Network Build Script${NC}

Usage: ./build.sh [COMMAND] [OPTIONS]

${YELLOW}COMMANDS:${NC}
    debug           Build with Debug configuration (default)
    release         Build with Release configuration (optimized)
    relwithdebinfo  Build with optimizations + debug symbols
    asan            Build with AddressSanitizer (memory error detection)
    ubsan           Build with UndefinedBehaviorSanitizer
    tsan            Build with ThreadSanitizer (data race detection)
    msan            Build with MemorySanitizer (uninitialized memory)
    tidy            Build with clang-tidy static analysis
    format          Build with clang-format enabled
    test            Build and run tests
    bench           Run benchmarks (uses debug build)
    bench-release   Run benchmarks in Release mode (recommended)
    run             Run the executable (default: debug build)
    clean           Remove all build directories
    help            Show this help message

${YELLOW}OPTIONS:${NC}
    -g, --generator GENERATOR   CMake generator (default: Ninja)
    -j, --jobs N                Number of parallel jobs (default: $(nproc 2>/dev/null || echo 4))
    --no-cuda                   Build without CUDA support
    --make                      Use Make instead of Ninja

${YELLOW}EXAMPLES:${NC}
    ./build.sh debug                    # Build debug version
    ./build.sh release --jobs 8         # Build release with 8 parallel jobs
    ./build.sh asan                     # Build with AddressSanitizer
    ./build.sh tidy                     # Build with clang-tidy checks
    ./build.sh test                     # Build and run all tests
    ./build.sh bench                    # Run benchmarks (debug)
    ./build.sh bench-release            # Run benchmarks (Release, recommended)
    ./build.sh run                      # Run debug executable
    ./build.sh run release              # Run release executable
    ./build.sh run -- --arg1 --arg2     # Run with arguments
    ./build.sh clean                    # Clean all build artifacts

${YELLOW}SANITIZERS:${NC}
    - ASan: Detects memory leaks, buffer overflows, use-after-free
    - UBSan: Detects undefined behavior (integer overflow, etc.)
    - TSan: Detects data races (for multi-threaded code)
    - MSan: Detects uninitialized memory reads (Clang only)

${YELLOW}BENCHMARKS:${NC}
    - Always use 'bench-release' for accurate performance measurements
    - Pass Google Benchmark flags: ./build.sh bench-release -- --benchmark_filter=Alloc
    - Common flags: --benchmark_repetitions=N, --benchmark_format=csv

${YELLOW}NOTES:${NC}
    - Sanitizers are incompatible with CUDA. CPU-only builds will be used.
    - Benchmarks should be run in Release mode for accurate timing.

EOF
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to clean build directories
clean_builds() {
    print_info "Cleaning build directories..."
    rm -rf build build-* compile_commands.json
    print_success "Clean complete!"
}

# Function to configure and build
build_project() {
    local build_dir=$1
    local build_type=$2
    local extra_args=$3

    print_info "Build Directory: ${build_dir}"
    print_info "Build Type: ${build_type}"
    print_info "Generator: ${GENERATOR}"
    print_info "Jobs: ${JOBS}"

    # Check for Ninja
    if [ "$GENERATOR" = "Ninja" ] && ! command_exists ninja; then
        print_warning "Ninja not found, falling back to Make"
        GENERATOR="Unix Makefiles"
    fi

    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"

    # Configure
    print_info "Running CMake configuration..."
    cmake .. \
        -G "$GENERATOR" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        $extra_args \
        $CMAKE_ARGS

    # Build
    print_info "Building project..."
    cmake --build . --parallel $JOBS

    cd ..
    
    # Create symlink to compile_commands.json for IDE integration
    if [ -f "$build_dir/compile_commands.json" ]; then
        ln -sf "$build_dir/compile_commands.json" compile_commands.json
        print_info "Created symlink to compile_commands.json"
    fi

    print_success "Build complete! Executable: ${build_dir}/loom"
}

# Parse command line arguments
COMMAND=${1:-debug}
shift || true  # Remove first argument, ignore error if no args

# Special handling for commands that take arbitrary args (bench, bench-release, run)
if [[ "$COMMAND" == "bench" || "$COMMAND" == "bench-release" || "$COMMAND" == "run" ]]; then
    # Don't parse remaining arguments, pass them through
    :
else
    # Parse options for other commands
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--generator)
                GENERATOR="$2"
                shift 2
                ;;
            -j|--jobs)
                JOBS="$2"
                shift 2
                ;;
            --no-cuda)
                ENABLE_CUDA=OFF
                shift
                ;;
            --make)
                GENERATOR="Unix Makefiles"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
fi

# Execute command
case $COMMAND in
    debug)
        BUILD_DIR="build"
        BUILD_TYPE="Debug"
        build_project "$BUILD_DIR" "$BUILD_TYPE" ""
        ;;
    
    release)
        BUILD_DIR="build-release"
        BUILD_TYPE="Release"
        build_project "$BUILD_DIR" "$BUILD_TYPE" ""
        ;;
    
    relwithdebinfo)
        BUILD_DIR="build-relwithdebinfo"
        BUILD_TYPE="RelWithDebInfo"
        build_project "$BUILD_DIR" "$BUILD_TYPE" ""
        ;;
    
    asan)
        print_warning "Building with AddressSanitizer (CPU only)"
        BUILD_DIR="build-asan"
        BUILD_TYPE="Debug"
        EXTRA_ARGS="-DENABLE_SANITIZER_ADDRESS=ON"
        build_project "$BUILD_DIR" "$BUILD_TYPE" "$EXTRA_ARGS"
        print_info "Run with: ASAN_OPTIONS=detect_leaks=1 ./${BUILD_DIR}/loom"
        ;;
    
    ubsan)
        print_warning "Building with UndefinedBehaviorSanitizer (CPU only)"
        BUILD_DIR="build-ubsan"
        BUILD_TYPE="Debug"
        EXTRA_ARGS="-DENABLE_SANITIZER_UNDEFINED=ON"
        build_project "$BUILD_DIR" "$BUILD_TYPE" "$EXTRA_ARGS"
        ;;
    
    tsan)
        print_warning "Building with ThreadSanitizer (CPU only)"
        BUILD_DIR="build-tsan"
        BUILD_TYPE="Debug"
        EXTRA_ARGS="-DENABLE_SANITIZER_THREAD=ON"
        build_project "$BUILD_DIR" "$BUILD_TYPE" "$EXTRA_ARGS"
        ;;
    
    msan)
        if ! command_exists clang++; then
            print_error "MemorySanitizer requires Clang compiler"
            exit 1
        fi
        print_warning "Building with MemorySanitizer (Clang only, CPU only)"
        BUILD_DIR="build-msan"
        BUILD_TYPE="Debug"
        EXTRA_ARGS="-DENABLE_SANITIZER_MEMORY=ON -DCMAKE_CXX_COMPILER=clang++"
        build_project "$BUILD_DIR" "$BUILD_TYPE" "$EXTRA_ARGS"
        ;;
    
    tidy)
        if ! command_exists clang-tidy; then
            print_error "clang-tidy not found. Please install it first."
            exit 1
        fi
        print_info "Building with clang-tidy static analysis..."
        BUILD_DIR="build-tidy"
        BUILD_TYPE="Debug"
        EXTRA_ARGS="-DENABLE_CLANG_TIDY=ON"
        build_project "$BUILD_DIR" "$BUILD_TYPE" "$EXTRA_ARGS"
        ;;
    
    format)
        if ! command_exists clang-format; then
            print_error "clang-format not found. Please install it first."
            exit 1
        fi
        print_info "Building with clang-format enabled..."
        BUILD_DIR="build"
        BUILD_TYPE="Debug"
        EXTRA_ARGS="-DENABLE_CLANG_FORMAT=ON"
        build_project "$BUILD_DIR" "$BUILD_TYPE" "$EXTRA_ARGS"
        
        print_info "Running format check..."
        cd "$BUILD_DIR"
        cmake --build . --target format-check || {
            print_warning "Format check failed. Run './build.sh format-fix' to apply fixes."
        }
        cd ..
        ;;
    
    format-check)
        BUILD_DIR="build"
        if [ ! -d "$BUILD_DIR" ]; then
            print_error "Build directory not found. Run './build.sh format' first."
            exit 1
        fi
        cd "$BUILD_DIR"
        cmake --build . --target format-check
        cd ..
        ;;
    
    format-fix)
        BUILD_DIR="build"
        if [ ! -d "$BUILD_DIR" ]; then
            print_info "Creating build directory for formatting..."
            mkdir -p "$BUILD_DIR"
            cd "$BUILD_DIR"
            cmake .. -G "$GENERATOR" -DENABLE_CLANG_FORMAT=ON
            cd ..
        fi
        print_info "Applying code formatting..."
        cd "$BUILD_DIR"
        cmake --build . --target format-fix
        cd ..
        print_success "Code formatted successfully!"
        ;;
    
    test)
        # Build first if not exists
        if [ ! -d "build" ]; then
            print_info "Build directory not found, building first..."
            build_project "build" "Debug" ""
        fi
        
        print_info "Running tests..."
        cd build
        ctest --output-on-failure
        cd ..
        print_success "Tests complete!"
        ;;
    
    bench)
        # Build first if not exists
        if [ ! -d "build" ]; then
            print_info "Build directory not found, building first..."
            build_project "build" "Debug" ""
        fi
        
        print_warning "Running benchmarks in Debug mode (for accurate results, use 'bench-release')"
        print_info "Running benchmarks..."
        
        # Check if benchmark executable exists
        if [ ! -f "build/benchmarks/benchmarks" ]; then
            print_error "Benchmark executable not found. Build failed?"
            exit 1
        fi
        
        cd build
        ./benchmarks/benchmarks "$@"
        cd ..
        print_success "Benchmarks complete!"
        ;;
    
    bench-release)
        # Build release if not exists
        if [ ! -d "build-release" ]; then
            print_info "Release build not found, building first..."
            build_project "build-release" "Release" ""
        fi
        
        print_info "Running benchmarks in Release mode..."
        
        # Check if benchmark executable exists
        if [ ! -f "build-release/benchmarks/benchmarks" ]; then
            print_error "Benchmark executable not found. Build failed?"
            exit 1
        fi
        
        cd build-release
        ./benchmarks/benchmarks "$@"
        cd ..
        print_success "Benchmarks complete!"
        ;;
    
    run)
        # Determine which build to run
        RUN_BUILD_TYPE=${1:-debug}
        
        # Map build type to directory
        case $RUN_BUILD_TYPE in
            debug)
                RUN_BUILD_DIR="build"
                ;;
            release)
                RUN_BUILD_DIR="build-release"
                ;;
            relwithdebinfo)
                RUN_BUILD_DIR="build-relwithdebinfo"
                ;;
            asan)
                RUN_BUILD_DIR="build-asan"
                ;;
            ubsan)
                RUN_BUILD_DIR="build-ubsan"
                ;;
            tsan)
                RUN_BUILD_DIR="build-tsan"
                ;;
            msan)
                RUN_BUILD_DIR="build-msan"
                ;;
            tidy)
                RUN_BUILD_DIR="build-tidy"
                ;;
            --)
                # No build type specified, just arguments
                RUN_BUILD_DIR="build"
                ;;
            *)
                # Assume it's the start of arguments, use default build
                RUN_BUILD_DIR="build"
                # Put the argument back for later
                set -- "$RUN_BUILD_TYPE" "$@"
                ;;
        esac
        
        # Skip the build type argument if it was consumed
        if [ "$RUN_BUILD_TYPE" != "--" ] && [ "$RUN_BUILD_TYPE" = "${1:-}" ]; then
            shift || true
        fi
        
        # Skip -- separator if present
        if [ "${1:-}" = "--" ]; then
            shift
        fi
        
        EXECUTABLE="${RUN_BUILD_DIR}/loom"
        
        # Check if executable exists
        if [ ! -f "$EXECUTABLE" ]; then
            print_error "Executable not found: $EXECUTABLE"
            print_info "Build it first with: ./build.sh ${RUN_BUILD_TYPE}"
            exit 1
        fi
        
        print_info "Running: $EXECUTABLE $@"
        "./$EXECUTABLE" "$@"
        ;;
    
    clean)
        clean_builds
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

print_success "Done!"


