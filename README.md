# C++ MNIST Neural Network

This project implements a neural network from scratch to solve the MNIST digit recognition problem. It includes implementations for both CPU (learning purpose) and GPU (CUDA) without using external linear algebra libraries.

## Project Structure

- `src/`: Source code
  - `common/`: Shared logic (Data loading, Matrix math)
  - `cpu/`: CPU-specific implementation
  - `gpu/`: CUDA kernels
- `include/`: Header files
- `tests/`: GoogleTest suite
- `docker/`: Dockerfiles for local and cloud environments
- `.devcontainer/`: VS Code Dev Container configuration
- `.vscode/`: VS Code debugging and task configurations

## Quick Start

The project includes a convenient build script that handles all build configurations:

```bash
# Simple debug build (fastest for development)
./build.sh debug

# Optimized release build
./build.sh release

# Build and run tests
./build.sh test

# Build with AddressSanitizer (memory error detection)
./build.sh asan

# Show all available options
./build.sh help
```

## Requirements

### Essential
- CMake 3.14+
- C++17 compatible compiler (GCC, Clang, or MSVC)
- Ninja build system (recommended) or Make

### Optional
- CUDA Toolkit (for GPU acceleration)
- clang-format (for code formatting)
- clang-tidy (for static analysis)
- GDB or LLDB (for debugging)

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake ninja-build clang-format clang-tidy gdb
```

**macOS (Homebrew):**
```bash
brew install cmake ninja llvm
```

## Building

### Using the Build Script (Recommended)

The `build.sh` script provides easy access to all build configurations:

```bash
# Debug build with symbols
./build.sh debug

# Optimized release build
./build.sh release

# Release with debug info (for profiling)
./build.sh relwithdebinfo

# Build with sanitizers
./build.sh asan              # AddressSanitizer
./build.sh ubsan             # UndefinedBehaviorSanitizer
./build.sh tsan              # ThreadSanitizer
./build.sh msan              # MemorySanitizer (Clang only)

# Build with static analysis
./build.sh tidy              # Run clang-tidy checks

# Code formatting
./build.sh format-check      # Check formatting
./build.sh format-fix        # Auto-format code

# Build options
./build.sh release --jobs 8  # Use 8 parallel jobs
./build.sh debug --no-cuda   # Build without CUDA
./build.sh debug --make      # Use Make instead of Ninja

# Clean all build artifacts
./build.sh clean
```

### Manual CMake Build

If you prefer manual control:

```bash
mkdir build && cd build

# Basic build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build . --parallel

# With sanitizers
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZER_ADDRESS=ON
cmake --build . --parallel

# With clang-tidy
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_CLANG_TIDY=ON
cmake --build . --parallel

# With clang-format
cmake .. -G Ninja -DENABLE_CLANG_FORMAT=ON
cmake --build . --target format-fix
```

## Development Tools

### Code Formatting (clang-format)

The project uses clang-format for consistent code style:

```bash
# Check if code is properly formatted
./build.sh format-check

# Automatically format all code
./build.sh format-fix
```

Configuration is in `.clang-format` (based on Google C++ Style Guide with customizations).

### Static Analysis (clang-tidy)

Catch bugs and enforce best practices:

```bash
./build.sh tidy
```

Configuration is in `.clang-tidy`. The build will show warnings for:
- Potential bugs
- Performance issues
- Modernization suggestions
- Code readability issues

### Sanitizers

Runtime error detection tools:

#### AddressSanitizer (ASan)
Detects memory errors: leaks, buffer overflows, use-after-free

```bash
./build.sh asan
ASAN_OPTIONS=detect_leaks=1 ./build-asan/mnist_net
```

#### UndefinedBehaviorSanitizer (UBSan)
Detects undefined behavior: integer overflow, null pointer dereference

```bash
./build.sh ubsan
./build-ubsan/mnist_net
```

#### ThreadSanitizer (TSan)
Detects data races in multi-threaded code

```bash
./build.sh tsan
./build-tsan/mnist_net
```

#### MemorySanitizer (MSan)
Detects uninitialized memory reads (Clang only)

```bash
./build.sh msan
./build-msan/mnist_net
```

**Note:** Sanitizers are incompatible with CUDA. CPU-only builds will be used.

### Debugging

#### VS Code Integration

The project includes preconfigured debugging tasks in `.vscode/launch.json`:

1. Open the project in VS Code
2. Set breakpoints in your code
3. Press `F5` or use the "Run and Debug" panel
4. Choose a debug configuration:
   - `(gdb) Launch mnist_net` - Debug main application
   - `(gdb) Launch Tests` - Debug unit tests
   - `(gdb) Launch with ASan` - Debug with AddressSanitizer

#### Command Line Debugging

```bash
# Build with debug symbols
./build.sh debug

# Launch GDB
gdb ./build/mnist_net

# Common GDB commands
(gdb) break main          # Set breakpoint
(gdb) run                 # Start program
(gdb) next                # Step over
(gdb) step                # Step into
(gdb) print variable      # Print variable value
(gdb) backtrace           # Show call stack
```

### Profiling

#### CPU Profiling with gprof

```bash
# Build with profiling enabled
mkdir build-profile && cd build-profile
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-pg" -DCMAKE_EXE_LINKER_FLAGS="-pg"
cmake --build . --parallel

# Run program (generates gmon.out)
./mnist_net

# Generate profile report
gprof mnist_net gmon.out > profile.txt
```

#### CPU Profiling with perf

```bash
# Build with debug info
./build.sh relwithdebinfo

# Record performance data
perf record -g ./build-relwithdebinfo/mnist_net

# View report
perf report
```

#### GPU Profiling (CUDA)

```bash
# Using NVIDIA Nsight Systems
nsys profile --stats=true ./build/mnist_net

# Using nvprof (legacy)
nvprof ./build/mnist_net

# Using cuda-memcheck for memory errors
cuda-memcheck ./build/mnist_net
```

## Testing

```bash
# Build and run all tests
./build.sh test

# Run tests manually
cd build
ctest --output-on-failure

# Run specific test
./tests/unit_tests --gtest_filter=TestName.*

# Run tests with sanitizers
./build.sh asan
cd build-asan
ctest --output-on-failure
```

## Docker Support

### CPU-Only Container (Development)

The CPU container includes all development tools (ninja, clang-format, clang-tidy, sanitizers, gdb, valgrind):

```bash
# Build the image
docker build -t mnist-cpu -f docker/Dockerfile.cpu .

# Run interactively with mounted source
docker run -it -v $(pwd):/workspace mnist-cpu bash

# Inside the container, use the build script
./build.sh debug
./build.sh asan
./build.sh format-fix
```

### CUDA Container (Production)

The GPU container includes all development tools plus CUDA support. It automatically builds both release and debug versions:

```bash
# Build the image
docker build -t mnist-gpu -f docker/Dockerfile.gpu .

# Run the release build
docker run --gpus all mnist-gpu

# Run the debug build
docker run --gpus all mnist-gpu ./build/mnist_net

# Run interactively for development
docker run --gpus all -it mnist-gpu bash

# Inside the container
./build.sh test                    # Run tests
./build.sh tidy                    # Static analysis (CPU only)
nsys profile ./build/mnist_net     # Profile with Nsight Systems
```

### Available Tools in Containers

Both containers include:
- ✅ **Ninja** - Fast parallel builds
- ✅ **clang-format** - Code formatting
- ✅ **clang-tidy** - Static analysis
- ✅ **GCC & Clang** - Multiple compilers
- ✅ **GDB** - Debugging
- ✅ **Valgrind** - Memory profiling
- ✅ **Sanitizers** - ASan, UBSan, TSan (CPU container)

GPU container additionally includes:
- ✅ **CUDA Toolkit** - GPU development
- ✅ **NVIDIA Nsight** - GPU profiling (use `nsys`)
- ✅ **cuda-memcheck** - GPU memory checking

### VS Code Dev Container (Recommended for Development)

The easiest way to develop with all tools preconfigured:

1. Install [VS Code](https://code.visualstudio.com/) and [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open this folder in VS Code
3. Click "Reopen in Container" when prompted
4. All tools, extensions, and settings are automatically configured!

**Features:**
- One-click development environment
- All VS Code C++ extensions pre-installed
- Debugging configured (press F5)
- Build tasks integrated (Ctrl+Shift+B)
- IntelliSense with compile_commands.json
- No manual setup required

See `.devcontainer/README.md` for details.

## Build System Details

### Build Types

- **Debug**: No optimization, full debug symbols, assertions enabled
- **Release**: Full optimization (-O3), no debug symbols
- **RelWithDebInfo**: Optimization + debug symbols (best for profiling)
- **MinSizeRel**: Optimize for size

### CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ENABLE_CLANG_TIDY` | Enable clang-tidy static analysis | OFF |
| `ENABLE_CLANG_FORMAT` | Enable clang-format targets | OFF |
| `ENABLE_SANITIZER_ADDRESS` | Enable AddressSanitizer | OFF |
| `ENABLE_SANITIZER_UNDEFINED` | Enable UndefinedBehaviorSanitizer | OFF |
| `ENABLE_SANITIZER_THREAD` | Enable ThreadSanitizer | OFF |
| `ENABLE_SANITIZER_MEMORY` | Enable MemorySanitizer | OFF |

### Ninja vs Make

Ninja is faster for incremental builds and recommended:

```bash
# Using Ninja (default)
./build.sh debug

# Using Make
./build.sh debug --make
```

## Contributing

Before submitting code:

1. Format your code: `./build.sh format-fix`
2. Check for issues: `./build.sh tidy`
3. Run tests: `./build.sh test`
4. Test with sanitizers: `./build.sh asan`

## Troubleshooting

### Ninja not found
```bash
# Install Ninja
sudo apt-get install ninja-build  # Ubuntu/Debian
brew install ninja                 # macOS

# Or use Make instead
./build.sh debug --make
```

### clang-tidy errors
```bash
# Install clang-tidy
sudo apt-get install clang-tidy
```

### CUDA not detected
```bash
# Check CUDA installation
nvcc --version

# Build without CUDA
./build.sh debug --no-cuda
```

### Sanitizer crashes
Sanitizers have runtime overhead and may slow down execution significantly. This is expected behavior.

