# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Loom is a neural network framework implemented from scratch in C++20 for the MNIST digit recognition problem. The project includes both CPU (for learning) and GPU (CUDA) implementations without using external linear algebra libraries. The executable is named `loom`, though the CMake project is still named `mnist_neural_net`.

## Build Commands

The project uses a comprehensive `build.sh` script as the primary interface:

```bash
# Development builds
./build.sh debug              # Debug build with symbols (default)
./build.sh release            # Optimized release build

# Testing
./build.sh test               # Build and run all tests
cd build && ctest --output-on-failure                    # Run tests manually
./build/tests/unit_tests --gtest_filter=TestName.*      # Run specific test

# Benchmarking
./build.sh bench-release      # Run benchmarks (always use release mode)
./build.sh bench-release -- --benchmark_filter=Alloc    # Run specific benchmarks

# Code quality
./build.sh format-fix         # Auto-format all code (clang-format)
./build.sh format-check       # Check formatting without changes
./build.sh tidy               # Run clang-tidy static analysis

# Sanitizers for debugging
./build.sh asan               # AddressSanitizer (memory errors)
./build.sh ubsan              # UndefinedBehaviorSanitizer
./build.sh tsan               # ThreadSanitizer (data races)
./build.sh msan               # MemorySanitizer (Clang only)

# Running
./build/loom                  # Run the main executable

# Cleaning
./build.sh clean              # Remove all build directories
```

### Manual CMake Build

```bash
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build . --parallel
```

## Architecture

### Core Abstractions

The framework is built around several key abstractions:

1. **Device** (`include/common/device.h`)
   - Represents computation device: CPU, CUDA, or MPS
   - Used throughout the framework to route operations to appropriate backends

2. **Allocator** (`include/common/memory/allocator.h`)
   - Abstract interface for memory allocation on different devices
   - Implementations: `BasicAllocator` (simple malloc/free), `PoolingAllocator` (memory pooling)

3. **AllocatorRegistry** (`include/common/registry/allocator_registry.h`)
   - Global singleton registry mapping Device -> Allocator
   - Thread-safe with mutex protection
   - Automatically creates default allocators on first access

4. **Storage** (`include/common/tensor/storage.h`)
   - Low-level data container with reference counting (shared_ptr)
   - Manages raw memory via Allocator
   - Type-safe accessor with `as<T>()` template method using TensorStorageType concept
   - Supports shallow copy (copy constructor) and deep copy (clone method)

5. **Tensor** (`include/common/tensor/tensor.h`)
   - High-level n-dimensional array with shape, stride, and offset
   - Wraps Storage with shared ownership
   - Supports view operations (reshape, transpose, slice) without copying data
   - Factory methods: `zeros()`, `ones()`, `rand()`, `randn()`, `full()`

### Memory Management Strategy

- **Reference Counting**: Storage uses `shared_ptr` for automatic memory management
- **Pooling**: `PoolingAllocator` uses hash map + vectors for O(1) pool lookups
- **Size Headers**: Block size stored in header prefix (no separate metadata map)
- **Zero-size Sentinel**: Shared static sentinel for zero-size allocations
- **Alignment**: Configurable alignment support for SIMD operations

### Type System

- **DType** (`include/common/dtypes.h`): Enum for tensor data types (FLOAT32, INT32, etc.)
- **TensorStorageType** concept: Compile-time validation of valid tensor storage types
- **dtype_traits**: Template metaprogramming to map C++ types to DType enum values

### Directory Structure

```
src/
├── common/          # Core abstractions (Device, Tensor, Storage, Logger)
│   ├── memory/      # Allocator interface and utilities
│   ├── registry/    # AllocatorRegistry for device->allocator mapping
│   ├── tensor/      # Tensor and Storage implementations
│   └── dataloader/  # MNIST data loading
├── cpu/             # CPU-specific implementations (BasicAllocator, PoolingAllocator)
└── gpu/             # CUDA kernels and GPU implementations

include/
├── common/          # Headers mirroring src/common structure
├── cpu/             # CPU allocator headers
└── gpu/             # GPU headers

tests/               # GoogleTest unit tests (one file per component)
benchmarks/          # Google Benchmark performance tests
```

## Development Guidelines

### Adding New Allocators

1. Inherit from `Allocator` base class
2. Implement `allocate()`, `deallocate()`, and `device()` methods
3. Register with `AllocatorRegistry` (typically in a static initializer)
4. Add corresponding unit tests in `tests/`
5. Add performance benchmarks in `benchmarks/`

### Working with Tensors

- Tensors use shallow copying by default (shared Storage)
- Use `clone()` for deep copy when needed
- View operations (reshape, transpose, etc.) share underlying Storage
- Check `isContiguous()` before operations that require contiguous memory
- Use `contiguous()` to create contiguous copy if needed

### Code Style

- **Formatting**: Google C++ Style Guide with customizations (see `.clang-format`)
- **Naming**:
  - Classes: PascalCase (e.g., `PoolingAllocator`)
  - Member variables: mCamelCase prefix (e.g., `mFreeBlocks`)
  - Static members: sCamelCase prefix (e.g., `sZeroSizeSentinel`)
  - Functions: camelCase (e.g., `allocate()`)
- **Attributes**: Use `[[nodiscard]]` for methods that return important values
- **Namespaces**: All code in `loom` namespace

### Testing Strategy

- Unit tests use GoogleTest framework
- One test file per component (e.g., `test_storage.cpp`, `test_pooling_allocator.cpp`)
- Tests cover: basic functionality, edge cases, error handling, thread safety
- Sanitizers (ASan, UBSan, TSan) used in CI for memory safety

### Benchmarking

- Use Google Benchmark for performance testing
- Always run benchmarks in Release mode (`./build.sh bench-release`)
- Benchmark naming: `BM_ComponentName_Operation` (e.g., `BM_PoolingAllocator_AllocDealloc`)
- Use `benchmark::DoNotOptimize()` to prevent compiler optimizations
- Export to JSON: `--benchmark_format=json > results.json`

## Important Implementation Details

### Pooling Allocator Design

The `PoolingAllocator` uses a header-based size tracking approach:
- Each allocated block has a hidden header prefix containing the block size
- User receives pointer to data after the header
- On deallocation, reads the header to determine size and returns block to correct pool
- Uses `std::unordered_map<size_t, std::vector<void*>>` for O(1) lookups
- Alignment is maintained by adjusting the total allocation size

### CUDA Support

- Conditional compilation with `USE_CUDA` macro
- CUDA detection in CMakeLists.txt via `check_language(CUDA)`
- Sanitizers automatically disable CUDA (CPU-only builds)
- Separate `.cu` files for CUDA kernels in `src/gpu/`

### Logger System

- Thread-safe singleton logger (`include/common/logger.h`)
- Supports multiple named instances (e.g., "System", "CPU", "GPU")
- Log levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- Output to console, file, or both
- ANSI color support for terminal output

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Builds in Docker containers for reproducibility
- Matrix build: Debug and Release configurations
- Runs all tests with ctest
- Static analysis with clang-tidy (separate job)
- Uses Docker layer caching for faster builds

## Docker Support

- **CPU Container** (`docker/Dockerfile.cpu`): All dev tools (Ninja, clang-format, clang-tidy, sanitizers, gdb, valgrind)
- **GPU Container** (`docker/Dockerfile.gpu`): CPU tools + CUDA toolkit, Nsight profiling
- **Dev Container** (`.devcontainer/`): VS Code remote development with all extensions preconfigured
