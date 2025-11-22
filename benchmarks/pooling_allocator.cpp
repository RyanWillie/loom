

/**
 * Allocator Performance Benchmarks
 *
 * Compare performance of different allocator implementations:
 * - BasicAllocator: Simple malloc/free wrapper
 * - PoolingAllocator: Advanced allocator with pooling (when implemented)
 *
 * Add your own benchmarks here following the pattern below.
 */

#include "common/device.h"
#include "cpu/basic_allocator.h"
#include "cpu/pooling_allocator.h"
#include <benchmark/benchmark.h>

using namespace loom;

//! Single Allocation Benchmarks

static void BM_PoolingAllocator_SingleAlloc_64B(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        void* ptr = allocator.allocate(64);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr);
    }
}
BENCHMARK(BM_PoolingAllocator_SingleAlloc_64B);

static void BM_PoolingAllocator_SingleAlloc_1KB(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        void* ptr = allocator.allocate(1024);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr);
    }
}
BENCHMARK(BM_PoolingAllocator_SingleAlloc_1KB);

static void BM_PoolingAllocator_SingleAlloc_1MB(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        void* ptr = allocator.allocate(1024 * 1024);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr);
    }
}
BENCHMARK(BM_PoolingAllocator_SingleAlloc_1MB);

static void BM_PoolingAllocator_SingleAlloc_10MB(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        void* ptr = allocator.allocate(10 * 1024 * 1024);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr);
    }
}
BENCHMARK(BM_PoolingAllocator_SingleAlloc_10MB);

static void BM_PoolingAllocator_SingleAlloc_100MB(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        void* ptr = allocator.allocate(100 * 1024 * 1024);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr);
    }
}
BENCHMARK(BM_PoolingAllocator_SingleAlloc_100MB);

static void BM_PoolingAllocator_SingleAlloc_1GB(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        void* ptr = allocator.allocate(1024 * 1024 * 1024);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr);
    }
}
BENCHMARK(BM_PoolingAllocator_SingleAlloc_1GB);

//! Multiple Allocation Benchmarks

static void BM_PoolingAllocator_MultipleAllocations_10MB(benchmark::State& state) {
    PoolingAllocator allocator;  // Use default constructor

    for (auto _ : state) {
        std::vector<void*> ptrs;
        for (int i = 0; i < state.range(0); i++) {
            ptrs.push_back(allocator.allocate(10 * 1024 * 1024));
        }

        for (auto ptr : ptrs) {
            benchmark::DoNotOptimize(ptr);
            allocator.deallocate(ptr);
        }
    }
}
BENCHMARK(BM_PoolingAllocator_MultipleAllocations_10MB)
    ->Range(8, 128)        // 8 to 128 allocations (80MB to 1.25GB)
    ->RangeMultiplier(2);  // Double each time: 8, 16, 32, 64, 128

//! Memory Reuse Benchmarks - Tests allocator's ability to reuse freed memory

static void BM_PoolingAllocator_MemoryReuse_1MB(benchmark::State& state) {
    PoolingAllocator allocator;
    const size_t block_size = 1024 * 1024;  // 1MB
    const int num_blocks = state.range(0);

    for (auto _ : state) {
        // Phase 1: Allocate N blocks
        std::vector<void*> ptrs;
        for (int i = 0; i < num_blocks; i++) {
            ptrs.push_back(allocator.allocate(block_size));
        }

        // Phase 2: Deallocate all blocks
        for (auto ptr : ptrs) {
            allocator.deallocate(ptr);
        }

        // Phase 3: Reallocate N blocks (should reuse freed memory)
        std::vector<void*> reused_ptrs;
        for (int i = 0; i < num_blocks; i++) {
            reused_ptrs.push_back(allocator.allocate(block_size));
        }

        // Cleanup
        for (auto ptr : reused_ptrs) {
            benchmark::DoNotOptimize(ptr);
            allocator.deallocate(ptr);
        }
    }
}
BENCHMARK(BM_PoolingAllocator_MemoryReuse_1MB)
    ->Arg(8)     // 8MB total
    ->Arg(16)    // 16MB total
    ->Arg(32)    // 32MB total
    ->Arg(64)    // 64MB total
    ->Arg(128);  // 128MB total

static void BM_PoolingAllocator_MemoryReuse_10MB(benchmark::State& state) {
    PoolingAllocator allocator;
    const size_t block_size = 10 * 1024 * 1024;  // 10MB
    const int num_blocks = state.range(0);

    for (auto _ : state) {
        // Phase 1: Allocate N blocks
        std::vector<void*> ptrs;
        for (int i = 0; i < num_blocks; i++) {
            ptrs.push_back(allocator.allocate(block_size));
        }

        // Phase 2: Deallocate all blocks
        for (auto ptr : ptrs) {
            allocator.deallocate(ptr);
        }

        // Phase 3: Reallocate N blocks (should reuse freed memory)
        std::vector<void*> reused_ptrs;
        for (int i = 0; i < num_blocks; i++) {
            reused_ptrs.push_back(allocator.allocate(block_size));
        }

        // Cleanup
        for (auto ptr : reused_ptrs) {
            benchmark::DoNotOptimize(ptr);
            allocator.deallocate(ptr);
        }
    }
}
BENCHMARK(BM_PoolingAllocator_MemoryReuse_10MB)
    ->Arg(4)    // 40MB total
    ->Arg(8)    // 80MB total
    ->Arg(16)   // 160MB total
    ->Arg(32);  // 320MB total

static void BM_PoolingAllocator_MemoryReuse_MixedSizes(benchmark::State& state) {
    PoolingAllocator allocator;
    // Mix of small, medium, and large allocations
    const std::vector<size_t> sizes = {
        1024,             // 1KB
        64 * 1024,        // 64KB
        1024 * 1024,      // 1MB
        10 * 1024 * 1024  // 10MB
    };

    for (auto _ : state) {
        // Phase 1: Allocate blocks of various sizes
        std::vector<void*> ptrs;
        for (int i = 0; i < state.range(0); i++) {
            size_t size = sizes[i % sizes.size()];
            ptrs.push_back(allocator.allocate(size));
        }

        // Phase 2: Deallocate all
        for (auto ptr : ptrs) {
            allocator.deallocate(ptr);
        }

        // Phase 3: Reallocate same pattern
        std::vector<void*> reused_ptrs;
        for (int i = 0; i < state.range(0); i++) {
            size_t size = sizes[i % sizes.size()];
            reused_ptrs.push_back(allocator.allocate(size));
        }

        // Cleanup
        for (auto ptr : reused_ptrs) {
            benchmark::DoNotOptimize(ptr);
            allocator.deallocate(ptr);
        }
    }
}
BENCHMARK(BM_PoolingAllocator_MemoryReuse_MixedSizes)
    ->Arg(16)    // 16 allocations (mixed sizes)
    ->Arg(64)    // 64 allocations
    ->Arg(256);  // 256 allocations