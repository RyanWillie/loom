# Performance Benchmarks

Performance tests for comparing different implementations and optimizations.

## Directory Structure

```
benchmarks/
├── CMakeLists.txt           # Build configuration
├── bench_main.cpp           # Main entry point (Google Benchmark)
├── bench_allocators.cpp     # Allocator performance tests
└── README.md                # This file
```

## Running Benchmarks

### Build and run all benchmarks:
```bash
cd build
./benchmarks/benchmarks
```

### Run specific benchmark:
```bash
./benchmarks/benchmarks --benchmark_filter=BasicAllocator
```

### Get more detailed output:
```bash
./benchmarks/benchmarks --benchmark_repetitions=10
```

### Export results to CSV:
```bash
./benchmarks/benchmarks --benchmark_format=csv > results.csv
```

### Export results to JSON:
```bash
./benchmarks/benchmarks --benchmark_format=json > results.json
```

## Writing Benchmarks

### Basic Pattern:
```cpp
#include <benchmark/benchmark.h>

static void BM_YourBenchmark(benchmark::State& state) {
    // Setup (runs once)
    YourClass obj;
    
    // This loop runs many times
    for (auto _ : state) {
        // Code to benchmark
        obj.operation();
        
        // Prevent compiler optimization
        benchmark::DoNotOptimize(obj);
    }
}
BENCHMARK(BM_YourBenchmark);
```

### Parameterized Benchmark:
```cpp
static void BM_AllocSize(benchmark::State& state) {
    Allocator alloc;
    const size_t size = state.range(0);
    
    for (auto _ : state) {
        void* ptr = alloc.allocate(size);
        benchmark::DoNotOptimize(ptr);
        alloc.deallocate(ptr);
    }
}
BENCHMARK(BM_AllocSize)->Range(64, 1<<20);  // 64B to 1MB
```

### With Setup/Teardown:
```cpp
static void BM_WithSetup(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        // Setup per iteration (not measured)
        auto data = prepareData();
        state.ResumeTiming();
        
        // Measured code
        processData(data);
    }
}
BENCHMARK(BM_WithSetup);
```

## Tips

1. **Always use `benchmark::DoNotOptimize()`** to prevent compiler from optimizing away your code
2. **Use `state.PauseTiming()` / `ResumeTiming()`** to exclude setup/teardown from measurements
3. **Run multiple repetitions** (`--benchmark_repetitions`) for statistical significance
4. **Compare implementations side-by-side** by naming them consistently (e.g., `BM_Basic_*` vs `BM_Pooled_*`)
5. **Build in Release mode** for accurate performance measurements

## Useful Flags

- `--benchmark_filter=<regex>` - Run only matching benchmarks
- `--benchmark_repetitions=N` - Run each benchmark N times
- `--benchmark_min_time=<time>` - Minimum time to run each benchmark
- `--benchmark_format=<console|json|csv>` - Output format
- `--benchmark_out=<file>` - Write results to file
- `--benchmark_list_tests` - List all benchmarks without running

## Visualizing Results

### Quick Start

```bash
# 1. Run benchmarks and export to JSON
./build.sh bench-release --benchmark_format=json > benchmarks/results.json

# 2. Install Python dependencies (first time only)
pip3 install matplotlib numpy

# 3. Generate visualizations
python3 benchmarks/visualize_benchmarks.py benchmarks/results.json
```

This will create `benchmarks/visualizations/` directory with:
- **single_allocations.png** - Bar chart comparing single allocation performance
- **memory_reuse_*.png** - Line plots showing memory reuse patterns
- **speedup_comparison.png** - Speedup ratios (>1.0 means CPUAllocator is faster)
- **summary.txt** - Text table with all results

### Comparing Different Runs

```bash
# Run baseline
./build.sh bench-release --benchmark_format=json > benchmarks/baseline.json

# Make changes to your allocator...

# Run comparison
./build.sh bench-release --benchmark_format=json > benchmarks/improved.json

# Visualize both
python3 benchmarks/visualize_benchmarks.py benchmarks/baseline.json
python3 benchmarks/visualize_benchmarks.py benchmarks/improved.json
```

## References

- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [Google Benchmark User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)

