#!/usr/bin/env python3
"""
Benchmark Visualization Tool

Parses Google Benchmark JSON output and creates comparison charts.

Usage:
    # Export benchmarks to JSON
    ./build.sh bench-release --benchmark_format=json > benchmarks/results.json
    
    # Generate visualizations
    python3 benchmarks/visualize_benchmarks.py benchmarks/results.json
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_benchmark_name(name):
    """Parse benchmark name into components."""
    parts = name.split('/')
    base_name = parts[0]
    param = parts[1] if len(parts) > 1 else None
    
    # Extract allocator type and test name
    if base_name.startswith('BM_BasicAllocator_'):
        allocator = 'BasicAllocator'
        test = base_name[len('BM_BasicAllocator_'):]
    elif base_name.startswith('BM_CPUAllocator_'):
        allocator = 'CPUAllocator'
        test = base_name[len('BM_CPUAllocator_'):]
    else:
        allocator = 'Unknown'
        test = base_name
    
    return allocator, test, param


def load_benchmark_results(json_file):
    """Load and parse benchmark JSON results."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    benchmarks = data.get('benchmarks', [])
    
    # Group benchmarks by test type
    grouped = defaultdict(lambda: {'BasicAllocator': [], 'CPUAllocator': []})
    
    for bench in benchmarks:
        name = bench['name']
        allocator, test, param = parse_benchmark_name(name)
        
        result = {
            'name': name,
            'time': bench['cpu_time'],  # nanoseconds
            'iterations': bench['iterations'],
            'param': int(param) if param else None
        }
        
        grouped[test][allocator].append(result)
    
    return grouped


def format_time(ns):
    """Format time in appropriate unit."""
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns/1000:.2f} μs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    else:
        return f"{ns/1_000_000_000:.2f} s"


def plot_single_allocations(grouped_data, output_dir):
    """Plot single allocation benchmarks."""
    test_name = 'SingleAlloc'
    tests = [k for k in grouped_data.keys() if test_name in k]
    
    if not tests:
        return
    
    # Extract allocation sizes
    sizes = []
    basic_times = []
    cpu_times = []
    
    for test in sorted(tests):
        # Extract size from test name (e.g., "SingleAlloc_1MB")
        size_str = test.replace('SingleAlloc_', '')
        
        basic = grouped_data[test]['BasicAllocator']
        cpu = grouped_data[test]['CPUAllocator']
        
        if basic and cpu:
            sizes.append(size_str)
            basic_times.append(basic[0]['time'])
            cpu_times.append(cpu[0]['time'])
    
    if not sizes:
        return
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, basic_times, width, label='BasicAllocator', alpha=0.8)
    bars2 = ax.bar(x + width/2, cpu_times, width, label='CPUAllocator', alpha=0.8)
    
    ax.set_xlabel('Allocation Size', fontsize=12)
    ax.set_ylabel('Time (ns)', fontsize=12)
    ax.set_title('Single Allocation Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_time(height),
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'single_allocations.png', dpi=300)
    print(f"✓ Generated: {output_dir / 'single_allocations.png'}")
    plt.close()


def plot_memory_reuse(grouped_data, output_dir):
    """Plot memory reuse benchmarks."""
    test_patterns = ['MemoryReuse_1MB', 'MemoryReuse_10MB', 'MemoryReuse_MixedSizes']
    
    for pattern in test_patterns:
        tests = [k for k in grouped_data.keys() if pattern in k]
        if not tests or len(tests) > 1:
            continue
        
        test = tests[0]
        basic = grouped_data[test]['BasicAllocator']
        cpu = grouped_data[test]['CPUAllocator']
        
        if not basic or not cpu:
            continue
        
        # Extract parameters and times
        basic_params = [b['param'] for b in basic if b['param']]
        basic_times = [b['time'] for b in basic if b['param']]
        cpu_params = [c['param'] for c in cpu if c['param']]
        cpu_times = [c['time'] for c in cpu if c['param']]
        
        if not basic_params or not cpu_params:
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(basic_params, basic_times, 'o-', label='BasicAllocator', 
                linewidth=2, markersize=8, alpha=0.8)
        ax.plot(cpu_params, cpu_times, 's-', label='CPUAllocator', 
                linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Number of Allocations', fontsize=12)
        ax.set_ylabel('Time (ns)', fontsize=12)
        ax.set_title(f'Memory Reuse Performance: {pattern}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        
        plt.tight_layout()
        filename = f'memory_reuse_{pattern.lower()}.png'
        plt.savefig(output_dir / filename, dpi=300)
        print(f"✓ Generated: {output_dir / filename}")
        plt.close()


def plot_speedup(grouped_data, output_dir):
    """Plot speedup ratio (BasicAllocator / CPUAllocator)."""
    tests_with_params = {}
    
    for test_name, allocators in grouped_data.items():
        basic = allocators['BasicAllocator']
        cpu = allocators['CPUAllocator']
        
        if not basic or not cpu:
            continue
        
        # Match by parameters
        for b in basic:
            for c in cpu:
                if b['param'] == c['param']:
                    if test_name not in tests_with_params:
                        tests_with_params[test_name] = []
                    
                    speedup = b['time'] / c['time']
                    tests_with_params[test_name].append({
                        'param': b['param'],
                        'speedup': speedup
                    })
    
    if not tests_with_params:
        return
    
    # Create speedup comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(tests_with_params)))
    
    for (test_name, results), color in zip(tests_with_params.items(), colors):
        if results[0]['param'] is None:
            continue
        
        params = [r['param'] for r in results]
        speedups = [r['speedup'] for r in results]
        
        ax.plot(params, speedups, 'o-', label=test_name, 
                linewidth=2, markersize=8, alpha=0.8, color=color)
    
    # Add reference line at 1.0 (no speedup)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No Speedup (1.0x)')
    
    ax.set_xlabel('Number of Allocations', fontsize=12)
    ax.set_ylabel('Speedup (BasicAllocator / CPUAllocator)', fontsize=12)
    ax.set_title('Allocator Performance Speedup (>1.0 = CPUAllocator is faster)', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'speedup_comparison.png'}")
    plt.close()


def generate_summary_table(grouped_data, output_dir):
    """Generate a text summary table."""
    output_file = output_dir / 'summary.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for test_name in sorted(grouped_data.keys()):
            basic = grouped_data[test_name]['BasicAllocator']
            cpu = grouped_data[test_name]['CPUAllocator']
            
            if not basic or not cpu:
                continue
            
            f.write(f"\n{test_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Param':<10} {'BasicAllocator':<20} {'CPUAllocator':<20} {'Speedup':<10}\n")
            f.write("-" * 80 + "\n")
            
            for b in basic:
                matching_cpu = [c for c in cpu if c['param'] == b['param']]
                if matching_cpu:
                    c = matching_cpu[0]
                    speedup = b['time'] / c['time']
                    param_str = str(b['param']) if b['param'] else 'N/A'
                    
                    f.write(f"{param_str:<10} {format_time(b['time']):<20} "
                           f"{format_time(c['time']):<20} {speedup:.2f}x\n")
    
    print(f"✓ Generated: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_benchmarks.py <benchmark_results.json>")
        print("\nFirst, export benchmarks to JSON:")
        print("  ./build.sh bench-release --benchmark_format=json > benchmarks/results.json")
        print("\nThen visualize:")
        print("  python3 benchmarks/visualize_benchmarks.py benchmarks/results.json")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    output_dir = json_file.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading benchmark results...")
    grouped_data = load_benchmark_results(json_file)
    
    print(f"\nGenerating visualizations in: {output_dir}/")
    print("-" * 60)
    
    plot_single_allocations(grouped_data, output_dir)
    plot_memory_reuse(grouped_data, output_dir)
    plot_speedup(grouped_data, output_dir)
    generate_summary_table(grouped_data, output_dir)
    
    print("-" * 60)
    print(f"\n✅ Done! Visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.iterdir()):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()

