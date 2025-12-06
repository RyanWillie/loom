#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "loom/device.h"
#include "loom/memory/allocator.h"
#include "loom/memory/utils.h"

namespace loom {

struct PoolingAllocatorConfig {
    size_t alignment = 0;
};

/**
 * @brief Pooling based allocator for CPU
 *
 * This allocator is a pooling allocator for CPU Memory.
 * - It supports alignment and pinned memory allocation.
 * - It uses a pool of memory blocks to allocate memory.
 * - It uses a hash map with vectors for O(1) lookups (exact size match).
 * - It stores block size in a header prefix (no separate metadata map).
 *
 * Optimizations:
 * - Hash map + vector for O(1) pool lookups (vs O(log n) with multimap)
 * - Header prefix for size tracking (vs separate hash map)
 */
class PoolingAllocator : public Allocator {
  public:
    PoolingAllocator(const PoolingAllocatorConfig& config = PoolingAllocatorConfig());
    ~PoolingAllocator() override;
    PoolingAllocator(const PoolingAllocator&) = delete;
    PoolingAllocator& operator=(const PoolingAllocator&) = delete;
    PoolingAllocator(PoolingAllocator&&) = delete;
    PoolingAllocator& operator=(PoolingAllocator&&) = delete;

    void* allocate(size_t bytes) override;
    void deallocate(void* data) override;

    [[nodiscard]] Device device() const override;

  private:
    void* AllocateBlock(size_t bytes);

    PoolingAllocatorConfig mConfig;

    // Pool: maps size -> list of free blocks of that exact size
    // O(1) lookup instead of O(log n) with multimap
    std::unordered_map<size_t, std::vector<void*>> mFreeBlocks;

    // Sentinel for zero-size allocations
    // Shared across all PoolingAllocator instances
    static char sZeroSizeSentinel;
};
}  // namespace loom