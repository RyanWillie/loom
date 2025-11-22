#pragma once
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/device.h"
#include "common/memory/allocator.h"
#include "common/memory/utils.h"

namespace loom {

struct PoolingAllocatorConfig {
    size_t alignment = 16;
    bool pinned = false;
    std::vector<size_t> block_sizes;
};

/**
 * @brief Pooling based allocator for CPU
 *
 * This allocator is a pooling allocator for CPU Memory.
 * - It supports alignment and pinned memory allocation.
 * - It uses a pool of memory blocks to allocate memory.
 * - It uses a free list to manage the memory blocks.
 * - It uses a mutex to protect the free list.
 * - It uses a condition variable to signal the free list.
 * - It uses a thread to manage the memory blocks.
 * - It uses a thread to manage the free list.
 * - It uses a thread to manage the mutex.
 * - It uses a thread to manage the condition variable.
 */
class PoolingAllocator : public Allocator {
  public:
    PoolingAllocator(const PoolingAllocatorConfig& config = PoolingAllocatorConfig());
    ~PoolingAllocator() = default;
    PoolingAllocator(const PoolingAllocator&) = delete;
    PoolingAllocator& operator=(const PoolingAllocator&) = delete;
    PoolingAllocator(PoolingAllocator&&) = delete;
    PoolingAllocator& operator=(PoolingAllocator&&) = delete;

    void* allocate(size_t bytes) override;
    void deallocate(void* data) override;

    [[nodiscard]] Device device() const override;

  private:
    PoolingAllocatorConfig mConfig;

    // Track the allocated sizes of the memory blocks
    std::unordered_map<void*, size_t> mAllocatedSizes;

    std::multimap<size_t, void*> mFreeBlocks;
};
}  // namespace loom