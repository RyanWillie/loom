#pragma once

#include "loom/device.h"
#include "loom/memory/allocator.h"

namespace loom {

/**
 * @brief Basic allocator for CPU
 *
 * This allocator is a simple allocator for CPU.
 * - It does not support any special features.
 * - It is used for basic allocation and deallocation of memory.
 * - It just wraps the std::malloc and std::free functions.
 */
class BasicAllocator : public Allocator {
  public:
    BasicAllocator(const Device& device);
    ~BasicAllocator() = default;

    void* allocate(size_t bytes) override;
    void deallocate(void* data) override;

    [[nodiscard]] Device device() const override;
};
}  // namespace loom