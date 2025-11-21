#pragma once

#include "common/device.h"
#include "common/tensor/allocator.h"

namespace loom {

/**
 * @brief Basic allocator for CPU
 *
 * This allocator is a simple allocator for CPU. It allocates memory from the CPU.
 * It is a simple allocator that does not support any special features.
 * It is used for basic allocation and deallocation of memory.
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