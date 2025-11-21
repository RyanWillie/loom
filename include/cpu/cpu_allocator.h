#pragma once
#include <memory>
#include <vector>

#include "common/device.h"
#include "common/tensor/allocator.h"

namespace loom {
struct CPUAllocatorConfig {
    size_t alignment = 16;
    bool pinned = false;
};

class CPUAllocator : public Allocator {
  public:
    CPUAllocator(const CPUAllocatorConfig& config = CPUAllocatorConfig());
    ~CPUAllocator() = default;
    CPUAllocator(const CPUAllocator&) = delete;
    CPUAllocator& operator=(const CPUAllocator&) = delete;
    CPUAllocator(CPUAllocator&&) = delete;
    CPUAllocator& operator=(CPUAllocator&&) = delete;

    void* allocate(size_t bytes) override;
    void deallocate(void* data) override;

    [[nodiscard]] Device device() const override;

  private:
    CPUAllocatorConfig mConfig;
};
}  // namespace loom