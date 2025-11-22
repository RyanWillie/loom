#include "cpu/pooling_allocator.h"

#include <cstdlib>

namespace loom {
PoolingAllocator::PoolingAllocator(const PoolingAllocatorConfig& config) : mConfig(config) {}

void* PoolingAllocator::allocate(size_t bytes) {
    return malloc(bytes);
}

void PoolingAllocator::deallocate(void* data) {
    free(data);
}

Device PoolingAllocator::device() const {
    return Device(DeviceType::CPU);
}
}  // namespace loom