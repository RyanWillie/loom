#include "cpu/cpu_allocator.h"

#include <cstdlib>

namespace loom {
CPUAllocator::CPUAllocator(const CPUAllocatorConfig& config) : mConfig(config) {}

void* CPUAllocator::allocate(size_t bytes) {
    return malloc(bytes);
}

void CPUAllocator::deallocate(void* data) {
    free(data);
}

Device CPUAllocator::device() const {
    return Device(DeviceType::CPU);
}
}  // namespace loom