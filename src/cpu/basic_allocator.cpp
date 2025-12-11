#include "loom/memory/basic_allocator.h"

#include <cstdlib>
#include <stdexcept>

namespace loom {
BasicAllocator::BasicAllocator(const Device& device) {
    if (!device.isCPU()) {
        throw std::runtime_error("BasicAllocator can only be created on CPU");
    }
}

void* BasicAllocator::allocate(size_t bytes) {
    return std::malloc(bytes);
}

void BasicAllocator::deallocate(void* data) {
    std::free(data);
}

Device BasicAllocator::device() const {
    return Device(DeviceType::CPU);
}
}  // namespace loom