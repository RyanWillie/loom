#include "loom/registry/allocator_registry.h"

#include <stdexcept>

#include "loom/device.h"
#include "loom/memory/basic_allocator.h"
#include "loom/memory/pooling_allocator.h"

namespace loom {

// Static members
std::map<loom::Device, std::shared_ptr<Allocator>> AllocatorRegistry::mAllocators = {};
std::mutex AllocatorRegistry::mMutex;

void AllocatorRegistry::clear() {
    std::lock_guard<std::mutex> lock(mMutex);
    mAllocators.clear();
}

void AllocatorRegistry::set(const loom::Device& device, std::shared_ptr<Allocator> allocator) {
    std::lock_guard<std::mutex> lock(mMutex);
    mAllocators[device] = allocator;
}

std::shared_ptr<Allocator> AllocatorRegistry::get(const loom::Device& device) {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mAllocators.find(device) == mAllocators.end()) {
        // Create a default allocator for the device
        if (device.isCPU()) {
            mAllocators[device] = std::make_shared<PoolingAllocator>();
        } else if (device.isCUDA()) {
            // mAllocators[device] = std::make_shared<CUDAAllocator>();
            throw std::runtime_error("CUDA allocator not implemented");
        } else if (device.isMPS()) {
            // mAllocators[device] = std::make_shared<MPSAllocator>();
            throw std::runtime_error("MPS allocator not implemented");
        } else {
            throw std::runtime_error("Unsupported device type");
        }
    }
    return mAllocators[device];
}

bool AllocatorRegistry::exists(const loom::Device& device) {
    std::lock_guard<std::mutex> lock(mMutex);
    return mAllocators.find(device) != mAllocators.end();
}

}  // namespace loom