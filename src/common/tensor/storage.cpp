#include "common/tensor/storage.h"

#include <cstring>
#include <stdexcept>

#include "common/device.h"
#include "common/dtypes.h"
#include "common/registry/allocator_registry.h"
#include "common/tensor/allocator.h"

namespace loom {
using loom::Allocator;
using loom::AllocatorRegistry;
using loom::Device;
using loom::DType;

Storage::Storage(size_t size, DType dtype, const Device& device)
    : mSize(size), mAllocator(AllocatorRegistry::get(device)), mDevice(device), mDataType(dtype) {
    const size_t bytes = size * sizeOf(dtype);
    mData = std::shared_ptr<void>(mAllocator->allocate(bytes), [allocator = mAllocator](void* ptr) {
        allocator->deallocate(ptr);
    });
    if (!mData) {
        throw std::runtime_error("Failed to allocate storage");
    }
}

Storage Storage::clone() const {
    if (!mData) {
        throw std::runtime_error("Cannot clone storage with no data");
    }

    // Create a new storage with the same size, data type, and device
    Storage newStorage(mSize, mDataType, mDevice);
    if (mDevice.isCPU() || mDevice.isMPS()) {
        // Copy the data from the current storage to the new storage using memcpy
        std::memcpy(newStorage.mData.get(), mData.get(), mSize * sizeOf(mDataType));
    } else {
        throw std::runtime_error("Unsupported device type for cloning");
    }
    return newStorage;
}

std::shared_ptr<void> Storage::data() const {
    return mData;
}

size_t Storage::size() const {
    return mSize;
}

size_t Storage::sizeInBytes() const {
    return mSize * sizeOf(mDataType);
}

std::shared_ptr<Allocator> Storage::allocator() const {
    return mAllocator;
}

Device Storage::device() const {
    return mDevice;
}

DType Storage::dtype() const {
    return mDataType;
}

}  // namespace loom