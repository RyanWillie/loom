#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/memory/allocator.h"
#include "loom/type_traits.h"

namespace loom {
using loom::Allocator;
using loom::Device;
using loom::DType;

class Storage {
  public:
    // @param size: Number of elements in the storage
    // @param dtype: Data type of the storage
    // @param device: Device location of the storage
    Storage(size_t size, DType dtype, const Device& device);
    ~Storage() = default;

    // Shallow copy of the storage
    // @param other: The storage to copy
    Storage(const Storage& other) = default;
    // Copy assignment operator
    Storage& operator=(const Storage& other) = default;
    // Move constructor
    Storage(Storage&& other) noexcept = default;
    // Move assignment operator
    Storage& operator=(Storage&& other) noexcept = default;

    // Deep copy of the storage
    [[nodiscard]] Storage clone() const;

    // Returns shared pointer to the raw data
    [[nodiscard]] std::shared_ptr<void> data() const;

    // Type-safe accessor with compile-time and runtime validation
    // T must be a valid tensor storage type (float, int32_t, etc.)
    // Throws std::runtime_error if storage dtype doesn't match requested type
    template <TensorStorageType T>
    std::shared_ptr<T> as() const {
        // Compile-time: TensorStorageType concept ensures T is valid
        // Runtime: Check that T matches the storage's actual dtype
        constexpr DType expected_dtype = dtype_traits<T>::value;
        if (mDataType != expected_dtype) {
            throw std::runtime_error(std::string("Type mismatch: requested ") +
                                     name(expected_dtype) + ", but storage contains " +
                                     name(mDataType));
        }
        // Use aliasing constructor: shares ownership with mData
        return std::shared_ptr<T>(mData, static_cast<T*>(mData.get()));
    }

    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t sizeInBytes() const;
    [[nodiscard]] std::shared_ptr<Allocator> allocator() const;
    [[nodiscard]] Device device() const;
    [[nodiscard]] DType dtype() const;

  private:
    std::shared_ptr<void> mData;
    size_t mSize;
    std::shared_ptr<Allocator> mAllocator;
    Device mDevice;   // Device location of the storage
    DType mDataType;  // Data type of the storage
};
}  // namespace loom
