#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace loom {

// ============================================================================
// Debug bounds checking - compiled out in release builds (when NDEBUG is defined)
// ============================================================================
#ifndef NDEBUG
#define LOOM_ACCESSOR_BOUNDS_CHECK(index, size)                                                   \
    do {                                                                                          \
        if ((index) >= (size)) {                                                                  \
            throw std::out_of_range("Accessor index " + std::to_string(index) +                   \
                                    " out of bounds for dimension size " + std::to_string(size)); \
        }                                                                                         \
    } while (0)
#else
#define LOOM_ACCESSOR_BOUNDS_CHECK(index, size) ((void)0)
#endif

// Forward declaration for recursive accessor types
template <typename T, size_t N>
class TensorAccessor;

// ============================================================================
// 1D Accessor - Base case that returns T& directly
// ============================================================================
template <typename T>
class TensorAccessor<T, 1> {
  public:
    TensorAccessor(T* data, const size_t* stride, const size_t* shape)
        : mData(data), mStride(stride[0]), mSize(shape[0]) {}

    // Returns a reference to the element at index i
    T& operator[](size_t i) {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mSize);
        return mData[i * mStride];
    }

    // Const version for read-only access
    const T& operator[](size_t i) const {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mSize);
        return mData[i * mStride];
    }

    // Size of this dimension
    [[nodiscard]] size_t size() const { return mSize; }

  private:
    T* mData;
    size_t mStride;
    size_t mSize;
};

// ============================================================================
// 2D Accessor - Returns 1D accessor
// ============================================================================
template <typename T>
class TensorAccessor<T, 2> {
  public:
    TensorAccessor(T* data, const size_t* stride, const size_t* shape)
        : mData(data), mStride(stride), mShape(shape) {}

    // Returns a 1D accessor for row i
    TensorAccessor<T, 1> operator[](size_t i) {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 1>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Const version
    TensorAccessor<T, 1> operator[](size_t i) const {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 1>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Size of this dimension
    [[nodiscard]] size_t size() const { return mShape[0]; }

  private:
    T* mData;
    const size_t* mStride;
    const size_t* mShape;
};

// ============================================================================
// 3D Accessor - Returns 2D accessor
// ============================================================================
template <typename T>
class TensorAccessor<T, 3> {
  public:
    TensorAccessor(T* data, const size_t* stride, const size_t* shape)
        : mData(data), mStride(stride), mShape(shape) {}

    // Returns a 2D accessor
    TensorAccessor<T, 2> operator[](size_t i) {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 2>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Const version
    TensorAccessor<T, 2> operator[](size_t i) const {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 2>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Size of this dimension
    [[nodiscard]] size_t size() const { return mShape[0]; }

  private:
    T* mData;
    const size_t* mStride;
    const size_t* mShape;
};

// ============================================================================
// 4D Accessor - Returns 3D accessor
// ============================================================================
template <typename T>
class TensorAccessor<T, 4> {
  public:
    TensorAccessor(T* data, const size_t* stride, const size_t* shape)
        : mData(data), mStride(stride), mShape(shape) {}

    // Returns a 3D accessor
    TensorAccessor<T, 3> operator[](size_t i) {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 3>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Const version
    TensorAccessor<T, 3> operator[](size_t i) const {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 3>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Size of this dimension
    [[nodiscard]] size_t size() const { return mShape[0]; }

  private:
    T* mData;
    const size_t* mStride;
    const size_t* mShape;
};

// ============================================================================
// 5D Accessor - Returns 4D accessor (for batch+channel+3D data)
// ============================================================================
template <typename T>
class TensorAccessor<T, 5> {
  public:
    TensorAccessor(T* data, const size_t* stride, const size_t* shape)
        : mData(data), mStride(stride), mShape(shape) {}

    // Returns a 4D accessor
    TensorAccessor<T, 4> operator[](size_t i) {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 4>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Const version
    TensorAccessor<T, 4> operator[](size_t i) const {
        LOOM_ACCESSOR_BOUNDS_CHECK(i, mShape[0]);
        return TensorAccessor<T, 4>(mData + i * mStride[0], mStride + 1, mShape + 1);
    }

    // Size of this dimension
    [[nodiscard]] size_t size() const { return mShape[0]; }

  private:
    T* mData;
    const size_t* mStride;
    const size_t* mShape;
};

}  // namespace loom
