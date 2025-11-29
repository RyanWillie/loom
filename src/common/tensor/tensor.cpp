#include "common/tensor/tensor.h"

#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include "common/logger.h"

namespace loom {

Tensor::Tensor(const std::vector<size_t>& shape, const loom::DType dtype,
               const loom::Device& device)
    : mShape(shape), mStride(Tensor::calculateStride(shape)), mOffset(0) {
    // Calculate total number of elements
    size_t numElements = 1;
    for (size_t dim : shape) {
        numElements *= dim;
    }
    // Allocate storage for all elements
    mStorage = std::make_shared<Storage>(numElements, dtype, device);
}

Tensor::Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape,
               std::vector<size_t> stride, size_t offset)
    : mStorage(storage), mShape(shape), mStride(stride), mOffset(offset) {}

Tensor Tensor::zeros(const std::vector<size_t>& shape, const loom::DType dtype,
                     const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to 0
    tensor.zero();
    return tensor;
}

Tensor& Tensor::zero() {
    std::memset(mStorage->data().get(), 0, mStorage->sizeInBytes());
    return *this;
}

Tensor Tensor::ones(const std::vector<size_t>& shape, const loom::DType dtype,
                    const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to 1
    tensor.one();
    return tensor;
}

Tensor& Tensor::one() {
    loom::DType dt = dtype();
    size_t num_elements = mStorage->size();

    dispatchByDType(dt, mStorage->data().get(), num_elements, [](auto* data, size_t size) {
        using T = std::remove_pointer_t<decltype(data)>;
        std::fill_n(data, size, static_cast<T>(1));
    });
    return *this;
}

Tensor Tensor::rand(const std::vector<size_t>& shape, const loom::DType dtype,
                    const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to random
    tensor.rand();
    return tensor;
}

Tensor& Tensor::rand() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
    return *this;
}

Tensor Tensor::randn(const std::vector<size_t>& shape, const loom::DType dtype,
                     const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to random
    tensor.randn();
    return tensor;
}

Tensor& Tensor::randn() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
    return *this;
}

Tensor& Tensor::uniform(const double min, const double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
    return *this;
}

Tensor Tensor::full(const std::vector<size_t>& shape, const double value, const loom::DType dtype,
                    const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to value
    tensor.fill(value);
    return tensor;
}

Tensor& Tensor::fill(const double value) {
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        std::fill_n(ptr, n, static_cast<T>(value));
    });
    return *this;
}

Tensor Tensor::clone() const {
    Storage storage = mStorage->clone();
    return {std::make_shared<Storage>(storage), mShape, mStride, mOffset};
}

std::vector<size_t> Tensor::calculateStride(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }

    std::vector<size_t> stride(shape.size());
    size_t acc = 1;

    // Calculate strides in row-major order (C-style)
    // For shape [2, 3, 4], strides are [12, 4, 1]
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        stride[i] = acc;
        acc *= shape[i];
    }
    return stride;
}

size_t Tensor::calculateOffset(const std::vector<size_t>& indices,
                               const std::vector<size_t>& stride) {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * stride[i];
    }
    return offset;
}

std::vector<size_t> Tensor::broadcastShape(const std::vector<size_t>& a,
                                           const std::vector<size_t>& b) {
    size_t max_ndim = std::max(a.size(), b.size());
    std::vector<size_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        // Get dimensions from right (or 1 if dimension doesn't exist)
        size_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dim_a == dim_b) {
            result[max_ndim - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            result[max_ndim - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            result[max_ndim - 1 - i] = dim_a;
        } else {
            throw std::runtime_error("Shapes not broadcastable: dimension " +
                                     std::to_string(max_ndim - 1 - i) + " has sizes " +
                                     std::to_string(dim_a) + " and " + std::to_string(dim_b));
        }
    }
    return result;
}

std::vector<size_t> Tensor::broadcastStrides(const std::vector<size_t>& original_shape,
                                             const std::vector<size_t>& original_stride,
                                             const std::vector<size_t>& target_shape) {
    // Create strides for broadcasting: stride is 0 for broadcast dimensions
    std::vector<size_t> result(target_shape.size(), 0);

    size_t shape_offset = target_shape.size() - original_shape.size();

    for (size_t i = 0; i < original_shape.size(); ++i) {
        size_t target_idx = shape_offset + i;
        if (original_shape[i] == target_shape[target_idx]) {
            // Same size: use original stride
            result[target_idx] = original_stride[i];
        } else if (original_shape[i] == 1) {
            // Broadcast dimension: stride = 0 (repeat the value)
            result[target_idx] = 0;
        } else {
            throw std::runtime_error("Cannot broadcast shape");
        }
    }
    // Dimensions prepended (shape_offset) remain 0, which is correct for broadcasting

    return result;
}

const std::vector<size_t>& Tensor::shape() const {
    return mShape;
}

const std::vector<size_t>& Tensor::stride() const {
    return mStride;
}

size_t Tensor::offset() const {
    return mOffset;
}

loom::DType Tensor::dtype() const {
    return mStorage->dtype();
}

loom::Device Tensor::device() const {
    return mStorage->device();
}

size_t Tensor::numel() const {
    return mStorage->size();
}

size_t Tensor::size(const size_t dim) const {
    return mShape[dim];
}

size_t Tensor::ndim() const {
    return mShape.size();
}

bool Tensor::isContiguous() const {
    // Edge case: empty or single element tensors are always contiguous
    if (mShape.empty() || numel() <= 1) {
        return true;
    }

    // A tensor is contiguous if:
    // 1. Offset is 0 (starts at beginning of storage)
    // 2. Strides match row-major layout
    std::vector<size_t> expectedStride = calculateStride(mShape);
    return mOffset == 0 && mStride == expectedStride;
}

double Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() only works on single-element tensors, got " +
                                 std::to_string(numel()) + " elements");
    }

    double result = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), 1,
                    [&](auto* ptr, size_t) { result = static_cast<double>(ptr[mOffset]); });
    return result;
}

Tensor Tensor::contiguous() const {
    // If already contiguous, return shallow copy
    if (isContiguous()) {
        return *this;
    }

    // Create new contiguous tensor
    Tensor result(mShape, dtype(), device());

    // Copy data element by element, respecting strides
    dispatchByDType(dtype(), result.mStorage->data().get(), numel(),
                    [this](auto* dst_ptr, size_t total_elements) {
                        using T = std::remove_pointer_t<decltype(dst_ptr)>;
                        const T* src_ptr = static_cast<const T*>(mStorage->data().get());

                        // For contiguous result, just iterate linearly
                        size_t dst_idx = 0;

                        // Need to iterate through all multi-dimensional indices
                        std::vector<size_t> indices(mShape.size(), 0);

                        for (size_t i = 0; i < total_elements; ++i) {
                            // Calculate source offset using strides
                            size_t src_offset = mOffset;
                            for (size_t dim = 0; dim < indices.size(); ++dim) {
                                src_offset += indices[dim] * mStride[dim];
                            }

                            // Copy element
                            dst_ptr[dst_idx++] = src_ptr[src_offset];

                            // Increment multi-dimensional index
                            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                                indices[dim]++;
                                if (indices[dim] < mShape[dim]) {
                                    break;  // No carry
                                }
                                indices[dim] = 0;  // Carry to next dimension
                            }
                        }
                    });

    return result;
}

Tensor Tensor::toDevice(const Device& device) const {
    // If same device, return shallow copy
    if (device.type() == this->device().type()) {
        return *this;
    }

    // Different device - would need to copy data across devices
    // For now, throw an error for non-CPU devices
    throw std::runtime_error("Device transfer not yet implemented for non-CPU devices");
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (mShape == other.mShape) {
        return clone() += other;
    }
    return broadcastOp(other, [](auto a, auto b) { return a + b; });
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (mShape == other.mShape) {
        return clone() -= other;
    }
    return broadcastOp(other, [](auto a, auto b) { return a - b; });
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (mShape == other.mShape) {
        return clone() *= other;
    }
    return broadcastOp(other, [](auto a, auto b) { return a * b; });
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (mShape == other.mShape) {
        return clone() /= other;
    }
    return broadcastOp(other, [](auto a, auto b) { return a / b; });
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for addition");
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* other_ptr = static_cast<const T*>(other.mStorage->data().get());

        for (size_t i = 0; i < n; ++i) {
            ptr[i] += other_ptr[i];
        }
    });
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for subtraction");
    }

    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for subtraction");
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* other_ptr = static_cast<const T*>(other.mStorage->data().get());

        for (size_t i = 0; i < n; ++i) {
            ptr[i] -= other_ptr[i];
        }
    });
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for multiplication");
    }
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for multiplication");
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* other_ptr = static_cast<const T*>(other.mStorage->data().get());

        for (size_t i = 0; i < n; ++i) {
            ptr[i] *= other_ptr[i];
        }
    });
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for division");
    }
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for division");
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* other_ptr = static_cast<const T*>(other.mStorage->data().get());

        for (size_t i = 0; i < n; ++i) {
            ptr[i] /= other_ptr[i];
        }
    });
    return *this;
}

Tensor& Tensor::operator+=(const double scalar) {
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        T scalar_value = static_cast<T>(scalar);
        for (size_t i = 0; i < n; ++i) {
            ptr[i] += scalar_value;
        }
    });
    return *this;
}

Tensor& Tensor::operator-=(const double scalar) {
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        T scalar_value = static_cast<T>(scalar);
        for (size_t i = 0; i < n; ++i) {
            ptr[i] -= scalar_value;
        }
    });
    return *this;
}

Tensor& Tensor::operator*=(const double scalar) {
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        T scalar_value = static_cast<T>(scalar);
        for (size_t i = 0; i < n; ++i) {
            ptr[i] *= scalar_value;
        }
    });
    return *this;
}

Tensor& Tensor::operator/=(const double scalar) {
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        T scalar_value = static_cast<T>(scalar);
        for (size_t i = 0; i < n; ++i) {
            ptr[i] /= scalar_value;
        }
    });
    return *this;
}

Tensor Tensor::operator+(const double scalar) const {
    return clone() += scalar;
}

Tensor Tensor::operator-(const double scalar) const {
    return clone() -= scalar;
}

Tensor Tensor::operator*(const double scalar) const {
    return clone() *= scalar;
}

Tensor Tensor::operator/(const double scalar) const {
    return clone() /= scalar;
}

Tensor Tensor::flatten() const {
    return reshape({numel()});
}

Tensor Tensor::reshape(const std::vector<size_t>& shape) const {
    if (!isContiguous()) {
        throw std::runtime_error("Tensor is not contiguous");
    }

    // Check new shape has same number of elements as original
    size_t new_numel = 1;
    for (size_t dim : shape) {
        new_numel *= dim;
    }
    if (new_numel != numel()) {
        throw std::runtime_error("Invalid shape for reshape");
    }

    // Calculate new stride
    std::vector<size_t> new_stride = calculateStride(shape);

    return Tensor(mStorage, shape, new_stride, mOffset);
}

// Remove single-element dimensions
Tensor Tensor::squeeze() const {
    std::vector<size_t> new_shape;
    std::vector<size_t> new_stride;
    for (size_t i = 0; i < mShape.size(); ++i) {
        if (mShape[i] != 1) {
            new_shape.push_back(mShape[i]);
            new_stride.push_back(mStride[i]);
        }
    }

    // Edge case: if all dimensions are 1, return a 1D tensor
    if (new_shape.empty()) {
        return reshape({1});
    }

    return {mStorage, new_shape, new_stride, mOffset};
}

Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) {
        dim += static_cast<int>(mShape.size());
    }
    if (dim < 0 || dim >= static_cast<int>(mShape.size())) {
        throw std::runtime_error("Dimension out of range for squeeze");
    }

    if (mShape[static_cast<size_t>(dim)] != 1) {
        throw std::runtime_error("Dimension is not 1 for squeeze");
    }
    // Calculate new stride
    std::vector<size_t> new_shape;
    std::vector<size_t> new_stride;
    for (size_t i = 0; i < mShape.size(); ++i) {
        if (i != static_cast<size_t>(dim)) {
            new_shape.push_back(mShape[i]);
            new_stride.push_back(mStride[i]);
        }
    }

    if (new_shape.empty()) {
        new_shape = {1};
        new_stride = {1};
    }

    return Tensor(mStorage, new_shape, new_stride, mOffset);
}

Tensor Tensor::unsqueeze(int dim) const {
    // 1. Handle negative indexing
    int ndim_new = static_cast<int>(mShape.size()) + 1;
    if (dim < 0) {
        dim += ndim_new;
    }

    // 2. Validate dimension (can be 0 to ndim inclusive!)
    if (dim < 0 || dim > static_cast<int>(mShape.size())) {
        throw std::runtime_error("Dimension out of range");
    }

    // 3. Insert size-1 dimension at position
    std::vector<size_t> new_shape = mShape;
    new_shape.insert(new_shape.begin() + dim, 1);

    // 4. Insert stride (use stride of next dimension, or 1 if last)
    std::vector<size_t> new_stride = mStride;
    size_t stride_value = (dim < static_cast<int>(mStride.size())) ? mStride[dim] : 1;
    new_stride.insert(new_stride.begin() + dim, stride_value);

    return Tensor(mStorage, new_shape, new_stride, mOffset);
}

Tensor Tensor::transpose() const {
    if (mShape.size() < 2) {
        throw std::runtime_error("Tensor must have at least 2 dimensions for transpose");
    }

    return transpose(static_cast<int>(mShape.size()) - 2, static_cast<int>(mShape.size()) - 1);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 == dim1) {
        return *this;
    }

    if (dim0 < 0) {
        dim0 += static_cast<int>(mShape.size());
    }
    if (dim1 < 0) {
        dim1 += static_cast<int>(mShape.size());
    }

    if (dim0 < 0 || dim0 >= static_cast<int>(mShape.size()) || dim1 < 0 ||
        dim1 >= static_cast<int>(mShape.size())) {
        throw std::runtime_error("Dimension out of range for transpose");
    }

    std::vector<size_t> new_shape = mShape;
    std::vector<size_t> new_stride = mStride;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_stride[dim0], new_stride[dim1]);
    return Tensor(mStorage, new_shape, new_stride, mOffset);
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    // 1. Validate dims has correct size
    if (dims.size() != mShape.size()) {
        throw std::runtime_error("Permute: dims must have same length as tensor dimensions");
    }

    // 2. Check each dimension appears exactly once
    std::vector<bool> seen(dims.size(), false);
    for (int dim : dims) {
        int d = dim;
        // Handle negative indexing
        if (d < 0) {
            d += static_cast<int>(mShape.size());
        }

        // Validate range and uniqueness
        if (d < 0 || d >= static_cast<int>(mShape.size()) || seen[d]) {
            throw std::runtime_error(
                "Invalid permutation: each dimension must appear exactly once");
        }
        seen[d] = true;
    }

    // 3. Build new shape and stride based on permutation
    std::vector<size_t> new_shape(dims.size());
    std::vector<size_t> new_stride(dims.size());

    for (size_t i = 0; i < dims.size(); ++i) {
        int d = dims[i];
        if (d < 0) {
            d += static_cast<int>(mShape.size());
        }

        new_shape[i] = mShape[d];
        new_stride[i] = mStride[d];
    }

    return Tensor(mStorage, new_shape, new_stride, mOffset);
}

Tensor Tensor::slice(int dim, const size_t start, const size_t end) const {
    // 1. Handle negative dim (like Python: -1 = last dim)
    if (dim < 0) {
        dim += static_cast<int>(mShape.size());
    }

    // 2. Validate dim is in range [0, ndim)
    if (dim < 0 || dim >= static_cast<int>(mShape.size())) {
        throw std::out_of_range("Dimension out of range for slice");
    }

    // 3. Validate start < end && end <= shape[dim]
    if (start >= end || end > mShape[static_cast<size_t>(dim)]) {
        throw std::out_of_range("Slice start and end indices are invalid");
    }

    // 4. Build new shape (same as mShape, but dim has size end-start)
    std::vector<size_t> new_shape = mShape;
    new_shape[static_cast<size_t>(dim)] = end - start;
    std::vector<size_t> new_stride = mStride;

    // 5. Calculate new offset: mOffset + start * mStride[dim]
    size_t new_offset = mOffset + start * mStride[static_cast<size_t>(dim)];
    // 6. Return new Tensor with same storage, new shape, same stride, new offset
    return {mStorage, new_shape, new_stride, new_offset};
}

Tensor Tensor::sum() const {
    // Sum all elements of the tensor
    double sum = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        for (size_t i = 0; i < n; ++i) {
            sum += static_cast<T>(ptr[i]);
        }
    });
    return Tensor::full({1}, sum, dtype(), device());
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for sum");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
            // else: skip this dimension
        } else {
            out_shape.push_back(mShape[i]);
        }
    }

    // Handle edge case: reducing to scalar
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());

    // 4. Get total number of output elements
    size_t out_numel = result.numel();

    // 5. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            // Insert the reduced dimension back
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;  // Will iterate over this
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Sum over the reduced dimension
                            T acc = 0;
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;

                                // Calculate input offset using strides
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }

                                acc += src_ptr[in_offset];
                            }

                            dst_ptr[out_linear] = acc;
                        }
                    });

    return result;
}

// ============================================================================
// Mean Reductions
// ============================================================================

Tensor Tensor::mean() const {
    double total = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            total += static_cast<double>(ptr[i]);
        }
    });
    return Tensor::full({1}, total / static_cast<double>(numel()), dtype(), device());
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for mean");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
        } else {
            out_shape.push_back(mShape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());
    size_t out_numel = result.numel();

    // 4. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Sum over the reduced dimension, then divide
                            T acc = 0;
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }
                                acc += src_ptr[in_offset];
                            }

                            dst_ptr[out_linear] = acc / static_cast<T>(reduce_size);
                        }
                    });

    return result;
}

// ============================================================================
// Max Reductions
// ============================================================================

Tensor Tensor::max() const {
    double max_val = std::numeric_limits<double>::lowest();
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            double val = static_cast<double>(ptr[i]);
            if (val > max_val)
                max_val = val;
        }
    });
    return Tensor::full({1}, max_val, dtype(), device());
}

Tensor Tensor::max(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for max");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
        } else {
            out_shape.push_back(mShape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());
    size_t out_numel = result.numel();

    // 4. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Find max over the reduced dimension
                            T acc = std::numeric_limits<T>::lowest();
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }
                                if (src_ptr[in_offset] > acc) {
                                    acc = src_ptr[in_offset];
                                }
                            }

                            dst_ptr[out_linear] = acc;
                        }
                    });

    return result;
}

// ============================================================================
// Min Reductions
// ============================================================================

Tensor Tensor::min() const {
    double min_val = std::numeric_limits<double>::max();
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            double val = static_cast<double>(ptr[i]);
            if (val < min_val)
                min_val = val;
        }
    });
    return Tensor::full({1}, min_val, dtype(), device());
}

Tensor Tensor::min(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for min");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
        } else {
            out_shape.push_back(mShape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());
    size_t out_numel = result.numel();

    // 4. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Find min over the reduced dimension
                            T acc = std::numeric_limits<T>::max();
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }
                                if (src_ptr[in_offset] < acc) {
                                    acc = src_ptr[in_offset];
                                }
                            }

                            dst_ptr[out_linear] = acc;
                        }
                    });

    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    if (dtype() != other.dtype()) {
        throw std::runtime_error("Dot product requires matching dtypes");
    }

    if (ndim() != 1 || other.ndim() != 1) {
        throw std::runtime_error("Dot product requires 1D tensors");
    }
    if (mShape[0] != other.mShape[0]) {
        throw std::runtime_error("Dot product requires matching sizes");
    }

    double result = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* other_ptr = static_cast<const T*>(other.mStorage->data().get());
        for (size_t i = 0; i < n; ++i) {
            result += static_cast<double>(ptr[i]) * static_cast<double>(other_ptr[i]);
        }
    });

    return Tensor::full({1}, result, dtype(), device());
}

Tensor Tensor::matmul(const Tensor& other) const {
    // 1. Validate: both must be 2D
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }

    // 2. Validate: inner dimensions must match [M, K] × [K, N]
    if (mShape[1] != other.mShape[0]) {
        throw std::runtime_error("matmul: inner dimensions must match. Got [" +
                                 std::to_string(mShape[0]) + ", " + std::to_string(mShape[1]) +
                                 "] × [" + std::to_string(other.mShape[0]) + ", " +
                                 std::to_string(other.mShape[1]) + "]");
    }

    // 3. Validate: same dtype
    if (dtype() != other.dtype()) {
        throw std::runtime_error("matmul requires matching dtypes");
    }

    size_t M = mShape[0];
    size_t K = mShape[1];
    size_t N = other.mShape[1];

    // 4. Create output tensor [M, N]
    Tensor result = Tensor::zeros({M, N}, dtype(), device());

    // 5. Compute: triple nested loop (naïve O(M×N×K) implementation)
    dispatchByDType(dtype(), result.mStorage->data().get(), M * N, [&](auto* c_ptr, size_t) {
        using T = std::remove_pointer_t<decltype(c_ptr)>;
        const T* a_ptr = static_cast<const T*>(mStorage->data().get());
        const T* b_ptr = static_cast<const T*>(other.mStorage->data().get());

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    // A[i,k] × B[k,j]
                    sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                }
                c_ptr[i * N + j] = sum;
            }
        }
    });

    return result;
}

// ============================================================================
// Visualization
// ============================================================================

void Tensor::print(const std::string& name) const {
    auto& logger = Logger::getInstance("Tensor");

    // Build shape string
    std::ostringstream shape_str;
    shape_str << "[";
    for (size_t i = 0; i < mShape.size(); ++i) {
        shape_str << mShape[i];
        if (i < mShape.size() - 1)
            shape_str << ", ";
    }
    shape_str << "]";

    // Build stride string
    std::ostringstream stride_str;
    stride_str << "[";
    for (size_t i = 0; i < mStride.size(); ++i) {
        stride_str << mStride[i];
        if (i < mStride.size() - 1)
            stride_str << ", ";
    }
    stride_str << "]";

    // Print metadata
    logger.info("========================================");
    logger.info("{}", name);
    logger.info("========================================");
    logger.info("Shape:      {}", shape_str.str());
    logger.info("Stride:     {}", stride_str.str());
    logger.info("Offset:     {}", mOffset);
    logger.info("DType:      {}", loom::name(dtype()));
    logger.info("Device:     {}", device().toString());
    logger.info("Elements:   {}", numel());
    logger.info("Contiguous: {}", isContiguous() ? "Yes" : "No");
    logger.info("----------------------------------------");

    // Limit output for large tensors
    const size_t MAX_ELEMENTS = 100;
    bool truncated = numel() > MAX_ELEMENTS;
    size_t elements_to_print = truncated ? MAX_ELEMENTS : numel();

    if (numel() == 0) {
        logger.info("Data: (empty)");
        logger.info("========================================");
        return;
    }

    // Print data based on dimensionality
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* data = static_cast<const T*>(ptr);

        std::ostringstream data_str;
        data_str << std::fixed << std::setprecision(4);

        if (ndim() == 1) {
            // 1D: Print as simple list
            data_str << "[";
            for (size_t i = 0; i < elements_to_print; ++i) {
                data_str << data[mOffset + i * mStride[0]];
                if (i < elements_to_print - 1)
                    data_str << ", ";
            }
            if (truncated)
                data_str << ", ...";
            data_str << "]";
            logger.info("Data: {}", data_str.str());

        } else if (ndim() == 2) {
            // 2D: Print as matrix
            logger.info("Data:");
            size_t rows = mShape[0];
            size_t cols = mShape[1];

            for (size_t i = 0; i < rows && i * cols < elements_to_print; ++i) {
                data_str.str("");
                data_str << "  [";
                size_t max_cols = std::min(cols, (elements_to_print - i * cols));
                for (size_t j = 0; j < max_cols; ++j) {
                    size_t idx = mOffset + i * mStride[0] + j * mStride[1];
                    data_str << std::setw(8) << data[idx];
                    if (j < max_cols - 1)
                        data_str << ", ";
                }
                if (truncated && max_cols < cols)
                    data_str << ", ...";
                data_str << "]";
                logger.info("{}", data_str.str());
            }
            if (truncated && rows * cols > elements_to_print) {
                logger.info("  ...");
            }

        } else if (ndim() == 3) {
            // 3D: Print as separate matrices
            logger.info("Data:");
            size_t depth = mShape[0];
            size_t rows = mShape[1];
            size_t cols = mShape[2];

            for (size_t d = 0; d < depth; ++d) {
                logger.info("  Matrix [{}]:", d);
                for (size_t i = 0; i < rows; ++i) {
                    data_str.str("");
                    data_str << "    [";
                    for (size_t j = 0; j < cols; ++j) {
                        size_t idx = mOffset + d * mStride[0] + i * mStride[1] + j * mStride[2];
                        data_str << std::setw(8) << data[idx];
                        if (j < cols - 1)
                            data_str << ", ";
                    }
                    data_str << "]";
                    logger.info("{}", data_str.str());
                }
            }

        } else {
            // 4D+: Print flattened view with indices
            logger.info("Data (first {} elements, shape too complex for matrix view):",
                        elements_to_print);
            data_str << "[";
            for (size_t i = 0; i < elements_to_print; ++i) {
                // For now, just print in storage order (contiguous assumption)
                data_str << data[mOffset + i];
                if (i < elements_to_print - 1)
                    data_str << ", ";
                // Add newline every 10 elements for readability
                if ((i + 1) % 10 == 0 && i < elements_to_print - 1) {
                    data_str << "\n ";
                }
            }
            if (truncated)
                data_str << ", ...";
            data_str << "]";
            logger.info("{}", data_str.str());
        }
    });

    logger.info("========================================");
}

}  // namespace loom
