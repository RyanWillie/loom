#include "common/tensor/tensor.h"

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

std::vector<size_t> Tensor::calculateStride(const std::vector<size_t>& shape) {
    std::vector<size_t> stride(shape.size());
    size_t size = 1;
    for (size_t i = shape.size() - 1; i != 0; --i) {
        stride[i] = size;
        size *= shape[i];
    }
    return stride;
}

size_t Tensor::calculateOffset(const std::vector<size_t>& shape,
                               const std::vector<size_t>& stride) {
    size_t offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        offset += shape[i] * stride[i];
    }
    return offset;
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
}  // namespace loom
