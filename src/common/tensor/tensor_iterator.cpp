#include "loom/tensor/tensor_iterator.h"

namespace loom {

TensorIterator::TensorIterator(const std::vector<size_t>& shape,
                               const std::vector<size_t>& stride, size_t base_offset)
    : mIndices(shape.size(), 0)
    , mShape(shape)
    , mStride(stride)
    , mBaseOffset(base_offset)
    , mCurrentOffset(base_offset)
    , mPosition(0)
    , mTotalElements(computeNumel(shape)) {}

void TensorIterator::reset() {
    std::fill(mIndices.begin(), mIndices.end(), 0);
    mCurrentOffset = mBaseOffset;
    mPosition = 0;
}

size_t TensorIterator::computeNumel(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    size_t total = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            return 0;  // Any zero dimension means zero elements
        }
        total *= dim;
    }
    return total;
}

}  // namespace loom
