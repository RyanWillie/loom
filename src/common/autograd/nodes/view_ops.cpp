#include "loom/autograd/nodes/view_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// ReshapeBackward
// ============================================================================

ReshapeBackward::ReshapeBackward(const std::vector<size_t>& input_shape)
    : mInputShape(input_shape) {
    mName = "ReshapeBackward";
}

std::vector<Tensor> ReshapeBackward::backward(const Tensor& gradOutput) {
    // Simply reshape gradient back to original input shape
    return {gradOutput.reshape(mInputShape)};
}

// ============================================================================
// TransposeBackward
// ============================================================================

TransposeBackward::TransposeBackward() : mDim0(-2), mDim1(-1), mSimpleTranspose(true) {
    mName = "TransposeBackward";
}

TransposeBackward::TransposeBackward(int dim0, int dim1)
    : mDim0(dim0), mDim1(dim1), mSimpleTranspose(false) {
    mName = "TransposeBackward";
}

std::vector<Tensor> TransposeBackward::backward(const Tensor& gradOutput) {
    // Transpose is its own inverse - apply same transpose to gradient
    if (mSimpleTranspose) {
        return {gradOutput.transpose()};
    } else {
        return {gradOutput.transpose(mDim0, mDim1)};
    }
}

// ============================================================================
// FlattenBackward
// ============================================================================

FlattenBackward::FlattenBackward(const std::vector<size_t>& input_shape)
    : mInputShape(input_shape) {
    mName = "FlattenBackward";
}

std::vector<Tensor> FlattenBackward::backward(const Tensor& gradOutput) {
    // Flatten is just reshape to 1D, so backward is reshape to original
    return {gradOutput.reshape(mInputShape)};
}

// ============================================================================
// SqueezeBackward
// ============================================================================

SqueezeBackward::SqueezeBackward(int dim) : mDim(dim) {
    mName = "SqueezeBackward";
}

std::vector<Tensor> SqueezeBackward::backward(const Tensor& gradOutput) {
    // Squeeze removes a dimension, so backward adds it back
    return {gradOutput.unsqueeze(mDim)};
}

// ============================================================================
// UnsqueezeBackward
// ============================================================================

UnsqueezeBackward::UnsqueezeBackward(int dim) : mDim(dim) {
    mName = "UnsqueezeBackward";
}

std::vector<Tensor> UnsqueezeBackward::backward(const Tensor& gradOutput) {
    // Unsqueeze adds a dimension, so backward removes it
    return {gradOutput.squeeze(mDim)};
}

// ============================================================================
// PermuteBackward
// ============================================================================

PermuteBackward::PermuteBackward(const std::vector<int>& dims) : mDims(dims) {
    mName = "PermuteBackward";
}

std::vector<Tensor> PermuteBackward::backward(const Tensor& gradOutput) {
    // Permute backward: apply inverse permutation to gradient
    //
    // If forward permutation was [2, 0, 1], meaning:
    //   new_dim[0] = old_dim[2]
    //   new_dim[1] = old_dim[0]
    //   new_dim[2] = old_dim[1]
    //
    // Then inverse permutation is [1, 2, 0], meaning:
    //   recovered_dim[0] = permuted_dim[1]  (get back old_dim[0])
    //   recovered_dim[1] = permuted_dim[2]  (get back old_dim[1])
    //   recovered_dim[2] = permuted_dim[0]  (get back old_dim[2])
    //
    // Algorithm: inverse[mDims[i]] = i

    std::vector<int> inverse_dims(mDims.size());
    for (size_t i = 0; i < mDims.size(); ++i) {
        int d = mDims[i];
        // Handle negative indexing
        if (d < 0) {
            d += static_cast<int>(mDims.size());
        }
        inverse_dims[d] = static_cast<int>(i);
    }

    return {gradOutput.permute(inverse_dims)};
}

// ============================================================================
// SliceBackward
// ============================================================================

SliceBackward::SliceBackward(const std::vector<size_t>& input_shape, int dim, size_t start,
                             size_t end)
    : mInputShape(input_shape), mDim(dim), mStart(start), mEnd(end) {
    mName = "SliceBackward";
}

std::vector<Tensor> SliceBackward::backward(const Tensor& gradOutput) {
    // Slice backward: scatter gradient to correct positions in input
    //
    // Strategy:
    // 1. Create zeros tensor with input shape
    // 2. Manually copy gradOutput values to the sliced region
    //
    // This is more complex than other view ops because we need to scatter
    // the gradient to specific positions, not just reshape/permute it.

    // Create zero gradient with input shape
    Tensor grad_input =
        Tensor::zeros(mInputShape, gradOutput.dtype(), gradOutput.device()).contiguous();

    // Handle negative dim indexing
    int dim = mDim;
    if (dim < 0) {
        dim += static_cast<int>(mInputShape.size());
    }

    // We need to copy gradOutput into the sliced region of grad_input
    // This requires iterating over all elements and computing indices
    //
    // For simplicity with the current Tensor API, we'll use the fact that
    // we can get a slice view and then manually copy element by element

    Tensor grad_input_flat = grad_input.flatten();
    Tensor grad_output_flat = gradOutput.contiguous().flatten();

    auto grad_input_acc = grad_input_flat.accessor<float, 1>();
    auto grad_output_acc = grad_output_flat.accessor<float, 1>();

    // Compute strides for multi-dimensional indexing
    std::vector<size_t> strides(mInputShape.size());
    strides[mInputShape.size() - 1] = 1;
    for (int i = static_cast<int>(mInputShape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * mInputShape[i + 1];
    }

    // Iterate over all elements in gradOutput
    size_t grad_output_size = grad_output_flat.numel();
    for (size_t out_idx = 0; out_idx < grad_output_size; ++out_idx) {
        // Convert flat index in gradOutput to multi-dimensional index
        std::vector<size_t> multi_idx(mInputShape.size());
        const auto& output_shape = gradOutput.shape();

        size_t temp = out_idx;
        for (int d = static_cast<int>(output_shape.size()) - 1; d >= 0; --d) {
            multi_idx[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        // Adjust the sliced dimension by adding start offset
        multi_idx[dim] += mStart;

        // Convert multi-dimensional index to flat index in grad_input
        size_t in_idx = 0;
        for (size_t d = 0; d < mInputShape.size(); ++d) {
            in_idx += multi_idx[d] * strides[d];
        }

        // Copy gradient value
        grad_input_acc[in_idx] = grad_output_acc[out_idx];
    }

    return {grad_input};
}

}  // namespace autograd
}  // namespace loom
