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

}  // namespace autograd
}  // namespace loom
