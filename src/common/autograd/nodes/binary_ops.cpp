#include "loom/autograd/nodes/binary_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// Helper Function: unbroadcast
// ============================================================================
// This function reverses broadcasting by summing over dimensions that were expanded
// Example: grad shape {64, 10} -> original shape {10} => sum over dimension 0
// Handles two cases:
//   Case 1: Prepended dimensions (e.g., {10} broadcast to {64, 10})
//   Case 2: Expanded dimensions (e.g., {1, 10} broadcast to {64, 10})

Tensor unbroadcast(const Tensor& grad, const std::vector<size_t>& original_shape) {
    Tensor result = grad;

    // Remove prepended dimensions by summing
    while (result.shape().size() > original_shape.size()) {
        result = result.sum(0, false);
    }

    // Sum over dimensions that were broadcast from size 1
    for (size_t i = 0; i < original_shape.size(); ++i) {
        if (original_shape[i] == 1 && result.shape()[i] > 1) {
            result = result.sum(i, true);
        }
    }

    return result.reshape(original_shape);
}

// ============================================================================
// AddBackward: z = x + y
// ============================================================================

AddBackward::AddBackward(const std::vector<size_t>& shape_x, const std::vector<size_t>& shape_y)
    : mShapeX(shape_x), mShapeY(shape_y) {
    mName = "AddBackward";
}

std::vector<Tensor> AddBackward::backward(const Tensor& gradOutput) {
    // Mathematical rule: ∂(x + y)/∂x = 1, ∂(x + y)/∂y = 1
    // Therefore: grad_x = gradOutput, grad_y = gradOutput
    // (with unbroadcasting to match original input shapes)

    Tensor grad_x = unbroadcast(gradOutput, mShapeX);
    Tensor grad_y = unbroadcast(gradOutput, mShapeY);
    return {grad_x, grad_y};
}

// ============================================================================
// SubBackward: z = x - y
// ============================================================================

SubBackward::SubBackward(const std::vector<size_t>& shape_x, const std::vector<size_t>& shape_y)
    : mShapeX(shape_x), mShapeY(shape_y) {
    mName = "SubBackward";
}

std::vector<Tensor> SubBackward::backward(const Tensor& gradOutput) {
    // Mathematical rule: ∂(x - y)/∂x = 1, ∂(x - y)/∂y = -1
    // Therefore: grad_x = gradOutput, grad_y = -gradOutput
    // (with unbroadcasting to match original input shapes)

    Tensor grad_x = unbroadcast(gradOutput, mShapeX);
    Tensor grad_y = unbroadcast(gradOutput * -1.0, mShapeY);
    return {grad_x, grad_y};
}

// ============================================================================
// MulBackward: z = x * y
// ============================================================================

MulBackward::MulBackward(const Tensor& x, const Tensor& y, const std::vector<size_t>& shape_x,
                         const std::vector<size_t>& shape_y)
    : mSavedX(x), mSavedY(y), mShapeX(shape_x), mShapeY(shape_y) {
    mName = "MulBackward";
}

std::vector<Tensor> MulBackward::backward(const Tensor& gradOutput) {
    // Check that saved tensors haven't been modified in-place
    checkVersions();

    // Mathematical rule: ∂(x * y)/∂x = y, ∂(x * y)/∂y = x
    // Therefore: grad_x = gradOutput * y, grad_y = gradOutput * x
    // (with unbroadcasting to match original input shapes)

    Tensor grad_x = unbroadcast(gradOutput * mSavedY, mShapeX);
    Tensor grad_y = unbroadcast(gradOutput * mSavedX, mShapeY);
    return {grad_x, grad_y};
}

// ============================================================================
// DivBackward: z = x / y
// ============================================================================

DivBackward::DivBackward(const Tensor& x, const Tensor& y, const std::vector<size_t>& shape_x,
                         const std::vector<size_t>& shape_y)
    : mSavedX(x), mSavedY(y), mShapeX(shape_x), mShapeY(shape_y) {
    mName = "DivBackward";
}

std::vector<Tensor> DivBackward::backward(const Tensor& gradOutput) {
    // Check that saved tensors haven't been modified in-place
    checkVersions();

    // Mathematical rule: ∂(x / y)/∂x = 1/y, ∂(x / y)/∂y = -x/y²
    // Therefore: grad_x = gradOutput / y, grad_y = gradOutput * (-x/y²)
    // (with unbroadcasting to match original input shapes)

    Tensor grad_x = unbroadcast(gradOutput / mSavedY, mShapeX);
    Tensor grad_y = unbroadcast(gradOutput * -1.0 * mSavedX / (mSavedY * mSavedY), mShapeY);
    return {grad_x, grad_y};
}

// ============================================================================
// DotBackward: z = dot(x, y)
// ============================================================================

DotBackward::DotBackward(const Tensor& x, const Tensor& y) : mSavedX(x), mSavedY(y) {
    mName = "DotBackward";
}

std::vector<Tensor> DotBackward::backward(const Tensor& gradOutput) {
    // Check that saved tensors haven't been modified in-place
    checkVersions();

    // Dot product backward:
    // Forward:  z = sum(x[i] * y[i])  (scalar result)
    // Backward: ∂L/∂x = ∂L/∂z * y
    //           ∂L/∂y = ∂L/∂z * x
    //
    // Since the output is a scalar, gradOutput is also a scalar.
    // We multiply each input vector by the scalar gradient.

    // Extract scalar gradient value
    double grad_scalar = gradOutput.item();

    // Compute gradients: scale each saved input by the scalar gradient
    Tensor grad_x = mSavedY * grad_scalar;
    Tensor grad_y = mSavedX * grad_scalar;

    return {grad_x, grad_y};
}

// ============================================================================
// ScalarMulBackward: z = x * scalar
// ============================================================================

std::vector<Tensor> ScalarMulBackward::backward(const Tensor& gradOutput) {
    // ∂(x * scalar)/∂x = scalar
    // grad_x = gradOutput * scalar
    return {gradOutput * mScalar};
}

// ============================================================================
// ScalarDivBackward: z = x / scalar
// ============================================================================

std::vector<Tensor> ScalarDivBackward::backward(const Tensor& gradOutput) {
    // ∂(x / scalar)/∂x = 1/scalar
    // grad_x = gradOutput / scalar
    return {gradOutput / mScalar};
}

// ============================================================================
// ScalarAddBackward: z = x + scalar
// ============================================================================

std::vector<Tensor> ScalarAddBackward::backward(const Tensor& gradOutput) {
    // ∂(x + scalar)/∂x = 1
    // grad_x = gradOutput
    return {gradOutput};
}

// ============================================================================
// ScalarSubBackward: z = x - scalar
// ============================================================================

std::vector<Tensor> ScalarSubBackward::backward(const Tensor& gradOutput) {
    // ∂(x - scalar)/∂x = 1
    // grad_x = gradOutput
    return {gradOutput};
}

}  // namespace autograd
}  // namespace loom
