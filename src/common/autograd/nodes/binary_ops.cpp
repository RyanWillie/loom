#include "loom/autograd/nodes/binary_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// Helper Function: unbroadcast
// ============================================================================
// This function reverses broadcasting by summing over dimensions that were expanded
// Example: grad shape {64, 10} -> original shape {10} => sum over dimension 0

Tensor unbroadcast(const Tensor& grad, const std::vector<size_t>& original_shape) {
    // TODO(human): Implement unbroadcast logic
    //
    // Algorithm:
    // 1. Start with result = grad
    // 2. While result has more dimensions than original_shape:
    //    - Sum over dimension 0 (this removes prepended dimensions)
    //    - Use result.sum(0, false) - false means don't keep dimension
    // 3. For each dimension i in original_shape:
    //    - If original_shape[i] == 1 AND result.size(i) > 1:
    //      * This dimension was broadcast from size 1 to something larger
    //      * Sum over dimension i, keeping it as size 1
    //      * Use result.sum(i, true) - true means keep dimension as 1
    // 4. Reshape result to match original_shape exactly
    // 5. Return result
    //
    // Hint: You'll need to handle two cases:
    //   Case 1: Prepended dimensions (e.g., {10} broadcast to {64, 10})
    //   Case 2: Expanded dimensions (e.g., {1, 10} broadcast to {64, 10})

    Tensor result = grad;
    while (result.shape().size() > original_shape.size()) {
        result = result.sum(0, false);
    }

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
    // TODO(human): Implement AddBackward
    //
    // Mathematical rule: ∂(x + y)/∂x = 1, ∂(x + y)/∂y = 1
    // This means: grad_x = gradOutput * 1 = gradOutput
    //             grad_y = gradOutput * 1 = gradOutput
    //
    // Algorithm:
    // 1. Compute grad_x by unbroadcasting gradOutput to mShapeX
    //    - grad_x = unbroadcast(gradOutput, mShapeX)
    // 2. Compute grad_y by unbroadcasting gradOutput to mShapeY
    //    - grad_y = unbroadcast(gradOutput, mShapeY)
    // 3. Return vector containing {grad_x, grad_y}

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
    // TODO(human): Implement SubBackward
    //
    // Mathematical rule: ∂(x - y)/∂x = 1, ∂(x - y)/∂y = -1
    // This means: grad_x =  gradOutput
    //             grad_y = -gradOutput
    //
    // Algorithm:
    // 1. Compute grad_x = unbroadcast(gradOutput, mShapeX)
    // 2. Compute grad_y = unbroadcast(-gradOutput, mShapeY)
    //    - Note the negation! Subtraction negates the second gradient
    // 3. Return {grad_x, grad_y}
    //
    // Hint: You can negate a tensor with: -gradOutput or gradOutput * -1.0

    Tensor grad_x = unbroadcast(gradOutput, mShapeX);
    Tensor grad_y = unbroadcast(gradOutput * -1.0, mShapeY);  // Negate using multiplication
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

    // TODO(human): Implement MulBackward
    //
    // Mathematical rule: ∂(x * y)/∂x = y, ∂(x * y)/∂y = x
    // This means: grad_x = gradOutput * y
    //             grad_y = gradOutput * x
    //
    // Algorithm:
    // 1. Compute grad_x:
    //    a. Multiply gradOutput * mSavedY (element-wise)
    //    b. Unbroadcast to mShapeX
    // 2. Compute grad_y:
    //    a. Multiply gradOutput * mSavedX (element-wise)
    //    b. Unbroadcast to mShapeY
    // 3. Return {grad_x, grad_y}
    //
    // Key insight: We use the SAVED tensors (mSavedX, mSavedY) because
    // the gradient depends on the input values!

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

    // TODO(human): Implement DivBackward
    //
    // Mathematical rule: ∂(x / y)/∂x = 1/y, ∂(x / y)/∂y = -x/y²
    // This means: grad_x = gradOutput * (1/y)
    //             grad_y = gradOutput * (-x/y²)
    //
    // Algorithm:
    // 1. Compute grad_x:
    //    a. Calculate gradOutput / mSavedY
    //    b. Unbroadcast to mShapeX
    // 2. Compute grad_y:
    //    a. Calculate -mSavedX / (mSavedY * mSavedY)
    //    b. Multiply by gradOutput
    //    c. Unbroadcast to mShapeY
    // 3. Return {grad_x, grad_y}
    //
    // Hint: For y², you can use mSavedY * mSavedY
    // Hint: Be careful with the order of operations for the chain rule!

    Tensor grad_x = unbroadcast(gradOutput / mSavedY, mShapeX);
    Tensor grad_y = unbroadcast(gradOutput * -1.0 * mSavedX / (mSavedY * mSavedY), mShapeY);
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
