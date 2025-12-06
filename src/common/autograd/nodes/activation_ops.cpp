#include "loom/autograd/nodes/activation_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// ReLUBackward: y = max(0, x)
// ============================================================================

ReLUBackward::ReLUBackward(const Tensor& x) : mSavedInput(x) {
    mName = "ReLUBackward";
}

std::vector<Tensor> ReLUBackward::backward(const Tensor& gradOutput) {
    // Check that saved tensors haven't been modified in-place
    checkVersions();

    // ReLU backward: gradient flows through only where input was positive
    // ∂L/∂x = ∂L/∂y * (x > 0)

    // Create mask tensor with same shape (1 where input > 0, else 0)
    Tensor mask = Tensor::zeros(mSavedInput.shape(), mSavedInput.dtype(), mSavedInput.device());

    // Flatten both tensors to iterate over all elements regardless of dimensionality
    // contiguous() ensures the data is laid out sequentially in memory
    Tensor input_flat = mSavedInput.contiguous().flatten();
    Tensor mask_flat = mask.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto mask_acc = mask_flat.accessor<float, 1>();

    size_t n = input_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        if (input_acc[i] > 0.0f) {
            mask_acc[i] = 1.0f;
        }
    }

    // Apply mask to gradient (element-wise multiplication)
    // mask still has the original shape since flatten() returns a view
    Tensor grad_input = gradOutput * mask;

    return {grad_input};
}

}  // namespace autograd
}  // namespace loom
