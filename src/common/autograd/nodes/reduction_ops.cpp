#include "loom/autograd/nodes/reduction_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// SumBackward: y = sum(x)
// ============================================================================

SumBackward::SumBackward(const std::vector<size_t>& input_shape) : mInputShape(input_shape) {
    mName = "SumBackward";
}

std::vector<Tensor> SumBackward::backward(const Tensor& gradOutput) {
    // Sum backward: broadcast gradient uniformly to all input elements
    //
    // The gradient of sum is trivial: every input element contributed
    // equally (with derivative 1), so we just broadcast the output gradient
    // to match the input shape.
    //
    // Example 1: If input was [2, 3] and sum() gave scalar [1],
    //            gradOutput is [1] and we expand it to [2, 3]
    // Example 2: If input was [2, 3] and sum(dim=1, keepdim=True) gave [2, 1],
    //            gradOutput is [2, 1] and we expand it to [2, 3]

    // Handle both scalar and non-scalar gradOutput
    if (gradOutput.numel() == 1) {
        // Scalar case - fill with the scalar value
        Tensor grad_input =
            Tensor::full(mInputShape, gradOutput.item(), gradOutput.dtype(), gradOutput.device());
        return {grad_input};
    } else {
        // Non-scalar case - broadcast gradOutput to input shape
        // This happens with sum(dim, keepdim=True)
        Tensor grad_input = Tensor::zeros(mInputShape, gradOutput.dtype(), gradOutput.device());

        // Use broadcasting: each element of gradOutput is copied to multiple positions in grad_input
        grad_input = grad_input + gradOutput;  // Broadcasting addition

        return {grad_input};
    }
}

// ============================================================================
// MeanBackward: y = mean(x) = sum(x) / n
// ============================================================================

MeanBackward::MeanBackward(const std::vector<size_t>& input_shape, size_t num_elements)
    : mInputShape(input_shape), mNumElements(num_elements) {
    mName = "MeanBackward";
}

std::vector<Tensor> MeanBackward::backward(const Tensor& gradOutput) {
    // Mean backward: broadcast gradient divided by number of elements
    //
    // Since mean = sum / n, by chain rule:
    // ∂L/∂x_i = ∂L/∂y * ∂y/∂x_i = ∂L/∂y * (1/n)
    //
    // So each element gets gradient / n

    // Handle both scalar and non-scalar gradOutput
    if (gradOutput.numel() == 1) {
        // Scalar case - fill with the scalar value divided by n
        double grad_value = gradOutput.item() / static_cast<double>(mNumElements);
        Tensor grad_input =
            Tensor::full(mInputShape, grad_value, gradOutput.dtype(), gradOutput.device());
        return {grad_input};
    } else {
        // Non-scalar case - broadcast gradOutput to input shape, then divide by n
        // This happens with mean(dim, keepdim=True)
        Tensor grad_input = Tensor::zeros(mInputShape, gradOutput.dtype(), gradOutput.device());

        // Use broadcasting: each element of gradOutput is copied to multiple positions
        grad_input = grad_input + gradOutput;  // Broadcasting addition

        // Divide by number of elements that were averaged
        grad_input = grad_input / static_cast<double>(mNumElements);

        return {grad_input};
    }
}

}  // namespace autograd
}  // namespace loom
