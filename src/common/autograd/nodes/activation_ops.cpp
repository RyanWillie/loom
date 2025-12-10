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

// ============================================================================
// ExpBackward: y = exp(x)
// ============================================================================

ExpBackward::ExpBackward(const Tensor& output) : mSavedOutput(output) {
    mName = "ExpBackward";
}

std::vector<Tensor> ExpBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Exp backward: ∂L/∂x = ∂L/∂y * exp(x) = ∂L/∂y * y
    // We saved y = exp(x) to avoid recomputing
    Tensor grad_input = gradOutput * mSavedOutput;

    return {grad_input};
}

// ============================================================================
// LogBackward: y = log(x)
// ============================================================================

LogBackward::LogBackward(const Tensor& input) : mSavedInput(input) {
    mName = "LogBackward";
}

std::vector<Tensor> LogBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Log backward: ∂L/∂x = ∂L/∂y * (1/x) = ∂L/∂y / x
    Tensor grad_input = gradOutput / mSavedInput;

    return {grad_input};
}

// ============================================================================
// MaxBackward: y = max(x, dim)
// ============================================================================

MaxBackward::MaxBackward(const Tensor& input, const Tensor& output, int dim, bool keepdim)
    : mSavedInput(input), mSavedOutput(output), mDim(dim), mKeepDim(keepdim) {
    mName = "MaxBackward";
}

std::vector<Tensor> MaxBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Max backward: gradients only flow to positions that were the maximum
    // ∂L/∂x[i] = ∂L/∂y[j] if x[i] == max(...) at position j, else 0
    //
    // Strategy:
    //  1. Ensure output and gradOutput have keepdim shape for broadcasting
    //  2. Create mask where input == output (broadcasted)
    //  3. Multiply gradOutput (broadcasted) by mask

    Tensor output_for_broadcast = mSavedOutput;
    Tensor grad_for_broadcast = gradOutput;

    // If keepdim was false, we need to unsqueeze along the reduced dimension
    // to enable broadcasting back to the input shape
    if (!mKeepDim) {
        // Adjust dim for negative indexing
        int dim = mDim;
        if (dim < 0) {
            dim += static_cast<int>(mSavedInput.ndim());
        }

        output_for_broadcast = mSavedOutput.unsqueeze(dim);
        grad_for_broadcast = gradOutput.unsqueeze(dim);
    }

    // Create mask where input equals the max value
    // The mask will be 1.0 for positions that were the max, 0.0 otherwise
    //
    // We need to compare input with output, but output may have a reduced shape.
    // So we use element-wise comparison with broadcasting.
    Tensor mask = Tensor::zeros(mSavedInput.shape(), mSavedInput.dtype(), mSavedInput.device());

    // Flatten tensors for iteration
    Tensor input_flat = mSavedInput.contiguous().flatten();
    Tensor mask_flat = mask.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto mask_acc = mask_flat.accessor<float, 1>();

    size_t n = input_flat.numel();

    // We need to map each input position to its corresponding output position
    // For this, we iterate through input indices and compute the corresponding
    // output index by setting the reduced dimension to 0 (since output has size 1 there)
    const auto& input_shape = mSavedInput.shape();
    const auto& output_shape = output_for_broadcast.shape();

    Tensor output_flat = output_for_broadcast.contiguous().flatten();
    auto output_acc = output_flat.accessor<float, 1>();

    // Convert flat index to multi-dimensional index for input
    for (size_t flat_idx = 0; flat_idx < n; ++flat_idx) {
        // Compute multi-dimensional index from flat index
        std::vector<size_t> multi_idx(input_shape.size());
        size_t temp = flat_idx;
        for (int d = static_cast<int>(input_shape.size()) - 1; d >= 0; --d) {
            multi_idx[d] = temp % input_shape[d];
            temp /= input_shape[d];
        }

        // Compute corresponding output index (broadcasted)
        size_t output_flat_idx = 0;
        size_t output_stride = 1;
        for (int d = static_cast<int>(output_shape.size()) - 1; d >= 0; --d) {
            size_t idx = multi_idx[d];
            // If output dim is 1, use index 0 (broadcasting)
            if (output_shape[d] == 1) {
                idx = 0;
            }
            output_flat_idx += idx * output_stride;
            output_stride *= output_shape[d];
        }

        // Mark positions where input equals the max value
        if (input_acc[flat_idx] == output_acc[output_flat_idx]) {
            mask_acc[flat_idx] = 1.0f;
        }
    }

    // Apply mask to gradient (element-wise multiplication with broadcasting)
    Tensor grad_input = grad_for_broadcast * mask;

    return {grad_input};
}

// ============================================================================
// MinBackward: y = min(x, dim)
// ============================================================================

MinBackward::MinBackward(const Tensor& input, const Tensor& output, int dim, bool keepdim)
    : mSavedInput(input), mSavedOutput(output), mDim(dim), mKeepDim(keepdim) {
    mName = "MinBackward";
}

std::vector<Tensor> MinBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Min backward: gradients only flow to positions that were the minimum
    // ∂L/∂x[i] = ∂L/∂y[j] if x[i] == min(...) at position j, else 0
    //
    // Strategy: Same as MaxBackward but check for minimum values

    Tensor output_for_broadcast = mSavedOutput;
    Tensor grad_for_broadcast = gradOutput;

    // If keepdim was false, unsqueeze along the reduced dimension
    if (!mKeepDim) {
        int dim = mDim;
        if (dim < 0) {
            dim += static_cast<int>(mSavedInput.ndim());
        }

        output_for_broadcast = mSavedOutput.unsqueeze(dim);
        grad_for_broadcast = gradOutput.unsqueeze(dim);
    }

    // Create mask where input equals the min value
    Tensor mask = Tensor::zeros(mSavedInput.shape(), mSavedInput.dtype(), mSavedInput.device());

    Tensor input_flat = mSavedInput.contiguous().flatten();
    Tensor mask_flat = mask.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto mask_acc = mask_flat.accessor<float, 1>();

    size_t n = input_flat.numel();

    const auto& input_shape = mSavedInput.shape();
    const auto& output_shape = output_for_broadcast.shape();

    Tensor output_flat = output_for_broadcast.contiguous().flatten();
    auto output_acc = output_flat.accessor<float, 1>();

    // Convert flat index to multi-dimensional index for input
    for (size_t flat_idx = 0; flat_idx < n; ++flat_idx) {
        std::vector<size_t> multi_idx(input_shape.size());
        size_t temp = flat_idx;
        for (int d = static_cast<int>(input_shape.size()) - 1; d >= 0; --d) {
            multi_idx[d] = temp % input_shape[d];
            temp /= input_shape[d];
        }

        // Compute corresponding output index (broadcasted)
        size_t output_flat_idx = 0;
        size_t output_stride = 1;
        for (int d = static_cast<int>(output_shape.size()) - 1; d >= 0; --d) {
            size_t idx = multi_idx[d];
            if (output_shape[d] == 1) {
                idx = 0;
            }
            output_flat_idx += idx * output_stride;
            output_stride *= output_shape[d];
        }

        // Mark positions where input equals the min value
        if (input_acc[flat_idx] == output_acc[output_flat_idx]) {
            mask_acc[flat_idx] = 1.0f;
        }
    }

    // Apply mask to gradient
    Tensor grad_input = grad_for_broadcast * mask;

    return {grad_input};
}

// ============================================================================
// SigmoidBackward: y = 1 / (1 + exp(-x))
// ============================================================================

SigmoidBackward::SigmoidBackward(const Tensor& output) : mSavedOutput(output) {
    mName = "SigmoidBackward";
}

std::vector<Tensor> SigmoidBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Sigmoid backward: ∂L/∂x = ∂L/∂y * y * (1 - y)
    // where y = sigmoid(x)
    //
    // We saved y to avoid recomputing sigmoid(x)

    Tensor one = Tensor::ones(mSavedOutput.shape(), mSavedOutput.dtype(),
                              mSavedOutput.device());
    Tensor one_minus_y = one - mSavedOutput;  // (1 - y)
    Tensor sigmoid_grad = mSavedOutput * one_minus_y;  // y * (1 - y)
    Tensor grad_input = gradOutput * sigmoid_grad;

    return {grad_input};
}

// ============================================================================
// TanhBackward: y = tanh(x)
// ============================================================================

TanhBackward::TanhBackward(const Tensor& output) : mSavedOutput(output) {
    mName = "TanhBackward";
}

std::vector<Tensor> TanhBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Tanh backward: ∂L/∂x = ∂L/∂y * (1 - y²)
    // where y = tanh(x)
    //
    // We saved y to avoid recomputing tanh(x)

    Tensor one = Tensor::ones(mSavedOutput.shape(), mSavedOutput.dtype(),
                              mSavedOutput.device());
    Tensor y_squared = mSavedOutput * mSavedOutput;  // y²
    Tensor tanh_grad = one - y_squared;  // (1 - y²)
    Tensor grad_input = gradOutput * tanh_grad;

    return {grad_input};
}

// ============================================================================
// SoftmaxBackward: y = softmax(x, dim)
// ============================================================================

SoftmaxBackward::SoftmaxBackward(const Tensor& output, int dim)
    : mSavedOutput(output), mDim(dim) {
    mName = "SoftmaxBackward";
}

std::vector<Tensor> SoftmaxBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Softmax backward: ∂L/∂x_i = y_i * (∂L/∂y_i - sum_j(y_j * ∂L/∂y_j))
    // where y = softmax(x, dim)
    //
    // Formula derivation:
    //   For softmax: s_i = exp(x_i) / sum_k(exp(x_k))
    //   Jacobian: ∂s_i/∂x_j = s_i * (δ_ij - s_j)
    //   Chain rule: ∂L/∂x_i = sum_j (∂L/∂s_j * ∂s_j/∂x_i)
    //                       = sum_j (∂L/∂s_j * s_j * (δ_ji - s_i))
    //                       = ∂L/∂s_i * s_i - s_i * sum_j(∂L/∂s_j * s_j)
    //                       = s_i * (∂L/∂s_i - sum_j(s_j * ∂L/∂s_j))

    // Compute element-wise product: y * gradOutput
    Tensor y_grad = mSavedOutput * gradOutput;

    // Sum along the softmax dimension with keepdim=true for broadcasting
    Tensor sum_y_grad = y_grad.sum(mDim, true);

    // Compute: gradOutput - sum(y * gradOutput, dim, keepdim=true)
    Tensor grad_diff = gradOutput - sum_y_grad;

    // Final gradient: y * (gradOutput - sum)
    Tensor grad_input = mSavedOutput * grad_diff;

    return {grad_input};
}

// ============================================================================
// LogSoftmaxBackward: y = log_softmax(x, dim)
// ============================================================================

LogSoftmaxBackward::LogSoftmaxBackward(const Tensor& output, int dim)
    : mSavedOutput(output), mDim(dim) {
    mName = "LogSoftmaxBackward";
}

std::vector<Tensor> LogSoftmaxBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // LogSoftmax backward: ∂L/∂x_i = ∂L/∂y_i - exp(y_i) * sum_j(∂L/∂y_j)
    // where y = log_softmax(x, dim)
    //
    // Key insight: log_softmax(x) = x - log(sum(exp(x)))
    // So: exp(log_softmax(x)) = softmax(x)
    //
    // Derivation:
    //   Let z = log(sum(exp(x_k)))  [log-sum-exp]
    //   Then y_i = x_i - z
    //   ∂y_i/∂x_j = δ_ij - ∂z/∂x_j
    //             = δ_ij - exp(x_j) / sum(exp(x_k))
    //             = δ_ij - softmax(x)_j
    //             = δ_ij - exp(y_j)
    //   ∂L/∂x_i = sum_j (∂L/∂y_j * ∂y_j/∂x_i)
    //           = ∂L/∂y_i - sum_j(∂L/∂y_j * exp(y_j))
    //           = ∂L/∂y_i - exp(y_i) * sum_j(∂L/∂y_j)  [if i == j]

    // Compute sum of gradients along the dimension with keepdim=true
    Tensor sum_grad = gradOutput.sum(mDim, true);

    // Compute exp(y) = softmax(x)
    Tensor softmax = mSavedOutput.exp();

    // Compute: gradOutput - exp(y) * sum(gradOutput)
    Tensor grad_input = gradOutput - softmax * sum_grad;

    return {grad_input};
}

}  // namespace autograd
}  // namespace loom
