#pragma once

#include <memory>
#include <vector>

#include "loom/autograd/node.h"
#include "loom/tensor/tensor.h"

namespace loom {

// Forward declaration
class Tensor;

namespace autograd {

// ============================================================================
// ReLU Activation Backward Node
// ============================================================================

// ReLUBackward: Backward pass for ReLU activation
// Forward:  y = max(0, x)  (element-wise)
// Backward: ∂L/∂x_i = ∂L/∂y_i * 1  if x_i > 0
//                   = 0             if x_i ≤ 0
//
// Key insight: ReLU acts as a binary gate - gradients flow through where
// the input was positive, and are blocked where input was negative.
//
// Example: If x = [-1, 2, -3, 4] and ∂L/∂y = [a, b, c, d], then
//          ∂L/∂x = [0, b, 0, d]  (gradients blocked at indices 0 and 2)
class ReLUBackward : public Node {
  public:
    // Constructor: Save copy of input tensor
    // We need the input values to determine where to apply the mask (x > 0)
    ReLUBackward(const Tensor& x);

    // Compute gradient for input given gradient of output
    // Given: gradOutput = ∂L/∂y where y = ReLU(x)
    // Return: ∂L/∂x = gradOutput * (x > 0)
    //
    // Implementation: Create a mask tensor where mask[i] = 1 if x[i] > 0 else 0
    //                 Then return gradOutput * mask (element-wise)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ReLUBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedInput;  // Saved copy of input to determine gradient mask
};

// ============================================================================
// Exp Activation Backward Node
// ============================================================================

// ExpBackward: Backward pass for exponential function
// Forward:  y = exp(x) = e^x  (element-wise)
// Backward: ∂L/∂x = ∂L/∂y * exp(x) = ∂L/∂y * y
//
// Key insight: The derivative of exp(x) is exp(x), so we can save the
// output instead of recomputing exp(x) during backward pass.
class ExpBackward : public Node {
  public:
    // Save the output y = exp(x) for efficient backward pass
    ExpBackward(const Tensor& output);

    // Given: gradOutput = ∂L/∂y where y = exp(x)
    // Return: ∂L/∂x = gradOutput * y
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ExpBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedOutput;  // Saved y = exp(x)
};

// ============================================================================
// Log Activation Backward Node
// ============================================================================

// LogBackward: Backward pass for natural logarithm
// Forward:  y = log(x) = ln(x)  (element-wise)
// Backward: ∂L/∂x = ∂L/∂y * (1/x)
//
// Key insight: The derivative of log(x) is 1/x, so we need to save
// the input x for the backward pass.
class LogBackward : public Node {
  public:
    // Save the input x for backward pass
    LogBackward(const Tensor& input);

    // Given: gradOutput = ∂L/∂y where y = log(x)
    // Return: ∂L/∂x = gradOutput / x
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "LogBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedInput;  // Saved input x
};

// ============================================================================
// Max Reduction Backward Node
// ============================================================================

// MaxBackward: Backward pass for max reduction along a dimension
// Forward:  y[i] = max(x[i, :]) along specified dimension
// Backward: ∂L/∂x[i,j] = ∂L/∂y[i]  if x[i,j] == max(x[i,:])
//                       = 0         otherwise
//
// Key insight: Gradients only flow to the positions that contained the
// maximum value. If there are ties (multiple elements with same max),
// gradients flow to all tied positions.
//
// Example: If x = [[1, 3, 2], [4, 1, 4]] and we take max along dim=1:
//          y = [3, 4]
//          If ∂L/∂y = [a, b], then ∂L/∂x = [[0, a, 0], [b/2, 0, b/2]]
//          (Note: gradients split among ties in row 1)
class MaxBackward : public Node {
  public:
    // Save input, output, and reduction parameters
    // We need both input and output to identify where max occurred
    MaxBackward(const Tensor& input, const Tensor& output, int dim, bool keepdim);

    // Given: gradOutput = ∂L/∂y where y = max(x, dim)
    // Return: ∂L/∂x with gradients scattered to max positions
    //
    // Implementation:
    //  1. Broadcast output to input shape if needed (keepdim handling)
    //  2. Create mask where input == output (broadcasted)
    //  3. Broadcast gradOutput to input shape
    //  4. Return gradOutput * mask (gradients only flow to max positions)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "MaxBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedInput;   // Input tensor to identify max positions
    Tensor mSavedOutput;  // Output tensor to compare against
    int mDim;             // Dimension that was reduced
    bool mKeepDim;        // Whether keepdim was true
};

// ============================================================================
// Min Reduction Backward Node
// ============================================================================

// MinBackward: Backward pass for min reduction along a dimension
// Forward:  y[i] = min(x[i, :]) along specified dimension
// Backward: ∂L/∂x[i,j] = ∂L/∂y[i]  if x[i,j] == min(x[i,:])
//                       = 0         otherwise
//
// Key insight: Gradients only flow to the positions that contained the
// minimum value. If there are ties (multiple elements with same min),
// gradients flow to all tied positions.
//
// Example: If x = [[3, 1, 2], [1, 4, 1]] and we take min along dim=1:
//          y = [1, 1]
//          If ∂L/∂y = [a, b], then ∂L/∂x = [[0, a, 0], [b/2, 0, b/2]]
//          (Note: gradients split among ties in row 1)
class MinBackward : public Node {
  public:
    // Save input, output, and reduction parameters
    // We need both input and output to identify where min occurred
    MinBackward(const Tensor& input, const Tensor& output, int dim, bool keepdim);

    // Given: gradOutput = ∂L/∂y where y = min(x, dim)
    // Return: ∂L/∂x with gradients scattered to min positions
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "MinBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedInput;   // Input tensor to identify min positions
    Tensor mSavedOutput;  // Output tensor to compare against
    int mDim;             // Dimension that was reduced
    bool mKeepDim;        // Whether keepdim was true
};

// ============================================================================
// Sigmoid Activation Backward Node
// ============================================================================

// SigmoidBackward: Backward pass for sigmoid activation
// Forward:  y = 1 / (1 + exp(-x))  (element-wise)
// Backward: ∂L/∂x = ∂L/∂y * y * (1 - y)
//
// Key insight: The derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x)),
// so we save the output y = sigmoid(x) rather than recomputing it.
//
// Numerical stability: For very large |x|, sigmoid saturates (y ≈ 0 or y ≈ 1),
// making gradients very small (vanishing gradient problem).
class SigmoidBackward : public Node {
  public:
    // Save the output y = sigmoid(x) for efficient backward pass
    SigmoidBackward(const Tensor& output);

    // Given: gradOutput = ∂L/∂y where y = sigmoid(x)
    // Return: ∂L/∂x = gradOutput * y * (1 - y)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "SigmoidBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedOutput;  // Saved y = sigmoid(x)
};

// ============================================================================
// Tanh Activation Backward Node
// ============================================================================

// TanhBackward: Backward pass for hyperbolic tangent
// Forward:  y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// Backward: ∂L/∂x = ∂L/∂y * (1 - y²)
//
// Key insight: The derivative of tanh is 1 - tanh²(x), so we save
// the output y = tanh(x) rather than recomputing it.
class TanhBackward : public Node {
  public:
    // Save the output y = tanh(x) for efficient backward pass
    TanhBackward(const Tensor& output);

    // Given: gradOutput = ∂L/∂y where y = tanh(x)
    // Return: ∂L/∂x = gradOutput * (1 - y²)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "TanhBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedOutput;  // Saved y = tanh(x)
};

// ============================================================================
// Softmax Activation Backward Node
// ============================================================================

// SoftmaxBackward: Backward pass for softmax normalization
// Forward:  y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))  along dimension
// Backward: ∂L/∂x_i = y_i * (∂L/∂y_i - sum_j(y_j * ∂L/∂y_j))
//
// Key insight: Softmax creates dependencies between all elements along the
// dimension, so the gradient involves a weighted sum.
//
// Formula derivation: For softmax s_i = exp(x_i) / sum(exp(x_j)):
//   ∂s_i/∂x_j = s_i * (δ_ij - s_j)  where δ_ij is Kronecker delta
//   ∂L/∂x_i = sum_j (∂L/∂s_j * ∂s_j/∂x_i) = s_i * (∂L/∂s_i - sum_j(s_j * ∂L/∂s_j))
class SoftmaxBackward : public Node {
  public:
    // Save the output y = softmax(x) and the dimension
    SoftmaxBackward(const Tensor& output, int dim);

    // Given: gradOutput = ∂L/∂y where y = softmax(x, dim)
    // Return: ∂L/∂x using the formula above
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "SoftmaxBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedOutput;  // Saved y = softmax(x)
    int mDim;             // Dimension along which softmax was applied
};

// ============================================================================
// LogSoftmax Activation Backward Node
// ============================================================================

// LogSoftmaxBackward: Backward pass for log-softmax
// Forward:  y_i = x_i - max(x) - log(sum(exp(x_j - max(x))))  along dimension
// Backward: ∂L/∂x_i = ∂L/∂y_i - exp(y_i) * sum_j(∂L/∂y_j)
//
// Key insight: LogSoftmax is more numerically stable than log(softmax(x))
// for classification tasks, and its gradient is simpler.
//
// This is the preferred activation for multi-class classification when
// paired with NLLLoss (negative log likelihood).
class LogSoftmaxBackward : public Node {
  public:
    // Save the output y = log_softmax(x) and the dimension
    LogSoftmaxBackward(const Tensor& output, int dim);

    // Given: gradOutput = ∂L/∂y where y = log_softmax(x, dim)
    // Return: ∂L/∂x = gradOutput - exp(y) * sum(gradOutput, dim, keepdim=true)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "LogSoftmaxBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mSavedOutput;  // Saved y = log_softmax(x)
    int mDim;             // Dimension along which log_softmax was applied
};

}  // namespace autograd
}  // namespace loom
