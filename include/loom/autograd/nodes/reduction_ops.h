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
// Sum Reduction Backward Node
// ============================================================================

// SumBackward: Backward pass for sum reduction
// Forward:  y = sum(x)  (reduces to scalar or along dimension)
// Backward: ∂L/∂x_i = ∂L/∂y  for all i
//
// Key insight: Every element of x contributed equally to y, so the gradient
// is simply broadcast back to all elements uniformly.
//
// Example: If x = [[1, 2], [3, 4]], y = sum(x) = 10, and ∂L/∂y = 5, then
//          ∂L/∂x = [[5, 5], [5, 5]]  (gradient broadcast to all elements)
class SumBackward : public Node {
  public:
    // Constructor: Save original input shape for gradient broadcasting
    SumBackward(const std::vector<size_t>& input_shape);

    // Compute gradient for input given gradient of output
    // Given: gradOutput = ∂L/∂y where y = sum(x)
    // Return: ∂L/∂x = broadcast(gradOutput, input_shape)
    //
    // Implementation: Since sum reduces to scalar, gradOutput is typically
    // a scalar tensor. We need to expand it to match the input shape.
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "SumBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    std::vector<size_t> mInputShape;  // Original shape before reduction
};

// ============================================================================
// Mean Reduction Backward Node
// ============================================================================

// MeanBackward: Backward pass for mean reduction
// Forward:  y = mean(x) = sum(x) / n
// Backward: ∂L/∂x_i = ∂L/∂y / n  for all i
//
// Key insight: Like sum, but divided by n since mean = sum/n
// The chain rule gives us: ∂L/∂x_i = ∂L/∂y * ∂y/∂x_i = ∂L/∂y * (1/n)
//
// Example: If x = [[1, 2], [3, 4]], y = mean(x) = 2.5, and ∂L/∂y = 4, then
//          ∂L/∂x = [[1, 1], [1, 1]]  (4 / 4 elements = 1 for each)
class MeanBackward : public Node {
  public:
    // Constructor: Save input shape and number of elements
    MeanBackward(const std::vector<size_t>& input_shape, size_t num_elements);

    // Compute gradient for input given gradient of output
    // Given: gradOutput = ∂L/∂y where y = mean(x)
    // Return: ∂L/∂x = broadcast(gradOutput / n, input_shape)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "MeanBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    std::vector<size_t> mInputShape;  // Original shape before reduction
    size_t mNumElements;              // Number of elements (n)
};

}  // namespace autograd
}  // namespace loom
