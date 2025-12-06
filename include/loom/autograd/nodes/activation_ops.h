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

}  // namespace autograd
}  // namespace loom
