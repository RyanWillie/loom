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
// Matrix Multiplication Backward Node
// ============================================================================

// MatmulBackward: Backward pass for matrix multiplication
// Forward:  C = A @ B   where A:[M,K], B:[K,N], C:[M,N]
// Backward: ∂L/∂A = (∂L/∂C) @ B^T   [M,N] @ [N,K] = [M,K]
//           ∂L/∂B = A^T @ (∂L/∂C)   [K,M] @ [M,N] = [K,N]
//
// Key insight: We multiply the incoming gradient by the OTHER input's transpose
// This is how gradients flow backward through matrix multiplications
//
// Example: For a linear layer y = xW, if we have dL/dy, then:
//   - dL/dx = (dL/dy) @ W^T   (gradient flows back to input)
//   - dL/dW = x^T @ (dL/dy)   (gradient accumulates to weights)
class MatmulBackward : public Node {
  public:
    // Constructor: Save copies of input tensors
    // We NEED the actual values of A and B to compute gradients
    // Unlike addition (where gradient is independent of input values),
    // matmul gradient depends on the input matrices
    MatmulBackward(const Tensor& a, const Tensor& b);

    // Compute gradients for inputs given gradient of output
    // Given: gradOutput = dL/dC where C = A @ B
    // Return: {grad_a, grad_b} where:
    //   grad_a = dL/dA = (dL/dC) @ B^T
    //   grad_b = dL/dB = A^T @ (dL/dC)
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "MatmulBackward"; }
    size_t numInputs() const override { return 2; }

  private:
    Tensor mSavedA;  // Saved copy of A for backward pass
    Tensor mSavedB;  // Saved copy of B for backward pass
};

}  // namespace autograd
}  // namespace loom
