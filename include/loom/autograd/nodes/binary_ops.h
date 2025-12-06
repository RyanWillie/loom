#pragma once

#include <memory>
#include <vector>

#include "loom/autograd/node.h"
#include "loom/tensor/tensor.h"  // Need full definition for member variables

namespace loom {

// Forward declaration (keep for consistency, but tensor.h already declares it)
class Tensor;

namespace autograd {

// ============================================================================
// Helper Functions
// ============================================================================

// Unbroadcast gradient back to original shape by summing over broadcast dimensions
// Example: If input was {10} but grad is {64, 10} due to broadcasting,
//          sum over batch dimension to get back to {10}
Tensor unbroadcast(const Tensor& grad, const std::vector<size_t>& original_shape);

// ============================================================================
// Binary Operation Backward Nodes
// ============================================================================

// AddBackward: Backward pass for addition operation
// Forward:  z = x + y
// Backward: ∂L/∂x = ∂L/∂z * 1 = ∂L/∂z
//           ∂L/∂y = ∂L/∂z * 1 = ∂L/∂z
// Note: Addition gradient just passes through unchanged
class AddBackward : public Node {
  public:
    // Constructor: Save original shapes for unbroadcasting
    // We don't need to save x or y values - addition gradient doesn't depend on inputs
    AddBackward(const std::vector<size_t>& shape_x, const std::vector<size_t>& shape_y);

    // Compute gradients for inputs given gradient of output
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "AddBackward"; }
    size_t numInputs() const override { return 2; }

  private:
    std::vector<size_t> mShapeX;
    std::vector<size_t> mShapeY;
};

// SubBackward: Backward pass for subtraction operation
// Forward:  z = x - y
// Backward: ∂L/∂x = ∂L/∂z * 1  = ∂L/∂z
//           ∂L/∂y = ∂L/∂z * -1 = -∂L/∂z
// Note: Second input gradient is negated
class SubBackward : public Node {
  public:
    SubBackward(const std::vector<size_t>& shape_x, const std::vector<size_t>& shape_y);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "SubBackward"; }
    size_t numInputs() const override { return 2; }

  private:
    std::vector<size_t> mShapeX;
    std::vector<size_t> mShapeY;
};

// MulBackward: Backward pass for element-wise multiplication
// Forward:  z = x * y
// Backward: ∂L/∂x = ∂L/∂z * y
//           ∂L/∂y = ∂L/∂z * x
// Note: Must save input values for gradient computation
class MulBackward : public Node {
  public:
    // Constructor: Save copies of input tensors
    // We NEED the values of x and y to compute gradients
    MulBackward(const Tensor& x, const Tensor& y, const std::vector<size_t>& shape_x,
                const std::vector<size_t>& shape_y);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "MulBackward"; }
    size_t numInputs() const override { return 2; }

  private:
    Tensor mSavedX;  // Saved copy of x for backward pass
    Tensor mSavedY;  // Saved copy of y for backward pass
    std::vector<size_t> mShapeX;
    std::vector<size_t> mShapeY;
};

// DivBackward: Backward pass for element-wise division
// Forward:  z = x / y
// Backward: ∂L/∂x = ∂L/∂z * (1/y)
//           ∂L/∂y = ∂L/∂z * (-x/y²)
// Note: Must save both inputs for gradient computation
class DivBackward : public Node {
  public:
    DivBackward(const Tensor& x, const Tensor& y, const std::vector<size_t>& shape_x,
                const std::vector<size_t>& shape_y);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "DivBackward"; }
    size_t numInputs() const override { return 2; }

  private:
    Tensor mSavedX;  // Saved copy of x for backward pass
    Tensor mSavedY;  // Saved copy of y for backward pass
    std::vector<size_t> mShapeX;
    std::vector<size_t> mShapeY;
};

// ============================================================================
// Scalar Operation Backward Nodes
// ============================================================================

// ScalarMulBackward: Backward pass for scalar multiplication
// Forward:  z = x * scalar
// Backward: ∂L/∂x = ∂L/∂z * scalar
class ScalarMulBackward : public Node {
  public:
    explicit ScalarMulBackward(double scalar) : mScalar(scalar) {}

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ScalarMulBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    double mScalar;
};

// ScalarDivBackward: Backward pass for scalar division
// Forward:  z = x / scalar
// Backward: ∂L/∂x = ∂L/∂z * (1/scalar)
class ScalarDivBackward : public Node {
  public:
    explicit ScalarDivBackward(double scalar) : mScalar(scalar) {}

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ScalarDivBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    double mScalar;
};

// ScalarAddBackward: Backward pass for scalar addition
// Forward:  z = x + scalar
// Backward: ∂L/∂x = ∂L/∂z * 1 = ∂L/∂z
class ScalarAddBackward : public Node {
  public:
    ScalarAddBackward() = default;

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ScalarAddBackward"; }
    size_t numInputs() const override { return 1; }
};

// ScalarSubBackward: Backward pass for scalar subtraction
// Forward:  z = x - scalar
// Backward: ∂L/∂x = ∂L/∂z * 1 = ∂L/∂z
class ScalarSubBackward : public Node {
  public:
    ScalarSubBackward() = default;

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ScalarSubBackward"; }
    size_t numInputs() const override { return 1; }
};

}  // namespace autograd
}  // namespace loom
