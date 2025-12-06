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
// View Operation Backward Nodes
// ============================================================================
// View operations (reshape, transpose, etc.) don't copy data - they just
// change how the tensor is interpreted. The backward pass simply reverses
// the view transformation on the gradient.

// ReshapeBackward: Backward pass for reshape operation
// Forward:  y = x.reshape(new_shape)
// Backward: ∂L/∂x = (∂L/∂y).reshape(original_shape)
class ReshapeBackward : public Node {
  public:
    ReshapeBackward(const std::vector<size_t>& input_shape);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "ReshapeBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    std::vector<size_t> mInputShape;
};

// TransposeBackward: Backward pass for transpose operation
// Forward:  y = x.transpose() or x.transpose(dim0, dim1)
// Backward: ∂L/∂x = (∂L/∂y).transpose() with same dimensions
class TransposeBackward : public Node {
  public:
    // For simple transpose (swap last two dims)
    TransposeBackward();
    // For transpose with specific dimensions
    TransposeBackward(int dim0, int dim1);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "TransposeBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    int mDim0;
    int mDim1;
    bool mSimpleTranspose;  // true if using default (last two dims)
};

// FlattenBackward: Backward pass for flatten operation
// Forward:  y = x.flatten()
// Backward: ∂L/∂x = (∂L/∂y).reshape(original_shape)
// Note: Flatten is just reshape to 1D, so backward is reshape back
class FlattenBackward : public Node {
  public:
    FlattenBackward(const std::vector<size_t>& input_shape);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "FlattenBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    std::vector<size_t> mInputShape;
};

// SqueezeBackward: Backward pass for squeeze operation
// Forward:  y = x.squeeze(dim) - removes dimension of size 1
// Backward: ∂L/∂x = (∂L/∂y).unsqueeze(dim) - adds dimension back
class SqueezeBackward : public Node {
  public:
    SqueezeBackward(int dim);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "SqueezeBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    int mDim;
};

// UnsqueezeBackward: Backward pass for unsqueeze operation
// Forward:  y = x.unsqueeze(dim) - adds dimension of size 1
// Backward: ∂L/∂x = (∂L/∂y).squeeze(dim) - removes dimension
class UnsqueezeBackward : public Node {
  public:
    UnsqueezeBackward(int dim);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "UnsqueezeBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    int mDim;
};

}  // namespace autograd
}  // namespace loom
