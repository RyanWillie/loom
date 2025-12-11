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

// PermuteBackward: Backward pass for permute operation
// Forward:  y = x.permute(dims)  - reorders dimensions according to dims
// Backward: ∂L/∂x = (∂L/∂y).permute(inverse_dims)
//
// Key insight: Permute backward uses the inverse permutation.
// If forward permutation was [2, 0, 1], the inverse is [1, 2, 0]
//
// Example: If x.shape = [2, 3, 4] and we do permute([2, 0, 1]):
//          y.shape = [4, 2, 3]
//          If ∂L/∂y has shape [4, 2, 3], then ∂L/∂x must have shape [2, 3, 4]
//          We achieve this by permute(∂L/∂y, [1, 2, 0]) which reverses the reordering
class PermuteBackward : public Node {
  public:
    // Save the forward permutation to compute inverse in backward
    PermuteBackward(const std::vector<int>& dims);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "PermuteBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    std::vector<int> mDims;  // Forward permutation
};

// SliceBackward: Backward pass for slice operation
// Forward:  y = x.slice(dim, start, end)  - extracts a range along dimension
// Backward: ∂L/∂x = zeros with x.shape, then ∂L/∂x[..., start:end, ...] = ∂L/∂y
//
// Key insight: Slice backward scatters the gradient to the correct positions.
// All positions outside the slice get zero gradient.
//
// Example: If x.shape = [10, 20] and we do slice(0, 2, 5):
//          y.shape = [3, 20] (rows 2, 3, 4)
//          If ∂L/∂y has shape [3, 20], then ∂L/∂x has shape [10, 20] where:
//          - Rows 0-1: all zeros
//          - Rows 2-4: values from ∂L/∂y
//          - Rows 5-9: all zeros
class SliceBackward : public Node {
  public:
    // Save input shape and slice parameters
    SliceBackward(const std::vector<size_t>& input_shape, int dim, size_t start, size_t end);

    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "SliceBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    std::vector<size_t> mInputShape;  // Original input shape
    int mDim;                          // Dimension that was sliced
    size_t mStart;                     // Start index of slice
    size_t mEnd;                       // End index of slice
};

}  // namespace autograd
}  // namespace loom
