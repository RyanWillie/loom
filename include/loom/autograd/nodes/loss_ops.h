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
// NLL Extract Backward Node
// ============================================================================

// NLLExtractBackward: Backward pass for extracting negative log likelihood
// Forward:  nll[i] = -log_probs[i, target[i]]
// Backward: ∂L/∂log_probs[i,j] = -∂L/∂nll[i] if j == target[i], else 0
//
// This is the backward pass for indexing into log probabilities by target class.
class NLLExtractBackward : public Node {
  public:
    // Save targets and shape information for backward pass
    NLLExtractBackward(const Tensor& targets, size_t batch_size, size_t num_classes);

    // Given: gradOutput = ∂L/∂nll of shape [batch_size]
    // Return: ∂L/∂log_probs of shape [batch_size, num_classes]
    //
    // For each sample i, the gradient flows only to log_probs[i, target[i]]:
    //   grad_log_probs[i, target[i]] = -gradOutput[i]
    //   grad_log_probs[i, j] = 0  for j != target[i]
    std::vector<Tensor> backward(const Tensor& gradOutput) override;

    std::string name() const override { return "NLLExtractBackward"; }
    size_t numInputs() const override { return 1; }

  private:
    Tensor mTargets;         // Saved target indices
    size_t mBatchSize;       // Batch size
    size_t mNumClasses;      // Number of classes
};

}  // namespace autograd
}  // namespace loom
