#pragma once

#include <memory>
#include <string>
#include <vector>

namespace loom {

// Forward declaration (avoid circular dependency with tensor.h)
class Tensor;

namespace autograd {

// Node: Base class for all backward function nodes in the computation graph
// Each operation (Add, Mul, Matmul, etc.) will inherit from this and implement backward()
class Node : public std::enable_shared_from_this<Node> {
  public:
    virtual ~Node() = default;

    // Compute gradients for inputs given gradient of output
    // Returns vector of gradients (one per input tensor)
    // Example: For z = x + y, backward(dL/dz) returns [dL/dx, dL/dy]
    virtual std::vector<Tensor> backward(const Tensor& gradOutput) = 0;

    // Get name for debugging (e.g., "AddBackward", "MulBackward")
    virtual std::string name() const = 0;

    // Get number of inputs this node expects
    // Example: Add has 2 inputs, ReLU has 1 input
    virtual size_t numInputs() const = 0;

    // Store references to next functions (predecessor nodes in computation graph)
    // These are the grad_fn's of the input tensors
    void setNextFunctions(const std::vector<std::shared_ptr<Node>>& next) { mNextFunctions = next; }

    // Get predecessor nodes for graph traversal
    const std::vector<std::shared_ptr<Node>>& nextFunctions() const { return mNextFunctions; }

    // Store references to input tensors (for gradient accumulation)
    // These are the actual tensors that were inputs to the operation
    // Also captures their versions for in-place operation detection
    void setInputTensors(const std::vector<std::shared_ptr<Tensor>>& inputs);

    // Get input tensors for gradient accumulation
    const std::vector<std::shared_ptr<Tensor>>& inputTensors() const { return mInputTensors; }

  protected:
    // Check that saved tensors haven't been modified in-place
    // Throws runtime_error if version mismatch detected
    void checkVersions() const;

    // Predecessor nodes in the computation graph
    // Used by the backward engine for topological sort
    std::vector<std::shared_ptr<Node>> mNextFunctions;

    // Input tensors that produced this node's output
    // Used by the backward engine to accumulate gradients
    // Stored as shared_ptr to keep tensors alive during backward pass
    std::vector<std::shared_ptr<Tensor>> mInputTensors;

    // Versions of input tensors at the time they were saved
    // Used to detect in-place modifications
    std::vector<uint64_t> mSavedVersions;

    // Cached name for debugging
    std::string mName;
};

}  // namespace autograd
}  // namespace loom