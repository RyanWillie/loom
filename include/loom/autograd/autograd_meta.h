#pragma once
#include <memory>

namespace loom {

// Forward declarations (avoid circular dependencies)
class Tensor;

namespace autograd {

// Forward declaration
class Node;

// AutogradMeta: Stores gradient tracking metadata for tensors
// This is a simple data struct (not a class) for Phase 1 simplicity
struct AutogradMeta {
    // Accumulated gradient tensor (same shape as parent tensor)
    // nullptr until first backward pass
    std::shared_ptr<Tensor> grad = nullptr;

    // Backward function that created this tensor
    // nullptr for leaf tensors (user-created, not from operations)
    std::shared_ptr<Node> gradFn = nullptr;

    // Whether this tensor requires gradient computation
    bool requiresGrad = false;

    // Whether this is a leaf tensor (user-created, not result of operation)
    bool isLeaf = true;

    // Version counter for in-place operation detection (Phase 7)
    uint64_t version = 0;

    // Constructors
    AutogradMeta() = default;

    AutogradMeta(bool requires_grad, bool is_leaf)
        : requiresGrad(requires_grad), isLeaf(is_leaf) {}
};

} // namespace autograd
} // namespace loom