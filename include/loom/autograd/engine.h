#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loom {

// Forward declarations
class Tensor;

namespace autograd {

// Forward declaration
class Node;

// Engine: Orchestrates the backward pass through the computation graph
// Implements reverse-mode automatic differentiation
class Engine {
  public:
    // Execute backward pass starting from root tensor
    // root: The tensor to compute gradients from (typically loss)
    // gradOutput: The gradient of the loss with respect to root (typically ones)
    static void backward(Tensor& root, const Tensor& gradOutput);

  private:
    // Perform topological sort of computation graph using DFS
    // Returns nodes in reverse topological order (root first)
    static std::vector<std::shared_ptr<Node>> topologicalSort(
        const std::shared_ptr<Node>& root_node);

    // DFS helper for topological sort
    // Recursively visits all predecessor nodes
    static void topologicalSortDFS(const std::shared_ptr<Node>& node,
                                    std::unordered_set<Node*>& visited,
                                    std::vector<std::shared_ptr<Node>>& sorted);
};

}  // namespace autograd
}  // namespace loom
