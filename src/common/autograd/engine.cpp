#include "loom/autograd/engine.h"

#include <algorithm>
#include <stdexcept>

#include "loom/autograd/node.h"
#include "loom/logger.h"
#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

void Engine::backward(Tensor& root, const Tensor& gradOutput) {
    auto& logger = Logger::getInstance("Autograd");

    // STEP 1: Validate root tensor
    //   - Check if root.requiresGrad() is true
    //   - If false, throw an error (calling backward on non-grad tensor is an error)
    if (!root.requiresGrad()) {
        throw std::runtime_error(
            "element 0 of tensors does not require grad and does not have a grad_fn");
    }

    // STEP 2: Initialize gradient for root tensor
    //   - If root.grad() is nullptr:
    //       * Create new gradient tensor: Tensor::zeros(root.shape(), root.dtype(), root.device())
    //       * Store it in root's AutogradMeta->grad
    //   - Accumulate gradOutput into root's gradient using +=
    //       * Handle both same-shape and scalar gradOutput cases
    root.accumulateGrad(gradOutput);

    // STEP 3: Check if root is a leaf
    //   - If root.isLeaf(), log debug message and return
    //   - Leaf tensors have no predecessors to propagate to
    if (root.isLeaf()) {
        logger.debug("Root tensor is a leaf");
        return;
    }

    // STEP 4: Get root's gradient function
    //   - Get root.gradFn()
    //   - If nullptr, log warning and return
    if (!root.gradFn()) {
        logger.warning("Root tensor does not have a gradient function");
        return;
    }

    // STEP 5: Topological sort
    //   - Call topologicalSort(root.gradFn())
    //   - This gives you nodes in reverse topological order
    //   - Log the number of nodes found
    std::vector<std::shared_ptr<Node>> sorted = topologicalSort(root.gradFn());
    logger.info("Topological sort completed {} nodes", sorted.size());

    // STEP 6: Create gradient map for intermediate tensors
    //   - Use std::unordered_map<Node*, Tensor>
    //   - Initialize with root_node -> root.grad()
    std::unordered_map<Node*, Tensor> grad_map;
    grad_map.insert({root.gradFn().get(), *root.grad()});

    // STEP 7: Traverse nodes in reverse topological order
    //   - For each node:
    //       a. Get node's output gradient from grad_map
    //       b. Call node->backward(grad_output) to get input gradients
    //       c. For each input gradient:
    //          - Get corresponding input tensor (from node->inputTensors())
    //          - If input requires grad:
    //              * Initialize input's grad if nullptr
    //              * Accumulate gradient using +=
    //              * If input is non-leaf with gradFn:
    //                  - Add/accumulate to grad_map for further propagation
    for (const auto& node : sorted) {
        // Get the gradient for this node's output
        // Use .at() instead of [] since Tensor has no default constructor
        const auto& grad_output = grad_map.at(node.get());

        // Compute gradients for inputs
        const auto& grad_inputs = node->backward(grad_output);

        // Get references to the actual input tensors
        const auto& input_tensors = node->inputTensors();

        // Match each computed gradient to its corresponding input tensor
        for (size_t i = 0; i < grad_inputs.size(); ++i) {
            auto& input_tensor = input_tensors[i];
            const Tensor& grad_input = grad_inputs[i];

            // Only accumulate if this tensor tracks gradients
            if (input_tensor->requiresGrad()) {
                // Accumulate gradient (handles initialization automatically)
                input_tensor->accumulateGrad(grad_input);

                // If non-leaf tensor, propagate gradient to its grad_fn
                if (!input_tensor->isLeaf() && input_tensor->gradFn()) {
                    auto it = grad_map.find(input_tensor->gradFn().get());
                    if (it == grad_map.end()) {
                        // First time seeing this grad_fn - use insert to avoid operator[]
                        grad_map.insert({input_tensor->gradFn().get(), grad_input});
                    } else {
                        // Multiple paths lead to this node - accumulate
                        it->second += grad_input;
                    }
                }
            }
        }
    }

    // STEP 8: Log completion
    logger.info("Backward pass completed");
    return;
}

std::vector<std::shared_ptr<Node>> Engine::topologicalSort(const std::shared_ptr<Node>& root_node) {
    // STEP 1: Create visited set
    //   - Use std::unordered_set<Node*> to track visited nodes
    //   - Use raw pointers for set (shared_ptr for ownership)
    //
    std::unordered_set<Node*> visited;
    // STEP 2: Create result vector
    //   - Use std::vector<std::shared_ptr<Node>> for sorted nodes

    std::vector<std::shared_ptr<Node>> sorted;
    // STEP 3: Call DFS
    //   - topologicalSortDFS(root_node, visited, sorted)
    topologicalSortDFS(root_node, visited, sorted);
    // STEP 4: Reverse the result
    //   - DFS gives post-order, we need reverse
    //   - Use std::reverse(sorted.begin(), sorted.end())
    std::reverse(sorted.begin(), sorted.end());
    // STEP 5: Return sorted nodes
    return sorted;
}

void Engine::topologicalSortDFS(const std::shared_ptr<Node>& node,
                                std::unordered_set<Node*>& visited,
                                std::vector<std::shared_ptr<Node>>& sorted) {
    // STEP 1: Base case checks
    //   - If node is nullptr, return
    //   - If node already visited (check visited.count(node.get())), return
    if (!node || visited.count(node.get())) {
        return;
    }

    // STEP 2: Mark as visited
    //   - visited.insert(node.get())
    visited.insert(node.get());

    // STEP 3: Visit all predecessors (recursive step)
    //   - For each next_function in node->nextFunctions():
    //       * Recursively call topologicalSortDFS(next_function, visited, sorted)
    for (const auto& next_function : node->nextFunctions()) {
        topologicalSortDFS(next_function, visited, sorted);
    }
    // STEP 4: Add current node to sorted list (post-order)
    //   - sorted.push_back(node)
    sorted.push_back(node);
}

}  // namespace autograd
}  // namespace loom
