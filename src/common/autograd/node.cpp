#include "loom/autograd/node.h"

#include <stdexcept>

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

void Node::setInputTensors(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    mInputTensors = inputs;
    mSavedVersions.clear();

    // Capture current version of each input tensor
    for (const auto& input : inputs) {
        mSavedVersions.push_back(input->version());
    }
}

void Node::checkVersions() const {
    // Check that all saved tensors have the same version as when they were saved
    // If version changed, a tensor was modified in-place after being saved
    for (size_t i = 0; i < mInputTensors.size(); ++i) {
        uint64_t saved_version = mSavedVersions[i];
        uint64_t current_version = mInputTensors[i]->version();

        if (current_version != saved_version) {
            throw std::runtime_error(
                "RuntimeError: one of the variables needed for gradient "
                "computation has been modified by an in-place operation: "
                "[" + mName + "] is at version " + std::to_string(current_version) +
                "; expected version " + std::to_string(saved_version));
        }
    }
}

}  // namespace autograd
}  // namespace loom
