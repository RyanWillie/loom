#include "loom/autograd/nodes/loss_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// NLLExtractBackward: nll[i] = -log_probs[i, target[i]]
// ============================================================================

NLLExtractBackward::NLLExtractBackward(const Tensor& targets, size_t batch_size, size_t num_classes)
    : mTargets(targets), mBatchSize(batch_size), mNumClasses(num_classes) {
    mName = "NLLExtractBackward";
}

std::vector<Tensor> NLLExtractBackward::backward(const Tensor& gradOutput) {
    checkVersions();

    // Create gradient tensor for log_probs: [batch_size, num_classes]
    Tensor grad_log_probs = Tensor::zeros({mBatchSize, mNumClasses},
                                           gradOutput.dtype(), gradOutput.device());

    // Scatter gradients to the target indices
    // For each sample i: grad_log_probs[i, target[i]] = -gradOutput[i]
    auto grad_out_acc = gradOutput.accessor<float, 1>();
    auto targets_acc = mTargets.accessor<int64_t, 1>();
    auto grad_log_probs_acc = grad_log_probs.accessor<float, 2>();

    for (size_t i = 0; i < mBatchSize; ++i) {
        int64_t target_class = targets_acc[i];
        grad_log_probs_acc[i][target_class] = -grad_out_acc[i];
    }

    return {grad_log_probs};
}

}  // namespace autograd
}  // namespace loom
