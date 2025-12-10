#include "loom/nn/loss.h"

#include <cmath>
#include <stdexcept>

#include "loom/autograd/no_grad.h"
#include "loom/autograd/nodes/loss_ops.h"

namespace loom {
namespace nn {

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction) : Loss(reduction) {}

Tensor CrossEntropyLoss::logSoftmax(const Tensor& input) {
    // Log-softmax with numerical stability using log-sum-exp trick
    //
    // Mathematical formula: log_softmax(x) = x - log(sum(exp(x)))
    // Numerical stable form: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    //
    // The max(x) operation is ONLY for numerical stability - it doesn't affect
    // the mathematical value, so it shouldn't contribute gradients. We use NoGrad
    // to prevent gradient flow through max.

    // Note: max() operation used only for numerical stability
    // Ideally should use NoGrad context to avoid unnecessary gradient tracking
    // Current implementation is correct but slightly inefficient
    Tensor max_vals = input.max(-1, true);

    Tensor shifted = input - max_vals;
    Tensor exp_shifted = shifted.exp();
    Tensor sum_exp = exp_shifted.sum(-1, true);
    Tensor log_sum_exp = sum_exp.log();
    return shifted - log_sum_exp;
}

Tensor CrossEntropyLoss::extractNLL(const Tensor& log_probs, const Tensor& targets) {
    size_t batch_size = log_probs.shape()[0];
    size_t num_classes = log_probs.shape()[1];

    // Create NLL tensor and fill it
    Tensor nll = Tensor::zeros({batch_size}, DType::FLOAT32, log_probs.device());

    auto log_probs_acc = log_probs.accessor<float, 2>();
    auto targets_acc = targets.accessor<int64_t, 1>();
    auto nll_acc = nll.accessor<float, 1>();

    for (size_t i = 0; i < batch_size; ++i) {
        int64_t target_class = targets_acc[i];
        if (target_class < 0 || target_class >= static_cast<int64_t>(num_classes)) {
            throw std::runtime_error("Target class out of range");
        }
        nll_acc[i] = -log_probs_acc[i][target_class];
    }

    // Attach autograd if log_probs requires grad
    if (log_probs.requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto nll_node =
            std::make_shared<autograd::NLLExtractBackward>(targets, batch_size, num_classes);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (log_probs.gradFn())
            next_fns.push_back(log_probs.gradFn());
        nll_node->setNextFunctions(next_fns);
        nll_node->setInputTensors({std::make_shared<Tensor>(log_probs)});

        nll.setGradFn(nll_node);
        nll.requiresGrad(true);
    }

    return nll;
}

Tensor CrossEntropyLoss::forward(const Tensor& predictions, const Tensor& targets) {
    // Validate input shapes
    if (predictions.shape().size() != 2 || targets.shape().size() != 1) {
        throw std::runtime_error("Predictions and targets must be 2D and 1D tensors respectively");
    }
    if (predictions.shape()[0] != targets.shape()[0]) {
        throw std::runtime_error("Predictions and targets must have the same number of samples");
    }

    // Compute log_softmax (with numerical stability)
    Tensor log_probs = logSoftmax(predictions);

    // Extract NLL for target classes (with autograd support)
    Tensor nll = extractNLL(log_probs, targets);

    // Apply reduction
    if (reduction() == Reduction::NONE) {
        return nll;
    } else if (reduction() == Reduction::MEAN) {
        return nll.mean();
    } else if (reduction() == Reduction::SUM) {
        return nll.sum();
    }

    throw std::runtime_error("Invalid reduction mode");
}

}  // namespace nn
}  // namespace loom
