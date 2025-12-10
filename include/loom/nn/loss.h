#pragma once

#include <string>

#include "loom/nn/module.h"
#include "loom/tensor/tensor.h"

namespace loom {
namespace nn {

/**
 * @brief Reduction mode for loss functions.
 */
enum class Reduction {
    NONE,   // No reduction, return loss per sample
    MEAN,   // Average loss over batch
    SUM     // Sum loss over batch
};

/**
 * @brief Base class for loss functions.
 *
 * Loss functions inherit from Module but override forward() to take
 * two arguments: predictions and targets.
 */
class Loss : public Module {
  public:
    Loss(Reduction reduction = Reduction::MEAN) : mReduction(reduction) {}

    /**
     * @brief Compute loss from predictions and targets.
     *
     * @param predictions Model output (logits, probabilities, etc.)
     * @param targets Ground truth labels
     * @return Loss tensor (scalar if reduced, per-sample if Reduction::NONE)
     */
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0;

    /**
     * @brief Hide single-argument forward from Module.
     *
     * Loss requires two inputs, so single-argument forward is not valid.
     */
    Tensor forward(const Tensor&) override {
        throw std::runtime_error("Loss::forward requires predictions and targets");
    }

    /**
     * @brief Get reduction mode.
     */
    Reduction reduction() const { return mReduction; }

  protected:
    Reduction mReduction;
};

/**
 * @brief Cross-entropy loss for multi-class classification.
 *
 * Combines LogSoftmax and NegativeLogLikelihood in a numerically stable way.
 * This is the standard loss for classification tasks like MNIST.
 *
 * Formula:
 *   loss = -log(softmax(predictions)[target_class])
 *        = -log(exp(pred[target]) / sum(exp(pred)))
 *        = -pred[target] + log(sum(exp(pred)))
 *
 * Numerical stability:
 *   Use log-sum-exp trick to prevent overflow/underflow:
 *   log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
 *
 * Input shapes:
 *   predictions: [batch_size, num_classes] - raw logits (NOT probabilities!)
 *   targets: [batch_size] - class indices in range [0, num_classes)
 *
 * Output shape:
 *   - Reduction::NONE: [batch_size] - loss per sample
 *   - Reduction::MEAN/SUM: scalar
 *
 * Example:
 *   auto criterion = CrossEntropyLoss(Reduction::MEAN);
 *
 *   Tensor logits = model->forward(input);  // [32, 10]
 *   Tensor targets = ...;                    // [32] with values 0-9
 *   Tensor loss = criterion.forward(logits, targets);  // scalar
 *
 *   loss.backward();  // Compute gradients
 */
class CrossEntropyLoss : public Loss {
  public:
    /**
     * @brief Construct CrossEntropyLoss.
     *
     * @param reduction Reduction mode (default: MEAN)
     */
    explicit CrossEntropyLoss(Reduction reduction = Reduction::MEAN);

    /**
     * @brief Compute cross-entropy loss.
     *
     * @param predictions Logits of shape [batch_size, num_classes]
     * @param targets Class indices of shape [batch_size]
     * @return Loss tensor
     * @throws std::runtime_error if shapes are invalid or targets out of range
     */
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;

  private:
    /**
     * @brief Compute log-softmax along last dimension.
     *
     * Uses log-sum-exp trick for numerical stability:
     *   log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
     *
     * @param input Tensor of shape [batch_size, num_classes]
     * @return Log probabilities of same shape
     */
    Tensor logSoftmax(const Tensor& input);

    /**
     * @brief Extract negative log likelihood for target classes.
     *
     * For each sample i, extracts -log_probs[i, targets[i]]
     * Preserves autograd by manually setting up backward function.
     *
     * @param log_probs Log probabilities [batch_size, num_classes]
     * @param targets Target class indices [batch_size]
     * @return NLL for each sample [batch_size]
     */
    Tensor extractNLL(const Tensor& log_probs, const Tensor& targets);
};

}  // namespace nn
}  // namespace loom
