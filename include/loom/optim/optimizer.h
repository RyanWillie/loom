#pragma once

#include <memory>
#include <vector>

#include "loom/nn/parameter.h"

namespace loom {
namespace optim {

/**
 * @brief Abstract base class for all optimizers.
 *
 * Optimizers update model parameters based on their gradients. The typical
 * training loop pattern is:
 *
 *   optimizer.zeroGrad();      // Clear previous gradients
 *   loss = model.forward(...); // Forward pass
 *   loss.backward();           // Compute gradients
 *   optimizer.step();          // Update parameters
 *
 * All parameter updates MUST occur within a NoGrad context to prevent
 * building computation graphs during optimization.
 */
class Optimizer {
  protected:
    std::vector<std::shared_ptr<nn::Parameter>> mParameters;
    double mLearningRate;

  public:
    /**
     * @brief Construct optimizer with given parameters and learning rate.
     * @param parameters Vector of learnable parameters to optimize
     * @param lr Learning rate (step size)
     */
    Optimizer(const std::vector<std::shared_ptr<nn::Parameter>>& parameters, double lr);

    virtual ~Optimizer() = default;

    /**
     * @brief Perform a single optimization step (parameter update).
     *
     * Must be implemented by derived classes. Implementation MUST use
     * NoGrad context to disable autograd during parameter updates.
     */
    virtual void step() = 0;

    /**
     * @brief Zero out all parameter gradients.
     *
     * This should be called before each backward pass to clear accumulated
     * gradients from the previous iteration.
     */
    void zeroGrad();

    /**
     * @brief Get the current learning rate.
     */
    double learningRate() const { return mLearningRate; }

    /**
     * @brief Set a new learning rate (for learning rate scheduling).
     */
    void setLearningRate(double lr) { mLearningRate = lr; }

    /**
     * @brief Get the number of parameters being optimized.
     */
    size_t numParameters() const { return mParameters.size(); }
};

}  // namespace optim
}  // namespace loom
