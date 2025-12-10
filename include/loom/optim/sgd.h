#pragma once

#include "loom/optim/optimizer.h"

namespace loom {
namespace optim {

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 *
 * Implements vanilla SGD parameter updates:
 *   θ_new = θ_old - lr * ∇θ
 *
 * Where:
 *   θ     = parameter
 *   lr    = learning rate
 *   ∇θ    = gradient of loss with respect to parameter
 *
 * Example usage:
 *   auto params = model->parameters();
 *   SGD optimizer(params, 0.01);  // lr = 0.01
 *
 *   for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *       optimizer.zeroGrad();
 *       Tensor loss = loss_fn(model->forward(x), y);
 *       loss.backward();
 *       optimizer.step();
 *   }
 */
class SGD : public Optimizer {
  public:
    /**
     * @brief Construct SGD optimizer.
     * @param parameters Parameters to optimize
     * @param lr Learning rate (default: 0.01)
     */
    SGD(const std::vector<std::shared_ptr<nn::Parameter>>& parameters, double lr = 0.01);

    /**
     * @brief Perform SGD update step.
     *
     * Updates each parameter: p = p - lr * grad(p)
     * Uses NoGrad context to prevent building computation graphs.
     */
    void step() override;
};

}  // namespace optim
}  // namespace loom
