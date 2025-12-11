#include "loom/optim/optimizer.h"

namespace loom {
namespace optim {

Optimizer::Optimizer(const std::vector<std::shared_ptr<nn::Parameter>>& parameters, double lr)
    : mParameters(parameters), mLearningRate(lr) {}

void Optimizer::zeroGrad() {
    for (auto& parameter : mParameters) {
        parameter->zeroGrad();
    }
}

}  // namespace optim
}  // namespace loom
