#include "loom/optim/sgd.h"

#include "loom/autograd/no_grad.h"

namespace loom {
namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<nn::Parameter>>& parameters, double lr)
    : Optimizer(parameters, lr) {}

void SGD::step() {
    autograd::NoGrad no_grad;
    for (auto& parameter : mParameters) {
        if (parameter->grad()) {
            parameter->data() -= (*parameter->grad()) * mLearningRate;
        }
    }
}

}  // namespace optim
}  // namespace loom
