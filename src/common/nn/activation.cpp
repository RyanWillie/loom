#include "loom/nn/activation.h"

namespace loom {
namespace nn {

Tensor ReLU::forward(const Tensor& input) {
    // Simply delegate to Tensor's relu() method which handles autograd
    return input.relu();
}

}  // namespace nn
}  // namespace loom
