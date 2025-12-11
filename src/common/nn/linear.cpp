#include "loom/nn/linear.h"

#include <stdexcept>

namespace loom {
namespace nn {

Linear::Linear(size_t inFeatures, size_t outFeatures, bool bias)
    : mInFeatures(inFeatures), mOutFeatures(outFeatures) {
    mWeight = registerParameter("weight", Parameter::kaiming({mOutFeatures, mInFeatures}));
    if (bias) {
        mBias = registerParameter("bias", Parameter::zeros({mOutFeatures}));
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (input.shape().size() != 2 || input.shape()[1] != mInFeatures) {
        throw std::runtime_error("Input must be a 2D tensor with shape [batch_size, in_features]");
    }

    Tensor weight_t = mWeight->data().transpose();
    Tensor output = input.matmul(weight_t);
    if (mBias) {
        output = output + mBias->data();
    }
    return output;
}

}  // namespace nn
}  // namespace loom
