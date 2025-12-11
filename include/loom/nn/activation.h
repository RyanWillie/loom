#pragma once

#include "loom/nn/module.h"
#include "loom/tensor/tensor.h"

namespace loom {
namespace nn {

/**
 * @brief ReLU (Rectified Linear Unit) activation function module.
 *
 * Applies the ReLU activation function element-wise:
 *   ReLU(x) = max(0, x)
 *
 * This module can be used standalone or composed in Sequential containers.
 */
class ReLU : public Module {
   public:
    /**
     * @brief Construct a ReLU activation module.
     */
    ReLU() = default;

    /**
     * @brief Apply ReLU activation to input tensor.
     *
     * @param input Input tensor of any shape
     * @return Tensor Output tensor with ReLU applied element-wise
     */
    Tensor forward(const Tensor& input) override;
};

}  // namespace nn
}  // namespace loom
