#pragma once

#include <cstddef>
#include <memory>

#include "loom/nn/module.h"
#include "loom/nn/parameter.h"
#include "loom/tensor/tensor.h"

namespace loom {
namespace nn {

/**
 * @brief Fully connected (linear) layer: y = xW^T + b
 *
 * Applies a linear transformation to incoming data:
 *   output = input @ weight.T + bias
 *
 * Shape transformation:
 *   input:  [batch_size, in_features]
 *   weight: [out_features, in_features]
 *   bias:   [out_features]
 *   output: [batch_size, out_features]
 *
 * Weight initialization:
 *   - Uses Kaiming (He) initialization for weights (good for ReLU networks)
 *   - Initializes bias to zeros
 *
 * Example usage:
 *   auto layer = std::make_shared<Linear>(784, 128);  // MNIST input -> hidden
 *   Tensor x = Tensor::randn({32, 784});  // batch_size=32
 *   Tensor y = layer->forward(x);         // shape: [32, 128]
 */
class Linear : public Module {
  public:
    /**
     * @brief Construct a Linear layer.
     *
     * @param inFeatures Number of input features
     * @param outFeatures Number of output features
     * @param bias Whether to include a bias term (default: true)
     */
    Linear(size_t inFeatures, size_t outFeatures, bool bias = true);

    /**
     * @brief Forward pass: y = xW^T + b
     *
     * @param input Input tensor of shape [batch_size, in_features]
     * @return Output tensor of shape [batch_size, out_features]
     * @throws std::runtime_error if input is not 2D or has wrong feature size
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Get the weight parameter.
     */
    std::shared_ptr<Parameter> weight() const { return mWeight; }

    /**
     * @brief Get the bias parameter (may be nullptr if bias=false).
     */
    std::shared_ptr<Parameter> bias() const { return mBias; }

    /**
     * @brief Get number of input features.
     */
    size_t inFeatures() const { return mInFeatures; }

    /**
     * @brief Get number of output features.
     */
    size_t outFeatures() const { return mOutFeatures; }

  private:
    size_t mInFeatures;
    size_t mOutFeatures;
    std::shared_ptr<Parameter> mWeight;
    std::shared_ptr<Parameter> mBias;  // nullptr if bias=false
};

}  // namespace nn
}  // namespace loom
