#pragma once

#include "loom/tensor/tensor.h"
#include <memory>

namespace loom {
namespace nn {

/**
 * @brief Wrapper around Tensor for learnable parameters in neural networks.
 *
 * Parameter is a thin wrapper that marks a Tensor as trainable. It automatically
 * sets requiresGrad(true) and provides semantic clarity that this tensor should
 * be updated during optimization.
 *
 * Key differences from raw Tensor:
 * - Always requires gradients by default
 * - Type-safe: prevents accidentally registering non-learnable tensors
 * - Semantic meaning: clearly indicates this is a learnable weight
 * - Extensible: can add parameter-specific features (regularization, constraints)
 */
class Parameter {
private:
    Tensor mData;

public:
    /**
     * @brief Construct a Parameter from a Tensor.
     * @param data The underlying tensor data
     * @param requiresGrad Whether to track gradients (default: true)
     */
    explicit Parameter(const Tensor& data, bool requiresGrad = true);

    /**
     * @brief Get mutable reference to underlying tensor.
     */
    Tensor& data() { return mData; }

    /**
     * @brief Get const reference to underlying tensor.
     */
    const Tensor& data() const { return mData; }

    /**
     * @brief Get gradient tensor.
     */
    std::shared_ptr<Tensor> grad() { return mData.grad(); }

    /**
     * @brief Get const gradient tensor.
     */
    std::shared_ptr<const Tensor> grad() const { return mData.grad(); }

    /**
     * @brief Zero the gradient of this parameter.
     */
    void zeroGrad();

    // ========================================================================
    // Factory methods for common initializations
    // ========================================================================

    /**
     * @brief Create parameter initialized with zeros.
     * @param shape Shape of the parameter tensor
     * @param dtype Data type (default: FLOAT32)
     */
    static Parameter zeros(const std::vector<size_t>& shape,
                          DType dtype = DType::FLOAT32);

    /**
     * @brief Create parameter initialized with ones.
     * @param shape Shape of the parameter tensor
     * @param dtype Data type (default: FLOAT32)
     */
    static Parameter ones(const std::vector<size_t>& shape,
                         DType dtype = DType::FLOAT32);

    /**
     * @brief Create parameter with Kaiming (He) initialization.
     *
     * Scales weights by sqrt(2 / fan_in) for ReLU networks. This initialization
     * helps prevent vanishing/exploding gradients by maintaining variance across
     * layers during forward and backward passes.
     *
     * Reference: He et al., "Delving Deep into Rectifiers" (2015)
     *
     * @param shape Shape of the parameter tensor
     * @param dtype Data type (default: FLOAT32)
     */
    static Parameter kaiming(const std::vector<size_t>& shape,
                            DType dtype = DType::FLOAT32);

    /**
     * @brief Create parameter with Xavier (Glorot) initialization.
     *
     * Scales weights by sqrt(1 / fan_in) for tanh/sigmoid networks. This maintains
     * variance for networks with symmetric activation functions.
     *
     * Reference: Glorot & Bengio, "Understanding the difficulty of training deep
     * feedforward neural networks" (2010)
     *
     * @param shape Shape of the parameter tensor
     * @param dtype Data type (default: FLOAT32)
     */
    static Parameter xavier(const std::vector<size_t>& shape,
                           DType dtype = DType::FLOAT32);

    /**
     * @brief Create parameter with uniform random initialization in [low, high).
     * @param shape Shape of the parameter tensor
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @param dtype Data type (default: FLOAT32)
     */
    static Parameter uniform(const std::vector<size_t>& shape,
                            double low = 0.0,
                            double high = 1.0,
                            DType dtype = DType::FLOAT32);

    // ========================================================================
    // Device operations
    // ========================================================================

    /**
     * @brief Move parameter to a different device.
     * @param device Target device (CPU, CUDA, MPS)
     */
    void to(Device device);

    /**
     * @brief Get the device this parameter resides on.
     */
    Device device() const { return mData.device(); }

    /**
     * @brief Get the shape of the parameter tensor.
     */
    const std::vector<size_t>& shape() const { return mData.shape(); }

    /**
     * @brief Get the number of elements in the parameter.
     */
    size_t numel() const { return mData.numel(); }
};

}  // namespace nn
}  // namespace loom
