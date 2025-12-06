#include "loom/nn/parameter.h"

#include <cmath>
#include <stdexcept>

namespace loom {
namespace nn {

// ============================================================================
// Constructor
// ============================================================================

Parameter::Parameter(const Tensor& data, bool requiresGrad) : mData(data) {
    mData.requiresGrad(requiresGrad);
}

// ============================================================================
// Gradient Operations
// ============================================================================

void Parameter::zeroGrad() {
    mData.zeroGrad();
}

// ============================================================================
// Factory Methods - Simple Initializations
// ============================================================================

Parameter Parameter::zeros(const std::vector<size_t>& shape, DType dtype) {
    Tensor t = Tensor::zeros(shape, dtype);
    return Parameter(t, true);  // requiresGrad = true by default
}

Parameter Parameter::ones(const std::vector<size_t>& shape, DType dtype) {
    Tensor t = Tensor::ones(shape, dtype);
    return Parameter(t, true);
}

// ============================================================================
// Factory Methods - Weight Initialization (TODO: Implement these!)
// ============================================================================

Parameter Parameter::kaiming(const std::vector<size_t>& shape, DType dtype) {
    // TODO(human): Implement Kaiming (He) initialization
    //
    // Kaiming initialization is designed for ReLU networks.
    // Formula: weights ~ N(0, std²) where std = sqrt(2 / fan_in)
    //
    // Steps:
    // 1. Validate shape is not empty (throw std::runtime_error if empty)
    // 2. Compute fan_in = shape[0] (first dimension of weight matrix)
    // 3. Compute std = sqrt(2.0 / fan_in)
    // 4. Create random normal tensor: Tensor t = Tensor::randn(shape, dtype)
    // 5. Scale by std: t = t * std (or t *= std if in-place multiplication works)
    // 6. Return Parameter(t, true)
    //
    // Why shape[0]? For a Linear layer weight matrix [out_features, in_features],
    // fan_in is the number of inputs, which is the first dimension.

    if (shape.empty()) {
        throw std::runtime_error("Shape is empty");
    }

    size_t fan_in = shape[0];
    double std = std::sqrt(2.0 / fan_in);
    Tensor t = Tensor::randn(shape, dtype);
    t = t * std;
    return Parameter(t, true);
}

Parameter Parameter::xavier(const std::vector<size_t>& shape, DType dtype) {
    // TODO(human): Implement Xavier (Glorot) initialization
    //
    // Xavier initialization is designed for tanh/sigmoid networks.
    // Formula: weights ~ N(0, std²) where std = sqrt(1 / fan_in)
    //
    // Steps:
    // 1. Validate shape is not empty
    // 2. Compute fan_in = shape[0]
    // 3. Compute std = sqrt(1.0 / fan_in)
    // 4. Create random normal tensor: Tensor t = Tensor::randn(shape, dtype)
    // 5. Scale by std: t = t * std
    // 6. Return Parameter(t, true)
    //
    // The difference from Kaiming is the scaling factor: 1 vs 2 in the numerator.
    // This accounts for the different activation function properties.
    if (shape.empty()) {
        throw std::runtime_error("Shape is empty");
    }

    size_t fan_in = shape[0];
    double std = std::sqrt(1.0 / fan_in);
    Tensor t = Tensor::randn(shape, dtype);
    t = t * std;
    return Parameter(t, true);
}

Parameter Parameter::uniform(const std::vector<size_t>& shape, double low, double high,
                             DType dtype) {
    // Create tensor and initialize with uniform distribution
    Tensor t = Tensor::rand(shape, dtype);  // rand() gives uniform [0, 1)

    // Scale to [low, high): t = t * (high - low) + low
    t = t * (high - low) + low;

    return Parameter(t, true);
}

// ============================================================================
// Device Operations
// ============================================================================

void Parameter::to(Device device) {
    mData = mData.toDevice(device);
}

}  // namespace nn
}  // namespace loom
