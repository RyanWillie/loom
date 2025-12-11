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
// Factory Methods - Weight Initialization
// ============================================================================

Parameter Parameter::kaiming(const std::vector<size_t>& shape, DType dtype) {
    // Kaiming (He) initialization designed for ReLU networks
    // Formula: weights ~ N(0, std²) where std = sqrt(2 / fan_in)

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
    // Xavier (Glorot) initialization designed for tanh/sigmoid networks
    // Formula: weights ~ N(0, std²) where std = sqrt(1 / fan_in)
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
