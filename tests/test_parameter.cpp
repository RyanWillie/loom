#include <cmath>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/nn/parameter.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;
using namespace loom::nn;

class ParameterTest : public ::testing::Test {
  protected:
    ParameterTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("ParameterTest");
        logger.info("Test fixture initialized");
        // Set fixed seed for reproducible tests
        Tensor::manualSeed(42);
    }

    Device mCpuDevice;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(ParameterTest, ConstructWithTensor) {
    Tensor t = Tensor::ones({2, 3}, DType::FLOAT32, mCpuDevice);
    Parameter param(t);

    EXPECT_TRUE(param.data().requiresGrad());
    EXPECT_EQ(param.data().shape()[0], 2);
    EXPECT_EQ(param.data().shape()[1], 3);
}

TEST_F(ParameterTest, ConstructWithRequiresGradFalse) {
    Tensor t = Tensor::ones({2, 3}, DType::FLOAT32, mCpuDevice);
    Parameter param(t, false);

    EXPECT_FALSE(param.data().requiresGrad());
}

// ============================================================================
// Factory Method Tests - Basic Properties
// ============================================================================

TEST_F(ParameterTest, ZerosCreatesCorrectShape) {
    Parameter param = Parameter::zeros({3, 4}, DType::FLOAT32);

    EXPECT_EQ(param.data().shape()[0], 3);
    EXPECT_EQ(param.data().shape()[1], 4);
    EXPECT_TRUE(param.data().requiresGrad());
}

TEST_F(ParameterTest, OnesCreatesCorrectShape) {
    Parameter param = Parameter::ones({5, 2}, DType::FLOAT32);

    EXPECT_EQ(param.data().shape()[0], 5);
    EXPECT_EQ(param.data().shape()[1], 2);
    EXPECT_TRUE(param.data().requiresGrad());
}

TEST_F(ParameterTest, UniformCreatesCorrectShape) {
    Parameter param = Parameter::uniform({4, 3}, -1.0, 1.0, DType::FLOAT32);

    EXPECT_EQ(param.data().shape()[0], 4);
    EXPECT_EQ(param.data().shape()[1], 3);
    EXPECT_TRUE(param.data().requiresGrad());
}

// ============================================================================
// Statistical Validation Tests
// ============================================================================

TEST_F(ParameterTest, KaimingInitializationVariance) {
    // TODO(human): Test that Kaiming initialization produces correct variance
    //
    // Theory: For weights ~ N(0, std²) where std = sqrt(2 / fan_in),
    //         the variance should be Var(weights) = 2 / fan_in
    //
    // Test approach:
    // 1. Create a parameter with known shape, e.g., {100, 50}
    //    fan_in = 100, so expected_variance = 2 / 100 = 0.02
    // 2. Access the data: auto acc = param.data().accessor<float, 2>();
    // 3. Compute empirical mean and variance over all elements
    // 4. Check that:
    //    - Mean is close to 0 (within ±0.01 or so)
    //    - Variance is close to expected_variance (within ±10% tolerance)
    //
    // Formulas:
    //   mean = sum(x_i) / n
    //   variance = sum((x_i - mean)²) / n
    //
    // Use EXPECT_NEAR(actual, expected, tolerance) for floating point comparison

    Parameter param = Parameter::kaiming({100, 50}, DType::FLOAT32);

    auto mean = param.data().mean().item();
    auto variance = param.data().var().item();

    EXPECT_NEAR(mean, 0.0, 0.01);
    EXPECT_NEAR(variance, 2.0 / 100, 0.1);
}

TEST_F(ParameterTest, XavierInitializationVariance) {
    // TODO(human): Test that Xavier initialization produces correct variance
    //
    // Theory: For weights ~ N(0, std²) where std = sqrt(1 / fan_in),
    //         the variance should be Var(weights) = 1 / fan_in
    //
    // Test approach (same as Kaiming):
    // 1. Create parameter with shape {100, 50}
    //    fan_in = 100, so expected_variance = 1 / 100 = 0.01
    // 2. Compute empirical mean and variance
    // 3. Validate both are close to theoretical values
    //
    // The key difference from Kaiming: expected variance is 1/fan_in instead of 2/fan_in

    Parameter param = Parameter::xavier({100, 50}, DType::FLOAT32);
    auto mean = param.data().mean().item();
    auto variance = param.data().var().item();

    EXPECT_NEAR(mean, 0.0, 0.01);
    EXPECT_NEAR(variance, 1.0 / 100, 0.1);
}

TEST_F(ParameterTest, KaimingThrowsOnEmptyShape) {
    EXPECT_THROW({ Parameter param = Parameter::kaiming({}, DType::FLOAT32); }, std::runtime_error);
}

TEST_F(ParameterTest, XavierThrowsOnEmptyShape) {
    EXPECT_THROW({ Parameter param = Parameter::xavier({}, DType::FLOAT32); }, std::runtime_error);
}

// ============================================================================
// Gradient Operation Tests
// ============================================================================

TEST_F(ParameterTest, ZeroGrad) {
    Tensor t = Tensor::ones({2, 2}, DType::FLOAT32, mCpuDevice);
    t.requiresGrad(true);
    Parameter param(t);

    // Simulate gradient computation
    Tensor y = param.data() * 2.0;
    y.backward(Tensor::ones(y.shape(), y.dtype(), y.device()));

    // Gradients should exist
    ASSERT_NE(param.grad(), nullptr);

    // Zero the gradients
    param.zeroGrad();

    // Check gradient tensor is now zero
    auto grad = param.grad();
    ASSERT_NE(grad, nullptr);
    auto acc = grad->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(acc[i][j], 0.0f);
        }
    }
}

// ============================================================================
// Device Operation Tests
// ============================================================================

TEST_F(ParameterTest, DeviceProperty) {
    Parameter param = Parameter::ones({2, 3}, DType::FLOAT32);
    EXPECT_TRUE(param.device().isCPU());
}

TEST_F(ParameterTest, ToDevice) {
    Parameter param = Parameter::ones({2, 3}, DType::FLOAT32);
    EXPECT_TRUE(param.device().isCPU());

    // Moving to same device should work
    param.to(mCpuDevice);
    EXPECT_TRUE(param.device().isCPU());
}

// ============================================================================
// Shape and Numel Tests
// ============================================================================

TEST_F(ParameterTest, ShapeAccess) {
    Parameter param = Parameter::zeros({3, 4, 5}, DType::FLOAT32);

    const auto& shape = param.shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 5);
}

TEST_F(ParameterTest, NumelComputation) {
    Parameter param = Parameter::zeros({3, 4, 5}, DType::FLOAT32);
    EXPECT_EQ(param.numel(), 3 * 4 * 5);
}
