#include <gtest/gtest.h>

#include <cmath>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/tensor/tensor.h"

using namespace loom;

// ============================================================================
// Test Fixture
// ============================================================================

class MathOpsTest : public ::testing::Test {
  protected:
    MathOpsTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("MathOpsTest");
        logger.info("Test fixture initialized");
    }

    Device mCpuDevice;
};

// ============================================================================
// Exp Tests
// ============================================================================

TEST_F(MathOpsTest, ExpBasicCorrectness) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 0.0f;
    x_acc[1] = 1.0f;
    x_acc[2] = 2.0f;

    Tensor y = x.exp();

    auto y_acc = y.accessor<float, 1>();
    EXPECT_FLOAT_EQ(y_acc[0], std::exp(0.0f));  // 1.0
    EXPECT_FLOAT_EQ(y_acc[1], std::exp(1.0f));  // 2.718...
    EXPECT_FLOAT_EQ(y_acc[2], std::exp(2.0f));  // 7.389...
}

TEST_F(MathOpsTest, ExpNegativeValues) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = -1.0f;
    x_acc[1] = -2.0f;
    x_acc[2] = -10.0f;

    Tensor y = x.exp();

    auto y_acc = y.accessor<float, 1>();
    EXPECT_NEAR(y_acc[0], std::exp(-1.0f), 1e-6f);   // 0.368
    EXPECT_NEAR(y_acc[1], std::exp(-2.0f), 1e-6f);   // 0.135
    EXPECT_NEAR(y_acc[2], std::exp(-10.0f), 1e-6f);  // very small
}

TEST_F(MathOpsTest, ExpPreservesShape) {
    Tensor x = Tensor::randn({2, 3, 4}, DType::FLOAT32, mCpuDevice);
    Tensor y = x.exp();

    EXPECT_EQ(y.shape().size(), 3);
    EXPECT_EQ(y.shape()[0], 2);
    EXPECT_EQ(y.shape()[1], 3);
    EXPECT_EQ(y.shape()[2], 4);
}

TEST_F(MathOpsTest, ExpZero) {
    Tensor x = Tensor::zeros({5}, DType::FLOAT32, mCpuDevice);
    Tensor y = x.exp();

    auto y_acc = y.accessor<float, 1>();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(y_acc[i], 1.0f);  // exp(0) = 1
    }
}

TEST_F(MathOpsTest, ExpLargePositiveValues) {
    // Test that large values don't cause issues
    Tensor x = Tensor::zeros({2}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 10.0f;
    x_acc[1] = 20.0f;

    Tensor y = x.exp();

    auto y_acc = y.accessor<float, 1>();
    EXPECT_NEAR(y_acc[0], std::exp(10.0f), 1.0f);
    EXPECT_NEAR(y_acc[1], std::exp(20.0f), 1000.0f);
}

TEST_F(MathOpsTest, ExpGradientFlow) {
    Tensor x = Tensor::ones({3}, DType::FLOAT32, mCpuDevice);
    x.requiresGrad(true);

    Tensor y = x.exp();
    Tensor loss = y.sum();

    loss.backward();

    // Gradient of exp(x) is exp(x)
    auto grad = x.grad();
    ASSERT_NE(grad, nullptr);

    auto y_acc = y.accessor<float, 1>();
    auto grad_acc = grad->accessor<float, 1>();

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad_acc[i], y_acc[i], 1e-5f);
    }
}

TEST_F(MathOpsTest, ExpGradientNumericalCheck) {
    Tensor x = Tensor::zeros({1}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 2.0f;
    x.requiresGrad(true);

    Tensor y = x.exp();
    y.backward();

    auto grad = x.grad();
    ASSERT_NE(grad, nullptr);

    auto grad_acc = grad->accessor<float, 1>();

    // Analytical gradient: d/dx[exp(x)] = exp(x) = exp(2) ≈ 7.389
    float expected_grad = std::exp(2.0f);
    EXPECT_NEAR(grad_acc[0], expected_grad, 1e-5f);
}

// ============================================================================
// Log Tests
// ============================================================================

TEST_F(MathOpsTest, LogBasicCorrectness) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 1.0f;
    x_acc[1] = 2.718281828f;  // e
    x_acc[2] = 7.389056099f;  // e^2

    Tensor y = x.log();

    auto y_acc = y.accessor<float, 1>();
    EXPECT_NEAR(y_acc[0], 0.0f, 1e-5f);     // log(1) = 0
    EXPECT_NEAR(y_acc[1], 1.0f, 1e-5f);     // log(e) = 1
    EXPECT_NEAR(y_acc[2], 2.0f, 1e-5f);     // log(e^2) = 2
}

TEST_F(MathOpsTest, LogSmallValues) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 0.1f;
    x_acc[1] = 0.01f;
    x_acc[2] = 0.001f;

    Tensor y = x.log();

    auto y_acc = y.accessor<float, 1>();
    EXPECT_NEAR(y_acc[0], std::log(0.1f), 1e-5f);
    EXPECT_NEAR(y_acc[1], std::log(0.01f), 1e-5f);
    EXPECT_NEAR(y_acc[2], std::log(0.001f), 1e-5f);
}

TEST_F(MathOpsTest, LogPreservesShape) {
    Tensor x = Tensor::ones({2, 3, 4}, DType::FLOAT32, mCpuDevice);
    Tensor y = x.log();

    EXPECT_EQ(y.shape().size(), 3);
    EXPECT_EQ(y.shape()[0], 2);
    EXPECT_EQ(y.shape()[1], 3);
    EXPECT_EQ(y.shape()[2], 4);
}

TEST_F(MathOpsTest, LogOne) {
    Tensor x = Tensor::ones({5}, DType::FLOAT32, mCpuDevice);
    Tensor y = x.log();

    auto y_acc = y.accessor<float, 1>();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(y_acc[i], 0.0f, 1e-6f);  // log(1) = 0
    }
}

TEST_F(MathOpsTest, LogLargeValues) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 10.0f;
    x_acc[1] = 100.0f;
    x_acc[2] = 1000.0f;

    Tensor y = x.log();

    auto y_acc = y.accessor<float, 1>();
    EXPECT_NEAR(y_acc[0], std::log(10.0f), 1e-5f);
    EXPECT_NEAR(y_acc[1], std::log(100.0f), 1e-5f);
    EXPECT_NEAR(y_acc[2], std::log(1000.0f), 1e-5f);
}

TEST_F(MathOpsTest, LogGradientFlow) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 1.0f;
    x_acc[1] = 2.0f;
    x_acc[2] = 4.0f;
    x.requiresGrad(true);

    Tensor y = x.log();
    Tensor loss = y.sum();

    loss.backward();

    // Gradient of log(x) is 1/x
    auto grad = x.grad();
    ASSERT_NE(grad, nullptr);

    auto grad_acc = grad->accessor<float, 1>();
    EXPECT_NEAR(grad_acc[0], 1.0f / 1.0f, 1e-5f);  // 1.0
    EXPECT_NEAR(grad_acc[1], 1.0f / 2.0f, 1e-5f);  // 0.5
    EXPECT_NEAR(grad_acc[2], 1.0f / 4.0f, 1e-5f);  // 0.25
}

TEST_F(MathOpsTest, LogGradientNumericalCheck) {
    Tensor x = Tensor::zeros({1}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 5.0f;
    x.requiresGrad(true);

    Tensor y = x.log();
    y.backward();

    auto grad = x.grad();
    ASSERT_NE(grad, nullptr);

    auto grad_acc = grad->accessor<float, 1>();

    // Analytical gradient: d/dx[log(x)] = 1/x = 1/5 = 0.2
    EXPECT_NEAR(grad_acc[0], 0.2f, 1e-5f);
}

// ============================================================================
// Exp-Log Composition Tests
// ============================================================================

TEST_F(MathOpsTest, ExpLogInverse) {
    // log(exp(x)) should equal x
    Tensor x = Tensor::randn({5}, DType::FLOAT32, mCpuDevice);
    Tensor y = x.exp().log();

    auto x_acc = x.accessor<float, 1>();
    auto y_acc = y.accessor<float, 1>();

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(y_acc[i], x_acc[i], 1e-5f);
    }
}

TEST_F(MathOpsTest, LogExpInverse) {
    // exp(log(x)) should equal x (for x > 0)
    Tensor x = Tensor::zeros({5}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    for (size_t i = 0; i < 5; ++i) {
        x_acc[i] = 0.5f + static_cast<float>(i);  // Positive values
    }

    Tensor y = x.log().exp();

    auto y_acc = y.accessor<float, 1>();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(y_acc[i], x_acc[i], 1e-5f);
    }
}

TEST_F(MathOpsTest, ExpLogChainGradient) {
    // Test gradient through exp(log(x)) chain
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 1.0f;
    x_acc[1] = 2.0f;
    x_acc[2] = 3.0f;
    x.requiresGrad(true);

    Tensor y = x.log().exp();  // Should be identity
    Tensor loss = y.sum();

    loss.backward();

    // Gradient should be all ones (identity function)
    auto grad = x.grad();
    ASSERT_NE(grad, nullptr);

    auto grad_acc = grad->accessor<float, 1>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad_acc[i], 1.0f, 1e-4f);
    }
}

// ============================================================================
// Combined Operation Tests
// ============================================================================

TEST_F(MathOpsTest, LogSumExp) {
    // Test log(sum(exp(x))) pattern used in softmax
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 1.0f;
    x_acc[1] = 2.0f;
    x_acc[2] = 3.0f;

    Tensor exp_x = x.exp();
    Tensor sum_exp = exp_x.sum();
    Tensor log_sum_exp = sum_exp.log();

    // Manual calculation: exp(1) + exp(2) + exp(3) ≈ 2.718 + 7.389 + 20.086 = 30.193
    // log(30.193) ≈ 3.408
    float expected = std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f));
    EXPECT_NEAR(log_sum_exp.item(), expected, 1e-4f);
}

TEST_F(MathOpsTest, ExpWithArithmeticOps) {
    Tensor x = Tensor::ones({3}, DType::FLOAT32, mCpuDevice);
    Tensor y = (x * 2.0f).exp();  // exp(2x)

    auto y_acc = y.accessor<float, 1>();
    float expected = std::exp(2.0f);  // exp(2*1) = exp(2)

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(y_acc[i], expected, 1e-4f);
    }
}

TEST_F(MathOpsTest, LogWithArithmeticOps) {
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 1>();
    x_acc[0] = 2.0f;
    x_acc[1] = 4.0f;
    x_acc[2] = 8.0f;

    Tensor y = (x * 2.0f).log();  // log(2x)

    auto y_acc = y.accessor<float, 1>();
    EXPECT_NEAR(y_acc[0], std::log(4.0f), 1e-5f);
    EXPECT_NEAR(y_acc[1], std::log(8.0f), 1e-5f);
    EXPECT_NEAR(y_acc[2], std::log(16.0f), 1e-5f);
}
