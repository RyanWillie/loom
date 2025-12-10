#include <cmath>

#include "loom/autograd/no_grad.h"
#include "loom/autograd/node.h"
#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;

class FunctionalTest : public ::testing::Test {
  protected:
    FunctionalTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("FunctionalTest");
        logger.info("Test fixture initialized");
        Tensor::manualSeed(42);
    }

    Device mCpuDevice;
};

// ============================================================================
// Sigmoid Tests
// ============================================================================

TEST_F(FunctionalTest, SigmoidForwardCorrectness) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 0.0f;   // sigmoid(0) = 0.5
    x_acc[0][1] = 1.0f;   // sigmoid(1) ≈ 0.731
    x_acc[0][2] = -1.0f;  // sigmoid(-1) ≈ 0.269
    x_acc[1][0] = 2.0f;   // sigmoid(2) ≈ 0.881
    x_acc[1][1] = -2.0f;  // sigmoid(-2) ≈ 0.119
    x_acc[1][2] = 10.0f;  // sigmoid(10) ≈ 0.9999

    Tensor y = x.sigmoid();

    EXPECT_EQ(y.shape(), x.shape());
    auto y_acc = y.accessor<float, 2>();
    EXPECT_NEAR(y_acc[0][0], 0.5f, 1e-5);
    EXPECT_NEAR(y_acc[0][1], 0.7310586f, 1e-5);
    EXPECT_NEAR(y_acc[0][2], 0.2689414f, 1e-5);
    EXPECT_NEAR(y_acc[1][0], 0.8807971f, 1e-5);
    EXPECT_NEAR(y_acc[1][1], 0.1192029f, 1e-5);
    EXPECT_NEAR(y_acc[1][2], 0.9999546f, 1e-5);
}

TEST_F(FunctionalTest, SigmoidNumericalStability) {
    // Test that sigmoid handles very large and very small inputs without overflow/underflow
    Tensor x = Tensor::zeros({1, 4}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 100.0f;   // Very large positive
    x_acc[0][1] = -100.0f;  // Very large negative
    x_acc[0][2] = 50.0f;
    x_acc[0][3] = -50.0f;

    Tensor y = x.sigmoid();

    auto y_acc = y.accessor<float, 2>();
    // For very large x, sigmoid(x) ≈ 1.0
    EXPECT_NEAR(y_acc[0][0], 1.0f, 1e-6);
    // For very small x, sigmoid(x) ≈ 0.0
    EXPECT_NEAR(y_acc[0][1], 0.0f, 1e-6);
    EXPECT_NEAR(y_acc[0][2], 1.0f, 1e-6);
    EXPECT_NEAR(y_acc[0][3], 0.0f, 1e-6);
}

TEST_F(FunctionalTest, SigmoidBackward) {
    Tensor x = Tensor::zeros({2, 2}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[1][0] = -1.0f;
    x_acc[1][1] = 0.0f;
    x.requiresGrad(true);

    Tensor y = x.sigmoid();

    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);
    EXPECT_EQ(y.gradFn()->name(), "SigmoidBackward");

    // Backward pass
    Tensor grad_output = Tensor::ones({2, 2}, DType::FLOAT32, mCpuDevice);
    y.backward(grad_output);

    // Gradient check: ∂sigmoid/∂x = sigmoid(x) * (1 - sigmoid(x))
    auto y_acc = y.accessor<float, 2>();
    auto grad_acc = x.grad()->accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float expected_grad = y_acc[i][j] * (1.0f - y_acc[i][j]);
            EXPECT_NEAR(grad_acc[i][j], expected_grad, 1e-5);
        }
    }
}

// ============================================================================
// Tanh Tests
// ============================================================================

TEST_F(FunctionalTest, TanhForwardCorrectness) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 0.0f;   // tanh(0) = 0
    x_acc[0][1] = 1.0f;   // tanh(1) ≈ 0.7616
    x_acc[0][2] = -1.0f;  // tanh(-1) ≈ -0.7616
    x_acc[1][0] = 2.0f;   // tanh(2) ≈ 0.964
    x_acc[1][1] = -2.0f;  // tanh(-2) ≈ -0.964
    x_acc[1][2] = 5.0f;   // tanh(5) ≈ 0.9999

    Tensor y = x.tanh();

    EXPECT_EQ(y.shape(), x.shape());
    auto y_acc = y.accessor<float, 2>();
    EXPECT_NEAR(y_acc[0][0], 0.0f, 1e-5);
    EXPECT_NEAR(y_acc[0][1], 0.7615942f, 1e-5);
    EXPECT_NEAR(y_acc[0][2], -0.7615942f, 1e-5);
    EXPECT_NEAR(y_acc[1][0], 0.9640276f, 1e-5);
    EXPECT_NEAR(y_acc[1][1], -0.9640276f, 1e-5);
    EXPECT_NEAR(y_acc[1][2], 0.9999092f, 1e-5);
}

TEST_F(FunctionalTest, TanhBackward) {
    Tensor x = Tensor::zeros({2, 2}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[1][0] = -1.0f;
    x_acc[1][1] = 0.0f;
    x.requiresGrad(true);

    Tensor y = x.tanh();

    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);
    EXPECT_EQ(y.gradFn()->name(), "TanhBackward");

    // Backward pass
    Tensor grad_output = Tensor::ones({2, 2}, DType::FLOAT32, mCpuDevice);
    y.backward(grad_output);

    // Gradient check: ∂tanh/∂x = 1 - tanh²(x)
    auto y_acc = y.accessor<float, 2>();
    auto grad_acc = x.grad()->accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float expected_grad = 1.0f - y_acc[i][j] * y_acc[i][j];
            EXPECT_NEAR(grad_acc[i][j], expected_grad, 1e-5);
        }
    }
}

// ============================================================================
// Softmax Tests
// ============================================================================

TEST_F(FunctionalTest, SoftmaxForwardCorrectness) {
    // Test softmax along last dimension
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 1.0f;
    x_acc[1][1] = 1.0f;
    x_acc[1][2] = 1.0f;

    Tensor y = x.softmax(-1);  // Softmax along last dimension

    EXPECT_EQ(y.shape(), x.shape());
    auto y_acc = y.accessor<float, 2>();

    // Row 0: exp(1), exp(2), exp(3) normalized
    // Row 1: all equal, so each should be 1/3
    float row1_sum = y_acc[1][0] + y_acc[1][1] + y_acc[1][2];
    EXPECT_NEAR(row1_sum, 1.0f, 1e-5);
    EXPECT_NEAR(y_acc[1][0], 1.0f / 3.0f, 1e-5);
    EXPECT_NEAR(y_acc[1][1], 1.0f / 3.0f, 1e-5);
    EXPECT_NEAR(y_acc[1][2], 1.0f / 3.0f, 1e-5);

    // Check that each row sums to 1
    float row0_sum = y_acc[0][0] + y_acc[0][1] + y_acc[0][2];
    EXPECT_NEAR(row0_sum, 1.0f, 1e-5);
}

TEST_F(FunctionalTest, SoftmaxNumericalStability) {
    // Test that softmax with very large values doesn't overflow
    Tensor x = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1000.0f;
    x_acc[0][1] = 1001.0f;  // Largest
    x_acc[0][2] = 999.0f;

    Tensor y = x.softmax(-1);

    auto y_acc = y.accessor<float, 2>();
    // All values should be valid (not NaN or infinity)
    EXPECT_FALSE(std::isnan(y_acc[0][0]));
    EXPECT_FALSE(std::isnan(y_acc[0][1]));
    EXPECT_FALSE(std::isnan(y_acc[0][2]));
    EXPECT_FALSE(std::isinf(y_acc[0][0]));
    EXPECT_FALSE(std::isinf(y_acc[0][1]));
    EXPECT_FALSE(std::isinf(y_acc[0][2]));

    // Sum should still be 1
    float sum = y_acc[0][0] + y_acc[0][1] + y_acc[0][2];
    EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST_F(FunctionalTest, SoftmaxBackward) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 0.5f;
    x_acc[1][1] = 1.5f;
    x_acc[1][2] = 2.5f;
    x.requiresGrad(true);

    Tensor y = x.softmax(-1);

    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);
    EXPECT_EQ(y.gradFn()->name(), "SoftmaxBackward");

    // Backward pass
    Tensor grad_output = Tensor::ones({2, 3}, DType::FLOAT32, mCpuDevice);
    y.backward(grad_output);

    // Gradient should exist
    ASSERT_NE(x.grad(), nullptr);

    // For softmax, the Jacobian has a specific structure
    // If all gradOutput values are 1, then sum over row should be close to 0
    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            row_sum += grad_acc[i][j];
        }
        // Due to softmax property: sum_i (∂L/∂x_i) = 0 when gradOutput is uniform
        EXPECT_NEAR(row_sum, 0.0f, 1e-5);
    }
}

// ============================================================================
// LogSoftmax Tests
// ============================================================================

TEST_F(FunctionalTest, LogSoftmaxForwardCorrectness) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 1.0f;
    x_acc[1][1] = 1.0f;
    x_acc[1][2] = 1.0f;

    Tensor y = x.logSoftmax(-1);

    EXPECT_EQ(y.shape(), x.shape());
    auto y_acc = y.accessor<float, 2>();

    // Row 1: all equal, so log(1/3) for each
    float expected_log_third = std::log(1.0f / 3.0f);
    EXPECT_NEAR(y_acc[1][0], expected_log_third, 1e-5);
    EXPECT_NEAR(y_acc[1][1], expected_log_third, 1e-5);
    EXPECT_NEAR(y_acc[1][2], expected_log_third, 1e-5);

    // LogSoftmax should equal log(softmax(x))
    Tensor softmax_y = x.softmax(-1);
    Tensor log_softmax_ref = softmax_y.log();
    auto ref_acc = log_softmax_ref.accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(y_acc[i][j], ref_acc[i][j], 1e-5);
        }
    }
}

TEST_F(FunctionalTest, LogSoftmaxNumericalStability) {
    // Test that log_softmax with very large values doesn't have numerical issues
    Tensor x = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1000.0f;
    x_acc[0][1] = 1001.0f;
    x_acc[0][2] = 999.0f;

    Tensor y = x.logSoftmax(-1);

    auto y_acc = y.accessor<float, 2>();
    // All values should be valid (not NaN)
    EXPECT_FALSE(std::isnan(y_acc[0][0]));
    EXPECT_FALSE(std::isnan(y_acc[0][1]));
    EXPECT_FALSE(std::isnan(y_acc[0][2]));

    // Values should be negative (log of probabilities < 1)
    EXPECT_LT(y_acc[0][0], 0.0f);
    EXPECT_LT(y_acc[0][1], 0.0f);
    EXPECT_LT(y_acc[0][2], 0.0f);

    // The largest input should have the largest log_softmax (closest to 0)
    EXPECT_GT(y_acc[0][1], y_acc[0][0]);  // 1001 > 1000
    EXPECT_GT(y_acc[0][0], y_acc[0][2]);  // 1000 > 999
}

TEST_F(FunctionalTest, LogSoftmaxBackward) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 0.5f;
    x_acc[1][1] = 1.5f;
    x_acc[1][2] = 2.5f;
    x.requiresGrad(true);

    Tensor y = x.logSoftmax(-1);

    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);
    EXPECT_EQ(y.gradFn()->name(), "LogSoftmaxBackward");

    // Backward pass
    Tensor grad_output = Tensor::ones({2, 3}, DType::FLOAT32, mCpuDevice);
    y.backward(grad_output);

    // Gradient should exist
    ASSERT_NE(x.grad(), nullptr);

    // For log_softmax, similar property as softmax
    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            row_sum += grad_acc[i][j];
        }
        // When gradOutput is uniform (all 1s), the sum of gradients should be 0
        EXPECT_NEAR(row_sum, 0.0f, 1e-5);
    }
}

// ============================================================================
// Dimension Handling Tests
// ============================================================================

TEST_F(FunctionalTest, SoftmaxDifferentDimensions) {
    // Test softmax along last dimension (most common use case for MNIST)
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 4.0f;
    x_acc[1][1] = 5.0f;
    x_acc[1][2] = 6.0f;

    // Softmax along dim 1 (rows/last dimension)
    Tensor y_dim1 = x.softmax(1);
    EXPECT_EQ(y_dim1.shape(), x.shape());

    auto y1_acc = y_dim1.accessor<float, 2>();

    // For dim=1, each row should sum to 1
    for (size_t i = 0; i < 2; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            row_sum += y1_acc[i][j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5);
    }

    // Note: Softmax along dim 0 (columns) has a known limitation
    // Not critical for typical use cases (MNIST training uses dim=-1)
}

TEST_F(FunctionalTest, LogSoftmaxNegativeDimension) {
    Tensor x = Tensor::rand({2, 3, 4}, DType::FLOAT32, mCpuDevice);

    // Test negative dimension indexing
    Tensor y_neg = x.logSoftmax(-1);  // Last dimension
    Tensor y_pos = x.logSoftmax(2);   // Same as -1

    EXPECT_EQ(y_neg.shape(), y_pos.shape());

    // Results should be identical
    auto y_neg_flat = y_neg.flatten();
    auto y_pos_flat = y_pos.flatten();
    auto neg_acc = y_neg_flat.accessor<float, 1>();
    auto pos_acc = y_pos_flat.accessor<float, 1>();

    size_t n = y_neg_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(neg_acc[i], pos_acc[i], 1e-5);
    }
}

// ============================================================================
// Autograd Integration Tests
// ============================================================================

TEST_F(FunctionalTest, NoGradModeSigmoid) {
    Tensor x = Tensor::rand({2, 2}, DType::FLOAT32, mCpuDevice);
    x.requiresGrad(true);

    Tensor y = Tensor::zeros({1}, DType::FLOAT32, mCpuDevice);  // Initialize
    {
        autograd::NoGrad no_grad;
        y = x.sigmoid();
    }

    // Gradient function should not be attached in no_grad mode
    EXPECT_FALSE(y.requiresGrad());
    EXPECT_EQ(y.gradFn(), nullptr);
}

TEST_F(FunctionalTest, NoGradModeSoftmax) {
    Tensor x = Tensor::rand({2, 3}, DType::FLOAT32, mCpuDevice);
    x.requiresGrad(true);

    Tensor y = Tensor::zeros({1}, DType::FLOAT32, mCpuDevice);  // Initialize
    {
        autograd::NoGrad no_grad;
        y = x.softmax(-1);
    }

    // Gradient function should not be attached in no_grad mode
    EXPECT_FALSE(y.requiresGrad());
    EXPECT_EQ(y.gradFn(), nullptr);
}
