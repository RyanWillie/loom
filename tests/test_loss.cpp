#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/nn/loss.h"
#include "loom/tensor/tensor.h"

using namespace loom;
using namespace loom::nn;

// ============================================================================
// Test Fixture
// ============================================================================

class LossTest : public ::testing::Test {
  protected:
    LossTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("LossTest");
        logger.info("Test fixture initialized");
    }

    Device mCpuDevice;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(LossTest, ConstructWithDefaultReduction) {
    CrossEntropyLoss criterion;

    EXPECT_EQ(criterion.reduction(), Reduction::MEAN);
}

TEST_F(LossTest, ConstructWithMeanReduction) {
    CrossEntropyLoss criterion(Reduction::MEAN);

    EXPECT_EQ(criterion.reduction(), Reduction::MEAN);
}

TEST_F(LossTest, ConstructWithSumReduction) {
    CrossEntropyLoss criterion(Reduction::SUM);

    EXPECT_EQ(criterion.reduction(), Reduction::SUM);
}

TEST_F(LossTest, ConstructWithNoneReduction) {
    CrossEntropyLoss criterion(Reduction::NONE);

    EXPECT_EQ(criterion.reduction(), Reduction::NONE);
}

// ============================================================================
// Shape Validation Tests
// ============================================================================

TEST_F(LossTest, ValidShapes) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({4, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4}, DType::INT64, mCpuDevice);

    // Should not throw
    EXPECT_NO_THROW(criterion.forward(predictions, targets));
}

TEST_F(LossTest, ThrowsOn1DPredictions) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4}, DType::INT64, mCpuDevice);

    EXPECT_THROW(criterion.forward(predictions, targets), std::runtime_error);
}

TEST_F(LossTest, ThrowsOn3DPredictions) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({4, 10, 2}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4}, DType::INT64, mCpuDevice);

    EXPECT_THROW(criterion.forward(predictions, targets), std::runtime_error);
}

TEST_F(LossTest, ThrowsOn2DTargets) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({4, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4, 1}, DType::INT64, mCpuDevice);

    EXPECT_THROW(criterion.forward(predictions, targets), std::runtime_error);
}

TEST_F(LossTest, ThrowsOnBatchSizeMismatch) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({4, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({5}, DType::INT64, mCpuDevice);

    EXPECT_THROW(criterion.forward(predictions, targets), std::runtime_error);
}

TEST_F(LossTest, ThrowsOnTargetOutOfRange) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({2, 3}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({2}, DType::INT64, mCpuDevice);

    // Set second target to invalid class 5 (valid range is 0-2)
    auto targets_acc = targets.accessor<int64_t, 1>();
    targets_acc[0] = 0;
    targets_acc[1] = 5;  // Out of range!

    EXPECT_THROW(criterion.forward(predictions, targets), std::runtime_error);
}

TEST_F(LossTest, ThrowsOnNegativeTarget) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({2, 3}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({2}, DType::INT64, mCpuDevice);

    auto targets_acc = targets.accessor<int64_t, 1>();
    targets_acc[0] = -1;  // Negative!
    targets_acc[1] = 0;

    EXPECT_THROW(criterion.forward(predictions, targets), std::runtime_error);
}

// ============================================================================
// Numerical Correctness Tests
// ============================================================================

TEST_F(LossTest, SimpleNumericalCorrectness) {
    CrossEntropyLoss criterion(Reduction::NONE);

    // Create predictions with known values
    // logits = [[2.0, 1.0, 0.1]] (batch=1, classes=3)
    Tensor predictions = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = 2.0f;
    pred_acc[0][1] = 1.0f;
    pred_acc[0][2] = 0.1f;

    // Target class 0
    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 0;

    Tensor loss = criterion.forward(predictions, targets);

    // Manual calculation:
    // max = 2.0
    // shifted = [0.0, -1.0, -1.9]
    // exp_shifted = [1.0, 0.368, 0.150]
    // sum_exp = 1.518
    // log_sum_exp = 0.417
    // log_softmax = [2.0, 1.0, 0.1] - 2.0 - 0.417 = [-0.417, -1.417, -2.317]
    // nll for class 0 = -(-0.417) = 0.417

    auto loss_acc = loss.accessor<float, 1>();
    EXPECT_NEAR(loss_acc[0], 0.417f, 0.01f);
}

TEST_F(LossTest, NumericalCorrectnessTargetClass1) {
    CrossEntropyLoss criterion(Reduction::NONE);

    Tensor predictions = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = 1.0f;
    pred_acc[0][1] = 2.0f;
    pred_acc[0][2] = 0.5f;

    // Target class 1 (highest logit)
    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 1;

    Tensor loss = criterion.forward(predictions, targets);

    // When target has highest logit, loss should be small
    auto loss_acc = loss.accessor<float, 1>();
    EXPECT_LT(loss_acc[0], 1.0f);  // Should be relatively small
}

TEST_F(LossTest, NumericalCorrectnessWrongPrediction) {
    CrossEntropyLoss criterion(Reduction::NONE);

    Tensor predictions = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = 0.1f;
    pred_acc[0][1] = 0.2f;
    pred_acc[0][2] = 2.0f;  // Highest logit at class 2

    // Target class 0 (low logit) - model is wrong!
    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 0;

    Tensor loss = criterion.forward(predictions, targets);

    // When model predicts wrong class, loss should be larger
    auto loss_acc = loss.accessor<float, 1>();
    EXPECT_GT(loss_acc[0], 1.0f);  // Should be relatively large
}

TEST_F(LossTest, PerfectPredictionHasSmallLoss) {
    CrossEntropyLoss criterion(Reduction::NONE);

    // Logits strongly favor class 0
    Tensor predictions = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = 10.0f;  // Very confident
    pred_acc[0][1] = 0.0f;
    pred_acc[0][2] = 0.0f;

    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 0;  // Correct class

    Tensor loss = criterion.forward(predictions, targets);

    auto loss_acc = loss.accessor<float, 1>();
    EXPECT_LT(loss_acc[0], 0.1f);  // Very confident correct prediction = low loss
}

// ============================================================================
// Reduction Mode Tests
// ============================================================================

TEST_F(LossTest, ReductionNoneReturnsBatchSize) {
    CrossEntropyLoss criterion(Reduction::NONE);

    Tensor predictions = Tensor::randn({5, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({5}, DType::INT64, mCpuDevice);

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_EQ(loss.shape().size(), 1);
    EXPECT_EQ(loss.shape()[0], 5);  // One loss per sample
}

TEST_F(LossTest, ReductionMeanReturnsScalar) {
    CrossEntropyLoss criterion(Reduction::MEAN);

    Tensor predictions = Tensor::randn({5, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({5}, DType::INT64, mCpuDevice);

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_EQ(loss.numel(), 1);  // Scalar
}

TEST_F(LossTest, ReductionSumReturnsScalar) {
    CrossEntropyLoss criterion(Reduction::SUM);

    Tensor predictions = Tensor::randn({5, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({5}, DType::INT64, mCpuDevice);

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_EQ(loss.numel(), 1);  // Scalar
}

TEST_F(LossTest, ReductionMeanVsSumRelationship) {
    Tensor predictions = Tensor::randn({4, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4}, DType::INT64, mCpuDevice);

    CrossEntropyLoss criterion_mean(Reduction::MEAN);
    CrossEntropyLoss criterion_sum(Reduction::SUM);

    Tensor loss_mean = criterion_mean.forward(predictions, targets);
    Tensor loss_sum = criterion_sum.forward(predictions, targets);

    // sum = mean * batch_size
    float expected_sum = loss_mean.item() * 4.0f;
    EXPECT_NEAR(loss_sum.item(), expected_sum, 0.001f);
}

TEST_F(LossTest, ReductionMeanVsNoneRelationship) {
    Tensor predictions = Tensor::randn({4, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4}, DType::INT64, mCpuDevice);

    CrossEntropyLoss criterion_mean(Reduction::MEAN);
    CrossEntropyLoss criterion_none(Reduction::NONE);

    Tensor loss_mean = criterion_mean.forward(predictions, targets);
    Tensor loss_none = criterion_none.forward(predictions, targets);

    // mean should equal average of individual losses
    float manual_mean = 0.0f;
    auto loss_none_acc = loss_none.accessor<float, 1>();
    for (size_t i = 0; i < 4; ++i) {
        manual_mean += loss_none_acc[i];
    }
    manual_mean /= 4.0f;

    EXPECT_NEAR(loss_mean.item(), manual_mean, 0.001f);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(LossTest, HandlesLargeLogits) {
    CrossEntropyLoss criterion;

    // Very large logits that would overflow without log-sum-exp trick
    Tensor predictions = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = 1000.0f;
    pred_acc[0][1] = 999.0f;
    pred_acc[0][2] = 998.0f;
    pred_acc[1][0] = 500.0f;
    pred_acc[1][1] = 501.0f;
    pred_acc[1][2] = 499.0f;

    Tensor targets = Tensor::zeros({2}, DType::INT64, mCpuDevice);

    // Should not overflow or produce NaN/Inf
    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_FALSE(std::isnan(loss.item()));
    EXPECT_FALSE(std::isinf(loss.item()));
    EXPECT_GT(loss.item(), 0.0f);  // Loss should be positive
}

TEST_F(LossTest, HandlesLargeNegativeLogits) {
    CrossEntropyLoss criterion;

    // Very negative logits
    Tensor predictions = Tensor::zeros({2, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = -1000.0f;
    pred_acc[0][1] = -999.0f;
    pred_acc[0][2] = -998.0f;
    pred_acc[1][0] = -500.0f;
    pred_acc[1][1] = -501.0f;
    pred_acc[1][2] = -499.0f;

    Tensor targets = Tensor::zeros({2}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 2;  // Highest (least negative) class
    target_acc[1] = 0;

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_FALSE(std::isnan(loss.item()));
    EXPECT_FALSE(std::isinf(loss.item()));
}

TEST_F(LossTest, HandlesUniformLogits) {
    CrossEntropyLoss criterion(Reduction::NONE);

    // All logits equal - uniform probability
    Tensor predictions = Tensor::full({1, 4}, 1.0f, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);

    Tensor loss = criterion.forward(predictions, targets);

    // Loss should be -log(1/4) = log(4) ≈ 1.386
    auto loss_acc = loss.accessor<float, 1>();
    EXPECT_NEAR(loss_acc[0], std::log(4.0f), 0.01f);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(LossTest, GradientFlowsToPredictions) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({3, 5}, DType::FLOAT32, mCpuDevice);
    predictions.requiresGrad(true);

    Tensor targets = Tensor::zeros({3}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 0;
    target_acc[1] = 1;
    target_acc[2] = 2;

    Tensor loss = criterion.forward(predictions, targets);
    loss.backward();

    // Predictions should have gradients
    auto grad = predictions.grad();
    ASSERT_NE(grad, nullptr);
    EXPECT_EQ(grad->shape()[0], 3);
    EXPECT_EQ(grad->shape()[1], 5);
}

TEST_F(LossTest, GradientHasCorrectSign) {
    CrossEntropyLoss criterion(Reduction::MEAN);

    // Set values BEFORE calling requiresGrad to avoid in-place modification issues
    Tensor predictions = Tensor::zeros({1, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();
    pred_acc[0][0] = 2.0f;
    pred_acc[0][1] = 1.0f;
    pred_acc[0][2] = 0.5f;

    // Now mark for gradient tracking
    predictions.requiresGrad(true);

    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 0;  // Target is highest logit

    Tensor loss = criterion.forward(predictions, targets);
    loss.backward();

    auto grad = predictions.grad();
    auto grad_acc = grad->accessor<float, 2>();

    // Gradient for correct class should be negative (softmax - 1)
    // Gradient for incorrect classes should be positive (softmax)
    EXPECT_LT(grad_acc[0][0], 0.0f);  // Correct class
    EXPECT_GT(grad_acc[0][1], 0.0f);  // Incorrect class
    EXPECT_GT(grad_acc[0][2], 0.0f);  // Incorrect class
}

// ============================================================================
// MNIST-Scale Tests
// ============================================================================

TEST_F(LossTest, MNISTScaleBatch) {
    CrossEntropyLoss criterion;

    // Typical MNIST batch: 64 samples, 10 classes
    Tensor predictions = Tensor::randn({64, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({64}, DType::INT64, mCpuDevice);

    // Set targets to random valid classes
    auto target_acc = targets.accessor<int64_t, 1>();
    for (size_t i = 0; i < 64; ++i) {
        target_acc[i] = i % 10;  // Classes 0-9
    }

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_EQ(loss.numel(), 1);  // Scalar loss
    EXPECT_GT(loss.item(), 0.0f);
    EXPECT_FALSE(std::isnan(loss.item()));
}

TEST_F(LossTest, SingleClassWorks) {
    CrossEntropyLoss criterion;

    // Edge case: only one class (degenerate classification)
    Tensor predictions = Tensor::randn({4, 1}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({4}, DType::INT64, mCpuDevice);

    // All targets must be class 0
    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_EQ(loss.numel(), 1);
    // Loss should be very small since softmax(single value) = 1.0
    EXPECT_LT(loss.item(), 0.01f);
}

TEST_F(LossTest, LargeNumberOfClasses) {
    CrossEntropyLoss criterion;

    // Many classes (e.g., ImageNet-scale)
    Tensor predictions = Tensor::randn({8, 1000}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({8}, DType::INT64, mCpuDevice);

    auto target_acc = targets.accessor<int64_t, 1>();
    for (size_t i = 0; i < 8; ++i) {
        target_acc[i] = i * 100;  // Valid classes
    }

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_FALSE(std::isnan(loss.item()));
    EXPECT_FALSE(std::isinf(loss.item()));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(LossTest, SingleSampleBatch) {
    CrossEntropyLoss criterion;

    Tensor predictions = Tensor::randn({1, 10}, DType::FLOAT32, mCpuDevice);
    Tensor targets = Tensor::zeros({1}, DType::INT64, mCpuDevice);

    Tensor loss = criterion.forward(predictions, targets);

    EXPECT_EQ(loss.numel(), 1);
}

TEST_F(LossTest, AllSamplesCorrectClass) {
    CrossEntropyLoss criterion(Reduction::NONE);

    Tensor predictions = Tensor::zeros({3, 3}, DType::FLOAT32, mCpuDevice);
    auto pred_acc = predictions.accessor<float, 2>();

    // Make each sample strongly predict its target
    pred_acc[0][0] = 10.0f;  // Sample 0 -> class 0
    pred_acc[1][1] = 10.0f;  // Sample 1 -> class 1
    pred_acc[2][2] = 10.0f;  // Sample 2 -> class 2

    Tensor targets = Tensor::zeros({3}, DType::INT64, mCpuDevice);
    auto target_acc = targets.accessor<int64_t, 1>();
    target_acc[0] = 0;
    target_acc[1] = 1;
    target_acc[2] = 2;

    Tensor loss = criterion.forward(predictions, targets);

    // All losses should be small
    auto loss_acc = loss.accessor<float, 1>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_LT(loss_acc[i], 0.1f);
    }
}
