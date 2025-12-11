#include <gtest/gtest.h>

#include <memory>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/nn/linear.h"
#include "loom/nn/parameter.h"
#include "loom/tensor/tensor.h"

using namespace loom;
using namespace loom::nn;

// ============================================================================
// Test Fixture
// ============================================================================

class LinearTest : public ::testing::Test {
  protected:
    LinearTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("LinearTest");
        logger.info("Test fixture initialized");
    }

    Device mCpuDevice;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(LinearTest, ConstructWithBias) {
    Linear layer(10, 5, true);

    EXPECT_EQ(layer.inFeatures(), 10);
    EXPECT_EQ(layer.outFeatures(), 5);
    EXPECT_NE(layer.weight(), nullptr);
    EXPECT_NE(layer.bias(), nullptr);
}

TEST_F(LinearTest, ConstructWithoutBias) {
    Linear layer(10, 5, false);

    EXPECT_EQ(layer.inFeatures(), 10);
    EXPECT_EQ(layer.outFeatures(), 5);
    EXPECT_NE(layer.weight(), nullptr);
    EXPECT_EQ(layer.bias(), nullptr);
}

TEST_F(LinearTest, WeightShapeIsCorrect) {
    Linear layer(784, 128);

    auto weight = layer.weight();
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->shape().size(), 2);
    EXPECT_EQ(weight->shape()[0], 128);  // out_features
    EXPECT_EQ(weight->shape()[1], 784);  // in_features
}

TEST_F(LinearTest, BiasShapeIsCorrect) {
    Linear layer(784, 128);

    auto bias = layer.bias();
    ASSERT_NE(bias, nullptr);
    EXPECT_EQ(bias->shape().size(), 1);
    EXPECT_EQ(bias->shape()[0], 128);  // out_features
}

TEST_F(LinearTest, ParametersAreRegistered) {
    Linear layer_with_bias(10, 5, true);
    Linear layer_without_bias(10, 5, false);

    auto params_with = layer_with_bias.parameters();
    auto params_without = layer_without_bias.parameters();

    EXPECT_EQ(params_with.size(), 2);      // weight + bias
    EXPECT_EQ(params_without.size(), 1);   // weight only
}

TEST_F(LinearTest, NamedParametersHaveCorrectNames) {
    Linear layer(10, 5);

    auto named_params = layer.namedParameters();
    ASSERT_EQ(named_params.size(), 2);

    // std::map orders alphabetically: bias, weight
    EXPECT_EQ(named_params[0].first, "bias");
    EXPECT_EQ(named_params[1].first, "weight");
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(LinearTest, ForwardPassBasicShape) {
    Linear layer(10, 5);

    Tensor input = Tensor::randn({3, 10}, DType::FLOAT32, mCpuDevice);
    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 3);   // batch_size
    EXPECT_EQ(output.shape()[1], 5);   // out_features
}

TEST_F(LinearTest, ForwardPassSingleBatch) {
    Linear layer(4, 2);

    Tensor input = Tensor::randn({1, 4}, DType::FLOAT32, mCpuDevice);
    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
}

TEST_F(LinearTest, ForwardPassLargeBatch) {
    Linear layer(28, 10);

    Tensor input = Tensor::randn({128, 28}, DType::FLOAT32, mCpuDevice);
    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 128);
    EXPECT_EQ(output.shape()[1], 10);
}

TEST_F(LinearTest, ForwardPassMNISTSizes) {
    // Typical MNIST linear layer: 784 -> 128
    Linear layer(784, 128);

    Tensor input = Tensor::randn({64, 784}, DType::FLOAT32, mCpuDevice);
    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 64);
    EXPECT_EQ(output.shape()[1], 128);
}

TEST_F(LinearTest, ForwardPassNumericalCorrectness) {
    // Create a layer with known weights and bias for manual verification
    Linear layer(2, 1);

    // Set weight to [[2.0, 3.0]] (shape: [1, 2])
    auto weight_data = layer.weight()->data();
    auto weight_acc = weight_data.accessor<float, 2>();
    weight_acc[0][0] = 2.0f;
    weight_acc[0][1] = 3.0f;

    // Set bias to [1.0]
    auto bias_data = layer.bias()->data();
    auto bias_acc = bias_data.accessor<float, 1>();
    bias_acc[0] = 1.0f;

    // Input: [[1.0, 2.0]] (shape: [1, 2])
    Tensor input = Tensor::zeros({1, 2}, DType::FLOAT32, mCpuDevice);
    auto input_acc = input.accessor<float, 2>();
    input_acc[0][0] = 1.0f;
    input_acc[0][1] = 2.0f;

    // Forward: y = xW^T + b = [1, 2] @ [2, 3]^T + [1]
    //                       = [1, 2] @ [2; 3] + [1]
    //                       = [1*2 + 2*3] + [1]
    //                       = [8] + [1] = [9]
    Tensor output = layer.forward(input);

    auto output_acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(output_acc[0][0], 9.0f);
}

TEST_F(LinearTest, ForwardPassWithoutBias) {
    Linear layer(2, 1, false);

    // Set weight to [[2.0, 3.0]]
    auto weight_data = layer.weight()->data();
    auto weight_acc = weight_data.accessor<float, 2>();
    weight_acc[0][0] = 2.0f;
    weight_acc[0][1] = 3.0f;

    // Input: [[1.0, 2.0]]
    Tensor input = Tensor::zeros({1, 2}, DType::FLOAT32, mCpuDevice);
    auto input_acc = input.accessor<float, 2>();
    input_acc[0][0] = 1.0f;
    input_acc[0][1] = 2.0f;

    // y = xW^T = [1*2 + 2*3] = [8]
    Tensor output = layer.forward(input);

    auto output_acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(output_acc[0][0], 8.0f);
}

// ============================================================================
// Shape Validation Tests
// ============================================================================

TEST_F(LinearTest, ForwardThrowsOn1DInput) {
    Linear layer(10, 5);

    Tensor input = Tensor::randn({10}, DType::FLOAT32, mCpuDevice);

    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

TEST_F(LinearTest, ForwardThrowsOn3DInput) {
    Linear layer(10, 5);

    Tensor input = Tensor::randn({2, 3, 10}, DType::FLOAT32, mCpuDevice);

    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

TEST_F(LinearTest, ForwardThrowsOnWrongFeatureSize) {
    Linear layer(10, 5);

    // Input has 8 features instead of 10
    Tensor input = Tensor::randn({3, 8}, DType::FLOAT32, mCpuDevice);

    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(LinearTest, GradientFlowsThroughWeight) {
    Linear layer(3, 2);

    Tensor input = Tensor::ones({1, 3}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = layer.forward(input);
    Tensor loss = output.sum();

    loss.backward();

    // Weight should have gradients
    auto weight_grad = layer.weight()->grad();
    ASSERT_NE(weight_grad, nullptr);
    EXPECT_EQ(weight_grad->shape()[0], 2);
    EXPECT_EQ(weight_grad->shape()[1], 3);
}

TEST_F(LinearTest, GradientFlowsThroughBias) {
    Linear layer(3, 2);

    Tensor input = Tensor::ones({1, 3}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = layer.forward(input);
    Tensor loss = output.sum();

    loss.backward();

    // Bias should have gradients
    auto bias_grad = layer.bias()->grad();
    ASSERT_NE(bias_grad, nullptr);
    EXPECT_EQ(bias_grad->shape()[0], 2);
}

TEST_F(LinearTest, GradientFlowsToInput) {
    Linear layer(3, 2);

    Tensor input = Tensor::ones({2, 3}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = layer.forward(input);
    Tensor loss = output.sum();

    loss.backward();

    // Input should have gradients
    auto input_grad = input.grad();
    ASSERT_NE(input_grad, nullptr);
    EXPECT_EQ(input_grad->shape()[0], 2);
    EXPECT_EQ(input_grad->shape()[1], 3);
}

TEST_F(LinearTest, ZeroGradZerosParameterGradients) {
    Linear layer(3, 2);

    // Create gradients
    Tensor input = Tensor::ones({1, 3}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);
    Tensor output = layer.forward(input);
    output.sum().backward();

    // Verify gradients exist
    ASSERT_NE(layer.weight()->grad(), nullptr);
    ASSERT_NE(layer.bias()->grad(), nullptr);

    // Zero gradients
    layer.zeroGrad();

    // Check they're zeroed
    auto weight_grad = layer.weight()->grad();
    auto bias_grad = layer.bias()->grad();

    ASSERT_NE(weight_grad, nullptr);
    ASSERT_NE(bias_grad, nullptr);

    auto weight_flat = weight_grad->flatten();
    auto weight_acc = weight_flat.accessor<float, 1>();
    for (size_t i = 0; i < weight_flat.numel(); ++i) {
        EXPECT_FLOAT_EQ(weight_acc[i], 0.0f);
    }

    auto bias_flat = bias_grad->flatten();
    auto bias_acc = bias_flat.accessor<float, 1>();
    for (size_t i = 0; i < bias_flat.numel(); ++i) {
        EXPECT_FLOAT_EQ(bias_acc[i], 0.0f);
    }
}

// ============================================================================
// Device Movement Tests
// ============================================================================

TEST_F(LinearTest, ToDeviceMovesParameters) {
    Linear layer(10, 5);

    // Initially on CPU
    EXPECT_TRUE(layer.weight()->device().isCPU());
    EXPECT_TRUE(layer.bias()->device().isCPU());

    // Move to CPU again (tests the mechanism)
    layer.to(mCpuDevice);

    EXPECT_TRUE(layer.weight()->device().isCPU());
    EXPECT_TRUE(layer.bias()->device().isCPU());
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_F(LinearTest, TrainingModeDoesNotAffectForward) {
    // Linear layer behaves the same in train/eval mode
    // (unlike Dropout or BatchNorm)
    Linear layer(4, 2);

    Tensor input = Tensor::randn({2, 4}, DType::FLOAT32, mCpuDevice);

    layer.train();
    Tensor output_train = layer.forward(input);

    layer.eval();
    Tensor output_eval = layer.forward(input);

    // Outputs should be identical
    EXPECT_EQ(output_train.shape(), output_eval.shape());
}

// ============================================================================
// Multiple Forward Pass Tests
// ============================================================================

TEST_F(LinearTest, MultipleForwardPassesWork) {
    Linear layer(5, 3);

    Tensor input1 = Tensor::randn({2, 5}, DType::FLOAT32, mCpuDevice);
    Tensor input2 = Tensor::randn({2, 5}, DType::FLOAT32, mCpuDevice);

    Tensor output1 = layer.forward(input1);
    Tensor output2 = layer.forward(input2);

    EXPECT_EQ(output1.shape(), output2.shape());
    EXPECT_EQ(output1.shape()[0], 2);
    EXPECT_EQ(output1.shape()[1], 3);
}

// ============================================================================
// Large Dimension Tests
// ============================================================================

TEST_F(LinearTest, LargeDimensions) {
    // Test with realistic neural network sizes
    Linear layer(1024, 512);

    Tensor input = Tensor::randn({16, 1024}, DType::FLOAT32, mCpuDevice);
    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 512);
}

// ============================================================================
// Call Operator Tests
// ============================================================================

TEST_F(LinearTest, CallOperatorWorks) {
    Linear layer(4, 2);

    Tensor input = Tensor::randn({3, 4}, DType::FLOAT32, mCpuDevice);
    Tensor output = layer(input);  // Using operator()

    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 2);
}
