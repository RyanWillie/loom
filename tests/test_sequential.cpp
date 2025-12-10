#include <memory>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/nn/linear.h"
#include "loom/nn/sequential.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;
using namespace loom::nn;

// ============================================================================
// Test Fixture
// ============================================================================

class SequentialTest : public ::testing::Test {
  protected:
    SequentialTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("SequentialTest");
        logger.info("Test fixture initialized");
    }

    Device mCpuDevice;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(SequentialTest, ConstructEmpty) {
    Sequential model;

    EXPECT_EQ(model.size(), 0);
    EXPECT_TRUE(model.empty());
}

TEST_F(SequentialTest, ConstructFromInitializerList) {
    auto layer1 = std::make_shared<Linear>(10, 5);
    auto layer2 = std::make_shared<Linear>(5, 2);

    Sequential model({layer1, layer2});

    EXPECT_EQ(model.size(), 2);
    EXPECT_FALSE(model.empty());
}

TEST_F(SequentialTest, ConstructWithSingleLayer) {
    auto layer = std::make_shared<Linear>(10, 5);

    Sequential model({layer});

    EXPECT_EQ(model.size(), 1);
    EXPECT_FALSE(model.empty());
}

// ============================================================================
// add() Method Tests
// ============================================================================

TEST_F(SequentialTest, AddSingleModule) {
    Sequential model;
    auto layer = std::make_shared<Linear>(10, 5);

    model.add(layer);

    EXPECT_EQ(model.size(), 1);
    EXPECT_FALSE(model.empty());
}

TEST_F(SequentialTest, AddMultipleModules) {
    Sequential model;

    model.add(std::make_shared<Linear>(10, 8));
    model.add(std::make_shared<Linear>(8, 5));
    model.add(std::make_shared<Linear>(5, 2));

    EXPECT_EQ(model.size(), 3);
}

TEST_F(SequentialTest, AddReturnsReferenceForChaining) {
    Sequential model;

    // Builder pattern: chain multiple add() calls
    model.add(std::make_shared<Linear>(10, 8))
        .add(std::make_shared<Linear>(8, 5))
        .add(std::make_shared<Linear>(5, 2));

    EXPECT_EQ(model.size(), 3);
}

// ============================================================================
// Module Registration Tests
// ============================================================================

TEST_F(SequentialTest, ModulesAreRegisteredWithNumericNames) {
    Sequential model;

    model.add(std::make_shared<Linear>(10, 8));
    model.add(std::make_shared<Linear>(8, 5));

    auto named_params = model.namedParameters();

    // Should have names like "0.weight", "0.bias", "1.weight", "1.bias"
    bool found_0_weight = false;
    bool found_1_weight = false;

    for (const auto& [name, param] : named_params) {
        if (name == "0.weight")
            found_0_weight = true;
        if (name == "1.weight")
            found_1_weight = true;
    }

    EXPECT_TRUE(found_0_weight);
    EXPECT_TRUE(found_1_weight);
}

TEST_F(SequentialTest, ParametersCollectedFromAllModules) {
    Sequential model;

    model.add(std::make_shared<Linear>(10, 8));  // 2 params: weight, bias
    model.add(std::make_shared<Linear>(8, 5));   // 2 params: weight, bias
    model.add(std::make_shared<Linear>(5, 2));   // 2 params: weight, bias

    auto params = model.parameters();

    EXPECT_EQ(params.size(), 6);  // 3 layers * 2 params each
}

TEST_F(SequentialTest, ParametersCollectedFromLayerWithoutBias) {
    Sequential model;

    model.add(std::make_shared<Linear>(10, 8, true));  // weight + bias
    model.add(std::make_shared<Linear>(8, 5, false));  // weight only

    auto params = model.parameters();

    EXPECT_EQ(params.size(), 3);  // 2 weights + 1 bias
}

// ============================================================================
// Indexing Tests
// ============================================================================

TEST_F(SequentialTest, IndexingReturnsCorrectModule) {
    auto layer1 = std::make_shared<Linear>(10, 8);
    auto layer2 = std::make_shared<Linear>(8, 5);

    Sequential model({layer1, layer2});

    EXPECT_EQ(model[0], layer1);
    EXPECT_EQ(model[1], layer2);
}

TEST_F(SequentialTest, IndexingOutOfRangeThrows) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 5));

    EXPECT_THROW(model[1], std::out_of_range);
    EXPECT_THROW(model[100], std::out_of_range);
}

TEST_F(SequentialTest, IndexingOnEmptyModelThrows) {
    Sequential model;

    EXPECT_THROW(model[0], std::out_of_range);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(SequentialTest, ForwardPassSingleLayer) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 5));

    Tensor input = Tensor::randn({2, 10}, DType::FLOAT32, mCpuDevice);
    Tensor output = model.forward(input);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST_F(SequentialTest, ForwardPassMultipleLayers) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 8));
    model.add(std::make_shared<Linear>(8, 5));
    model.add(std::make_shared<Linear>(5, 2));

    Tensor input = Tensor::randn({3, 10}, DType::FLOAT32, mCpuDevice);
    Tensor output = model.forward(input);

    EXPECT_EQ(output.shape()[0], 3);  // batch_size preserved
    EXPECT_EQ(output.shape()[1], 2);  // final output size
}

TEST_F(SequentialTest, ForwardPassMNISTArchitecture) {
    // Typical MNIST architecture: 784 -> 128 -> 64 -> 10
    Sequential model;
    model.add(std::make_shared<Linear>(784, 128));
    model.add(std::make_shared<Linear>(128, 64));
    model.add(std::make_shared<Linear>(64, 10));

    Tensor input = Tensor::randn({32, 784}, DType::FLOAT32, mCpuDevice);
    Tensor output = model.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 10);
}

TEST_F(SequentialTest, ForwardThrowsOnEmptySequence) {
    Sequential model;

    Tensor input = Tensor::randn({2, 10}, DType::FLOAT32, mCpuDevice);

    EXPECT_THROW(model.forward(input), std::runtime_error);
}

TEST_F(SequentialTest, ForwardPassNumericalCorrectness) {
    // Create two layers with known weights for manual verification
    auto layer1 = std::make_shared<Linear>(2, 2);
    auto layer2 = std::make_shared<Linear>(2, 1);

    // Set layer1 weights to identity, bias to zero
    auto w1 = layer1->weight()->data().accessor<float, 2>();
    w1[0][0] = 1.0f;
    w1[0][1] = 0.0f;
    w1[1][0] = 0.0f;
    w1[1][1] = 1.0f;
    auto b1 = layer1->bias()->data().accessor<float, 1>();
    b1[0] = 0.0f;
    b1[1] = 0.0f;

    // Set layer2 to sum inputs: weight=[1, 1], bias=0
    auto w2 = layer2->weight()->data().accessor<float, 2>();
    w2[0][0] = 1.0f;
    w2[0][1] = 1.0f;
    auto b2 = layer2->bias()->data().accessor<float, 1>();
    b2[0] = 0.0f;

    Sequential model({layer1, layer2});

    // Input: [2.0, 3.0]
    Tensor input = Tensor::zeros({1, 2}, DType::FLOAT32, mCpuDevice);
    auto input_acc = input.accessor<float, 2>();
    input_acc[0][0] = 2.0f;
    input_acc[0][1] = 3.0f;

    // layer1: [2, 3] (identity)
    // layer2: 2 + 3 = 5
    Tensor output = model.forward(input);

    auto output_acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(output_acc[0][0], 5.0f);
}

TEST_F(SequentialTest, CallOperatorWorks) {
    Sequential model({std::make_shared<Linear>(10, 5)});

    Tensor input = Tensor::randn({2, 10}, DType::FLOAT32, mCpuDevice);
    Tensor output = model(input);  // Using operator()

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
}

// ============================================================================
// Multiple Forward Pass Tests
// ============================================================================

TEST_F(SequentialTest, MultipleForwardPassesWork) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 5));

    Tensor input1 = Tensor::randn({2, 10}, DType::FLOAT32, mCpuDevice);
    Tensor input2 = Tensor::randn({3, 10}, DType::FLOAT32, mCpuDevice);

    Tensor output1 = model.forward(input1);
    Tensor output2 = model.forward(input2);

    EXPECT_EQ(output1.shape()[0], 2);
    EXPECT_EQ(output2.shape()[0], 3);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(SequentialTest, GradientFlowsThroughAllLayers) {
    Sequential model;
    model.add(std::make_shared<Linear>(5, 3));
    model.add(std::make_shared<Linear>(3, 2));

    Tensor input = Tensor::ones({1, 5}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = model.forward(input);
    Tensor loss = output.sum();

    loss.backward();

    // All parameters should have gradients
    auto params = model.parameters();
    for (auto& param : params) {
        EXPECT_NE(param->grad(), nullptr);
    }

    // Input should have gradients
    EXPECT_NE(input.grad(), nullptr);
}

TEST_F(SequentialTest, ZeroGradZerosAllLayerGradients) {
    Sequential model;
    model.add(std::make_shared<Linear>(5, 3));
    model.add(std::make_shared<Linear>(3, 2));

    // Create gradients
    Tensor input = Tensor::ones({1, 5}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);
    Tensor output = model.forward(input);
    output.sum().backward();

    // Zero gradients
    model.zeroGrad();

    // Check all gradients are zero
    auto params = model.parameters();
    for (auto& param : params) {
        auto grad = param->grad();
        ASSERT_NE(grad, nullptr);

        auto flat = grad->flatten();
        auto acc = flat.accessor<float, 1>();
        for (size_t i = 0; i < flat.numel(); ++i) {
            EXPECT_FLOAT_EQ(acc[i], 0.0f);
        }
    }
}

// ============================================================================
// Device Movement Tests
// ============================================================================

TEST_F(SequentialTest, ToDeviceMovesAllParameters) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 8));
    model.add(std::make_shared<Linear>(8, 5));

    // Initially on CPU
    auto params = model.parameters();
    for (auto& param : params) {
        EXPECT_TRUE(param->device().isCPU());
    }

    // Move to CPU (tests the mechanism)
    model.to(mCpuDevice);

    // All should still be on CPU
    for (auto& param : params) {
        EXPECT_TRUE(param->device().isCPU());
    }
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_F(SequentialTest, TrainingModePropagatesToAllLayers) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 8));
    model.add(std::make_shared<Linear>(8, 5));

    model.train(true);
    EXPECT_TRUE(model.training());

    model.eval();
    EXPECT_FALSE(model.training());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(SequentialTest, SingleLayerBehavesLikeStandalone) {
    auto layer = std::make_shared<Linear>(10, 5);
    Sequential model({layer});

    Tensor input = Tensor::randn({2, 10}, DType::FLOAT32, mCpuDevice);

    Tensor output_sequential = model.forward(input);
    Tensor output_direct = layer->forward(input);

    EXPECT_EQ(output_sequential.shape(), output_direct.shape());
}

TEST_F(SequentialTest, VeryDeepNetwork) {
    Sequential model;

    // Create a 10-layer network
    model.add(std::make_shared<Linear>(100, 80));
    for (int i = 0; i < 8; ++i) {
        model.add(std::make_shared<Linear>(80, 80));
    }
    model.add(std::make_shared<Linear>(80, 10));

    EXPECT_EQ(model.size(), 10);

    Tensor input = Tensor::randn({4, 100}, DType::FLOAT32, mCpuDevice);
    Tensor output = model.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 10);
}

TEST_F(SequentialTest, MixedBiasConfiguration) {
    Sequential model;
    model.add(std::make_shared<Linear>(10, 8, true));  // with bias
    model.add(std::make_shared<Linear>(8, 5, false));  // without bias
    model.add(std::make_shared<Linear>(5, 2, true));   // with bias

    Tensor input = Tensor::randn({2, 10}, DType::FLOAT32, mCpuDevice);
    Tensor output = model.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 2);
}

// ============================================================================
// Regression Tests
// ============================================================================

TEST_F(SequentialTest, GradientFlowThroughLogSoftmaxPattern) {
    // Regression test: gradient accumulation bug with chained layers + log-softmax
    // Previously failed due to incorrect handling of non-contiguous tensors
    // in gradient accumulation (engine.cpp) and in-place operations (tensor.cpp)
    Sequential model;
    model.add(std::make_shared<Linear>(4, 3));
    model.add(std::make_shared<Linear>(3, 2));

    Tensor input = Tensor::ones({2, 4}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = model.forward(input);

    // Log-softmax pattern: max subtraction, exp, sum, log, final subtraction
    Tensor max_vals = output.max(-1, true);
    Tensor shifted = output - max_vals;
    Tensor exp_shifted = shifted.exp();
    Tensor sum_exp = exp_shifted.sum(-1, true);
    Tensor log_sum_exp = sum_exp.log();
    Tensor log_probs = shifted - log_sum_exp;

    // Backward pass
    log_probs.sum().backward();

    // Verify gradients flow correctly through all parameters
    auto params = model.parameters();
    for (auto& param : params) {
        auto grad = param->grad();
        ASSERT_NE(grad, nullptr) << "Gradient should exist for all parameters";

        // Check that at least one gradient element is non-zero
        // (Note: sum may be ~0 for log-softmax, which is mathematically correct)
        auto grad_flat = grad->flatten();
        auto grad_acc = grad_flat.accessor<float, 1>();
        float max_abs_grad = 0.0f;
        for (size_t i = 0; i < grad_flat.numel(); ++i) {
            max_abs_grad = std::max(max_abs_grad, std::abs(grad_acc[i]));
        }
        EXPECT_GT(max_abs_grad, 0.001f) << "Should have non-zero gradients";
    }
}

// ============================================================================
// Integration Tests for Common Patterns
// ============================================================================

TEST_F(SequentialTest, ResidualConnectionPattern) {
    // Common pattern: residual = x + f(x)
    // Tests gradient accumulation with multiple paths to same tensor

    auto layer = std::make_shared<Linear>(4, 4);
    Sequential model({layer});

    Tensor x = Tensor::randn({2, 4}, DType::FLOAT32, mCpuDevice);
    x.requiresGrad(true);

    // Residual connection: output = x + layer(x)
    Tensor fx = model.forward(x);
    Tensor residual = x + fx;  // x has two gradient paths!

    residual.sum().backward();

    // Verify both x and layer weights have gradients
    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(layer->weight()->grad(), nullptr);

    // x should receive gradients from both the direct path and through the layer
    float x_grad_sum = std::abs(x.grad()->sum().item());
    EXPECT_GT(x_grad_sum, 1e-6f) << "x should have non-zero gradients";

    // Layer weights should also have gradients
    float weight_grad_sum = std::abs(layer->weight()->grad()->sum().item());
    EXPECT_GT(weight_grad_sum, 1e-6f) << "Layer weights should have non-zero gradients";
}

TEST_F(SequentialTest, KeepDimReductionPattern) {
    // Test operations with keepdim=true (common in normalization, attention)
    // These create non-contiguous tensors that must be handled correctly

    Sequential model;
    model.add(std::make_shared<Linear>(4, 4));

    Tensor input = Tensor::randn({2, 4}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = model.forward(input);

    // Use max with keepdim (like in log-softmax)
    Tensor max_vals = output.max(-1, true);  // keepdim=true
    Tensor shifted = output - max_vals;      // Broadcast subtraction
    Tensor result = shifted * 2.0;

    // Backward pass
    result.sum().backward();

    // Verify gradients flow correctly
    auto params = model.parameters();
    for (auto& param : params) {
        ASSERT_NE(param->grad(), nullptr) << "Parameter should have gradient";

        // Check for non-zero gradients (use max instead of sum to avoid cancellation)
        auto grad_flat = param->grad()->flatten();
        auto grad_acc = grad_flat.accessor<float, 1>();
        float max_abs_grad = 0.0f;
        for (size_t i = 0; i < grad_flat.numel(); ++i) {
            max_abs_grad = std::max(max_abs_grad, std::abs(grad_acc[i]));
        }
        EXPECT_GT(max_abs_grad, 1e-6f) << "Parameter should have non-zero gradients";
    }
}

TEST_F(SequentialTest, MultipleSequentialForwardPasses) {
    // Test that a model can be used multiple times with different inputs
    // Important for batch processing and evaluation

    Sequential model;
    model.add(std::make_shared<Linear>(3, 2));

    // First forward pass
    Tensor input1 = Tensor::ones({2, 3}, DType::FLOAT32, mCpuDevice);
    input1.requiresGrad(true);
    Tensor output1 = model.forward(input1);
    output1.sum().backward();

    // Weights should have gradients
    auto params = model.parameters();
    ASSERT_FALSE(params.empty());
    ASSERT_NE(params[0]->grad(), nullptr);

    // Zero gradients for next iteration
    model.zeroGrad();

    // Second forward pass with different input
    Tensor input2 = Tensor::ones({3, 3}, DType::FLOAT32, mCpuDevice) * 2.0;
    input2.requiresGrad(true);
    Tensor output2 = model.forward(input2);
    output2.sum().backward();

    // Both passes should work correctly
    EXPECT_EQ(output1.shape()[0], 2);
    EXPECT_EQ(output2.shape()[0], 3);
}

TEST_F(SequentialTest, NestedOperationsWithTranspose) {
    // Test complex pattern with transpose operations in the middle
    // Mimics certain attention or transformation patterns

    auto layer1 = std::make_shared<Linear>(4, 3);
    auto layer2 = std::make_shared<Linear>(3, 2);

    Tensor input = Tensor::randn({2, 4}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    // Complex forward pass with transpose
    Tensor h1 = layer1->forward(input);        // [2, 3]
    Tensor h1_t = h1.transpose();              // [3, 2] - non-contiguous
    Tensor h1_back = h1_t.transpose();         // [2, 3] - back to original shape
    Tensor output = layer2->forward(h1_back);  // [2, 2]

    // Backward pass
    output.sum().backward();

    // Verify gradients exist for both layers
    ASSERT_NE(layer1->weight()->grad(), nullptr);
    ASSERT_NE(layer2->weight()->grad(), nullptr);

    // Check non-zero gradients
    float grad1_sum = std::abs(layer1->weight()->grad()->sum().item());
    float grad2_sum = std::abs(layer2->weight()->grad()->sum().item());

    EXPECT_GT(grad1_sum, 1e-6f);
    EXPECT_GT(grad2_sum, 1e-6f);
}

TEST_F(SequentialTest, BroadcastingWithSequential) {
    // Test broadcasting patterns common in neural networks
    // (e.g., adding biases, normalization constants)

    Sequential model;
    model.add(std::make_shared<Linear>(3, 2));

    Tensor input = Tensor::ones({4, 3}, DType::FLOAT32, mCpuDevice);
    input.requiresGrad(true);

    Tensor output = model.forward(input);  // [4, 2]

    // Add broadcast constant (simulates bias or normalization)
    Tensor constant = Tensor::ones({2}, DType::FLOAT32, mCpuDevice);
    constant.requiresGrad(true);

    Tensor result = output + constant;  // Broadcasting: [4, 2] + [2] -> [4, 2]

    result.sum().backward();

    // Both output and constant should have gradients
    ASSERT_NE(output.grad(), nullptr);
    ASSERT_NE(constant.grad(), nullptr);

    // Constant gradient should be accumulated from all rows
    auto const_grad_acc = constant.grad()->accessor<float, 1>();
    for (size_t j = 0; j < 2; ++j) {
        EXPECT_FLOAT_EQ(const_grad_acc[j], 4.0f);  // Gradient from 4 rows
    }
}
