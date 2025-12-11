#include <cmath>

#include "loom/autograd/node.h"
#include "loom/nn/linear.h"
#include "loom/nn/loss.h"
#include "loom/nn/parameter.h"
#include "loom/optim/sgd.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;
using namespace loom::optim;

class OptimizerTest : public ::testing::Test {
  protected:
    Device cpu_device{DeviceType::CPU};

    void SetUp() override {
        // Setup runs before each test
    }

    void TearDown() override {
        // Cleanup runs after each test
    }

    // Helper function to create test parameters
    std::vector<std::shared_ptr<nn::Parameter>> createTestParameters(size_t count = 3) {
        std::vector<std::shared_ptr<nn::Parameter>> params;
        for (size_t i = 0; i < count; ++i) {
            Tensor data = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
            data.requiresGrad(true);
            params.push_back(std::make_shared<nn::Parameter>(data));
        }
        return params;
    }

    // Helper to check if all values in a tensor are zero
    bool isAllZeros(const std::shared_ptr<Tensor>& tensor) {
        if (!tensor)
            return false;
        auto accessor = tensor->accessor<float, 2>();
        for (size_t i = 0; i < tensor->shape()[0]; ++i) {
            for (size_t j = 0; j < tensor->shape()[1]; ++j) {
                if (std::abs(accessor[i][j]) > 1e-6f) {
                    return false;
                }
            }
        }
        return true;
    }

    // Helper to compare tensor values
    bool tensorEqual(const Tensor& a, const Tensor& b, float tolerance = 1e-6f) {
        if (a.shape() != b.shape())
            return false;
        auto acc_a = a.accessor<float, 2>();
        auto acc_b = b.accessor<float, 2>();
        for (size_t i = 0; i < a.shape()[0]; ++i) {
            for (size_t j = 0; j < a.shape()[1]; ++j) {
                if (std::abs(acc_a[i][j] - acc_b[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
};

// Test 1: SGD Construction
TEST_F(OptimizerTest, SGDConstruction) {
    auto params = createTestParameters(3);
    SGD optimizer(params, 0.01);

    EXPECT_EQ(optimizer.learningRate(), 0.01);
    EXPECT_EQ(optimizer.numParameters(), 3);
}

// Test 2: Zero Gradients
TEST_F(OptimizerTest, ZeroGradClearsAllGradients) {
    auto params = createTestParameters();

    // Set gradients on all parameters
    for (auto& param : params) {
        param->data().backward(Tensor::ones(param->data().shape(), DType::FLOAT32, cpu_device));
        ASSERT_NE(param->grad(), nullptr);  // Verify gradients exist
    }

    // Zero gradients
    SGD optimizer(params, 0.01);
    optimizer.zeroGrad();

    // Verify all gradients are zero
    for (auto& param : params) {
        ASSERT_NE(param->grad(), nullptr);       // Gradient tensor should still exist
        EXPECT_TRUE(isAllZeros(param->grad()));  // But values should be zero
    }
}

// Test 3: Parameter Update
TEST_F(OptimizerTest, SGDUpdatesParameters) {
    // Create a parameter with all ones
    Tensor data = Tensor::ones({3}, DType::FLOAT32, cpu_device);
    data.requiresGrad(true);
    auto param = std::make_shared<nn::Parameter>(data);

    // Set gradient to all 2.0
    Tensor grad = Tensor::full({3}, 2.0f, DType::FLOAT32, cpu_device);
    param->data().backward(grad);

    // Create optimizer with lr=0.1 and perform step
    SGD optimizer({param}, 0.1);
    optimizer.step();

    // Verify update: new = old - lr * grad = 1.0 - 0.1 * 2.0 = 0.8
    auto updated_acc = param->data().accessor<float, 1>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(updated_acc[i], 0.8f);
    }
}

// Test 4: NoGrad Context Verification
TEST_F(OptimizerTest, StepDoesNotCreateComputationGraph) {
    Tensor data = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    data.requiresGrad(true);
    auto param = std::make_shared<nn::Parameter>(data);

    // Set gradient
    param->data().backward(Tensor::ones({2, 2}, DType::FLOAT32, cpu_device));

    // Verify param is a leaf before step
    EXPECT_TRUE(param->data().isLeaf());

    // Perform step
    SGD optimizer({param}, 0.01);
    optimizer.step();

    // Verify param is STILL a leaf (no graph created during update)
    EXPECT_TRUE(param->data().isLeaf());
    EXPECT_EQ(param->data().gradFn(), nullptr);
}

// Test 5: Multiple Parameters
TEST_F(OptimizerTest, SGDUpdatesMultipleParameters) {
    // Create parameters with different shapes
    Tensor data1 = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    data1.requiresGrad(true);
    auto p1 = std::make_shared<nn::Parameter>(data1);

    Tensor data2 = Tensor::ones({5}, DType::FLOAT32, cpu_device);
    data2.requiresGrad(true);
    auto p2 = std::make_shared<nn::Parameter>(data2);

    // Set different gradients
    p1->data().backward(Tensor::full({2, 3}, 1.0f, DType::FLOAT32, cpu_device));
    p2->data().backward(Tensor::full({5}, 2.0f, DType::FLOAT32, cpu_device));

    // Update both parameters with lr=0.1
    SGD optimizer({p1, p2}, 0.1);
    optimizer.step();

    // Verify p1: 1.0 - 0.1 * 1.0 = 0.9
    auto acc1 = p1->data().accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(acc1[i][j], 0.9f);
        }
    }

    // Verify p2: 1.0 - 0.1 * 2.0 = 0.8
    auto acc2 = p2->data().accessor<float, 1>();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(acc2[i], 0.8f);
    }
}

// Test 6: Skips Parameters Without Gradients
TEST_F(OptimizerTest, SkipsParametersWithoutGradients) {
    Tensor data1 = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    data1.requiresGrad(true);
    auto p1 = std::make_shared<nn::Parameter>(data1);

    Tensor data2 = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    data2.requiresGrad(true);
    auto p2 = std::make_shared<nn::Parameter>(data2);

    // Only p1 has gradient
    p1->data().backward(Tensor::ones({2, 3}, DType::FLOAT32, cpu_device));
    // p2 has no gradient

    // Store original p2 values
    auto original_p2 = p2->data().clone();

    SGD optimizer({p1, p2}, 0.1);
    optimizer.step();  // Should not crash

    // p1 should have changed: 1.0 - 0.1 * 1.0 = 0.9
    auto acc1 = p1->data().accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc1[0][0], 0.9f);

    // p2 should be unchanged (still 1.0)
    EXPECT_TRUE(tensorEqual(p2->data(), original_p2));
}

// Test 7: Learning Rate Modification
TEST_F(OptimizerTest, LearningRateCanBeModified) {
    auto params = createTestParameters(1);
    SGD optimizer(params, 0.01);

    // Verify initial learning rate
    EXPECT_EQ(optimizer.learningRate(), 0.01);

    // Change learning rate
    optimizer.setLearningRate(0.001);
    EXPECT_EQ(optimizer.learningRate(), 0.001);

    // Verify new LR is used in updates
    auto param = params[0];
    param->data().backward(Tensor::ones(param->data().shape(), DType::FLOAT32, cpu_device));
    optimizer.step();

    // Expected: 1.0 - 0.001 * 1.0 = 0.999
    auto acc = param->data().accessor<float, 2>();
    EXPECT_NEAR(acc[0][0], 0.999f, 1e-6f);
}

// Test 8: Integration Test with Linear Layer
TEST_F(OptimizerTest, IntegrationWithLinearLayer) {
    // Create a simple linear layer
    auto linear = std::make_shared<nn::Linear>(5, 3);
    auto params = linear->parameters();

    // Create optimizer
    SGD optimizer(params, 0.01);

    // Simulate training iteration
    Tensor input = Tensor::rand({2, 5}, DType::FLOAT32, cpu_device);
    input.requiresGrad(true);

    // Forward pass
    Tensor output = linear->forward(input);

    // Simple loss (just sum for testing)
    Tensor loss = output.sum();

    // Store weight before update
    auto weight_before = linear->weight()->data().clone();

    // Backward pass
    optimizer.zeroGrad();
    loss.backward();

    // Verify gradients exist
    for (auto& p : params) {
        ASSERT_NE(p->grad(), nullptr);
    }

    // Parameter update
    optimizer.step();
    auto weight_after = linear->weight()->data();

    // Verify weights changed (they shouldn't be exactly equal)
    EXPECT_FALSE(tensorEqual(weight_before, weight_after));
}

TEST_F(OptimizerTest, TwoLayerChain) {
    // Create a 2-layer network: 4 -> 3 -> 2 (NO ReLU, NO BIAS to simplify)
    auto layer1 = std::make_shared<nn::Linear>(4, 3, false);  // No bias
    auto layer2 = std::make_shared<nn::Linear>(3, 2, false);  // No bias

    // Create simple input and targets
    Tensor input = Tensor::ones({2, 4}, DType::FLOAT32, cpu_device);  // batch=2
    input.requiresGrad(true);

    Tensor targets = Tensor::zeros({2}, DType::INT64, cpu_device);
    auto targets_acc = targets.accessor<int64_t, 1>();
    targets_acc[0] = 0;
    targets_acc[1] = 1;

    // Forward pass WITHOUT ReLU
    Tensor hidden = layer1->forward(input);
    Tensor output = layer2->forward(hidden);

    auto criterion = std::make_shared<nn::CrossEntropyLoss>();
    Tensor loss = criterion->forward(output, targets);

    // Backward pass
    loss.backward();

    // Check gradients exist and are non-zero
    ASSERT_NE(layer1->weight()->grad(), nullptr);
    ASSERT_NE(layer2->weight()->grad(), nullptr);

    float grad1_sum = std::abs(layer1->weight()->grad()->sum().item());

    // For layer 2, check individual gradients (sum may be near 0 due to log-softmax in
    // CrossEntropy)
    auto grad2_acc = layer2->weight()->grad()->accessor<float, 2>();
    float grad2_max_abs = 0.0f;
    for (size_t i = 0; i < layer2->weight()->data().shape()[0]; ++i) {
        for (size_t j = 0; j < layer2->weight()->data().shape()[1]; ++j) {
            grad2_max_abs = std::max(grad2_max_abs, std::abs(grad2_acc[i][j]));
        }
    }

    EXPECT_GT(grad1_sum, 0.01f) << "Layer 1 should have non-zero gradients";
    EXPECT_GT(grad2_max_abs, 0.01f) << "Layer 2 should have non-zero individual gradients";
}

// Test 10: End-to-End MNIST-like Training with ReLU
TEST_F(OptimizerTest, EndToEndMNISTLikeTraining) {
    // Create a 2-layer MLP: 10 -> 5 -> 3 (small version of MNIST-like network)
    auto layer1 = std::make_shared<nn::Linear>(10, 5);
    auto layer2 = std::make_shared<nn::Linear>(5, 3);

    // Collect all parameters
    std::vector<std::shared_ptr<nn::Parameter>> all_params;
    auto params1 = layer1->parameters();
    auto params2 = layer2->parameters();
    all_params.insert(all_params.end(), params1.begin(), params1.end());
    all_params.insert(all_params.end(), params2.begin(), params2.end());

    // Create optimizer and loss
    SGD optimizer(all_params, 0.1);  // Aggressive LR for fast testing
    auto criterion = std::make_shared<nn::CrossEntropyLoss>();

    // Create synthetic data: batch_size=4, input_dim=10, num_classes=3
    Tensor input = Tensor::rand({4, 10}, DType::FLOAT32, cpu_device);
    input.requiresGrad(true);

    // Create targets (class indices for CrossEntropyLoss)
    Tensor targets = Tensor::zeros({4}, DType::INT64, cpu_device);
    auto targets_acc = targets.accessor<int64_t, 1>();
    targets_acc[0] = 0;  // Sample 0 -> class 0
    targets_acc[1] = 1;  // Sample 1 -> class 1
    targets_acc[2] = 2;  // Sample 2 -> class 2
    targets_acc[3] = 0;  // Sample 3 -> class 0

    // Perform multiple training steps
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    const int num_iterations = 20;

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Forward pass with ReLU
        optimizer.zeroGrad();
        Tensor hidden = layer1->forward(input);
        Tensor activated = hidden.relu();  // ReLU activation
        Tensor output = layer2->forward(activated);
        Tensor loss = criterion->forward(output, targets);

        if (iter == 0) {
            initial_loss = loss.item();
        }
        if (iter == num_iterations - 1) {
            final_loss = loss.item();
        }

        // Backward pass
        loss.backward();

        // Verify gradients exist
        ASSERT_NE(layer1->weight()->grad(), nullptr);
        ASSERT_NE(layer2->weight()->grad(), nullptr);

        // Verify gradients on first iteration
        if (iter == 0) {
            // Check individual gradients for layer 2 (sum may be near 0 due to log-softmax in
            // CrossEntropy)
            auto grad2_acc = layer2->weight()->grad()->accessor<float, 2>();
            float grad2_max_abs = 0.0f;
            for (size_t i = 0; i < layer2->weight()->data().shape()[0]; ++i) {
                for (size_t j = 0; j < layer2->weight()->data().shape()[1]; ++j) {
                    grad2_max_abs = std::max(grad2_max_abs, std::abs(grad2_acc[i][j]));
                }
            }

            float grad1_sum = std::abs(layer1->weight()->grad()->sum().item());

            EXPECT_GT(grad1_sum, 0.001f) << "Layer 1 should have non-zero gradients";
            EXPECT_GT(grad2_max_abs, 0.001f) << "Layer 2 should have non-zero individual gradients";
        }

        // Update parameters
        optimizer.step();
    }

    // After 20 iterations, loss should decrease significantly
    EXPECT_LT(final_loss, initial_loss * 0.8f)
        << "Loss should decrease by at least 20% after training. "
        << "Initial: " << initial_loss << ", Final: " << final_loss;

    // Test prediction accuracy after training
    Tensor hidden_test = layer1->forward(input);
    Tensor activated_test = hidden_test.relu();
    Tensor output_test = layer2->forward(activated_test);

    // Check that model produces reasonable predictions
    auto output_acc = output_test.accessor<float, 2>();
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        // Find predicted class (argmax)
        int pred_class = 0;
        float max_score = output_acc[i][0];
        for (int j = 1; j < 3; ++j) {
            if (output_acc[i][j] > max_score) {
                max_score = output_acc[i][j];
                pred_class = j;
            }
        }

        if (static_cast<int64_t>(pred_class) == targets_acc[i]) {
            correct++;
        }
    }

    // After 20 iterations on this simple problem, should get at least 2/4 correct
    EXPECT_GE(correct, 2)
        << "Model should learn to predict at least 50% correctly on this simple problem. "
        << "Got " << correct << "/4 correct";
}
