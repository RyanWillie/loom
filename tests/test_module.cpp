#include <gtest/gtest.h>

#include <memory>

#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/nn/module.h"
#include "loom/nn/parameter.h"
#include "loom/tensor/tensor.h"

using namespace loom;
using namespace loom::nn;

// ============================================================================
// Test Fixtures and Helper Modules
// ============================================================================

class ModuleTest : public ::testing::Test {
  protected:
    ModuleTest() : mCpuDevice(DeviceType::CPU) {}

    void SetUp() override {
        auto& logger = Logger::getInstance("ModuleTest");
        logger.info("Test fixture initialized");
    }

    Device mCpuDevice;
};

// Simple module with one parameter for testing
class SimpleModule : public Module {
  private:
    std::shared_ptr<Parameter> mWeight;

  public:
    SimpleModule(const std::vector<size_t>& shape) {
        mWeight = registerParameter("weight", Parameter::ones(shape));
    }

    Tensor forward(const Tensor& input) override {
        return input * mWeight->data();
    }

    std::shared_ptr<Parameter> weight() const { return mWeight; }
};

// Module with two parameters for testing
class TwoParamModule : public Module {
  private:
    std::shared_ptr<Parameter> mWeight;
    std::shared_ptr<Parameter> mBias;

  public:
    TwoParamModule(const std::vector<size_t>& weight_shape,
                   const std::vector<size_t>& bias_shape) {
        mWeight = registerParameter("weight", Parameter::ones(weight_shape));
        mBias = registerParameter("bias", Parameter::zeros(bias_shape));
    }

    Tensor forward(const Tensor& input) override {
        return input * mWeight->data() + mBias->data();
    }
};

// Hierarchical module containing submodules
class HierarchicalModule : public Module {
  private:
    std::shared_ptr<Module> mLayer1;
    std::shared_ptr<Module> mLayer2;

  public:
    HierarchicalModule() {
        mLayer1 = registerModule("layer1", std::make_shared<TwoParamModule>(
            std::vector<size_t>{3, 4}, std::vector<size_t>{4}));
        mLayer2 = registerModule("layer2", std::make_shared<SimpleModule>(
            std::vector<size_t>{4, 2}));
    }

    Tensor forward(const Tensor& input) override {
        auto x = mLayer1->forward(input);
        return mLayer2->forward(x);
    }
};

// ============================================================================
// Parameter Registration Tests
// ============================================================================

TEST_F(ModuleTest, RegisterParameter) {
    SimpleModule module({2, 3});

    auto params = module.parameters();
    EXPECT_EQ(params.size(), 1);
    EXPECT_EQ(params[0]->shape()[0], 2);
    EXPECT_EQ(params[0]->shape()[1], 3);
}

TEST_F(ModuleTest, RegisterMultipleParameters) {
    TwoParamModule module({3, 4}, {4});

    auto params = module.parameters();
    EXPECT_EQ(params.size(), 2);
}

TEST_F(ModuleTest, DuplicateParameterNameThrows) {
    class BadModule : public Module {
      public:
        BadModule() {
            registerParameter("weight", Parameter::ones({2, 2}));
            // This should throw
            EXPECT_THROW(
                registerParameter("weight", Parameter::ones({3, 3})),
                std::runtime_error
            );
        }
        Tensor forward(const Tensor& input) override { return input; }
    };

    BadModule module;
}

// ============================================================================
// Module Registration Tests
// ============================================================================

TEST_F(ModuleTest, RegisterModule) {
    HierarchicalModule module;

    auto params = module.parameters();
    // layer1 has 2 params (weight, bias), layer2 has 1 param (weight)
    EXPECT_EQ(params.size(), 3);
}

TEST_F(ModuleTest, DuplicateModuleNameThrows) {
    class BadModule : public Module {
      public:
        BadModule() {
            registerModule("layer", std::make_shared<SimpleModule>(std::vector<size_t>{2, 2}));
            // This should throw
            EXPECT_THROW(
                registerModule("layer", std::make_shared<SimpleModule>(std::vector<size_t>{3, 3})),
                std::runtime_error
            );
        }
        Tensor forward(const Tensor& input) override { return input; }
    };

    BadModule module;
}

// ============================================================================
// Recursive Parameter Collection Tests
// ============================================================================

TEST_F(ModuleTest, ParametersReturnsOwnParameters) {
    SimpleModule module({2, 3});

    auto params = module.parameters();
    EXPECT_EQ(params.size(), 1);
}

TEST_F(ModuleTest, ParametersReturnsSubmoduleParameters) {
    HierarchicalModule module;

    auto params = module.parameters();
    EXPECT_EQ(params.size(), 3);  // layer1: weight+bias, layer2: weight
}

TEST_F(ModuleTest, ParametersOrderIsDFS) {
    // Parameters are returned in alphabetical order within each module (std::map)
    // layer1: bias, weight (alphabetical)
    // layer2: weight
    HierarchicalModule module;

    auto params = module.parameters();
    ASSERT_EQ(params.size(), 3);

    // layer1.bias is {4}
    EXPECT_EQ(params[0]->shape()[0], 4);

    // layer1.weight is {3, 4}
    EXPECT_EQ(params[1]->shape()[0], 3);
    EXPECT_EQ(params[1]->shape()[1], 4);

    // layer2.weight is {4, 2}
    EXPECT_EQ(params[2]->shape()[0], 4);
    EXPECT_EQ(params[2]->shape()[1], 2);
}

// ============================================================================
// Named Parameters Tests
// ============================================================================

TEST_F(ModuleTest, NamedParametersReturnsCorrectNames) {
    TwoParamModule module({3, 4}, {4});

    auto named_params = module.namedParameters();
    ASSERT_EQ(named_params.size(), 2);

    // std::map orders alphabetically: bias, weight
    EXPECT_EQ(named_params[0].first, "bias");
    EXPECT_EQ(named_params[1].first, "weight");
}

TEST_F(ModuleTest, NamedParametersReturnsHierarchicalNames) {
    HierarchicalModule module;

    auto named_params = module.namedParameters();
    ASSERT_EQ(named_params.size(), 3);

    // std::map orders alphabetically within each module
    EXPECT_EQ(named_params[0].first, "layer1.bias");
    EXPECT_EQ(named_params[1].first, "layer1.weight");
    EXPECT_EQ(named_params[2].first, "layer2.weight");
}

TEST_F(ModuleTest, NamedParametersMatchesParameters) {
    HierarchicalModule module;

    auto params = module.parameters();
    auto named_params = module.namedParameters();

    ASSERT_EQ(params.size(), named_params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_EQ(params[i], named_params[i].second);
    }
}

// ============================================================================
// Device Movement Tests
// ============================================================================

TEST_F(ModuleTest, ToDeviceMovesAllParameters) {
    SimpleModule module({2, 3});

    EXPECT_TRUE(module.parameters()[0]->device().isCPU());

    module.to(mCpuDevice);  // Move to CPU (same device, but tests the mechanism)

    EXPECT_TRUE(module.parameters()[0]->device().isCPU());
}

TEST_F(ModuleTest, ToDeviceMovesSubmoduleParameters) {
    HierarchicalModule module;

    // All parameters should be on CPU initially
    for (auto& param : module.parameters()) {
        EXPECT_TRUE(param->device().isCPU());
    }

    module.to(mCpuDevice);

    // All should still be on CPU after move
    for (auto& param : module.parameters()) {
        EXPECT_TRUE(param->device().isCPU());
    }
}

// ============================================================================
// Gradient Zeroing Tests
// ============================================================================

TEST_F(ModuleTest, ZeroGradZerosOwnParameters) {
    SimpleModule module({2, 2});

    // Create some gradients
    auto weight = module.weight();
    weight->data().requiresGrad(true);

    Tensor y = weight->data() * 2.0;
    y.backward(Tensor::ones(y.shape(), y.dtype(), y.device()));

    // Gradient should exist and be non-zero
    ASSERT_NE(weight->grad(), nullptr);

    // Zero gradients
    module.zeroGrad();

    // Gradient tensor should be zero
    auto grad = weight->grad();
    ASSERT_NE(grad, nullptr);
    auto acc = grad->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(acc[i][j], 0.0f);
        }
    }
}

TEST_F(ModuleTest, ZeroGradZerosSubmoduleParameters) {
    HierarchicalModule module;

    // Create gradients for all parameters
    for (auto& param : module.parameters()) {
        param->data().requiresGrad(true);
        Tensor y = param->data() * 2.0;
        y.backward(Tensor::ones(y.shape(), y.dtype(), y.device()));
    }

    // All should have gradients
    for (auto& param : module.parameters()) {
        ASSERT_NE(param->grad(), nullptr);
    }

    // Zero all gradients
    module.zeroGrad();

    // All gradients should be zero
    for (auto& param : module.parameters()) {
        auto grad = param->grad();
        ASSERT_NE(grad, nullptr);

        // Check all elements are zero
        auto flat = grad->flatten();
        auto acc = flat.accessor<float, 1>();
        for (size_t i = 0; i < flat.numel(); ++i) {
            EXPECT_FLOAT_EQ(acc[i], 0.0f);
        }
    }
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_F(ModuleTest, DefaultTrainingMode) {
    SimpleModule module({2, 2});
    EXPECT_TRUE(module.training());
}

TEST_F(ModuleTest, SetTrainingMode) {
    SimpleModule module({2, 2});

    module.train(true);
    EXPECT_TRUE(module.training());

    module.train(false);
    EXPECT_FALSE(module.training());
}

TEST_F(ModuleTest, EvalMode) {
    SimpleModule module({2, 2});

    module.eval();
    EXPECT_FALSE(module.training());

    module.train();
    EXPECT_TRUE(module.training());
}

TEST_F(ModuleTest, TrainingModePropagates) {
    HierarchicalModule module;

    module.train(true);
    EXPECT_TRUE(module.training());

    module.eval();
    EXPECT_FALSE(module.training());
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(ModuleTest, ForwardPassWorks) {
    SimpleModule module({3});

    Tensor input = Tensor::ones({3}, DType::FLOAT32, mCpuDevice);
    Tensor output = module.forward(input);

    EXPECT_EQ(output.shape()[0], 3);
}

TEST_F(ModuleTest, CallOperatorWorks) {
    SimpleModule module({3});

    Tensor input = Tensor::ones({3}, DType::FLOAT32, mCpuDevice);
    Tensor output = module(input);  // Using operator()

    EXPECT_EQ(output.shape()[0], 3);
}
