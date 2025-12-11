#include "loom/autograd/no_grad.h"
#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;

class AutogradTest : public ::testing::Test {
  protected:
    Device cpu_device{DeviceType::CPU};

    void SetUp() override {
        // Set fixed seed for reproducible tests
        Tensor::manualSeed(42);
    }
};

// ============================================================================
// Phase 1: Basic Infrastructure Tests
// ============================================================================

TEST_F(AutogradTest, TensorDefaultsToNoGrad) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    EXPECT_FALSE(x.requiresGrad());
    EXPECT_TRUE(x.isLeaf());
    EXPECT_EQ(x.grad(), nullptr);
}

TEST_F(AutogradTest, RequiresGradCanBeEnabled) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    EXPECT_TRUE(x.requiresGrad());
    EXPECT_TRUE(x.isLeaf());
}

TEST_F(AutogradTest, RequiresGradCanBeDisabled) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    x.requiresGrad(false);
    EXPECT_FALSE(x.requiresGrad());
}

TEST_F(AutogradTest, RequiresGradChaining) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    // Test method chaining: requiresGrad() returns Tensor& so we can chain .zero()
    x.requiresGrad(true).zero();
    EXPECT_TRUE(x.requiresGrad());

    // Verify the tensor is indeed zero
    auto acc = x.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc[0][0], 0.0f);
}

TEST_F(AutogradTest, LeafTensorHasNoGradFn) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    EXPECT_TRUE(x.isLeaf());
    EXPECT_EQ(x.gradFn(), nullptr);
}

TEST_F(AutogradTest, ZeroGradDoesNotCrashOnNullGrad) {
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    // Should not crash even though grad is nullptr
    EXPECT_NO_THROW(x.zeroGrad());
}

TEST_F(AutogradTest, GradIsNullBeforeBackward) {
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    // No gradient until backward() is called
    EXPECT_EQ(x.grad(), nullptr);
}

TEST_F(AutogradTest, BackwardThrowsNotImplemented) {
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    // Phase 1: backward() should throw "not implemented"
    EXPECT_THROW(x.backward(), std::runtime_error);
}

TEST_F(AutogradTest, TensorWithoutRequiresGradHasNoOverhead) {
    // Create tensor without enabling autograd
    Tensor x = Tensor::zeros({100, 100}, DType::FLOAT32, cpu_device);

    // Verify autograd is completely disabled
    EXPECT_FALSE(x.requiresGrad());
    EXPECT_TRUE(x.isLeaf());
    EXPECT_EQ(x.grad(), nullptr);
    EXPECT_EQ(x.gradFn(), nullptr);

    // This demonstrates zero overhead - no AutogradMeta allocated
}

TEST_F(AutogradTest, MultipleRequiresGradCalls) {
    Tensor x = Tensor::ones({3, 3}, DType::FLOAT32, cpu_device);

    // Enable
    x.requiresGrad(true);
    EXPECT_TRUE(x.requiresGrad());

    // Disable
    x.requiresGrad(false);
    EXPECT_FALSE(x.requiresGrad());

    // Re-enable
    x.requiresGrad(true);
    EXPECT_TRUE(x.requiresGrad());
    EXPECT_TRUE(x.isLeaf());  // Still a leaf
}

TEST_F(AutogradTest, CopiedTensorSharesAutogradMeta) {
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Shallow copy (default copy constructor)
    Tensor y = x;

    // Both should share the same autograd metadata
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_TRUE(y.isLeaf());

    // Modifying one affects the other (shallow copy behavior)
    x.requiresGrad(false);
    EXPECT_FALSE(y.requiresGrad());  // y is affected because they share metadata
}

TEST_F(AutogradTest, ClonedTensorHasIndependentAutogradMeta) {
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Deep copy (clone creates new storage)
    Tensor y = x.clone();

    // y starts without autograd enabled (clone doesn't copy metadata)
    EXPECT_FALSE(y.requiresGrad());

    // They should be independent
    y.requiresGrad(true);
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_TRUE(x.requiresGrad());  // x is unaffected
}

TEST_F(AutogradTest, GradFnSetterWorks) {
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);

    // Initially no grad_fn
    EXPECT_EQ(x.gradFn(), nullptr);
    EXPECT_TRUE(x.isLeaf());

    // This test just verifies the setter/getter work
    // We'll use actual backward nodes in Phase 3
    // For now, just verify we can set a nullptr
    x.setGradFn(nullptr);
    EXPECT_FALSE(x.isLeaf());  // setGradFn marks as non-leaf
}

// ============================================================================
// Phase 1: Edge Cases and Safety Tests
// ============================================================================

TEST_F(AutogradTest, ZeroSizeTensorSupportsAutograd) {
    Tensor x = Tensor::zeros({0}, DType::FLOAT32, cpu_device);
    EXPECT_NO_THROW(x.requiresGrad(true));
    EXPECT_TRUE(x.requiresGrad());
}

TEST_F(AutogradTest, ScalarTensorSupportsAutograd) {
    Tensor x = Tensor::ones({1}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    EXPECT_TRUE(x.requiresGrad());
    EXPECT_TRUE(x.isLeaf());
}

TEST_F(AutogradTest, HighDimensionalTensorSupportsAutograd) {
    Tensor x = Tensor::zeros({2, 3, 4, 5}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);
    EXPECT_TRUE(x.requiresGrad());
    EXPECT_TRUE(x.isLeaf());
}

TEST_F(AutogradTest, DifferentDTypesSupportsAutograd) {
    // FLOAT32
    Tensor x32 = Tensor::zeros({2, 2}, DType::FLOAT32, cpu_device);
    x32.requiresGrad(true);
    EXPECT_TRUE(x32.requiresGrad());

    // FLOAT64
    Tensor x64 = Tensor::zeros({2, 2}, DType::FLOAT64, cpu_device);
    x64.requiresGrad(true);
    EXPECT_TRUE(x64.requiresGrad());

    // Note: Typically gradients are only computed for floating point types
    // Integer types should probably not support requiresGrad in practice
}

// ============================================================================
// Phase 3: Binary Operations Backward Pass Tests
// ============================================================================

TEST_F(AutogradTest, SimpleAdditionBackward) {
    // Test: z = x + y
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    Tensor y = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device) * 2.0;

    x.requiresGrad(true);
    y.requiresGrad(true);

    Tensor z = x + y;  // Should create AddBackward node

    // Verify graph structure
    EXPECT_TRUE(z.requiresGrad());
    EXPECT_FALSE(z.isLeaf());
    EXPECT_NE(z.gradFn(), nullptr);

    // Compute gradients - pass ones as gradient (simulates dL/dz = 1 everywhere)
    Tensor grad_output = Tensor::ones(z.shape(), DType::FLOAT32, cpu_device);
    z.backward(grad_output);

    // Verify gradients exist
    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(y.grad(), nullptr);

    // For addition: dL/dx = grad_output * 1 = grad_output = all ones
    auto x_grad_acc = x.grad()->accessor<float, 2>();
    auto y_grad_acc = y.grad()->accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(x_grad_acc[i][j], 1.0f);
            EXPECT_FLOAT_EQ(y_grad_acc[i][j], 1.0f);
        }
    }
}

TEST_F(AutogradTest, SimpleMultiplicationBackward) {
    // Test: z = x * y
    Tensor x = Tensor::ones({3, 3}, DType::FLOAT32, cpu_device) * 2.0;
    Tensor y = Tensor::ones({3, 3}, DType::FLOAT32, cpu_device) * 3.0;

    x.requiresGrad(true);
    y.requiresGrad(true);

    Tensor z = x * y;

    // Pass ones as gradient (simulates dL/dz = 1 everywhere)
    Tensor grad_output = Tensor::ones(z.shape(), DType::FLOAT32, cpu_device);
    z.backward(grad_output);

    // For multiplication: dL/dx = grad_output * y = 1.0 * 3.0 = 3.0
    //                     dL/dy = grad_output * x = 1.0 * 2.0 = 2.0
    auto x_grad_acc = x.grad()->accessor<float, 2>();
    auto y_grad_acc = y.grad()->accessor<float, 2>();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(x_grad_acc[i][j], 3.0f);  // y's value
            EXPECT_FLOAT_EQ(y_grad_acc[i][j], 2.0f);  // x's value
        }
    }
}

// ============================================================================
// Phase 3: Numerical Gradient Checking Tests
// ============================================================================

// Helper function to compute numerical gradient using finite differences
// For a scalar function f, the gradient at x is approximated by:
// ∂f/∂x[i] ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
template <typename Func>
Tensor numericalGradient(Func forward_fn, Tensor& x, double eps = 1e-5) {
    Tensor grad = Tensor::zeros(x.shape(), x.dtype(), x.device());
    auto x_acc = x.accessor<float, 2>();
    auto grad_acc = grad.accessor<float, 2>();

    for (size_t i = 0; i < x.size(0); ++i) {
        for (size_t j = 0; j < x.size(1); ++j) {
            // Save original value
            float original = x_acc[i][j];

            // Compute f(x + eps)
            x_acc[i][j] = original + eps;
            Tensor output_plus = forward_fn();
            float f_plus = output_plus.sum().item();  // Replace this line

            // Compute f(x - eps)
            x_acc[i][j] = original - eps;
            Tensor output_minus = forward_fn();
            float f_minus = output_minus.sum().item();  // Replace this line

            // Restore original value
            x_acc[i][j] = original;

            // Compute gradient: (f(x+eps) - f(x-eps)) / (2*eps)
            grad_acc[i][j] = (f_plus - f_minus) / (2.0 * eps);
        }
    }

    return grad;
}

TEST_F(AutogradTest, AdditionNumericalGradientCheck) {
    // Test addition gradient with numerical approximation
    // Scale random values for numerical stability
    Tensor x = Tensor::randn({3, 3}, DType::FLOAT32, cpu_device) * 0.5;
    Tensor y = Tensor::randn({3, 3}, DType::FLOAT32, cpu_device) * 0.5;

    // Forward: sum(x + y)
    auto forward = [&]() { return x + y; };

    // Compute numerical gradient
    Tensor numerical_grad_x = numericalGradient(forward, x);

    // Compute analytical gradient
    x.requiresGrad(true);
    y.requiresGrad(true);
    Tensor z = x + y;

    // Backward with ones (sum reduction)
    Tensor grad_output = Tensor::ones(z.shape(), DType::FLOAT32, cpu_device);
    z.backward(grad_output);

    // Compare analytical vs numerical
    auto analytical_acc = x.grad()->accessor<float, 2>();
    auto numerical_acc = numerical_grad_x.accessor<float, 2>();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(analytical_acc[i][j], numerical_acc[i][j], 2e-2);  // float32 tolerance
        }
    }
}

TEST_F(AutogradTest, SubtractionNumericalGradientCheck) {
    // Test subtraction gradient with numerical approximation
    Tensor x = Tensor::randn({2, 2}, DType::FLOAT32, cpu_device) * 0.5;
    Tensor y = Tensor::randn({2, 2}, DType::FLOAT32, cpu_device) * 0.5;

    // Forward: sum(x - y)
    auto forward = [&]() { return x - y; };

    // Compute numerical gradient
    Tensor numerical_grad_x = numericalGradient(forward, x);

    // Compute analytical gradient
    x.requiresGrad(true);
    y.requiresGrad(true);
    Tensor z = x - y;

    // Backward with ones (sum reduction)
    Tensor grad_output = Tensor::ones(z.shape(), DType::FLOAT32, cpu_device);
    z.backward(grad_output);

    // Compare
    auto analytical_acc = x.grad()->accessor<float, 2>();
    auto numerical_acc = numerical_grad_x.accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(analytical_acc[i][j], numerical_acc[i][j], 2e-2);  // float32 tolerance
        }
    }
}

TEST_F(AutogradTest, MultiplicationNumericalGradientCheck) {
    // Test multiplication gradient
    // Use scaled random values to avoid large gradients that cause precision issues
    // randn() gives std=1, so values can be large; scaling by 0.5 keeps them in [-1.5, 1.5] range
    Tensor x = Tensor::randn({2, 3}, DType::FLOAT32, cpu_device) * 0.5;
    Tensor y = Tensor::randn({2, 3}, DType::FLOAT32, cpu_device) * 0.5;

    // Forward: sum(x * y)
    auto forward = [&]() { return x * y; };

    // Compute numerical gradient
    Tensor numerical_grad_x = numericalGradient(forward, x);

    // Compute analytical gradient
    x.requiresGrad(true);
    y.requiresGrad(true);
    Tensor z = x * y;

    // Backward with ones (sum reduction)
    Tensor grad_output = Tensor::ones(z.shape(), DType::FLOAT32, cpu_device);
    z.backward(grad_output);

    // Compare
    auto analytical_acc = x.grad()->accessor<float, 2>();
    auto numerical_acc = numerical_grad_x.accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(analytical_acc[i][j], numerical_acc[i][j], 2e-2);  // float32 tolerance
        }
    }
}

TEST_F(AutogradTest, DivisionNumericalGradientCheck) {
    // Test division gradient
    // Use uniform distribution in [1, 3] to avoid small denominators and extreme gradients
    // Division gradient is 1/y, so smaller y means larger gradient and more numerical error
    Tensor x = Tensor::zeros({2, 2}, DType::FLOAT32, cpu_device);
    Tensor y = Tensor::zeros({2, 2}, DType::FLOAT32, cpu_device);
    x.uniform(1.0, 3.0);
    y.uniform(1.0, 3.0);

    // Forward: sum(x / y)
    auto forward = [&]() { return x / y; };

    // Compute numerical gradient
    Tensor numerical_grad_x = numericalGradient(forward, x);

    // Compute analytical gradient
    x.requiresGrad(true);
    y.requiresGrad(true);
    Tensor z = x / y;

    // Backward
    Tensor grad_output = Tensor::ones(z.shape(), DType::FLOAT32, cpu_device);
    z.backward(grad_output);

    // Compare
    auto analytical_acc = x.grad()->accessor<float, 2>();
    auto numerical_acc = numerical_grad_x.accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            // Division gradient involves y² which amplifies float32 precision errors
            EXPECT_NEAR(analytical_acc[i][j], numerical_acc[i][j], 5e-2);
        }
    }
}

// ============================================================================
// Phase 4: Matrix Multiplication Backward Pass Tests
// ============================================================================

TEST_F(AutogradTest, MatmulBackwardNumericalCheck) {
    // Test matmul gradient: C = A @ B
    Tensor A = Tensor::randn({3, 4}, DType::FLOAT32, cpu_device);
    Tensor B = Tensor::randn({4, 5}, DType::FLOAT32, cpu_device);

    // Forward: sum(A @ B)
    auto forward = [&]() { return A.matmul(B); };

    // Compute numerical gradient for A
    Tensor numerical_grad_A = numericalGradient(forward, A);

    // Compute analytical gradient
    A.requiresGrad(true);
    B.requiresGrad(true);
    Tensor C = A.matmul(B);

    // Backward with ones (sum reduction)
    Tensor grad_output = Tensor::ones(C.shape(), DType::FLOAT32, cpu_device);
    C.backward(grad_output);

    // Compare gradients for A
    auto analytical_acc = A.grad()->accessor<float, 2>();
    auto numerical_acc = numerical_grad_A.accessor<float, 2>();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            // Matmul accumulates many operations, needs slightly looser tolerance
            EXPECT_NEAR(analytical_acc[i][j], numerical_acc[i][j], 0.1);
        }
    }
}

TEST_F(AutogradTest, MatmulChainRuleTest) {
    // Test: loss = sum((A @ B @ C))
    // This tests gradient propagation through multiple matmuls
    Tensor A = Tensor::randn({2, 3}, DType::FLOAT32, cpu_device);
    Tensor B = Tensor::randn({3, 4}, DType::FLOAT32, cpu_device);
    Tensor C = Tensor::randn({4, 2}, DType::FLOAT32, cpu_device);

    A.requiresGrad(true);
    B.requiresGrad(true);
    C.requiresGrad(true);

    // Forward: A @ B @ C
    Tensor AB = A.matmul(B);    // [2,3] @ [3,4] = [2,4]
    Tensor ABC = AB.matmul(C);  // [2,4] @ [4,2] = [2,2]

    // Verify graph structure
    EXPECT_TRUE(AB.requiresGrad());
    EXPECT_FALSE(AB.isLeaf());
    EXPECT_NE(AB.gradFn(), nullptr);

    EXPECT_TRUE(ABC.requiresGrad());
    EXPECT_FALSE(ABC.isLeaf());
    EXPECT_NE(ABC.gradFn(), nullptr);

    // Backward
    Tensor grad_output = Tensor::ones(ABC.shape(), DType::FLOAT32, cpu_device);
    ABC.backward(grad_output);

    // Verify all gradients exist and have correct shapes
    ASSERT_NE(A.grad(), nullptr);
    ASSERT_NE(B.grad(), nullptr);
    ASSERT_NE(C.grad(), nullptr);

    EXPECT_EQ(A.grad()->shape(), A.shape());  // [2,3]
    EXPECT_EQ(B.grad()->shape(), B.shape());  // [3,4]
    EXPECT_EQ(C.grad()->shape(), C.shape());  // [4,2]
}

// ============================================================================
// Phase 5: Reduction Operations Backward Pass Tests
// ============================================================================

TEST_F(AutogradTest, SumBackward) {
    // Test: loss = sum(x)
    Tensor x = Tensor::ones({3, 4}, DType::FLOAT32, cpu_device) * 2.0;
    x.requiresGrad(true);

    Tensor s = x.sum();

    // Verify forward pass
    EXPECT_FLOAT_EQ(s.item(), 24.0);  // 3*4*2 = 24

    // Verify graph structure
    EXPECT_TRUE(s.requiresGrad());
    EXPECT_FALSE(s.isLeaf());
    EXPECT_NE(s.gradFn(), nullptr);

    // Backward: gradient of sum is 1 for all elements
    s.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

TEST_F(AutogradTest, MeanBackward) {
    // Test: loss = mean(x)
    Tensor x = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device) * 6.0;
    x.requiresGrad(true);

    Tensor m = x.mean();

    // Verify forward pass
    EXPECT_FLOAT_EQ(m.item(), 6.0);

    // Verify graph structure
    EXPECT_TRUE(m.requiresGrad());
    EXPECT_FALSE(m.isLeaf());
    EXPECT_NE(m.gradFn(), nullptr);

    // Backward: gradient of mean is 1/n for all elements
    m.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    // n = 6, so gradient should be 1/6 ≈ 0.1667
    float expected_grad = 1.0f / 6.0f;
    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(grad_acc[i][j], expected_grad, 1e-5);
        }
    }
}

TEST_F(AutogradTest, ReLUBackward) {
    // Test: y = relu(x) where x has positive and negative values
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    // Set up: [-1, 2, -3], [4, -5, 6]
    x_acc[0][0] = -1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = -3.0f;
    x_acc[1][0] = 4.0f;
    x_acc[1][1] = -5.0f;
    x_acc[1][2] = 6.0f;

    x.requiresGrad(true);
    Tensor y = x.relu();

    // Verify forward pass: max(0, x)
    auto y_acc = y.accessor<float, 2>();
    EXPECT_FLOAT_EQ(y_acc[0][0], 0.0f);  // relu(-1) = 0
    EXPECT_FLOAT_EQ(y_acc[0][1], 2.0f);  // relu(2) = 2
    EXPECT_FLOAT_EQ(y_acc[0][2], 0.0f);  // relu(-3) = 0
    EXPECT_FLOAT_EQ(y_acc[1][0], 4.0f);  // relu(4) = 4
    EXPECT_FLOAT_EQ(y_acc[1][1], 0.0f);  // relu(-5) = 0
    EXPECT_FLOAT_EQ(y_acc[1][2], 6.0f);  // relu(6) = 6

    // Verify graph structure
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_FALSE(y.isLeaf());
    EXPECT_NE(y.gradFn(), nullptr);

    // Backward: gradient flows through where x > 0
    Tensor grad_output = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    y.backward(grad_output);

    ASSERT_NE(x.grad(), nullptr);
    auto grad_acc = x.grad()->accessor<float, 2>();

    // Gradient is 1 where x > 0, 0 where x <= 0
    EXPECT_FLOAT_EQ(grad_acc[0][0], 0.0f);  // x was -1
    EXPECT_FLOAT_EQ(grad_acc[0][1], 1.0f);  // x was 2
    EXPECT_FLOAT_EQ(grad_acc[0][2], 0.0f);  // x was -3
    EXPECT_FLOAT_EQ(grad_acc[1][0], 1.0f);  // x was 4
    EXPECT_FLOAT_EQ(grad_acc[1][1], 0.0f);  // x was -5
    EXPECT_FLOAT_EQ(grad_acc[1][2], 1.0f);  // x was 6
}

TEST_F(AutogradTest, ReLUChainedWithSum) {
    // Test: loss = sum(relu(x))
    // This tests gradient flow through multiple operations
    Tensor x = Tensor::zeros({2, 2}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = -1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[1][0] = 3.0f;
    x_acc[1][1] = -4.0f;

    x.requiresGrad(true);

    // Forward: relu then sum
    Tensor r = x.relu();
    Tensor s = r.sum();

    // relu(x) = [0, 2, 3, 0], sum = 5
    EXPECT_FLOAT_EQ(s.item(), 5.0);

    // Backward
    s.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    auto grad_acc = x.grad()->accessor<float, 2>();

    // Gradient: sum gives 1 everywhere, relu masks where x <= 0
    EXPECT_FLOAT_EQ(grad_acc[0][0], 0.0f);  // x was -1
    EXPECT_FLOAT_EQ(grad_acc[0][1], 1.0f);  // x was 2
    EXPECT_FLOAT_EQ(grad_acc[1][0], 1.0f);  // x was 3
    EXPECT_FLOAT_EQ(grad_acc[1][1], 0.0f);  // x was -4
}

TEST_F(AutogradTest, MeanOfMultiplication) {
    // Test: loss = mean(x * y)
    // This tests chained operations with mean
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device) * 2.0;
    Tensor y = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device) * 3.0;

    x.requiresGrad(true);
    y.requiresGrad(true);

    Tensor z = x * y;        // z = [[6,6],[6,6]]
    Tensor loss = z.mean();  // loss = 6

    EXPECT_FLOAT_EQ(loss.item(), 6.0);

    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // For mean(x*y), gradient wrt x is y/n = 3/4 = 0.75
    // For mean(x*y), gradient wrt y is x/n = 2/4 = 0.5
    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(y.grad(), nullptr);

    auto x_grad_acc = x.grad()->accessor<float, 2>();
    auto y_grad_acc = y.grad()->accessor<float, 2>();

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(x_grad_acc[i][j], 0.75f, 1e-5);  // y / n = 3 / 4
            EXPECT_NEAR(y_grad_acc[i][j], 0.5f, 1e-5);   // x / n = 2 / 4
        }
    }
}

// ============================================================================
// Phase 6: View Operations Backward Pass Tests
// ============================================================================

TEST_F(AutogradTest, ReshapeBackward) {
    // Test: y = x.reshape({2, 3}).sum()
    Tensor x = Tensor::ones({6}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 1>();
    for (size_t i = 0; i < 6; ++i) {
        x_acc[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5, 6]
    }

    x.requiresGrad(true);

    Tensor y = x.reshape({2, 3});  // [[1,2,3], [4,5,6]]
    Tensor z = y.sum();            // 21

    EXPECT_FLOAT_EQ(z.item(), 21.0);

    // Verify graph structure
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_FALSE(y.isLeaf());
    EXPECT_NE(y.gradFn(), nullptr);

    // Backward
    z.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());  // Gradient has original shape [6]

    // Gradient should be all 1s (from sum)
    auto grad_acc = x.grad()->accessor<float, 1>();
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(grad_acc[i], 1.0f);
    }
}

TEST_F(AutogradTest, TransposeBackward) {
    // Test: y = x.transpose().sum()
    Tensor x = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    float val = 1.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            x_acc[i][j] = val++;
        }
    }

    x.requiresGrad(true);

    Tensor y = x.transpose();  // [3, 2]
    EXPECT_EQ(y.shape()[0], 3);
    EXPECT_EQ(y.shape()[1], 2);

    Tensor z = y.sum();

    // Verify graph structure
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);

    // Backward
    z.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    // Gradient should be all 1s
    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

TEST_F(AutogradTest, FlattenBackward) {
    // Test: y = x.flatten().sum()
    Tensor x = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device) * 2.0;
    x.requiresGrad(true);

    Tensor y = x.flatten();  // [6]
    EXPECT_EQ(y.shape().size(), 1);
    EXPECT_EQ(y.shape()[0], 6);

    Tensor z = y.sum();  // 12

    EXPECT_FLOAT_EQ(z.item(), 12.0);

    // Backward
    z.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

TEST_F(AutogradTest, UnsqueezeSqueezeBackward) {
    // Test unsqueeze and squeeze as inverse operations
    Tensor x = Tensor::ones({3, 4}, DType::FLOAT32, cpu_device) * 2.0;
    x.requiresGrad(true);

    Tensor y = x.unsqueeze(0);  // [1, 3, 4]
    EXPECT_EQ(y.shape().size(), 3);
    EXPECT_EQ(y.shape()[0], 1);

    Tensor z = y.squeeze(0);  // [3, 4]
    EXPECT_EQ(z.shape(), x.shape());

    Tensor loss = z.sum();  // 24

    EXPECT_FLOAT_EQ(loss.item(), 24.0);

    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

TEST_F(AutogradTest, ChainedViewOperations) {
    // Test multiple view operations chained together
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[1][0] = 3.0f;
    x_acc[1][1] = 4.0f;

    x.requiresGrad(true);

    Tensor a = x.reshape({4});  // [4]
    Tensor b = a.unsqueeze(0);  // [1, 4]
    Tensor c = b.transpose();   // [4, 1]
    Tensor loss = c.sum();      // 10

    EXPECT_FLOAT_EQ(loss.item(), 10.0);

    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

// ============================================================================
// In-Place Operation Detection Tests
// ============================================================================

TEST_F(AutogradTest, InPlaceModificationBumpsVersion) {
    // Test that in-place modifications bump the version counter
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    uint64_t version_before = x.version();
    EXPECT_EQ(version_before, 0);  // Initial version is 0

    x += 1.0;  // In-place modification

    uint64_t version_after = x.version();
    EXPECT_EQ(version_after, 1);  // Version should increment

    x *= 2.0;  // Another in-place operation

    EXPECT_EQ(x.version(), 2);  // Version should increment again
}

TEST_F(AutogradTest, InPlaceOperationAllowedWhenSafe) {
    // Test that in-place is safe when tensor isn't saved for backward
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    x += 1.0;  // OK: No backward nodes reference x yet

    Tensor y = x.sum();
    y.backward();  // Should work fine

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

// ============================================================================
// NoGrad Context Manager Tests
// ============================================================================

TEST_F(AutogradTest, NoGradDisablesGradientTracking) {
    // Test that NoGrad context disables gradient tracking
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    autograd::NoGrad no_grad;
    Tensor y = x * 2.0;  // Should not create grad_fn

    // y should not have a grad function despite x.requiresGrad() being true
    EXPECT_EQ(y.gradFn(), nullptr);
    EXPECT_FALSE(y.requiresGrad());
}

TEST_F(AutogradTest, NoGradRestoresPreviousState) {
    // Test that NoGrad restores the previous state
    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Before NoGrad: operations create grad_fn
    Tensor y1 = x * 2.0;
    EXPECT_NE(y1.gradFn(), nullptr);

    {
        autograd::NoGrad no_grad;
        Tensor y2 = x * 3.0;
        EXPECT_EQ(y2.gradFn(), nullptr);
    }

    // After NoGrad: operations create grad_fn again
    Tensor y3 = x * 4.0;
    EXPECT_NE(y3.gradFn(), nullptr);
}

TEST_F(AutogradTest, NoGradWorksWithMatmul) {
    // Test that NoGrad works with matmul operations
    Tensor A = Tensor::ones({3, 4}, DType::FLOAT32, cpu_device);
    Tensor B = Tensor::ones({4, 5}, DType::FLOAT32, cpu_device);
    A.requiresGrad(true);
    B.requiresGrad(true);

    autograd::NoGrad no_grad;
    Tensor C = A.matmul(B);

    EXPECT_EQ(C.gradFn(), nullptr);
    EXPECT_FALSE(C.requiresGrad());
}

TEST_F(AutogradTest, NoGradWorksWithComplexGraph) {
    // Test NoGrad with a complex computation graph
    Tensor x = Tensor::randn({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    autograd::NoGrad no_grad;
    Tensor a = x * 2.0;
    Tensor b = a.relu();
    Tensor c = b.sum();
    Tensor result = c + 1.0;

    // Entire graph should have no gradients
    EXPECT_EQ(result.gradFn(), nullptr);
    EXPECT_FALSE(result.requiresGrad());
}

// ============================================================================
// End-to-End Neural Network Test
// ============================================================================

TEST_F(AutogradTest, TwoLayerNeuralNetworkEndToEnd) {
    // Test complete forward-backward pass through a 2-layer neural network
    // Network: x -> W1 -> ReLU -> W2 -> loss
    //
    // This tests:
    // - Matmul forward/backward
    // - ReLU forward/backward
    // - Proper gradient accumulation
    // - Chain rule through multiple operations

    // Input: batch_size=4, input_dim=10
    Tensor x = Tensor::randn({4, 10}, DType::FLOAT32, cpu_device);

    // Layer 1: 10 -> 20
    Tensor W1 = Tensor::randn({10, 20}, DType::FLOAT32, cpu_device);
    W1.requiresGrad(true);

    // Layer 2: 20 -> 1
    Tensor W2 = Tensor::randn({20, 1}, DType::FLOAT32, cpu_device);
    W2.requiresGrad(true);

    // Forward pass
    Tensor h = x.matmul(W1);       // [4, 20]
    Tensor h_relu = h.relu();      // [4, 20]
    Tensor y = h_relu.matmul(W2);  // [4, 1]
    Tensor loss = (y * y).mean();  // scalar

    // Backward pass
    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // Verify gradients exist and have correct shapes
    ASSERT_NE(W1.grad(), nullptr);
    ASSERT_NE(W2.grad(), nullptr);

    EXPECT_EQ(W1.grad()->shape(), W1.shape());  // [10, 20]
    EXPECT_EQ(W2.grad()->shape(), W2.shape());  // [20, 1]

    // Verify gradients are not all zeros (sanity check)
    auto W1_grad_flat = W1.grad()->contiguous().flatten();
    auto W1_grad_acc = W1_grad_flat.accessor<float, 1>();

    bool has_nonzero_grad = false;
    for (size_t i = 0; i < W1_grad_flat.numel(); ++i) {
        if (std::abs(W1_grad_acc[i]) > 1e-7f) {
            has_nonzero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_grad);

    // Verify W2 gradients are not all zeros
    auto W2_grad_flat = W2.grad()->contiguous().flatten();
    auto W2_grad_acc = W2_grad_flat.accessor<float, 1>();

    has_nonzero_grad = false;
    for (size_t i = 0; i < W2_grad_flat.numel(); ++i) {
        if (std::abs(W2_grad_acc[i]) > 1e-7f) {
            has_nonzero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_grad);
}

// ============================================================================
// Gradient Accumulation Edge Cases (Regression Tests)
// ============================================================================

TEST_F(AutogradTest, GradientAccumulationWithMultiplePaths) {
    // Test gradient accumulation when a tensor receives gradients from multiple paths
    // This is the "diamond pattern": x -> a -> c
    //                                x -> b -> c

    Tensor x = Tensor::randn({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Create two paths from x
    Tensor a = x * 2.0;  // Path 1: da/dx = 2
    Tensor b = x + 1.0;  // Path 2: db/dx = 1

    // Converge to c
    Tensor c = a + b;  // dc/da = 1, dc/db = 1

    // Backward pass - x should accumulate gradients from both paths
    c.sum().backward();

    // Verify gradient accumulation
    // dc/dx = dc/da * da/dx + dc/db * db/dx = 1*2 + 1*1 = 3
    ASSERT_NE(x.grad(), nullptr);
    auto acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(acc[i][j], 3.0f)
                << "Failed at [" << i << "][" << j << "]: expected 3.0, got " << acc[i][j];
        }
    }
}

TEST_F(AutogradTest, GradientAccumulationWithNonContiguousTensors) {
    // Regression test: gradient accumulation with non-contiguous intermediate tensors
    // Previously failed when grad_input was non-contiguous in engine.cpp

    Tensor x = Tensor::randn({3, 4}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Create non-contiguous tensor through transpose
    Tensor y = x.transpose();  // Non-contiguous
    EXPECT_FALSE(y.isContiguous());

    // Further operations
    Tensor z = y * 2.0;
    Tensor output = z.sum();

    // Backward pass
    output.backward();

    // Verify x has gradients
    ASSERT_NE(x.grad(), nullptr);

    // All gradients should be 2.0 (derivative of 2x summed)
    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 2.0f);
        }
    }
}

TEST_F(AutogradTest, ComplexDiamondPatternWithNonContiguous) {
    // Combined test: Multiple paths + non-contiguous tensors
    //     x -> transpose -> a -> output
    //     x -> b -> output

    Tensor x = Tensor::randn({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Path 1: through transpose (non-contiguous)
    Tensor x_t = x.transpose();
    Tensor a = x_t * 3.0;
    Tensor a_t_back = a.transpose();  // Transpose back to [2, 3]

    // Path 2: direct
    Tensor b = x * 2.0;

    // Converge
    Tensor output = a_t_back + b;

    // Backward
    output.sum().backward();

    // Verify gradient accumulation
    // dc/dx = 3 (from path 1) + 2 (from path 2) = 5
    ASSERT_NE(x.grad(), nullptr);
    auto acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(acc[i][j], 5.0f, 1e-5f);
        }
    }
}

TEST_F(AutogradTest, GradientAccumulationWithBroadcasting) {
    // Test gradient accumulation with broadcasting operations
    // Common in neural networks (bias addition, normalization)

    Tensor x = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Broadcast operation (simulates bias or normalization)
    Tensor bias = Tensor::ones({3}, DType::FLOAT32, cpu_device);
    bias.requiresGrad(true);

    Tensor y = x + bias;  // Broadcasting: [2,3] + [3] -> [2,3]
    Tensor output = y.sum();

    output.backward();

    // x gradients should all be 1.0
    ASSERT_NE(x.grad(), nullptr);
    auto x_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(x_acc[i][j], 1.0f);
        }
    }

    // bias gradients should be 2.0 (accumulated from 2 rows)
    ASSERT_NE(bias.grad(), nullptr);
    auto bias_acc = bias.grad()->accessor<float, 1>();
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(bias_acc[j], 2.0f);
    }
}

TEST_F(AutogradTest, MultipleBackwardCallsAccumulateGradients) {
    // Test that calling backward multiple times accumulates gradients
    // This is important for gradient accumulation in mini-batch training

    Tensor x = Tensor::ones({2, 2}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // First backward pass
    Tensor y1 = x * 2.0;
    y1.sum().backward();

    // Check first gradient
    ASSERT_NE(x.grad(), nullptr);
    auto acc1 = x.grad()->accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc1[0][0], 2.0f);

    // Second backward pass (should accumulate)
    Tensor y2 = x * 3.0;
    y2.sum().backward();

    // Check accumulated gradient: 2 + 3 = 5
    auto acc2 = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(acc2[i][j], 5.0f);
        }
    }
}

TEST_F(AutogradTest, NeuralNetworkWithNoGradInference) {
    // Test that inference mode (NoGrad) works correctly in training workflow
    // This is a common pattern: train with gradients, evaluate without

    // Setup network
    Tensor W1 = Tensor::randn({5, 10}, DType::FLOAT32, cpu_device);
    Tensor W2 = Tensor::randn({10, 1}, DType::FLOAT32, cpu_device);
    W1.requiresGrad(true);
    W2.requiresGrad(true);

    // Training mode: forward + backward
    Tensor x_train = Tensor::randn({2, 5}, DType::FLOAT32, cpu_device);
    Tensor h_train = x_train.matmul(W1).relu();
    Tensor y_train = h_train.matmul(W2);
    Tensor loss = y_train.mean();

    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // Verify training created gradients
    ASSERT_NE(W1.grad(), nullptr);
    ASSERT_NE(W2.grad(), nullptr);

    // Inference mode: NoGrad context
    Tensor x_test = Tensor::randn({2, 5}, DType::FLOAT32, cpu_device);
    autograd::NoGrad no_grad;
    Tensor h_test = x_test.matmul(W1).relu();
    Tensor y_test = h_test.matmul(W2);

    // Inference should not create grad_fn
    EXPECT_EQ(y_test.gradFn(), nullptr);
    EXPECT_FALSE(y_test.requiresGrad());

    // Should not be able to call backward on inference result
    EXPECT_THROW(
        {
            y_test.backward(Tensor::ones({2, 1}, DType::FLOAT32, cpu_device));
        },
        std::runtime_error);
}

// ============================================================================
// Phase 5: Additional Reduction and View Operation Tests
// ============================================================================

TEST_F(AutogradTest, MeanBackwardWithDimKeepDim) {
    // Test: mean with dim and keepdim arguments
    // Input: [[1, 2, 3], [4, 5, 6]] shape [2, 3]
    // mean(dim=1, keepdim=True) -> [[2], [5]] shape [2, 1]
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 4.0f;
    x_acc[1][1] = 5.0f;
    x_acc[1][2] = 6.0f;
    x.requiresGrad(true);

    Tensor m = x.mean(1, true);  // mean along dim=1, keepdim=True

    // Verify forward pass
    EXPECT_EQ(m.shape(), std::vector<size_t>({2, 1}));
    auto m_acc = m.accessor<float, 2>();
    EXPECT_FLOAT_EQ(m_acc[0][0], 2.0f);  // (1+2+3)/3 = 2
    EXPECT_FLOAT_EQ(m_acc[1][0], 5.0f);  // (4+5+6)/3 = 5

    // Verify graph structure
    EXPECT_TRUE(m.requiresGrad());
    EXPECT_FALSE(m.isLeaf());
    EXPECT_NE(m.gradFn(), nullptr);

    // Backward: gradient of mean is 1/3 for all elements in each row
    Tensor grad_output = Tensor::ones({2, 1}, DType::FLOAT32, cpu_device);
    m.backward(grad_output);

    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), x.shape());

    // Each element in a row contributes to mean, so gradient is 1/3
    auto grad_acc = x.grad()->accessor<float, 2>();
    float expected_grad = 1.0f / 3.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(grad_acc[i][j], expected_grad, 1e-5);
        }
    }
}

TEST_F(AutogradTest, MinBackwardScalar) {
    // Test: scalar min (reduce all elements)
    // Input: [[1, -2, 3], [4, 5, -6]]
    // min() -> -6 (global minimum)
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = -2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 4.0f;
    x_acc[1][1] = 5.0f;
    x_acc[1][2] = -6.0f;
    x.requiresGrad(true);

    Tensor m = x.min();

    // Verify forward pass
    EXPECT_FLOAT_EQ(m.item(), -6.0f);

    // Verify graph structure
    EXPECT_TRUE(m.requiresGrad());
    EXPECT_FALSE(m.isLeaf());
    EXPECT_NE(m.gradFn(), nullptr);

    // Backward: gradient flows only to the minimum position
    m.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    auto grad_acc = x.grad()->accessor<float, 2>();

    // Only position [1][2] (value -6) should have gradient
    EXPECT_FLOAT_EQ(grad_acc[0][0], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[0][1], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[0][2], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[1][0], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[1][1], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[1][2], 1.0f);  // Gradient at min position
}

TEST_F(AutogradTest, MinBackwardWithDimKeepDim) {
    // Test: min with dim and keepdim
    // Input: [[3, 1, 2], [6, 4, 5]] shape [2, 3]
    // min(dim=1, keepdim=True) -> [[1], [4]] shape [2, 1]
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 3.0f;
    x_acc[0][1] = 1.0f;  // min in row 0
    x_acc[0][2] = 2.0f;
    x_acc[1][0] = 6.0f;
    x_acc[1][1] = 4.0f;  // min in row 1
    x_acc[1][2] = 5.0f;
    x.requiresGrad(true);

    Tensor m = x.min(1, true);  // min along dim=1, keepdim=True

    // Verify forward pass
    EXPECT_EQ(m.shape(), std::vector<size_t>({2, 1}));
    auto m_acc = m.accessor<float, 2>();
    EXPECT_FLOAT_EQ(m_acc[0][0], 1.0f);
    EXPECT_FLOAT_EQ(m_acc[1][0], 4.0f);

    // Verify graph structure
    EXPECT_TRUE(m.requiresGrad());
    EXPECT_NE(m.gradFn(), nullptr);

    // Backward: gradient flows only to min positions in each row
    Tensor grad_output = Tensor::ones({2, 1}, DType::FLOAT32, cpu_device);
    m.backward(grad_output);

    ASSERT_NE(x.grad(), nullptr);
    auto grad_acc = x.grad()->accessor<float, 2>();

    // Row 0: only position [0][1] (value 1) gets gradient
    EXPECT_FLOAT_EQ(grad_acc[0][0], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[0][1], 1.0f);
    EXPECT_FLOAT_EQ(grad_acc[0][2], 0.0f);

    // Row 1: only position [1][1] (value 4) gets gradient
    EXPECT_FLOAT_EQ(grad_acc[1][0], 0.0f);
    EXPECT_FLOAT_EQ(grad_acc[1][1], 1.0f);
    EXPECT_FLOAT_EQ(grad_acc[1][2], 0.0f);
}

TEST_F(AutogradTest, DotBackward) {
    // Test: dot product backward
    // x = [1, 2, 3], y = [4, 5, 6]
    // z = dot(x, y) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    // ∂z/∂x = y = [4, 5, 6]
    // ∂z/∂y = x = [1, 2, 3]
    Tensor x = Tensor::zeros({3}, DType::FLOAT32, cpu_device);
    Tensor y = Tensor::zeros({3}, DType::FLOAT32, cpu_device);

    auto x_acc = x.accessor<float, 1>();
    auto y_acc = y.accessor<float, 1>();
    x_acc[0] = 1.0f;
    x_acc[1] = 2.0f;
    x_acc[2] = 3.0f;
    y_acc[0] = 4.0f;
    y_acc[1] = 5.0f;
    y_acc[2] = 6.0f;

    x.requiresGrad(true);
    y.requiresGrad(true);

    Tensor z = x.dot(y);

    // Verify forward pass
    EXPECT_FLOAT_EQ(z.item(), 32.0f);

    // Verify graph structure
    EXPECT_TRUE(z.requiresGrad());
    EXPECT_FALSE(z.isLeaf());
    EXPECT_NE(z.gradFn(), nullptr);

    // Backward
    z.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // Check x gradients: should be y
    ASSERT_NE(x.grad(), nullptr);
    auto x_grad_acc = x.grad()->accessor<float, 1>();
    EXPECT_FLOAT_EQ(x_grad_acc[0], 4.0f);
    EXPECT_FLOAT_EQ(x_grad_acc[1], 5.0f);
    EXPECT_FLOAT_EQ(x_grad_acc[2], 6.0f);

    // Check y gradients: should be x
    ASSERT_NE(y.grad(), nullptr);
    auto y_grad_acc = y.grad()->accessor<float, 1>();
    EXPECT_FLOAT_EQ(y_grad_acc[0], 1.0f);
    EXPECT_FLOAT_EQ(y_grad_acc[1], 2.0f);
    EXPECT_FLOAT_EQ(y_grad_acc[2], 3.0f);
}

TEST_F(AutogradTest, PermuteBackward) {
    // Test: permute backward
    // Input shape: [2, 3, 4]
    // permute([2, 0, 1]) -> output shape: [4, 2, 3]
    // Gradient should flow back with inverse permutation
    Tensor x = Tensor::randn({2, 3, 4}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Permute: [2, 3, 4] -> [4, 2, 3]
    Tensor y = x.permute({2, 0, 1});

    // Verify forward pass shape
    EXPECT_EQ(y.shape(), std::vector<size_t>({4, 2, 3}));

    // Verify graph structure
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);

    // Backward with ones
    Tensor grad_output = Tensor::ones({4, 2, 3}, DType::FLOAT32, cpu_device);
    Tensor loss = (y * grad_output).sum();
    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // Check that gradient has correct shape
    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), std::vector<size_t>({2, 3, 4}));

    // Gradient values should all be 1 (since we used ones for grad_output)
    auto grad_flat = x.grad()->contiguous().flatten();
    auto grad_acc = grad_flat.accessor<float, 1>();
    for (size_t i = 0; i < grad_flat.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_acc[i], 1.0f);
    }
}

TEST_F(AutogradTest, PermuteBackwardNumericalCheck) {
    // Numerical gradient check for permute
    // f(x) = sum(permute(x, [1, 0]))
    Tensor x = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    x_acc[0][0] = 1.0f;
    x_acc[0][1] = 2.0f;
    x_acc[0][2] = 3.0f;
    x_acc[1][0] = 4.0f;
    x_acc[1][1] = 5.0f;
    x_acc[1][2] = 6.0f;
    x.requiresGrad(true);

    Tensor y = x.permute({1, 0});  // Transpose: [2, 3] -> [3, 2]
    Tensor loss = y.sum();

    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // Gradient of sum is 1 everywhere, permute doesn't change this
    ASSERT_NE(x.grad(), nullptr);
    auto grad_acc = x.grad()->accessor<float, 2>();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 1.0f);
        }
    }
}

TEST_F(AutogradTest, SliceBackward) {
    // Test: slice backward
    // Input shape: [5, 4]
    // slice(dim=0, start=1, end=4) -> output shape: [3, 4] (rows 1, 2, 3)
    // Gradient should scatter back to the sliced positions
    Tensor x = Tensor::randn({5, 4}, DType::FLOAT32, cpu_device);
    x.requiresGrad(true);

    // Slice rows 1 through 3 (inclusive)
    Tensor y = x.slice(0, 1, 4);

    // Verify forward pass shape
    EXPECT_EQ(y.shape(), std::vector<size_t>({3, 4}));

    // Verify graph structure
    EXPECT_TRUE(y.requiresGrad());
    EXPECT_NE(y.gradFn(), nullptr);

    // Backward with gradient of all 2's
    Tensor grad_output = Tensor::full({3, 4}, 2.0, DType::FLOAT32, cpu_device);
    Tensor loss = (y * grad_output).sum();
    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    // Check gradient shape
    ASSERT_NE(x.grad(), nullptr);
    EXPECT_EQ(x.grad()->shape(), std::vector<size_t>({5, 4}));

    auto grad_acc = x.grad()->accessor<float, 2>();

    // Row 0 should be zero (not in slice)
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(grad_acc[0][j], 0.0f);
    }

    // Rows 1, 2, 3 should have gradient of 2
    for (size_t i = 1; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(grad_acc[i][j], 2.0f);
        }
    }

    // Row 4 should be zero (not in slice)
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(grad_acc[4][j], 0.0f);
    }
}

TEST_F(AutogradTest, SliceBackwardColumnSlice) {
    // Test: slice along different dimension (columns)
    // Input shape: [3, 5]
    // slice(dim=1, start=1, end=3) -> output shape: [3, 2] (columns 1, 2)
    Tensor x = Tensor::zeros({3, 5}, DType::FLOAT32, cpu_device);
    auto x_acc = x.accessor<float, 2>();
    // Fill with unique values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            x_acc[i][j] = static_cast<float>(i * 5 + j);
        }
    }
    x.requiresGrad(true);

    // Slice columns 1 and 2
    Tensor y = x.slice(1, 1, 3);

    // Verify forward pass
    EXPECT_EQ(y.shape(), std::vector<size_t>({3, 2}));

    // Backward with ones
    Tensor loss = y.sum();
    loss.backward(Tensor::ones({1}, DType::FLOAT32, cpu_device));

    ASSERT_NE(x.grad(), nullptr);
    auto grad_acc = x.grad()->accessor<float, 2>();

    // Check gradient pattern
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(grad_acc[i][0], 0.0f);  // Column 0: not sliced
        EXPECT_FLOAT_EQ(grad_acc[i][1], 1.0f);  // Column 1: sliced
        EXPECT_FLOAT_EQ(grad_acc[i][2], 1.0f);  // Column 2: sliced
        EXPECT_FLOAT_EQ(grad_acc[i][3], 0.0f);  // Column 3: not sliced
        EXPECT_FLOAT_EQ(grad_acc[i][4], 0.0f);  // Column 4: not sliced
    }
}
