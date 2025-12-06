#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/logger.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// Test Fixture
// ============================================================================

class TensorTest : public ::testing::Test {
  protected:
    TensorTest() : cpu_device(DeviceType::CPU) {}

    void SetUp() override {
        // Pre-initialize logger to avoid race conditions
        auto& logger = Logger::getInstance("TensorTest");
        logger.info("Test fixture initialized");
    }

    Device cpu_device;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(TensorTest, ConstructWithValidShape) {
    EXPECT_NO_THROW({ Tensor tensor({2, 3}, DType::FLOAT32, cpu_device); });
}

TEST_F(TensorTest, ConstructedTensorHasCorrectShape) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.shape()[2], 4);
}

TEST_F(TensorTest, ConstructedTensorHasCorrectStrides) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    // Row-major strides: [12, 4, 1]
    EXPECT_EQ(tensor.stride()[0], 12);
    EXPECT_EQ(tensor.stride()[1], 4);
    EXPECT_EQ(tensor.stride()[2], 1);
}

TEST_F(TensorTest, ConstructedTensorHasZeroOffset) {
    Tensor tensor({2, 3}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.offset(), 0);
}

TEST_F(TensorTest, ConstructedTensorHasCorrectDType) {
    Tensor tensor({2, 3}, DType::FLOAT64, cpu_device);
    EXPECT_EQ(tensor.dtype(), DType::FLOAT64);
}

TEST_F(TensorTest, ConstructedTensorHasCorrectDevice) {
    Tensor tensor({2, 3}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.device().type(), DeviceType::CPU);
}

TEST_F(TensorTest, ConstructWithDifferentDTypes) {
    Tensor float32({2, 3}, DType::FLOAT32, cpu_device);
    Tensor int32({2, 3}, DType::INT32, cpu_device);
    Tensor uint8({2, 3}, DType::UINT8, cpu_device);

    EXPECT_EQ(float32.dtype(), DType::FLOAT32);
    EXPECT_EQ(int32.dtype(), DType::INT32);
    EXPECT_EQ(uint8.dtype(), DType::UINT8);
}

TEST_F(TensorTest, ConstructEmptyTensor) {
    Tensor tensor({0}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.numel(), 0);
}

// ============================================================================
// Static Factory Tests - Structural Properties
// ============================================================================

TEST_F(TensorTest, ZerosCreatesCorrectShape) {
    Tensor tensor = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.numel(), 6);
}

TEST_F(TensorTest, ZerosWorksWithDifferentDTypes) {
    Tensor float_tensor = Tensor::zeros({2, 2}, DType::FLOAT32, cpu_device);
    Tensor int_tensor = Tensor::zeros({2, 2}, DType::INT32, cpu_device);

    EXPECT_EQ(float_tensor.dtype(), DType::FLOAT32);
    EXPECT_EQ(int_tensor.dtype(), DType::INT32);
}

TEST_F(TensorTest, OnesCreatesCorrectShape) {
    Tensor tensor = Tensor::ones({3, 2}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 3);
    EXPECT_EQ(tensor.shape()[1], 2);
    EXPECT_EQ(tensor.numel(), 6);
}

TEST_F(TensorTest, OnesWorksWithIntegerTypes) {
    Tensor tensor = Tensor::ones({3, 3}, DType::INT32, cpu_device);
    EXPECT_EQ(tensor.dtype(), DType::INT32);
    EXPECT_EQ(tensor.numel(), 9);
}

TEST_F(TensorTest, FullCreatesCorrectShape) {
    Tensor tensor = Tensor::full({2, 2}, 5.0, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.numel(), 4);
}

TEST_F(TensorTest, RandCreatesCorrectShape) {
    Tensor tensor = Tensor::rand({10, 10}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.numel(), 100);
    EXPECT_EQ(tensor.shape()[0], 10);
    EXPECT_EQ(tensor.shape()[1], 10);
}

TEST_F(TensorTest, RandnCreatesCorrectShape) {
    Tensor tensor = Tensor::randn({5, 5}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.numel(), 25);
    EXPECT_EQ(tensor.shape()[0], 5);
    EXPECT_EQ(tensor.shape()[1], 5);
}

// ============================================================================
// In-Place Operation Tests (Behavior)
// ============================================================================

TEST_F(TensorTest, InPlaceOperationsReturnReference) {
    Tensor tensor({2, 2}, DType::FLOAT32, cpu_device);

    // Test method chaining
    Tensor& result1 = tensor.zero();
    Tensor& result2 = result1.one();
    Tensor& result3 = result2.fill(3.0);

    // All should reference the same object
    EXPECT_EQ(&result1, &tensor);
    EXPECT_EQ(&result2, &tensor);
    EXPECT_EQ(&result3, &tensor);
}

TEST_F(TensorTest, ZeroDoesNotChangeShape) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    tensor.zero();

    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.shape()[2], 4);
    EXPECT_EQ(tensor.numel(), 24);
}

TEST_F(TensorTest, OneDoesNotChangeDType) {
    Tensor tensor({2, 3}, DType::INT32, cpu_device);
    tensor.one();

    EXPECT_EQ(tensor.dtype(), DType::INT32);
}

TEST_F(TensorTest, FillDoesNotChangeProperties) {
    Tensor tensor({3, 3}, DType::FLOAT32, cpu_device);
    tensor.fill(2.5);

    EXPECT_EQ(tensor.numel(), 9);
    EXPECT_EQ(tensor.dtype(), DType::FLOAT32);
    EXPECT_TRUE(tensor.isContiguous());
}

TEST_F(TensorTest, UniformDoesNotChangeShape) {
    Tensor tensor({5, 5}, DType::FLOAT32, cpu_device);
    tensor.uniform(2.0, 5.0);

    EXPECT_EQ(tensor.numel(), 25);
    EXPECT_EQ(tensor.shape()[0], 5);
}

TEST_F(TensorTest, RandDoesNotChangeShape) {
    Tensor tensor({4, 4}, DType::FLOAT32, cpu_device);
    tensor.rand();

    EXPECT_EQ(tensor.numel(), 16);
    EXPECT_TRUE(tensor.isContiguous());
}

TEST_F(TensorTest, RandnDoesNotChangeShape) {
    Tensor tensor({3, 3}, DType::FLOAT32, cpu_device);
    tensor.randn();

    EXPECT_EQ(tensor.numel(), 9);
    EXPECT_EQ(tensor.dtype(), DType::FLOAT32);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST_F(TensorTest, NumelReturnsCorrectElementCount) {
    Tensor tensor1({2, 3}, DType::FLOAT32, cpu_device);
    Tensor tensor2({2, 3, 4}, DType::FLOAT32, cpu_device);

    EXPECT_EQ(tensor1.numel(), 6);
    EXPECT_EQ(tensor2.numel(), 24);
}

TEST_F(TensorTest, SizeReturnsDimensionSize) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);

    EXPECT_EQ(tensor.size(0), 2);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_EQ(tensor.size(2), 4);
}

TEST_F(TensorTest, NdimReturnsNumberOfDimensions) {
    Tensor tensor1({2}, DType::FLOAT32, cpu_device);
    Tensor tensor2({2, 3}, DType::FLOAT32, cpu_device);
    Tensor tensor3({2, 3, 4}, DType::FLOAT32, cpu_device);

    EXPECT_EQ(tensor1.ndim(), 1);
    EXPECT_EQ(tensor2.ndim(), 2);
    EXPECT_EQ(tensor3.ndim(), 3);
}

TEST_F(TensorTest, ShapeReturnsCorrectVector) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    const auto& shape = tensor.shape();

    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

TEST_F(TensorTest, StrideReturnsCorrectVector) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    const auto& stride = tensor.stride();

    EXPECT_EQ(stride.size(), 3);
    EXPECT_EQ(stride[0], 12);
    EXPECT_EQ(stride[1], 4);
    EXPECT_EQ(stride[2], 1);
}

// ============================================================================
// isContiguous Tests
// ============================================================================

TEST_F(TensorTest, NewTensorIsContiguous) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    EXPECT_TRUE(tensor.isContiguous());
}

TEST_F(TensorTest, EmptyTensorIsContiguous) {
    Tensor tensor({0}, DType::FLOAT32, cpu_device);
    EXPECT_TRUE(tensor.isContiguous());
}

TEST_F(TensorTest, SingleElementTensorIsContiguous) {
    Tensor tensor({1}, DType::FLOAT32, cpu_device);
    EXPECT_TRUE(tensor.isContiguous());
}

TEST_F(TensorTest, TensorRemainsContiguousAfterInPlaceOperations) {
    Tensor tensor({2, 3}, DType::FLOAT32, cpu_device);
    tensor.zero().one().fill(5.0);

    EXPECT_TRUE(tensor.isContiguous());
}

// ============================================================================
// Clone Tests
// ============================================================================

TEST_F(TensorTest, ClonePreservesShape) {
    Tensor original({2, 3, 4}, DType::INT32, cpu_device);
    Tensor cloned = original.clone();

    EXPECT_EQ(cloned.shape(), original.shape());
    EXPECT_EQ(cloned.stride(), original.stride());
    EXPECT_EQ(cloned.dtype(), original.dtype());
    EXPECT_EQ(cloned.numel(), original.numel());
}

TEST_F(TensorTest, ClonePreservesDType) {
    Tensor original({2, 3}, DType::FLOAT64, cpu_device);
    Tensor cloned = original.clone();

    EXPECT_EQ(cloned.dtype(), DType::FLOAT64);
}

TEST_F(TensorTest, ClonePreservesDevice) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    Tensor cloned = original.clone();

    EXPECT_EQ(cloned.device().type(), DeviceType::CPU);
}

TEST_F(TensorTest, CloneIsIndependent) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    original.fill(5.0);

    Tensor cloned = original.clone();

    // Modify clone - should not affect original
    cloned.fill(10.0);

    // Both should maintain their properties independently
    EXPECT_EQ(original.numel(), 6);
    EXPECT_EQ(cloned.numel(), 6);
    EXPECT_EQ(original.dtype(), DType::FLOAT32);
    EXPECT_EQ(cloned.dtype(), DType::FLOAT32);
}

// ============================================================================
// Contiguous Tests
// ============================================================================

TEST_F(TensorTest, ContiguousOnContiguousTensorPreservesProperties) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    Tensor result = original.contiguous();

    EXPECT_EQ(result.shape(), original.shape());
    EXPECT_TRUE(result.isContiguous());
    EXPECT_EQ(result.dtype(), original.dtype());
}

TEST_F(TensorTest, ContiguousReturnsCorrectShape) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor result = tensor.contiguous();

    EXPECT_EQ(result.shape(), tensor.shape());
    EXPECT_TRUE(result.isContiguous());
}

// ============================================================================
// toDevice Tests
// ============================================================================

TEST_F(TensorTest, ToDeviceSameDevicePreservesProperties) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    Tensor result = original.toDevice(cpu_device);

    EXPECT_EQ(result.shape(), original.shape());
    EXPECT_EQ(result.dtype(), original.dtype());
    EXPECT_EQ(result.device().type(), DeviceType::CPU);
}

// ============================================================================
// Copy Semantics Tests
// ============================================================================

TEST_F(TensorTest, CopyConstructorPreservesShape) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    Tensor copied(original);

    EXPECT_EQ(copied.shape(), original.shape());
    EXPECT_EQ(copied.dtype(), original.dtype());
    EXPECT_EQ(copied.numel(), original.numel());
}

TEST_F(TensorTest, CopyAssignmentPreservesShape) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    Tensor assigned({1, 1}, DType::INT32, cpu_device);

    assigned = original;

    EXPECT_EQ(assigned.shape(), original.shape());
    EXPECT_EQ(assigned.dtype(), original.dtype());
}

TEST_F(TensorTest, MoveConstructorPreservesProperties) {
    Tensor original({2, 3}, DType::FLOAT32, cpu_device);
    size_t expected_numel = original.numel();

    Tensor moved(std::move(original));

    EXPECT_EQ(moved.numel(), expected_numel);
    EXPECT_EQ(moved.shape()[0], 2);
    EXPECT_EQ(moved.shape()[1], 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TensorTest, OneDimensionalTensor) {
    Tensor tensor({10}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.ndim(), 1);
    EXPECT_EQ(tensor.numel(), 10);
    EXPECT_EQ(tensor.stride()[0], 1);
}

TEST_F(TensorTest, HighDimensionalTensor) {
    Tensor tensor({2, 3, 4, 5, 6}, DType::FLOAT32, cpu_device);
    EXPECT_EQ(tensor.ndim(), 5);
    EXPECT_EQ(tensor.numel(), 720);
}

TEST_F(TensorTest, LargeTensor) {
    // Test allocation of larger tensor (1MB)
    size_t size = 256 * 1024;  // 256K floats = 1MB
    EXPECT_NO_THROW({
        Tensor tensor({size}, DType::FLOAT32, cpu_device);
        EXPECT_EQ(tensor.numel(), size);
    });
}

TEST_F(TensorTest, MultipleTensorsDontInterfere) {
    Tensor t1({2, 3}, DType::FLOAT32, cpu_device);
    Tensor t2({4, 5}, DType::INT32, cpu_device);
    Tensor t3({3, 3}, DType::FLOAT64, cpu_device);

    t1.fill(1.0);
    t2.fill(2.0);
    t3.fill(3.0);

    // Each tensor should maintain its properties
    EXPECT_EQ(t1.numel(), 6);
    EXPECT_EQ(t2.numel(), 20);
    EXPECT_EQ(t3.numel(), 9);

    EXPECT_EQ(t1.dtype(), DType::FLOAT32);
    EXPECT_EQ(t2.dtype(), DType::INT32);
    EXPECT_EQ(t3.dtype(), DType::FLOAT64);
}

TEST_F(TensorTest, ChainedOperations) {
    Tensor tensor({3, 3}, DType::FLOAT32, cpu_device);

    // Chain multiple operations
    tensor.zero().one().fill(5.0).uniform(-1.0, 1.0).rand().randn();

    // Tensor should still be valid
    EXPECT_EQ(tensor.numel(), 9);
    EXPECT_TRUE(tensor.isContiguous());
    EXPECT_EQ(tensor.dtype(), DType::FLOAT32);
}

// ============================================================================
// Arithmetic Operations Tests
// ============================================================================

// Tensor-Tensor In-Place Operations
TEST_F(TensorTest, TensorInPlaceAddition) {
    Tensor t1 = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    Tensor& result = t1 += t2;

    EXPECT_EQ(&result, &t1);  // Should return reference to t1
    EXPECT_EQ(t1.shape(), t2.shape());
}

TEST_F(TensorTest, TensorInPlaceSubtraction) {
    Tensor t1 = Tensor::full({2, 3}, 5.0, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    t1 -= t2;

    EXPECT_EQ(t1.numel(), 6);
    EXPECT_EQ(t1.shape(), t2.shape());
}

TEST_F(TensorTest, TensorInPlaceMultiplication) {
    Tensor t1 = Tensor::full({2, 3}, 3.0, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    t1 *= t2;

    EXPECT_EQ(t1.numel(), 6);
}

TEST_F(TensorTest, TensorInPlaceDivision) {
    Tensor t1 = Tensor::full({2, 3}, 6.0, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    t1 /= t2;

    EXPECT_EQ(t1.numel(), 6);
}

// Tensor-Tensor Out-of-Place Operations
TEST_F(TensorTest, TensorOutOfPlaceAddition) {
    Tensor t1 = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    Tensor result = t1 + t2;

    // Original tensors should be unchanged
    EXPECT_EQ(t1.numel(), 6);
    EXPECT_EQ(t2.numel(), 6);
    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.shape(), t1.shape());
}

TEST_F(TensorTest, TensorOutOfPlaceSubtraction) {
    Tensor t1 = Tensor::full({2, 3}, 5.0, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    Tensor result = t1 - t2;

    EXPECT_EQ(result.shape(), t1.shape());
}

TEST_F(TensorTest, TensorOutOfPlaceMultiplication) {
    Tensor t1 = Tensor::full({2, 3}, 3.0, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    Tensor result = t1 * t2;

    EXPECT_EQ(result.numel(), 6);
}

TEST_F(TensorTest, TensorOutOfPlaceDivision) {
    Tensor t1 = Tensor::full({2, 3}, 6.0, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    Tensor result = t1 / t2;

    EXPECT_EQ(result.shape(), t1.shape());
}

// Tensor-Scalar In-Place Operations
TEST_F(TensorTest, ScalarInPlaceAddition) {
    Tensor tensor = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);

    Tensor& result = tensor += 5.0;

    EXPECT_EQ(&result, &tensor);
    EXPECT_EQ(tensor.numel(), 6);
}

TEST_F(TensorTest, ScalarInPlaceSubtraction) {
    Tensor tensor = Tensor::full({2, 3}, 10.0, DType::FLOAT32, cpu_device);

    tensor -= 3.0;

    EXPECT_EQ(tensor.numel(), 6);
}

TEST_F(TensorTest, ScalarInPlaceMultiplication) {
    Tensor tensor = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    tensor *= 3.0;

    EXPECT_EQ(tensor.numel(), 6);
}

TEST_F(TensorTest, ScalarInPlaceDivision) {
    Tensor tensor = Tensor::full({2, 3}, 6.0, DType::FLOAT32, cpu_device);

    tensor /= 2.0;

    EXPECT_EQ(tensor.numel(), 6);
}

// Tensor-Scalar Out-of-Place Operations
TEST_F(TensorTest, ScalarOutOfPlaceAddition) {
    Tensor tensor = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);

    Tensor result = tensor + 5.0;

    EXPECT_EQ(result.shape(), tensor.shape());
    EXPECT_EQ(tensor.numel(), 6);  // Original unchanged
}

TEST_F(TensorTest, ScalarOutOfPlaceSubtraction) {
    Tensor tensor = Tensor::full({2, 3}, 10.0, DType::FLOAT32, cpu_device);

    Tensor result = tensor - 3.0;

    EXPECT_EQ(result.numel(), 6);
}

TEST_F(TensorTest, ScalarOutOfPlaceMultiplication) {
    Tensor tensor = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    Tensor result = tensor * 3.0;

    EXPECT_EQ(result.shape(), tensor.shape());
}

TEST_F(TensorTest, ScalarOutOfPlaceDivision) {
    Tensor tensor = Tensor::full({2, 3}, 6.0, DType::FLOAT32, cpu_device);

    Tensor result = tensor / 2.0;

    EXPECT_EQ(result.numel(), 6);
}

// Error Handling Tests
TEST_F(TensorTest, ArithmeticThrowsOnShapeMismatch) {
    Tensor t1({2, 3}, DType::FLOAT32, cpu_device);
    Tensor t2({3, 2}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ t1 += t2; }, std::runtime_error);
    EXPECT_THROW({ t1 + t2; }, std::runtime_error);
}

TEST_F(TensorTest, ArithmeticThrowsOnDTypeMismatch) {
    Tensor t1({2, 3}, DType::FLOAT32, cpu_device);
    Tensor t2({2, 3}, DType::INT32, cpu_device);

    EXPECT_THROW({ t1 += t2; }, std::runtime_error);
    EXPECT_THROW({ t1 + t2; }, std::runtime_error);
}

// Chaining Tests
TEST_F(TensorTest, InPlaceArithmeticChaining) {
    Tensor tensor = Tensor::zeros({2, 3}, DType::FLOAT32, cpu_device);

    // Chain multiple in-place operations
    tensor += 1.0;
    tensor *= 2.0;
    tensor -= 0.5;
    tensor /= 1.5;

    EXPECT_EQ(tensor.numel(), 6);
    EXPECT_TRUE(tensor.isContiguous());
}

TEST_F(TensorTest, MixedArithmeticOperations) {
    Tensor t1 = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor t2 = Tensor::full({2, 3}, 2.0, DType::FLOAT32, cpu_device);

    // Mix tensor-tensor and tensor-scalar operations
    Tensor result = (t1 + t2) * 3.0;

    EXPECT_EQ(result.shape(), t1.shape());
}

// Different DTypes
TEST_F(TensorTest, ArithmeticWorksWithIntegerTypes) {
    Tensor t1 = Tensor::ones({2, 3}, DType::INT32, cpu_device);
    Tensor t2 = Tensor::ones({2, 3}, DType::INT32, cpu_device);

    t1 += t2;
    t1 += 5.0;  // Should truncate to 5

    EXPECT_EQ(t1.dtype(), DType::INT32);
    EXPECT_EQ(t1.numel(), 6);
}

TEST_F(TensorTest, ArithmeticPreservesContiguity) {
    Tensor tensor = Tensor::ones({2, 3, 4}, DType::FLOAT32, cpu_device);

    tensor += 1.0;
    EXPECT_TRUE(tensor.isContiguous());

    tensor *= Tensor::ones({2, 3, 4}, DType::FLOAT32, cpu_device);
    EXPECT_TRUE(tensor.isContiguous());
}

// ============================================================================
// View Operations Tests
// ============================================================================

// Flatten Tests
TEST_F(TensorTest, FlattenCreatesOneDimensionalView) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor flat = tensor.flatten();

    EXPECT_EQ(flat.ndim(), 1);
    EXPECT_EQ(flat.shape()[0], 24);
    EXPECT_EQ(flat.numel(), 24);
}

TEST_F(TensorTest, FlattenSharesStorage) {
    Tensor tensor = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor flat = tensor.flatten();

    // Modify flattened view
    flat.fill(5.0);

    // Original should be affected (shared storage)
    EXPECT_EQ(tensor.numel(), 6);
}

// Reshape Tests
TEST_F(TensorTest, ReshapeChangesShape) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor reshaped = tensor.reshape({6, 4});

    EXPECT_EQ(reshaped.shape()[0], 6);
    EXPECT_EQ(reshaped.shape()[1], 4);
    EXPECT_EQ(reshaped.numel(), 24);
}

TEST_F(TensorTest, ReshapePreservesElementCount) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor reshaped = tensor.reshape({8, 3});

    EXPECT_EQ(reshaped.numel(), tensor.numel());
}

TEST_F(TensorTest, ReshapeThrowsOnInvalidShape) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);

    // Wrong element count
    EXPECT_THROW({ tensor.reshape({5, 5}); }, std::runtime_error);
}

TEST_F(TensorTest, ReshapeRequiresContiguousTensor) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor transposed = tensor.transpose();

    // Transposed tensor is not contiguous
    EXPECT_FALSE(transposed.isContiguous());
    EXPECT_THROW({ transposed.reshape({4, 3, 2}); }, std::runtime_error);
}

// Squeeze Tests
TEST_F(TensorTest, SqueezeRemovesAllSingletonDimensions) {
    Tensor tensor({2, 1, 3, 1, 4}, DType::FLOAT32, cpu_device);
    Tensor squeezed = tensor.squeeze();

    EXPECT_EQ(squeezed.ndim(), 3);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
    EXPECT_EQ(squeezed.shape()[2], 4);
}

TEST_F(TensorTest, SqueezePreservesStrides) {
    Tensor tensor({2, 1, 3, 4}, DType::FLOAT32, cpu_device);
    // Original strides: [12, 12, 4, 1]

    Tensor squeezed = tensor.squeeze();
    // Should preserve non-singleton strides: [12, 4, 1]

    EXPECT_EQ(squeezed.stride()[0], 12);
    EXPECT_EQ(squeezed.stride()[1], 4);
    EXPECT_EQ(squeezed.stride()[2], 1);
}

TEST_F(TensorTest, SqueezeWithDimRemovesSpecificDimension) {
    Tensor tensor({2, 1, 3, 1, 4}, DType::FLOAT32, cpu_device);
    Tensor squeezed = tensor.squeeze(1);  // Remove only dim 1

    EXPECT_EQ(squeezed.ndim(), 4);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);  // dim 2 moved to position 1
    EXPECT_EQ(squeezed.shape()[2], 1);  // dim 3 still size-1
    EXPECT_EQ(squeezed.shape()[3], 4);
}

TEST_F(TensorTest, SqueezeThrowsOnNonSingletonDimension) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ tensor.squeeze(1); }, std::runtime_error);
}

TEST_F(TensorTest, SqueezeHandlesAllSingletonDimensions) {
    Tensor tensor({1, 1, 1}, DType::FLOAT32, cpu_device);
    Tensor squeezed = tensor.squeeze();

    // Should result in shape [1] not []
    EXPECT_EQ(squeezed.ndim(), 1);
    EXPECT_EQ(squeezed.shape()[0], 1);
}

// Unsqueeze Tests
TEST_F(TensorTest, UnsqueezeAddsNewDimension) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor unsqueezed = tensor.unsqueeze(1);

    EXPECT_EQ(unsqueezed.ndim(), 4);
    EXPECT_EQ(unsqueezed.shape()[0], 2);
    EXPECT_EQ(unsqueezed.shape()[1], 1);  // New dimension
    EXPECT_EQ(unsqueezed.shape()[2], 3);
    EXPECT_EQ(unsqueezed.shape()[3], 4);
}

TEST_F(TensorTest, UnsqueezeAtBeginning) {
    Tensor tensor({2, 3}, DType::FLOAT32, cpu_device);
    Tensor unsqueezed = tensor.unsqueeze(0);

    EXPECT_EQ(unsqueezed.shape()[0], 1);
    EXPECT_EQ(unsqueezed.shape()[1], 2);
    EXPECT_EQ(unsqueezed.shape()[2], 3);
}

TEST_F(TensorTest, UnsqueezeAtEnd) {
    Tensor tensor({2, 3}, DType::FLOAT32, cpu_device);
    Tensor unsqueezed = tensor.unsqueeze(2);

    EXPECT_EQ(unsqueezed.shape()[0], 2);
    EXPECT_EQ(unsqueezed.shape()[1], 3);
    EXPECT_EQ(unsqueezed.shape()[2], 1);
}

TEST_F(TensorTest, UnsqueezeNegativeIndexing) {
    Tensor tensor({2, 3}, DType::FLOAT32, cpu_device);
    Tensor unsqueezed = tensor.unsqueeze(-1);  // Same as unsqueeze(2)

    EXPECT_EQ(unsqueezed.ndim(), 3);
    EXPECT_EQ(unsqueezed.shape()[2], 1);
}

// Transpose Tests
TEST_F(TensorTest, TransposeSwapsLastTwoDimensions) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor transposed = tensor.transpose();

    EXPECT_EQ(transposed.shape()[0], 2);
    EXPECT_EQ(transposed.shape()[1], 4);  // Swapped
    EXPECT_EQ(transposed.shape()[2], 3);  // Swapped
}

TEST_F(TensorTest, TransposeSwapsStrides) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    // Original strides: [12, 4, 1]

    Tensor transposed = tensor.transpose();
    // Should swap strides of dims 1 and 2: [12, 1, 4]

    EXPECT_EQ(transposed.stride()[0], 12);
    EXPECT_EQ(transposed.stride()[1], 1);
    EXPECT_EQ(transposed.stride()[2], 4);
}

TEST_F(TensorTest, TransposeWithDimsSwapsSpecificDimensions) {
    Tensor tensor({2, 3, 4, 5}, DType::FLOAT32, cpu_device);
    Tensor transposed = tensor.transpose(0, 2);

    EXPECT_EQ(transposed.shape()[0], 4);  // Was dim 2
    EXPECT_EQ(transposed.shape()[1], 3);
    EXPECT_EQ(transposed.shape()[2], 2);  // Was dim 0
    EXPECT_EQ(transposed.shape()[3], 5);
}

TEST_F(TensorTest, TransposeCreatesNonContiguousTensor) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    EXPECT_TRUE(tensor.isContiguous());

    Tensor transposed = tensor.transpose();
    EXPECT_FALSE(transposed.isContiguous());
}

TEST_F(TensorTest, TransposeNegativeIndexing) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor t1 = tensor.transpose(-2, -1);  // Last two dims
    Tensor t2 = tensor.transpose();

    EXPECT_EQ(t1.shape(), t2.shape());
}

// Permute Tests
TEST_F(TensorTest, PermuteReordersDimensions) {
    Tensor tensor({2, 3, 4, 5}, DType::FLOAT32, cpu_device);
    Tensor permuted = tensor.permute({3, 0, 2, 1});

    EXPECT_EQ(permuted.shape()[0], 5);  // Was dim 3
    EXPECT_EQ(permuted.shape()[1], 2);  // Was dim 0
    EXPECT_EQ(permuted.shape()[2], 4);  // Was dim 2
    EXPECT_EQ(permuted.shape()[3], 3);  // Was dim 1
}

TEST_F(TensorTest, PermuteReordersStrides) {
    Tensor tensor({2, 3, 4, 5}, DType::FLOAT32, cpu_device);
    // Original strides: [60, 20, 5, 1]

    Tensor permuted = tensor.permute({3, 0, 2, 1});
    // Should reorder strides: [1, 60, 5, 20]

    EXPECT_EQ(permuted.stride()[0], 1);
    EXPECT_EQ(permuted.stride()[1], 60);
    EXPECT_EQ(permuted.stride()[2], 5);
    EXPECT_EQ(permuted.stride()[3], 20);
}

TEST_F(TensorTest, PermuteThrowsOnInvalidDimensionCount) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ tensor.permute({0, 1}); }, std::runtime_error);        // Too few
    EXPECT_THROW({ tensor.permute({0, 1, 2, 3}); }, std::runtime_error);  // Too many
}

TEST_F(TensorTest, PermuteThrowsOnInvalidPermutation) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ tensor.permute({0, 0, 2}); }, std::runtime_error);  // Duplicate
    EXPECT_THROW({ tensor.permute({0, 1, 5}); }, std::runtime_error);  // Out of range
}

TEST_F(TensorTest, PermuteWithNegativeIndexing) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor permuted = tensor.permute({-1, -2, -3});  // Reverse order

    EXPECT_EQ(permuted.shape()[0], 4);
    EXPECT_EQ(permuted.shape()[1], 3);
    EXPECT_EQ(permuted.shape()[2], 2);
}

// View Operations Share Storage
TEST_F(TensorTest, ViewOperationsShareStorage) {
    Tensor original = Tensor::ones({2, 3, 4}, DType::FLOAT32, cpu_device);

    Tensor squeezed = original.unsqueeze(0).squeeze(0);
    Tensor transposed = original.transpose();

    // All should share storage
    EXPECT_EQ(original.numel(), squeezed.numel());
    EXPECT_EQ(original.numel(), transposed.numel());
}

// Complex View Chaining
TEST_F(TensorTest, ChainedViewOperations) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);

    // Chain view operations (avoid flatten after transpose since that requires contiguous)
    Tensor result = tensor.unsqueeze(0).squeeze(0).unsqueeze(1).squeeze(1);

    // Should return to original shape
    EXPECT_EQ(result.shape(), tensor.shape());
    EXPECT_EQ(result.numel(), 24);
}

TEST_F(TensorTest, NonContiguousRequiresContiguousBeforeReshape) {
    Tensor tensor({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor transposed = tensor.transpose();

    EXPECT_FALSE(transposed.isContiguous());

    // Should work: make contiguous first, then reshape
    Tensor result = transposed.contiguous().reshape({4, 6});
    EXPECT_EQ(result.numel(), 24);
}

// ============================================================================
// Print Function Tests
// ============================================================================

TEST_F(TensorTest, PrintDisplaysTensorInformation) {
    Tensor tensor = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);

    // Should not throw
    EXPECT_NO_THROW({ tensor.print("Test Tensor"); });
}

TEST_F(TensorTest, PrintWorksWithDifferentShapes) {
    Tensor t1({5}, DType::FLOAT32, cpu_device);
    Tensor t2({2, 3}, DType::INT32, cpu_device);
    Tensor t3({2, 3, 4}, DType::FLOAT64, cpu_device);

    t1.fill(1.5);
    t2.fill(42);
    t3.fill(3.14159);

    EXPECT_NO_THROW({
        t1.print("1D Tensor");
        t2.print("2D Tensor");
        t3.print("3D Tensor");
    });
}

TEST_F(TensorTest, Print3DTensorAsMatrices) {
    // Create a small 3D tensor: 3 matrices of 2x3
    Tensor tensor = Tensor::randn({3, 2, 3}, DType::FLOAT32, cpu_device);

    // Should print as 3 separate 2x3 matrices
    EXPECT_NO_THROW({ tensor.print("3D Tensor (as separate matrices)"); });
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_F(TensorTest, BroadcastSameShape) {
    // Same shape - should use fast path but still work
    Tensor a = Tensor::full({2, 3}, 1.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({2, 3}, 2.0f, DType::FLOAT32, cpu_device);

    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_DOUBLE_EQ(c.sum().item(), 18.0);  // 6 elements * 3.0
}

TEST_F(TensorTest, BroadcastScalarTo2D) {
    // [1] + [2, 3] → [2, 3]
    Tensor a = Tensor::full({1}, 10.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({2, 3}, 1.0f, DType::FLOAT32, cpu_device);

    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_DOUBLE_EQ(c.sum().item(), 66.0);  // 6 elements * 11.0
}

TEST_F(TensorTest, BroadcastRowToMatrix) {
    // [3] + [2, 3] → [2, 3]
    Tensor a = Tensor::full({3}, 1.0f, DType::FLOAT32, cpu_device);
    auto acc_a = a.accessor<float, 1>();
    acc_a[0] = 1.0f;
    acc_a[1] = 2.0f;
    acc_a[2] = 3.0f;

    Tensor b = Tensor::full({2, 3}, 10.0f, DType::FLOAT32, cpu_device);

    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));

    auto acc_c = c.accessor<float, 2>();
    // Row 0: [1+10, 2+10, 3+10] = [11, 12, 13]
    EXPECT_FLOAT_EQ(acc_c[0][0], 11.0f);
    EXPECT_FLOAT_EQ(acc_c[0][1], 12.0f);
    EXPECT_FLOAT_EQ(acc_c[0][2], 13.0f);
    // Row 1: same values
    EXPECT_FLOAT_EQ(acc_c[1][0], 11.0f);
    EXPECT_FLOAT_EQ(acc_c[1][1], 12.0f);
    EXPECT_FLOAT_EQ(acc_c[1][2], 13.0f);
}

TEST_F(TensorTest, BroadcastColumnToMatrix) {
    // [2, 1] + [2, 3] → [2, 3]
    Tensor a = Tensor::full({2, 1}, 1.0f, DType::FLOAT32, cpu_device);
    auto acc_a = a.accessor<float, 2>();
    acc_a[0][0] = 100.0f;
    acc_a[1][0] = 200.0f;

    Tensor b = Tensor::full({2, 3}, 1.0f, DType::FLOAT32, cpu_device);

    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));

    auto acc_c = c.accessor<float, 2>();
    // Row 0: 100 + 1 = 101 for all columns
    EXPECT_FLOAT_EQ(acc_c[0][0], 101.0f);
    EXPECT_FLOAT_EQ(acc_c[0][1], 101.0f);
    EXPECT_FLOAT_EQ(acc_c[0][2], 101.0f);
    // Row 1: 200 + 1 = 201 for all columns
    EXPECT_FLOAT_EQ(acc_c[1][0], 201.0f);
    EXPECT_FLOAT_EQ(acc_c[1][1], 201.0f);
    EXPECT_FLOAT_EQ(acc_c[1][2], 201.0f);
}

TEST_F(TensorTest, BroadcastBothDimensions) {
    // [4, 1] + [3] → [4, 3]
    Tensor a = Tensor::full({4, 1}, 10.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3}, 0.0f, DType::FLOAT32, cpu_device);
    auto acc_b = b.accessor<float, 1>();
    acc_b[0] = 1.0f;
    acc_b[1] = 2.0f;
    acc_b[2] = 3.0f;

    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({4, 3}));
    EXPECT_EQ(c.numel(), 12u);

    auto acc_c = c.accessor<float, 2>();
    // Each row should be [11, 12, 13]
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(acc_c[i][0], 11.0f);
        EXPECT_FLOAT_EQ(acc_c[i][1], 12.0f);
        EXPECT_FLOAT_EQ(acc_c[i][2], 13.0f);
    }
}

TEST_F(TensorTest, BroadcastSubtraction) {
    Tensor a = Tensor::full({2, 3}, 10.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3}, 1.0f, DType::FLOAT32, cpu_device);

    Tensor c = a - b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_DOUBLE_EQ(c.sum().item(), 54.0);  // 6 * 9.0
}

TEST_F(TensorTest, BroadcastMultiplication) {
    Tensor a = Tensor::full({2, 3}, 5.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3}, 2.0f, DType::FLOAT32, cpu_device);

    Tensor c = a * b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_DOUBLE_EQ(c.sum().item(), 60.0);  // 6 * 10.0
}

TEST_F(TensorTest, BroadcastDivision) {
    Tensor a = Tensor::full({2, 3}, 10.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3}, 2.0f, DType::FLOAT32, cpu_device);

    Tensor c = a / b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_DOUBLE_EQ(c.sum().item(), 30.0);  // 6 * 5.0
}

TEST_F(TensorTest, Broadcast3D) {
    // [2, 1, 4] + [3, 4] → [2, 3, 4]
    Tensor a = Tensor::full({2, 1, 4}, 1.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3, 4}, 2.0f, DType::FLOAT32, cpu_device);

    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3, 4}));
    EXPECT_EQ(c.numel(), 24u);
    EXPECT_DOUBLE_EQ(c.sum().item(), 72.0);  // 24 * 3.0
}

TEST_F(TensorTest, BroadcastIncompatibleShapesThrows) {
    Tensor a = Tensor::full({2, 3}, 1.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({4}, 1.0f, DType::FLOAT32, cpu_device);  // 4 != 3

    EXPECT_THROW({ auto c = a + b; }, std::runtime_error);
}

TEST_F(TensorTest, BroadcastDTypeMismatchThrows) {
    Tensor a = Tensor::full({2, 3}, 1.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3}, 1.0, DType::FLOAT64, cpu_device);

    EXPECT_THROW({ auto c = a + b; }, std::runtime_error);
}

// ============================================================================
// Dot Product Tests
// ============================================================================

TEST_F(TensorTest, DotProductBasic) {
    Tensor a = Tensor::full({4}, 0.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({4}, 0.0f, DType::FLOAT32, cpu_device);

    auto acc_a = a.accessor<float, 1>();
    auto acc_b = b.accessor<float, 1>();

    // a = [1, 2, 3, 4]
    acc_a[0] = 1.0f;
    acc_a[1] = 2.0f;
    acc_a[2] = 3.0f;
    acc_a[3] = 4.0f;

    // b = [2, 3, 4, 5]
    acc_b[0] = 2.0f;
    acc_b[1] = 3.0f;
    acc_b[2] = 4.0f;
    acc_b[3] = 5.0f;

    Tensor c = a.dot(b);

    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    EXPECT_EQ(c.shape(), std::vector<size_t>({1}));
    EXPECT_DOUBLE_EQ(c.item(), 40.0);
}

TEST_F(TensorTest, DotProductZeros) {
    Tensor a = Tensor::zeros({5}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({5}, DType::FLOAT32, cpu_device);

    Tensor c = a.dot(b);
    EXPECT_DOUBLE_EQ(c.item(), 0.0);
}

TEST_F(TensorTest, DotProductOnes) {
    Tensor a = Tensor::ones({10}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({10}, DType::FLOAT32, cpu_device);

    Tensor c = a.dot(b);
    // 1*1 + 1*1 + ... (10 times) = 10
    EXPECT_DOUBLE_EQ(c.item(), 10.0);
}

TEST_F(TensorTest, DotProductSingleElement) {
    Tensor a = Tensor::full({1}, 3.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({1}, 4.0f, DType::FLOAT32, cpu_device);

    Tensor c = a.dot(b);
    EXPECT_DOUBLE_EQ(c.item(), 12.0);
}

TEST_F(TensorTest, DotProductNegativeValues) {
    Tensor a = Tensor::full({3}, 0.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3}, 0.0f, DType::FLOAT32, cpu_device);

    auto acc_a = a.accessor<float, 1>();
    auto acc_b = b.accessor<float, 1>();

    acc_a[0] = -1.0f;
    acc_a[1] = 2.0f;
    acc_a[2] = -3.0f;

    acc_b[0] = 4.0f;
    acc_b[1] = -5.0f;
    acc_b[2] = 6.0f;

    Tensor c = a.dot(b);
    // (-1)*4 + 2*(-5) + (-3)*6 = -4 - 10 - 18 = -32
    EXPECT_DOUBLE_EQ(c.item(), -32.0);
}

TEST_F(TensorTest, DotProductDifferentSizesThrows) {
    Tensor a = Tensor::ones({5}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({3}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ a.dot(b); }, std::runtime_error);
}

TEST_F(TensorTest, DotProductNon1DThrows) {
    Tensor a = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ a.dot(b); }, std::runtime_error);
}

TEST_F(TensorTest, DotProductDTypeMismatchThrows) {
    Tensor a = Tensor::ones({5}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({5}, DType::FLOAT64, cpu_device);

    EXPECT_THROW({ a.dot(b); }, std::runtime_error);
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST_F(TensorTest, MatmulBasic) {
    // [2, 3] × [3, 2] → [2, 2]
    Tensor a = Tensor::full({2, 3}, 0.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::full({3, 2}, 0.0f, DType::FLOAT32, cpu_device);

    auto acc_a = a.accessor<float, 2>();
    auto acc_b = b.accessor<float, 2>();

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    acc_a[0][0] = 1;
    acc_a[0][1] = 2;
    acc_a[0][2] = 3;
    acc_a[1][0] = 4;
    acc_a[1][1] = 5;
    acc_a[1][2] = 6;

    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]
    acc_b[0][0] = 7;
    acc_b[0][1] = 8;
    acc_b[1][0] = 9;
    acc_b[1][1] = 10;
    acc_b[2][0] = 11;
    acc_b[2][1] = 12;

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 2}));

    auto acc_c = c.accessor<float, 2>();

    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    EXPECT_FLOAT_EQ(acc_c[0][0], 58.0f);
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    EXPECT_FLOAT_EQ(acc_c[0][1], 64.0f);
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    EXPECT_FLOAT_EQ(acc_c[1][0], 139.0f);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    EXPECT_FLOAT_EQ(acc_c[1][1], 154.0f);
}

TEST_F(TensorTest, MatmulIdentity) {
    // A × I = A
    Tensor a = Tensor::full({3, 3}, 0.0f, DType::FLOAT32, cpu_device);
    Tensor identity = Tensor::zeros({3, 3}, DType::FLOAT32, cpu_device);

    auto acc_a = a.accessor<float, 2>();
    auto acc_i = identity.accessor<float, 2>();

    // Fill A with values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            acc_a[i][j] = static_cast<float>(i * 3 + j + 1);
        }
    }

    // Identity matrix
    acc_i[0][0] = 1;
    acc_i[1][1] = 1;
    acc_i[2][2] = 1;

    Tensor c = a.matmul(identity);

    EXPECT_EQ(c.shape(), std::vector<size_t>({3, 3}));

    auto acc_c = c.accessor<float, 2>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(acc_c[i][j], acc_a[i][j]);
        }
    }
}

TEST_F(TensorTest, MatmulOnes) {
    // [2, 3] of ones × [3, 4] of ones → [2, 4] where each element = 3
    Tensor a = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({3, 4}, DType::FLOAT32, cpu_device);

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 4}));
    // Each element should be 3 (sum of 3 ones × ones)
    EXPECT_DOUBLE_EQ(c.sum().item(), 24.0);  // 8 elements × 3
}

TEST_F(TensorTest, MatmulVectorTimesMatrix) {
    // [1, 3] × [3, 2] → [1, 2] (row vector × matrix)
    Tensor a = Tensor::full({1, 3}, 0.0f, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({3, 2}, DType::FLOAT32, cpu_device);

    auto acc_a = a.accessor<float, 2>();
    acc_a[0][0] = 1;
    acc_a[0][1] = 2;
    acc_a[0][2] = 3;

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape(), std::vector<size_t>({1, 2}));

    auto acc_c = c.accessor<float, 2>();
    // Each column sums to 1+2+3 = 6
    EXPECT_FLOAT_EQ(acc_c[0][0], 6.0f);
    EXPECT_FLOAT_EQ(acc_c[0][1], 6.0f);
}

TEST_F(TensorTest, MatmulInnerDimensionMismatchThrows) {
    Tensor a = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({4, 2}, DType::FLOAT32, cpu_device);  // 4 != 3

    EXPECT_THROW({ a.matmul(b); }, std::runtime_error);
}

TEST_F(TensorTest, MatmulNon2DThrows) {
    Tensor a = Tensor::ones({2, 3, 4}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({4, 2}, DType::FLOAT32, cpu_device);

    EXPECT_THROW({ a.matmul(b); }, std::runtime_error);
}

TEST_F(TensorTest, MatmulDTypeMismatchThrows) {
    Tensor a = Tensor::ones({2, 3}, DType::FLOAT32, cpu_device);
    Tensor b = Tensor::ones({3, 2}, DType::FLOAT64, cpu_device);

    EXPECT_THROW({ a.matmul(b); }, std::runtime_error);
}

// ============================================================================
// Variance Tests
// ============================================================================

TEST_F(TensorTest, VarianceGlobal) {
    Device cpu_device{DeviceType::CPU};
    Tensor t = Tensor::full({3, 3}, 2.0, DType::FLOAT32, cpu_device);
    Tensor var = t.var();
    EXPECT_NEAR(var.item(), 0.0, 1e-6);  // All same values -> 0 variance
}

TEST_F(TensorTest, VarianceNonZero) {
    Device cpu_device{DeviceType::CPU};
    // Create tensor [1, 2, 3, 4, 5] - known variance
    Tensor t = Tensor::zeros({5}, DType::FLOAT32, cpu_device);
    auto acc = t.accessor<float, 1>();
    for (int i = 0; i < 5; ++i) {
        acc[i] = static_cast<float>(i + 1);
    }
    
    // Mean = 3.0, Variance = 2.0
    Tensor var = t.var();
    EXPECT_NEAR(var.item(), 2.0, 1e-5);
}
