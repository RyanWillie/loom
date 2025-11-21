#include "common/dtypes.h"
#include "common/type_traits.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// dtype_traits Tests
// ============================================================================

TEST(TypeTraitsTest, DTypeTraitsFloat) {
    EXPECT_EQ(dtype_traits<float>::value, DType::FLOAT32);
}

TEST(TypeTraitsTest, DTypeTraitsDouble) {
    EXPECT_EQ(dtype_traits<double>::value, DType::FLOAT64);
}

TEST(TypeTraitsTest, DTypeTraitsInt8) {
    EXPECT_EQ(dtype_traits<int8_t>::value, DType::INT8);
}

TEST(TypeTraitsTest, DTypeTraitsInt16) {
    EXPECT_EQ(dtype_traits<int16_t>::value, DType::INT16);
}

TEST(TypeTraitsTest, DTypeTraitsInt32) {
    EXPECT_EQ(dtype_traits<int32_t>::value, DType::INT32);
}

TEST(TypeTraitsTest, DTypeTraitsInt64) {
    EXPECT_EQ(dtype_traits<int64_t>::value, DType::INT64);
}

TEST(TypeTraitsTest, DTypeTraitsUInt8) {
    EXPECT_EQ(dtype_traits<uint8_t>::value, DType::UINT8);
}

TEST(TypeTraitsTest, DTypeTraitsUInt16) {
    EXPECT_EQ(dtype_traits<uint16_t>::value, DType::UINT16);
}

TEST(TypeTraitsTest, DTypeTraitsUInt32) {
    EXPECT_EQ(dtype_traits<uint32_t>::value, DType::UINT32);
}

TEST(TypeTraitsTest, DTypeTraitsUInt64) {
    EXPECT_EQ(dtype_traits<uint64_t>::value, DType::UINT64);
}

// ============================================================================
// HasDType Concept Tests (Compile-time)
// ============================================================================

// These tests verify that the HasDType concept works correctly at compile-time
// If these compile, the concept is working

TEST(TypeTraitsTest, HasDTypeFloat) {
    static_assert(HasDType<float>, "float should satisfy HasDType");
    SUCCEED();
}

TEST(TypeTraitsTest, HasDTypeDouble) {
    static_assert(HasDType<double>, "double should satisfy HasDType");
    SUCCEED();
}

TEST(TypeTraitsTest, HasDTypeInt32) {
    static_assert(HasDType<int32_t>, "int32_t should satisfy HasDType");
    SUCCEED();
}

TEST(TypeTraitsTest, HasDTypeAllSignedInts) {
    static_assert(HasDType<int8_t>, "int8_t should satisfy HasDType");
    static_assert(HasDType<int16_t>, "int16_t should satisfy HasDType");
    static_assert(HasDType<int32_t>, "int32_t should satisfy HasDType");
    static_assert(HasDType<int64_t>, "int64_t should satisfy HasDType");
    SUCCEED();
}

TEST(TypeTraitsTest, HasDTypeAllUnsignedInts) {
    static_assert(HasDType<uint8_t>, "uint8_t should satisfy HasDType");
    static_assert(HasDType<uint16_t>, "uint16_t should satisfy HasDType");
    static_assert(HasDType<uint32_t>, "uint32_t should satisfy HasDType");
    static_assert(HasDType<uint64_t>, "uint64_t should satisfy HasDType");
    SUCCEED();
}

// These should NOT satisfy HasDType (would fail at compile-time if used in constrained template)
TEST(TypeTraitsTest, UnsupportedTypesDoNotHaveDType) {
    static_assert(!HasDType<std::string>, "std::string should not satisfy HasDType");
    static_assert(!HasDType<void*>, "void* should not satisfy HasDType");
    static_assert(!HasDType<char>, "char should not satisfy HasDType");
    SUCCEED();
}

// ============================================================================
// TensorStorageType Concept Tests
// ============================================================================

TEST(TypeTraitsTest, TensorStorageTypeFloat) {
    static_assert(TensorStorageType<float>, "float should satisfy TensorStorageType");
    SUCCEED();
}

TEST(TypeTraitsTest, TensorStorageTypeDouble) {
    static_assert(TensorStorageType<double>, "double should satisfy TensorStorageType");
    SUCCEED();
}

TEST(TypeTraitsTest, TensorStorageTypeInt32) {
    static_assert(TensorStorageType<int32_t>, "int32_t should satisfy TensorStorageType");
    SUCCEED();
}

TEST(TypeTraitsTest, TensorStorageTypeAllInts) {
    static_assert(TensorStorageType<int8_t>);
    static_assert(TensorStorageType<int16_t>);
    static_assert(TensorStorageType<int32_t>);
    static_assert(TensorStorageType<int64_t>);
    static_assert(TensorStorageType<uint8_t>);
    static_assert(TensorStorageType<uint16_t>);
    static_assert(TensorStorageType<uint32_t>);
    static_assert(TensorStorageType<uint64_t>);
    SUCCEED();
}

// Test that const types are rejected
TEST(TypeTraitsTest, TensorStorageTypeRejectsConstTypes) {
    static_assert(!TensorStorageType<const float>,
                  "const float should not satisfy TensorStorageType");
    static_assert(!TensorStorageType<const int32_t>,
                  "const int32_t should not satisfy TensorStorageType");
    SUCCEED();
}

// Test that reference types are rejected
TEST(TypeTraitsTest, TensorStorageTypeRejectsReferenceTypes) {
    static_assert(!TensorStorageType<float&>, "float& should not satisfy TensorStorageType");
    static_assert(!TensorStorageType<int32_t&>, "int32_t& should not satisfy TensorStorageType");
    SUCCEED();
}

// Test that unsupported types are rejected
TEST(TypeTraitsTest, TensorStorageTypeRejectsUnsupportedTypes) {
    static_assert(!TensorStorageType<std::string>,
                  "std::string should not satisfy TensorStorageType");
    static_assert(!TensorStorageType<void*>, "void* should not satisfy TensorStorageType");
    static_assert(!TensorStorageType<std::vector<int>>,
                  "std::vector should not satisfy TensorStorageType");
    SUCCEED();
}

// ============================================================================
// Concept Usage in Templates
// ============================================================================

// Helper template function to test concept constraints
template <TensorStorageType T>
bool isValidTensorType() {
    return true;
}

TEST(TypeTraitsTest, ConceptConstrainedFunctionWorksWithFloat) {
    EXPECT_TRUE(isValidTensorType<float>());
}

TEST(TypeTraitsTest, ConceptConstrainedFunctionWorksWithInt32) {
    EXPECT_TRUE(isValidTensorType<int32_t>());
}

TEST(TypeTraitsTest, ConceptConstrainedFunctionWorksWithAllTypes) {
    EXPECT_TRUE(isValidTensorType<float>());
    EXPECT_TRUE(isValidTensorType<double>());
    EXPECT_TRUE(isValidTensorType<int8_t>());
    EXPECT_TRUE(isValidTensorType<int16_t>());
    EXPECT_TRUE(isValidTensorType<int32_t>());
    EXPECT_TRUE(isValidTensorType<int64_t>());
    EXPECT_TRUE(isValidTensorType<uint8_t>());
    EXPECT_TRUE(isValidTensorType<uint16_t>());
    EXPECT_TRUE(isValidTensorType<uint32_t>());
    EXPECT_TRUE(isValidTensorType<uint64_t>());
}

// These would fail to compile if uncommented (demonstrating compile-time safety):
// TEST(TypeTraitsTest, ConceptConstrainedFunctionRejectsString) {
//     EXPECT_TRUE(isValidTensorType<std::string>());  // Won't compile!
// }

// ============================================================================
// Type Properties Tests
// ============================================================================

TEST(TypeTraitsTest, AllTypesAreTriviallyCopyable) {
    static_assert(std::is_trivially_copyable_v<float>);
    static_assert(std::is_trivially_copyable_v<double>);
    static_assert(std::is_trivially_copyable_v<int32_t>);
    static_assert(std::is_trivially_copyable_v<uint8_t>);
    SUCCEED();
}

TEST(TypeTraitsTest, StringIsNotTriviallyCopyable) {
    static_assert(!std::is_trivially_copyable_v<std::string>);
    SUCCEED();
}

// ============================================================================
// Integration Test with DType
// ============================================================================

TEST(TypeTraitsTest, TypeTraitsMatchDTypeSizes) {
    // Verify that dtype_traits types match the sizes reported by sizeOf
    EXPECT_EQ(sizeof(float), sizeOf(dtype_traits<float>::value));
    EXPECT_EQ(sizeof(double), sizeOf(dtype_traits<double>::value));
    EXPECT_EQ(sizeof(int8_t), sizeOf(dtype_traits<int8_t>::value));
    EXPECT_EQ(sizeof(int16_t), sizeOf(dtype_traits<int16_t>::value));
    EXPECT_EQ(sizeof(int32_t), sizeOf(dtype_traits<int32_t>::value));
    EXPECT_EQ(sizeof(int64_t), sizeOf(dtype_traits<int64_t>::value));
    EXPECT_EQ(sizeof(uint8_t), sizeOf(dtype_traits<uint8_t>::value));
    EXPECT_EQ(sizeof(uint16_t), sizeOf(dtype_traits<uint16_t>::value));
    EXPECT_EQ(sizeof(uint32_t), sizeOf(dtype_traits<uint32_t>::value));
    EXPECT_EQ(sizeof(uint64_t), sizeOf(dtype_traits<uint64_t>::value));
}

TEST(TypeTraitsTest, TypeTraitsMatchDTypeNames) {
    // Verify that dtype_traits correctly map to the right DType enum values
    EXPECT_STREQ(name(dtype_traits<float>::value), "float32");
    EXPECT_STREQ(name(dtype_traits<double>::value), "float64");
    EXPECT_STREQ(name(dtype_traits<int32_t>::value), "int32");
    EXPECT_STREQ(name(dtype_traits<uint8_t>::value), "uint8");
}
