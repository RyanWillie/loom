#include "common/dtypes.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// DType Size Tests
// ============================================================================

TEST(DTypesTest, SizeOfFloat32) {
    EXPECT_EQ(sizeOf(DType::FLOAT32), 4);
}

TEST(DTypesTest, SizeOfFloat64) {
    EXPECT_EQ(sizeOf(DType::FLOAT64), 8);
}

TEST(DTypesTest, SizeOfFloat16) {
    EXPECT_EQ(sizeOf(DType::FLOAT16), 2);
}

TEST(DTypesTest, SizeOfBFloat16) {
    EXPECT_EQ(sizeOf(DType::BFLOAT16), 2);
}

TEST(DTypesTest, SizeOfInt8) {
    EXPECT_EQ(sizeOf(DType::INT8), 1);
}

TEST(DTypesTest, SizeOfInt16) {
    EXPECT_EQ(sizeOf(DType::INT16), 2);
}

TEST(DTypesTest, SizeOfInt32) {
    EXPECT_EQ(sizeOf(DType::INT32), 4);
}

TEST(DTypesTest, SizeOfInt64) {
    EXPECT_EQ(sizeOf(DType::INT64), 8);
}

TEST(DTypesTest, SizeOfUInt8) {
    EXPECT_EQ(sizeOf(DType::UINT8), 1);
}

TEST(DTypesTest, SizeOfUInt16) {
    EXPECT_EQ(sizeOf(DType::UINT16), 2);
}

TEST(DTypesTest, SizeOfUInt32) {
    EXPECT_EQ(sizeOf(DType::UINT32), 4);
}

TEST(DTypesTest, SizeOfUInt64) {
    EXPECT_EQ(sizeOf(DType::UINT64), 8);
}

// ============================================================================
// DType Name Tests
// ============================================================================

TEST(DTypesTest, NameOfFloat32) {
    EXPECT_STREQ(name(DType::FLOAT32), "float32");
}

TEST(DTypesTest, NameOfFloat64) {
    EXPECT_STREQ(name(DType::FLOAT64), "float64");
}

TEST(DTypesTest, NameOfFloat16) {
    EXPECT_STREQ(name(DType::FLOAT16), "float16");
}

TEST(DTypesTest, NameOfBFloat16) {
    EXPECT_STREQ(name(DType::BFLOAT16), "bfloat16");
}

TEST(DTypesTest, NameOfInt8) {
    EXPECT_STREQ(name(DType::INT8), "int8");
}

TEST(DTypesTest, NameOfInt32) {
    EXPECT_STREQ(name(DType::INT32), "int32");
}

TEST(DTypesTest, NameOfUInt8) {
    EXPECT_STREQ(name(DType::UINT8), "uint8");
}

TEST(DTypesTest, NameOfUInt32) {
    EXPECT_STREQ(name(DType::UINT32), "uint32");
}

// ============================================================================
// Type Classification Tests
// ============================================================================

TEST(DTypesTest, IsIntegerForIntTypes) {
    EXPECT_TRUE(isInteger(DType::INT8));
    EXPECT_TRUE(isInteger(DType::INT16));
    EXPECT_TRUE(isInteger(DType::INT32));
    EXPECT_TRUE(isInteger(DType::INT64));
    EXPECT_TRUE(isInteger(DType::UINT8));
    EXPECT_TRUE(isInteger(DType::UINT16));
    EXPECT_TRUE(isInteger(DType::UINT32));
    EXPECT_TRUE(isInteger(DType::UINT64));
}

TEST(DTypesTest, IsIntegerForFloatTypes) {
    EXPECT_FALSE(isInteger(DType::FLOAT32));
    EXPECT_FALSE(isInteger(DType::FLOAT64));
    EXPECT_FALSE(isInteger(DType::FLOAT16));
    EXPECT_FALSE(isInteger(DType::BFLOAT16));
}

TEST(DTypesTest, IsFloatingPointForFloatTypes) {
    EXPECT_TRUE(isFloatingPoint(DType::FLOAT32));
    EXPECT_TRUE(isFloatingPoint(DType::FLOAT64));
    EXPECT_TRUE(isFloatingPoint(DType::FLOAT16));
    EXPECT_TRUE(isFloatingPoint(DType::BFLOAT16));
}

TEST(DTypesTest, IsFloatingPointForIntTypes) {
    EXPECT_FALSE(isFloatingPoint(DType::INT8));
    EXPECT_FALSE(isFloatingPoint(DType::INT32));
    EXPECT_FALSE(isFloatingPoint(DType::UINT8));
}

TEST(DTypesTest, IsUnsignedForUnsignedTypes) {
    EXPECT_TRUE(isUnsigned(DType::UINT8));
    EXPECT_TRUE(isUnsigned(DType::UINT16));
    EXPECT_TRUE(isUnsigned(DType::UINT32));
    EXPECT_TRUE(isUnsigned(DType::UINT64));
}

TEST(DTypesTest, IsUnsignedForSignedTypes) {
    EXPECT_FALSE(isUnsigned(DType::INT8));
    EXPECT_FALSE(isUnsigned(DType::INT16));
    EXPECT_FALSE(isUnsigned(DType::INT32));
    EXPECT_FALSE(isUnsigned(DType::INT64));
    EXPECT_FALSE(isUnsigned(DType::FLOAT32));
}

TEST(DTypesTest, IsSignedForSignedIntTypes) {
    EXPECT_TRUE(isSigned(DType::INT8));
    EXPECT_TRUE(isSigned(DType::INT16));
    EXPECT_TRUE(isSigned(DType::INT32));
    EXPECT_TRUE(isSigned(DType::INT64));
}

TEST(DTypesTest, IsSignedForUnsignedTypes) {
    EXPECT_FALSE(isSigned(DType::UINT8));
    EXPECT_FALSE(isSigned(DType::UINT16));
    EXPECT_FALSE(isSigned(DType::UINT32));
    EXPECT_FALSE(isSigned(DType::UINT64));
}

TEST(DTypesTest, IsSignedForFloatTypes) {
    EXPECT_FALSE(isSigned(DType::FLOAT32));
    EXPECT_FALSE(isSigned(DType::FLOAT64));
}

// ============================================================================
// Type Consistency Tests
// ============================================================================

TEST(DTypesTest, FloatTypesAreNotIntegers) {
    EXPECT_TRUE(isFloatingPoint(DType::FLOAT32));
    EXPECT_FALSE(isInteger(DType::FLOAT32));
}

TEST(DTypesTest, IntTypesAreNotFloats) {
    EXPECT_TRUE(isInteger(DType::INT32));
    EXPECT_FALSE(isFloatingPoint(DType::INT32));
}

TEST(DTypesTest, SignedAndUnsignedAreMutuallyExclusive) {
    EXPECT_TRUE(isSigned(DType::INT8));
    EXPECT_FALSE(isUnsigned(DType::INT8));

    EXPECT_TRUE(isUnsigned(DType::UINT8));
    EXPECT_FALSE(isSigned(DType::UINT8));
}
