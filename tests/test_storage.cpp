#include <cstring>

#include "common/device.h"
#include "common/dtypes.h"
#include "common/tensor/storage.h"
#include "common/type_traits.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// Test Fixture
// ============================================================================

class StorageTest : public ::testing::Test {
  protected:
    StorageTest() : cpu_device(DeviceType::CPU) {}

    Device cpu_device;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(StorageTest, ConstructWithValidParameters) {
    EXPECT_NO_THROW({ Storage storage(100, DType::FLOAT32, cpu_device); });
}

TEST_F(StorageTest, ConstructedStorageHasCorrectSize) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    EXPECT_EQ(storage.size(), 100);
}

TEST_F(StorageTest, ConstructedStorageHasCorrectDType) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    EXPECT_EQ(storage.dtype(), DType::FLOAT32);
}

TEST_F(StorageTest, ConstructedStorageHasCorrectDevice) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    EXPECT_EQ(storage.device().type(), DeviceType::CPU);
}

TEST_F(StorageTest, ConstructedStorageHasCorrectSizeInBytes) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    EXPECT_EQ(storage.sizeInBytes(), 100 * sizeof(float));
}

TEST_F(StorageTest, DataPointerIsNotNull) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    EXPECT_NE(storage.data(), nullptr);
}

TEST_F(StorageTest, ConstructWithDifferentDTypes) {
    Storage storage_float32(10, DType::FLOAT32, cpu_device);
    Storage storage_float64(10, DType::FLOAT64, cpu_device);
    Storage storage_int32(10, DType::INT32, cpu_device);

    EXPECT_EQ(storage_float32.sizeInBytes(), 10 * 4);
    EXPECT_EQ(storage_float64.sizeInBytes(), 10 * 8);
    EXPECT_EQ(storage_int32.sizeInBytes(), 10 * 4);
}

TEST_F(StorageTest, ConstructZeroSizeStorage) {
    // Edge case: zero size storage
    EXPECT_NO_THROW({ Storage storage(0, DType::FLOAT32, cpu_device); });
}

// ============================================================================
// Copy Constructor Tests (Shallow Copy)
// ============================================================================

TEST_F(StorageTest, CopyConstructorCreatesShallowCopy) {
    Storage storage1(100, DType::FLOAT32, cpu_device);

    // Write some data
    float* data1 = static_cast<float*>(storage1.data().get());
    data1[0] = 42.0f;

    // Copy construct
    Storage storage2(storage1);

    // Should share the same data
    float* data2 = static_cast<float*>(storage2.data().get());
    EXPECT_EQ(data2[0], 42.0f);

    // Modify through storage2
    data2[0] = 99.0f;

    // Should be reflected in storage1 (shared data)
    EXPECT_EQ(data1[0], 99.0f);
}

TEST_F(StorageTest, CopyConstructorSharesProperties) {
    Storage storage1(100, DType::FLOAT32, cpu_device);
    Storage storage2(storage1);

    EXPECT_EQ(storage1.size(), storage2.size());
    EXPECT_EQ(storage1.dtype(), storage2.dtype());
    EXPECT_EQ(storage1.device().type(), storage2.device().type());
    EXPECT_EQ(storage1.sizeInBytes(), storage2.sizeInBytes());
}

// ============================================================================
// Copy Assignment Tests
// ============================================================================

TEST_F(StorageTest, CopyAssignmentWorks) {
    Storage storage1(100, DType::FLOAT32, cpu_device);
    Storage storage2(50, DType::INT32, cpu_device);

    float* data1 = static_cast<float*>(storage1.data().get());
    data1[0] = 42.0f;

    storage2 = storage1;

    EXPECT_EQ(storage2.size(), 100);
    EXPECT_EQ(storage2.dtype(), DType::FLOAT32);

    float* data2 = static_cast<float*>(storage2.data().get());
    EXPECT_EQ(data2[0], 42.0f);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(StorageTest, MoveConstructorWorks) {
    Storage storage1(100, DType::FLOAT32, cpu_device);

    float* data1 = static_cast<float*>(storage1.data().get());
    data1[0] = 42.0f;

    Storage storage2(std::move(storage1));

    EXPECT_EQ(storage2.size(), 100);
    EXPECT_EQ(storage2.dtype(), DType::FLOAT32);

    float* data2 = static_cast<float*>(storage2.data().get());
    EXPECT_EQ(data2[0], 42.0f);
}

TEST_F(StorageTest, MoveAssignmentWorks) {
    Storage storage1(100, DType::FLOAT32, cpu_device);
    Storage storage2(50, DType::INT32, cpu_device);

    float* data1 = static_cast<float*>(storage1.data().get());
    data1[0] = 42.0f;

    storage2 = std::move(storage1);

    EXPECT_EQ(storage2.size(), 100);
    EXPECT_EQ(storage2.dtype(), DType::FLOAT32);

    float* data2 = static_cast<float*>(storage2.data().get());
    EXPECT_EQ(data2[0], 42.0f);
}

// ============================================================================
// Clone Tests (Deep Copy)
// ============================================================================

TEST_F(StorageTest, CloneCreatesIndependentCopy) {
    Storage storage1(100, DType::FLOAT32, cpu_device);

    // Write some data
    float* data1 = static_cast<float*>(storage1.data().get());
    for (size_t i = 0; i < 100; ++i) {
        data1[i] = static_cast<float>(i);
    }

    // Clone
    Storage storage2 = storage1.clone();

    // Verify data was copied
    float* data2 = static_cast<float*>(storage2.data().get());
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(data2[i], static_cast<float>(i));
    }

    // Modify storage2
    data2[0] = 999.0f;

    // storage1 should be unchanged
    EXPECT_EQ(data1[0], 0.0f);
    EXPECT_EQ(data2[0], 999.0f);
}

TEST_F(StorageTest, ClonePreservesProperties) {
    Storage storage1(100, DType::FLOAT64, cpu_device);
    Storage storage2 = storage1.clone();

    EXPECT_EQ(storage1.size(), storage2.size());
    EXPECT_EQ(storage1.dtype(), storage2.dtype());
    EXPECT_EQ(storage1.device().type(), storage2.device().type());
    EXPECT_EQ(storage1.sizeInBytes(), storage2.sizeInBytes());
}

TEST_F(StorageTest, CloneWithDifferentDTypes) {
    // Test cloning with various dtypes
    Storage storage_float(100, DType::FLOAT32, cpu_device);
    Storage storage_int(100, DType::INT32, cpu_device);
    Storage storage_byte(100, DType::UINT8, cpu_device);

    auto cloned_float = storage_float.clone();
    auto cloned_int = storage_int.clone();
    auto cloned_byte = storage_byte.clone();

    EXPECT_EQ(cloned_float.dtype(), DType::FLOAT32);
    EXPECT_EQ(cloned_int.dtype(), DType::INT32);
    EXPECT_EQ(cloned_byte.dtype(), DType::UINT8);
}

// ============================================================================
// Type-Safe Access Tests (as<T>() with concepts)
// ============================================================================

TEST_F(StorageTest, AsFloatWithFloat32Storage) {
    Storage storage(100, DType::FLOAT32, cpu_device);

    EXPECT_NO_THROW({
        auto ptr = storage.as<float>();
        EXPECT_NE(ptr, nullptr);
    });
}

TEST_F(StorageTest, AsDoubleWithFloat64Storage) {
    Storage storage(100, DType::FLOAT64, cpu_device);

    EXPECT_NO_THROW({
        auto ptr = storage.as<double>();
        EXPECT_NE(ptr, nullptr);
    });
}

TEST_F(StorageTest, AsInt32WithInt32Storage) {
    Storage storage(100, DType::INT32, cpu_device);

    EXPECT_NO_THROW({
        auto ptr = storage.as<int32_t>();
        EXPECT_NE(ptr, nullptr);
    });
}

TEST_F(StorageTest, AsThrowsOnTypeMismatch) {
    Storage storage(100, DType::FLOAT32, cpu_device);

    // Trying to access as double when storage is float32 should throw
    EXPECT_THROW({ auto ptr = storage.as<double>(); }, std::runtime_error);
}

TEST_F(StorageTest, AsTypeMismatchErrorMessage) {
    Storage storage(100, DType::FLOAT32, cpu_device);

    try {
        auto ptr = storage.as<int32_t>();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        // Error message should mention type mismatch
        EXPECT_NE(error_msg.find("mismatch"), std::string::npos);
    }
}

TEST_F(StorageTest, AsAllowsDataAccess) {
    Storage storage(10, DType::FLOAT32, cpu_device);

    auto ptr = storage.as<float>();
    float* raw = ptr.get();

    // Write data
    for (int i = 0; i < 10; ++i) {
        raw[i] = static_cast<float>(i * 2.5f);
    }

    // Read it back
    EXPECT_FLOAT_EQ(raw[0], 0.0f);
    EXPECT_FLOAT_EQ(raw[1], 2.5f);
    EXPECT_FLOAT_EQ(raw[9], 22.5f);
}

// ============================================================================
// Data Access Tests
// ============================================================================

TEST_F(StorageTest, DataReturnsValidPointer) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    auto data_ptr = storage.data();

    EXPECT_NE(data_ptr, nullptr);
    EXPECT_NE(data_ptr.get(), nullptr);
}

TEST_F(StorageTest, DataCanBeWrittenAndRead) {
    Storage storage(100, DType::FLOAT32, cpu_device);

    float* data = static_cast<float*>(storage.data().get());

    // Write
    data[0] = 1.0f;
    data[50] = 50.0f;
    data[99] = 99.0f;

    // Read back
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[50], 50.0f);
    EXPECT_FLOAT_EQ(data[99], 99.0f);
}

TEST_F(StorageTest, DataSharedBetweenCopies) {
    Storage storage1(100, DType::INT32, cpu_device);
    Storage storage2 = storage1;  // Shallow copy

    int32_t* data1 = static_cast<int32_t*>(storage1.data().get());
    int32_t* data2 = static_cast<int32_t*>(storage2.data().get());

    // Should point to same memory
    EXPECT_EQ(data1, data2);

    data1[0] = 42;
    EXPECT_EQ(data2[0], 42);
}

// ============================================================================
// Allocator Tests
// ============================================================================

TEST_F(StorageTest, AllocatorIsNotNull) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    auto allocator = storage.allocator();

    EXPECT_NE(allocator, nullptr);
}

TEST_F(StorageTest, AllocatorDeviceMatchesStorageDevice) {
    Storage storage(100, DType::FLOAT32, cpu_device);
    auto allocator = storage.allocator();

    EXPECT_EQ(allocator->device().type(), cpu_device.type());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(StorageTest, LargeAllocation) {
    // Allocate 10 MB
    size_t large_size = 10 * 1024 * 1024 / sizeof(float);

    EXPECT_NO_THROW({
        Storage storage(large_size, DType::FLOAT32, cpu_device);
        EXPECT_EQ(storage.size(), large_size);
    });
}

TEST_F(StorageTest, MultipleStoragesIndependent) {
    Storage storage1(100, DType::FLOAT32, cpu_device);
    Storage storage2(200, DType::FLOAT64, cpu_device);
    Storage storage3(50, DType::INT32, cpu_device);

    // All should be independent
    EXPECT_NE(storage1.data().get(), storage2.data().get());
    EXPECT_NE(storage1.data().get(), storage3.data().get());
    EXPECT_NE(storage2.data().get(), storage3.data().get());
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(StorageTest, StorageDestructionFreesMemory) {
    // This test verifies that storage cleans up properly
    // We can't directly test memory deallocation, but we can verify no crashes
    {
        Storage storage(1000, DType::FLOAT32, cpu_device);
        float* data = static_cast<float*>(storage.data().get());
        data[0] = 42.0f;
    }  // storage destroyed here

    // If we get here without crash, memory was handled correctly
    SUCCEED();
}

TEST_F(StorageTest, SharedPtrKeepsMemoryAlive) {
    auto data_ptr = []() {
        Storage storage(100, DType::FLOAT32, Device(DeviceType::CPU));
        float* data = static_cast<float*>(storage.data().get());
        data[0] = 42.0f;
        return storage.data();  // Return shared_ptr
    }();                        // storage destroyed, but data_ptr keeps memory alive

    // Memory should still be valid
    float* data = static_cast<float*>(data_ptr.get());
    EXPECT_FLOAT_EQ(data[0], 42.0f);
}

// ============================================================================
// Comprehensive Integration Test
// ============================================================================

TEST_F(StorageTest, ComprehensiveWorkflow) {
    // Create storage
    Storage storage(100, DType::FLOAT32, cpu_device);

    // Write data using type-safe accessor
    auto ptr = storage.as<float>();
    float* data = ptr.get();
    for (size_t i = 0; i < 100; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Create shallow copy
    Storage shallow_copy = storage;
    auto shallow_ptr = shallow_copy.as<float>();
    EXPECT_EQ(shallow_ptr.get()[50], 50.0f);

    // Create deep copy
    Storage deep_copy = storage.clone();
    auto deep_ptr = deep_copy.as<float>();
    EXPECT_EQ(deep_ptr.get()[50], 50.0f);

    // Modify deep copy
    deep_ptr.get()[50] = 999.0f;

    // Original and shallow copy should be unchanged
    EXPECT_EQ(data[50], 50.0f);
    EXPECT_EQ(shallow_ptr.get()[50], 50.0f);

    // Deep copy should be modified
    EXPECT_EQ(deep_ptr.get()[50], 999.0f);
}
