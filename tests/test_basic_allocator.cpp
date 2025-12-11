#include <cstring>

#include "loom/device.h"
#include "loom/memory/basic_allocator.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// BasicAllocator Tests
// ============================================================================

class BasicAllocatorTest : public ::testing::Test {
  protected:
    Device cpu_device{DeviceType::CPU};
};

TEST_F(BasicAllocatorTest, CreateOnCPUDevice) {
    BasicAllocator alloc(cpu_device);
    EXPECT_EQ(alloc.device().type(), DeviceType::CPU);
}

TEST_F(BasicAllocatorTest, ThrowsOnGPUDevice) {
    Device gpu{DeviceType::CUDA, 0};
    EXPECT_THROW({ BasicAllocator alloc(gpu); }, std::runtime_error);
}

TEST_F(BasicAllocatorTest, AllocateAndDeallocate) {
    BasicAllocator alloc(cpu_device);

    void* ptr = alloc.allocate(1024);
    EXPECT_NE(ptr, nullptr);

    // Write to verify it's usable
    std::memset(ptr, 0xAB, 1024);

    // Verify we can read it back
    unsigned char* byte_ptr = static_cast<unsigned char*>(ptr);
    EXPECT_EQ(byte_ptr[0], 0xAB);
    EXPECT_EQ(byte_ptr[1023], 0xAB);

    alloc.deallocate(ptr);
    // No crash = success
}

TEST_F(BasicAllocatorTest, AllocateZeroBytes) {
    BasicAllocator alloc(cpu_device);

    void* ptr = alloc.allocate(0);
    // Behavior is implementation-defined, just verify no crash
    // Some implementations return nullptr, others return a valid pointer

    alloc.deallocate(ptr);
    // No crash = success
}

TEST_F(BasicAllocatorTest, AllocateLargeBlock) {
    BasicAllocator alloc(cpu_device);

    size_t large_size = 100 * 1024 * 1024;  // 100 MB
    void* ptr = alloc.allocate(large_size);
    EXPECT_NE(ptr, nullptr);

    // Write to first and last byte to verify it's usable
    unsigned char* byte_ptr = static_cast<unsigned char*>(ptr);
    byte_ptr[0] = 0xFF;
    byte_ptr[large_size - 1] = 0xFF;

    EXPECT_EQ(byte_ptr[0], 0xFF);
    EXPECT_EQ(byte_ptr[large_size - 1], 0xFF);

    alloc.deallocate(ptr);
}

TEST_F(BasicAllocatorTest, MultipleAllocations) {
    BasicAllocator alloc(cpu_device);

    // Allocate multiple blocks
    void* ptr1 = alloc.allocate(128);
    void* ptr2 = alloc.allocate(256);
    void* ptr3 = alloc.allocate(512);

    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);

    // Pointers should be different
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);

    // Deallocate in different order
    alloc.deallocate(ptr2);
    alloc.deallocate(ptr1);
    alloc.deallocate(ptr3);
}

TEST_F(BasicAllocatorTest, AllocateDeallocateAllocate) {
    BasicAllocator alloc(cpu_device);

    // Allocate, deallocate, then allocate again
    void* ptr1 = alloc.allocate(1024);
    EXPECT_NE(ptr1, nullptr);
    alloc.deallocate(ptr1);

    void* ptr2 = alloc.allocate(1024);
    EXPECT_NE(ptr2, nullptr);
    alloc.deallocate(ptr2);

    // No crashes = success
}

TEST_F(BasicAllocatorTest, DeviceMethodReturnsCPU) {
    BasicAllocator alloc(cpu_device);

    Device returned_device = alloc.device();
    EXPECT_EQ(returned_device.type(), DeviceType::CPU);
}
