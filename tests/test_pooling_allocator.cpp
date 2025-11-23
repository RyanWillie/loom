#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "cpu/pooling_allocator.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// Test Fixture
// ============================================================================

class PoolingAllocatorTest : public ::testing::Test {
  protected:
    PoolingAllocatorConfig config;

    void SetUp() override {
        config.alignment = 64;  // Standard cache line alignment
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(PoolingAllocatorTest, BasicAllocateAndDeallocate) {
    PoolingAllocator alloc(config);

    void* ptr = alloc.allocate(1024);
    ASSERT_NE(ptr, nullptr);

    // Verify alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % config.alignment, 0);

    // Should not crash
    alloc.deallocate(ptr);
}

TEST_F(PoolingAllocatorTest, AllocateMultipleSizes) {
    PoolingAllocator alloc(config);

    void* ptr64 = alloc.allocate(64);
    void* ptr128 = alloc.allocate(128);
    void* ptr1k = alloc.allocate(1024);
    void* ptr1m = alloc.allocate(1024 * 1024);

    ASSERT_NE(ptr64, nullptr);
    ASSERT_NE(ptr128, nullptr);
    ASSERT_NE(ptr1k, nullptr);
    ASSERT_NE(ptr1m, nullptr);

    // All should be different
    EXPECT_NE(ptr64, ptr128);
    EXPECT_NE(ptr128, ptr1k);
    EXPECT_NE(ptr1k, ptr1m);

    alloc.deallocate(ptr64);
    alloc.deallocate(ptr128);
    alloc.deallocate(ptr1k);
    alloc.deallocate(ptr1m);
}

TEST_F(PoolingAllocatorTest, DeviceType) {
    PoolingAllocator alloc(config);

    Device device = alloc.device();
    EXPECT_EQ(device.type(), DeviceType::CPU);
}

// ============================================================================
// Pool Reuse Tests
// ============================================================================

TEST_F(PoolingAllocatorTest, SimplePoolReuse) {
    PoolingAllocator alloc(config);

    // Allocate
    void* ptr1 = alloc.allocate(1024);
    ASSERT_NE(ptr1, nullptr);

    // Deallocate (goes to pool)
    alloc.deallocate(ptr1);

    // Allocate same size (should reuse from pool)
    void* ptr2 = alloc.allocate(1024);
    ASSERT_NE(ptr2, nullptr);

    // Should get the same pointer back (exact size match)
    EXPECT_EQ(ptr1, ptr2);

    alloc.deallocate(ptr2);
}

TEST_F(PoolingAllocatorTest, MultipleBlocksPoolReuse) {
    PoolingAllocator alloc(config);

    // Allocate multiple blocks of same size
    void* ptr1 = alloc.allocate(512);
    void* ptr2 = alloc.allocate(512);
    void* ptr3 = alloc.allocate(512);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);

    // All should be different (new allocations)
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);

    // Deallocate all (all go to pool)
    alloc.deallocate(ptr1);
    alloc.deallocate(ptr2);
    alloc.deallocate(ptr3);

    // Allocate again (should reuse from pool)
    void* ptr4 = alloc.allocate(512);
    void* ptr5 = alloc.allocate(512);
    void* ptr6 = alloc.allocate(512);

    ASSERT_NE(ptr4, nullptr);
    ASSERT_NE(ptr5, nullptr);
    ASSERT_NE(ptr6, nullptr);

    // Should reuse the original pointers (might be in different order)
    std::vector<void*> original = {ptr1, ptr2, ptr3};
    std::vector<void*> reused = {ptr4, ptr5, ptr6};
    std::sort(original.begin(), original.end());
    std::sort(reused.begin(), reused.end());
    EXPECT_EQ(original, reused);

    alloc.deallocate(ptr4);
    alloc.deallocate(ptr5);
    alloc.deallocate(ptr6);
}

TEST_F(PoolingAllocatorTest, DifferentSizesNoCrossReuse) {
    PoolingAllocator alloc(config);

    // Allocate different sizes
    void* ptr512 = alloc.allocate(512);
    void* ptr1024 = alloc.allocate(1024);

    ASSERT_NE(ptr512, nullptr);
    ASSERT_NE(ptr1024, nullptr);

    // Deallocate
    alloc.deallocate(ptr512);
    alloc.deallocate(ptr1024);

    // Allocate different size - should NOT reuse
    void* new_ptr256 = alloc.allocate(256);
    void* new_ptr2048 = alloc.allocate(2048);

    ASSERT_NE(new_ptr256, nullptr);
    ASSERT_NE(new_ptr2048, nullptr);

    // Should get new pointers (no exact size match)
    EXPECT_NE(new_ptr256, ptr512);
    EXPECT_NE(new_ptr256, ptr1024);
    EXPECT_NE(new_ptr2048, ptr512);
    EXPECT_NE(new_ptr2048, ptr1024);

    alloc.deallocate(new_ptr256);
    alloc.deallocate(new_ptr2048);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(PoolingAllocatorTest, AllocateZeroBytes) {
    PoolingAllocator alloc(config);

    // Zero-size allocations should return a valid sentinel pointer
    void* ptr = alloc.allocate(0);
    EXPECT_NE(ptr, nullptr);

    // Should be safely deallocatable
    EXPECT_NO_THROW(alloc.deallocate(ptr));
}

TEST_F(PoolingAllocatorTest, DeallocateNullptrThrows) {
    PoolingAllocator alloc(config);

    EXPECT_THROW(alloc.deallocate(nullptr), std::runtime_error);
}

// ============================================================================
// Alignment Tests
// ============================================================================

TEST_F(PoolingAllocatorTest, DefaultAlignmentDetection) {
    PoolingAllocatorConfig default_config;
    default_config.alignment = 0;  // Auto-detect

    PoolingAllocator alloc(default_config);

    void* ptr = alloc.allocate(128);
    ASSERT_NE(ptr, nullptr);

    // Should be aligned to at least 64 bytes (common cache line)
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 64, 0);

    alloc.deallocate(ptr);
}

TEST_F(PoolingAllocatorTest, RespectRequestedAlignment) {
    PoolingAllocatorConfig config128;
    config128.alignment = 128;

    PoolingAllocator alloc(config128);

    void* ptr = alloc.allocate(256);
    ASSERT_NE(ptr, nullptr);

    // Should be aligned to the requested alignment (128 bytes)
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 128, 0);

    alloc.deallocate(ptr);
}

TEST_F(PoolingAllocatorTest, AlignmentForSmallSizes) {
    PoolingAllocator alloc(config);

    // Even small allocations should be aligned
    for (size_t size = 1; size <= 64; size++) {
        void* ptr = alloc.allocate(size);
        ASSERT_NE(ptr, nullptr);

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % config.alignment, 0) << "Size " << size << " not aligned";

        alloc.deallocate(ptr);
    }
}

// ============================================================================
// Memory Pattern Tests (Verify we can read/write)
// ============================================================================

TEST_F(PoolingAllocatorTest, CanWriteAndReadMemory) {
    PoolingAllocator alloc(config);

    const size_t size = 1024;
    void* ptr = alloc.allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Write pattern
    unsigned char* data = static_cast<unsigned char*>(ptr);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<unsigned char>(i % 256);
    }

    // Verify pattern
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(data[i], static_cast<unsigned char>(i % 256))
            << "Memory corruption at byte " << i;
    }

    alloc.deallocate(ptr);
}

TEST_F(PoolingAllocatorTest, ReusedMemoryIsIndependent) {
    PoolingAllocator alloc(config);

    const size_t size = 512;

    // Allocate and write pattern
    void* ptr1 = alloc.allocate(size);
    ASSERT_NE(ptr1, nullptr);
    std::memset(ptr1, 0xAA, size);
    alloc.deallocate(ptr1);

    // Allocate again (should reuse)
    void* ptr2 = alloc.allocate(size);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_EQ(ptr1, ptr2);  // Should be same address

    // Old data should still be there (we don't zero memory)
    unsigned char* data = static_cast<unsigned char*>(ptr2);
    EXPECT_EQ(data[0], 0xAA);

    // Write new pattern
    std::memset(ptr2, 0x55, size);
    EXPECT_EQ(data[0], 0x55);

    alloc.deallocate(ptr2);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(PoolingAllocatorTest, ManyAllocationsAndDeallocations) {
    PoolingAllocator alloc(config);

    std::vector<void*> ptrs;
    const size_t num_allocs = 1000;

    // Allocate many blocks
    for (size_t i = 0; i < num_allocs; i++) {
        size_t size = 64 + (i % 10) * 128;  // 64, 192, 320, ...
        void* ptr = alloc.allocate(size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs) {
        alloc.deallocate(ptr);
    }
}

TEST_F(PoolingAllocatorTest, AlternatingAllocDealloc) {
    PoolingAllocator alloc(config);

    // Repeatedly allocate and deallocate same size
    for (int i = 0; i < 1000; i++) {
        void* ptr = alloc.allocate(1024);
        ASSERT_NE(ptr, nullptr);
        alloc.deallocate(ptr);
    }

    // Should work without issues (and pool should help performance)
}

TEST_F(PoolingAllocatorTest, RandomSizedAllocations) {
    PoolingAllocator alloc(config);

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> size_dist(1, 10000);

    std::vector<void*> allocated;

    // Random allocations and deallocations
    for (int i = 0; i < 500; i++) {
        if (allocated.empty() || gen() % 2 == 0) {
            // Allocate
            size_t size = size_dist(gen);
            void* ptr = alloc.allocate(size);
            ASSERT_NE(ptr, nullptr);
            allocated.push_back(ptr);
        } else {
            // Deallocate random element
            size_t idx = gen() % allocated.size();
            alloc.deallocate(allocated[idx]);
            allocated.erase(allocated.begin() + idx);
        }
    }

    // Clean up remaining
    for (void* ptr : allocated) {
        alloc.deallocate(ptr);
    }
}

// ============================================================================
// Large Allocation Tests
// ============================================================================

TEST_F(PoolingAllocatorTest, LargeAllocation) {
    PoolingAllocator alloc(config);

    // Allocate 10MB
    const size_t size = 10 * 1024 * 1024;
    void* ptr = alloc.allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Verify we can write to it
    std::memset(ptr, 0xFF, 1024);  // Write first KB

    alloc.deallocate(ptr);
}

TEST_F(PoolingAllocatorTest, MultipleLargeAllocations) {
    PoolingAllocator alloc(config);

    std::vector<void*> ptrs;

    // Allocate multiple 1MB blocks
    for (int i = 0; i < 10; i++) {
        void* ptr = alloc.allocate(1024 * 1024);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs) {
        alloc.deallocate(ptr);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(PoolingAllocatorTest, SingleByteAllocation) {
    PoolingAllocator alloc(config);

    void* ptr = alloc.allocate(1);
    ASSERT_NE(ptr, nullptr);

    // Should still be aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % config.alignment, 0);

    // Should be able to write
    *static_cast<unsigned char*>(ptr) = 42;
    EXPECT_EQ(*static_cast<unsigned char*>(ptr), 42);

    alloc.deallocate(ptr);
}

TEST_F(PoolingAllocatorTest, OddSizeAllocation) {
    PoolingAllocator alloc(config);

    // Odd sizes should work correctly
    std::vector<size_t> odd_sizes = {7, 13, 99, 1001, 4097};

    for (size_t size : odd_sizes) {
        void* ptr = alloc.allocate(size);
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % config.alignment, 0);
        alloc.deallocate(ptr);
    }
}
