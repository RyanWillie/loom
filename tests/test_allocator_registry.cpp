#include <cstring>
#include <thread>
#include <vector>

#include "common/device.h"
#include "common/memory/allocator.h"
#include "common/registry/allocator_registry.h"
#include "cpu/pooling_allocator.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// Test Fixture
// ============================================================================

class AllocatorRegistryTest : public ::testing::Test {
  protected:
    void TearDown() override {
        // Clean up registry after each test
        AllocatorRegistry::clear();
    }
};

// ============================================================================
// Basic Registry Operations Tests
// ============================================================================

TEST_F(AllocatorRegistryTest, GetCreatesDefaultPoolingAllocator) {
    Device cpu_device(DeviceType::CPU);

    // Registry should create default CPU allocator on first access
    auto allocator = AllocatorRegistry::get(cpu_device);

    ASSERT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->device().type(), DeviceType::CPU);
}

TEST_F(AllocatorRegistryTest, ExistsReturnsFalseInitially) {
    Device cpu_device(DeviceType::CPU);
    EXPECT_FALSE(AllocatorRegistry::exists(cpu_device));
}

TEST_F(AllocatorRegistryTest, ExistsReturnsTrueAfterGet) {
    Device cpu_device(DeviceType::CPU);

    // First call creates the allocator
    auto allocator = AllocatorRegistry::get(cpu_device);
    EXPECT_NE(allocator, nullptr);

    // Now it should exist
    EXPECT_TRUE(AllocatorRegistry::exists(cpu_device));
}

TEST_F(AllocatorRegistryTest, GetReturnsSameAllocatorOnMultipleCalls) {
    Device cpu_device(DeviceType::CPU);

    auto allocator1 = AllocatorRegistry::get(cpu_device);
    auto allocator2 = AllocatorRegistry::get(cpu_device);

    // Should return the same shared_ptr (same underlying object)
    EXPECT_EQ(allocator1, allocator2);
}

// ============================================================================
// Custom Allocator Tests
// ============================================================================

TEST_F(AllocatorRegistryTest, SetCustomAllocator) {
    Device cpu_device(DeviceType::CPU);

    // Create and set a custom allocator
    PoolingAllocatorConfig config{.alignment = 64};
    auto custom_allocator = std::make_shared<PoolingAllocator>(config);

    AllocatorRegistry::set(cpu_device, custom_allocator);

    EXPECT_TRUE(AllocatorRegistry::exists(cpu_device));

    // Get should return our custom allocator
    auto retrieved = AllocatorRegistry::get(cpu_device);
    EXPECT_EQ(retrieved, custom_allocator);
}

TEST_F(AllocatorRegistryTest, SetReplacesExistingAllocator) {
    Device cpu_device(DeviceType::CPU);

    // Get default allocator first
    auto default_allocator = AllocatorRegistry::get(cpu_device);

    // Set a new custom allocator
    auto custom_allocator = std::make_shared<PoolingAllocator>();
    AllocatorRegistry::set(cpu_device, custom_allocator);

    // Should now get the custom allocator
    auto retrieved = AllocatorRegistry::get(cpu_device);
    EXPECT_EQ(retrieved, custom_allocator);
    EXPECT_NE(retrieved, default_allocator);
}

// ============================================================================
// Multiple Device Tests
// ============================================================================

TEST_F(AllocatorRegistryTest, DifferentDevicesHaveDifferentAllocators) {
    Device cpu_device(DeviceType::CPU);
    Device cuda_device_0(DeviceType::CUDA, 0);

    auto cpu_allocator = std::make_shared<PoolingAllocator>();
    AllocatorRegistry::set(cpu_device, cpu_allocator);

    // CUDA device should not exist yet
    EXPECT_TRUE(AllocatorRegistry::exists(cpu_device));
    EXPECT_FALSE(AllocatorRegistry::exists(cuda_device_0));
}

TEST_F(AllocatorRegistryTest, DifferentCUDAIndicesAreSeparate) {
    Device cuda_0(DeviceType::CUDA, 0);
    Device cuda_1(DeviceType::CUDA, 1);

    auto allocator_0 = std::make_shared<PoolingAllocator>();  // Using CPU allocator as placeholder
    AllocatorRegistry::set(cuda_0, allocator_0);

    EXPECT_TRUE(AllocatorRegistry::exists(cuda_0));
    EXPECT_FALSE(AllocatorRegistry::exists(cuda_1));
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(AllocatorRegistryTest, ClearRemovesAllAllocators) {
    Device cpu_device(DeviceType::CPU);
    Device cuda_device(DeviceType::CUDA, 0);

    // Add some allocators
    auto cpu_allocator = AllocatorRegistry::get(cpu_device);
    EXPECT_NE(cpu_allocator, nullptr);
    auto cuda_allocator = std::make_shared<PoolingAllocator>();
    AllocatorRegistry::set(cuda_device, cuda_allocator);

    EXPECT_TRUE(AllocatorRegistry::exists(cpu_device));
    EXPECT_TRUE(AllocatorRegistry::exists(cuda_device));

    // Clear registry
    AllocatorRegistry::clear();

    // Nothing should exist now
    EXPECT_FALSE(AllocatorRegistry::exists(cpu_device));
    EXPECT_FALSE(AllocatorRegistry::exists(cuda_device));
}

TEST_F(AllocatorRegistryTest, ClearDoesNotAffectExistingSharedPtrs) {
    Device cpu_device(DeviceType::CPU);

    // Get allocator and keep a reference
    auto allocator = AllocatorRegistry::get(cpu_device);
    ASSERT_NE(allocator, nullptr);

    // Clear registry
    AllocatorRegistry::clear();

    // Our reference should still be valid (shared_ptr semantics)
    EXPECT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->device().type(), DeviceType::CPU);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(AllocatorRegistryTest, GetThrowsForCUDAWhenNotImplemented) {
    Device cuda_device(DeviceType::CUDA, 0);

    // CUDA allocator not implemented yet, should throw
    EXPECT_THROW(AllocatorRegistry::get(cuda_device), std::runtime_error);
}

TEST_F(AllocatorRegistryTest, GetThrowsForMPSWhenNotImplemented) {
    Device mps_device(DeviceType::MPS, 0);

    // MPS allocator not implemented yet, should throw
    EXPECT_THROW(AllocatorRegistry::get(mps_device), std::runtime_error);
}

// ============================================================================
// Allocator Functionality Tests (via Registry)
// ============================================================================

TEST_F(AllocatorRegistryTest, AllocatorCanAllocateAndDeallocate) {
    Device cpu_device(DeviceType::CPU);
    auto allocator = AllocatorRegistry::get(cpu_device);

    // Allocate memory
    void* ptr = allocator->allocate(1024);
    ASSERT_NE(ptr, nullptr);

    // Write to it (should not crash)
    std::memset(ptr, 0, 1024);

    // Deallocate
    EXPECT_NO_THROW(allocator->deallocate(ptr));
}

TEST_F(AllocatorRegistryTest, AllocatorDeviceMatchesRequestedDevice) {
    Device cpu_device(DeviceType::CPU, 0);
    auto allocator = AllocatorRegistry::get(cpu_device);

    Device allocator_device = allocator->device();
    EXPECT_EQ(allocator_device.type(), cpu_device.type());
    EXPECT_EQ(allocator_device.index(), cpu_device.index());
}

// ============================================================================
// Thread Safety Tests (Basic)
// ============================================================================

TEST_F(AllocatorRegistryTest, ConcurrentGetIsSafe) {
    Device cpu_device(DeviceType::CPU);

    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<Allocator>> allocators(10);

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&cpu_device, &allocators, i]() {
            allocators[i] = AllocatorRegistry::get(cpu_device);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All threads should have gotten the same allocator
    for (int i = 1; i < 10; ++i) {
        EXPECT_EQ(allocators[0], allocators[i]);
    }
}
