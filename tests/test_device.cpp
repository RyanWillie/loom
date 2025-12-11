#include "loom/device.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// Device Construction Tests
// ============================================================================

TEST(DeviceTest, ConstructCPUDevice) {
    Device device(DeviceType::CPU);
    EXPECT_EQ(device.type(), DeviceType::CPU);
    EXPECT_EQ(device.index(), 0);
}

TEST(DeviceTest, ConstructCUDADevice) {
    Device device(DeviceType::CUDA);
    EXPECT_EQ(device.type(), DeviceType::CUDA);
    EXPECT_EQ(device.index(), 0);
}

TEST(DeviceTest, ConstructMPSDevice) {
    Device device(DeviceType::MPS);
    EXPECT_EQ(device.type(), DeviceType::MPS);
    EXPECT_EQ(device.index(), 0);
}

TEST(DeviceTest, ConstructDeviceWithIndex) {
    Device device(DeviceType::CUDA, 2);
    EXPECT_EQ(device.type(), DeviceType::CUDA);
    EXPECT_EQ(device.index(), 2);
}

TEST(DeviceTest, DefaultIndexIsZero) {
    Device device(DeviceType::CPU);
    EXPECT_EQ(device.index(), 0);
}

// ============================================================================
// Device Type Query Tests
// ============================================================================

TEST(DeviceTest, IsCPUReturnsTrueForCPU) {
    Device device(DeviceType::CPU);
    EXPECT_TRUE(device.isCPU());
    EXPECT_FALSE(device.isCUDA());
    EXPECT_FALSE(device.isMPS());
}

TEST(DeviceTest, IsCUDAReturnsTrueForCUDA) {
    Device device(DeviceType::CUDA);
    EXPECT_FALSE(device.isCPU());
    EXPECT_TRUE(device.isCUDA());
    EXPECT_FALSE(device.isMPS());
}

TEST(DeviceTest, IsMPSReturnsTrueForMPS) {
    Device device(DeviceType::MPS);
    EXPECT_FALSE(device.isCPU());
    EXPECT_FALSE(device.isCUDA());
    EXPECT_TRUE(device.isMPS());
}

// ============================================================================
// Device Comparison Tests
// ============================================================================

TEST(DeviceTest, EqualityOperatorSameType) {
    Device device1(DeviceType::CPU);
    Device device2(DeviceType::CPU);
    EXPECT_TRUE(device1 == device2);
    EXPECT_FALSE(device1 != device2);
}

TEST(DeviceTest, EqualityOperatorDifferentType) {
    Device device1(DeviceType::CPU);
    Device device2(DeviceType::CUDA);
    EXPECT_FALSE(device1 == device2);
    EXPECT_TRUE(device1 != device2);
}

TEST(DeviceTest, EqualityOperatorSameTypeAndIndex) {
    Device device1(DeviceType::CUDA, 1);
    Device device2(DeviceType::CUDA, 1);
    EXPECT_TRUE(device1 == device2);
}

TEST(DeviceTest, EqualityOperatorSameTypeDifferentIndex) {
    Device device1(DeviceType::CUDA, 0);
    Device device2(DeviceType::CUDA, 1);
    EXPECT_FALSE(device1 == device2);
    EXPECT_TRUE(device1 != device2);
}

TEST(DeviceTest, InequalityOperator) {
    Device device1(DeviceType::CPU);
    Device device2(DeviceType::CUDA);
    EXPECT_TRUE(device1 != device2);
    EXPECT_FALSE(device1 == device2);
}

// ============================================================================
// Device Ordering Tests (for map/set usage)
// ============================================================================

TEST(DeviceTest, LessThanOperatorByType) {
    Device cpu(DeviceType::CPU);
    Device cuda(DeviceType::CUDA);

    // Ordering should be consistent (exact order doesn't matter)
    bool cpu_less = cpu < cuda;
    bool cuda_less = cuda < cpu;

    // One must be less than the other (they're different types)
    EXPECT_TRUE(cpu_less || cuda_less);
    EXPECT_FALSE(cpu_less && cuda_less);
}

TEST(DeviceTest, LessThanOperatorByIndex) {
    Device cuda0(DeviceType::CUDA, 0);
    Device cuda1(DeviceType::CUDA, 1);
    Device cuda2(DeviceType::CUDA, 2);

    EXPECT_TRUE(cuda0 < cuda1);
    EXPECT_TRUE(cuda1 < cuda2);
    EXPECT_TRUE(cuda0 < cuda2);
}

TEST(DeviceTest, LessThanOperatorSameDevice) {
    Device device1(DeviceType::CPU, 0);
    Device device2(DeviceType::CPU, 0);

    EXPECT_FALSE(device1 < device2);
    EXPECT_FALSE(device2 < device1);
}

// ============================================================================
// Device String Representation Tests
// ============================================================================

TEST(DeviceTest, ToStringCPU) {
    Device device(DeviceType::CPU);
    std::string str = device.toString();
    EXPECT_NE(str.find("CPU"), std::string::npos);
    EXPECT_NE(str.find("0"), std::string::npos);
}

TEST(DeviceTest, ToStringCUDA) {
    Device device(DeviceType::CUDA, 2);
    std::string str = device.toString();
    EXPECT_NE(str.find("CUDA"), std::string::npos);
    EXPECT_NE(str.find("2"), std::string::npos);
}

TEST(DeviceTest, ToStringMPS) {
    Device device(DeviceType::MPS);
    std::string str = device.toString();
    EXPECT_NE(str.find("MPS"), std::string::npos);
}

// ============================================================================
// Copy and Move Semantics Tests
// ============================================================================

TEST(DeviceTest, CopyConstructor) {
    Device device1(DeviceType::CUDA, 3);
    Device device2(device1);

    EXPECT_EQ(device1.type(), device2.type());
    EXPECT_EQ(device1.index(), device2.index());
    EXPECT_TRUE(device1 == device2);
}

TEST(DeviceTest, CopyAssignment) {
    Device device1(DeviceType::CUDA, 3);
    Device device2(DeviceType::CPU, 0);

    device2 = device1;

    EXPECT_EQ(device1.type(), device2.type());
    EXPECT_EQ(device1.index(), device2.index());
    EXPECT_TRUE(device1 == device2);
}

TEST(DeviceTest, MoveConstructor) {
    Device device1(DeviceType::CUDA, 3);
    DeviceType original_type = device1.type();
    int original_index = device1.index();

    Device device2(std::move(device1));

    EXPECT_EQ(device2.type(), original_type);
    EXPECT_EQ(device2.index(), original_index);
}

TEST(DeviceTest, MoveAssignment) {
    Device device1(DeviceType::CUDA, 3);
    Device device2(DeviceType::CPU, 0);

    DeviceType original_type = device1.type();
    int original_index = device1.index();

    device2 = std::move(device1);

    EXPECT_EQ(device2.type(), original_type);
    EXPECT_EQ(device2.index(), original_index);
}

// ============================================================================
// Device in Container Tests (map/set compatibility)
// ============================================================================

TEST(DeviceTest, CanBeUsedInSet) {
    std::set<Device> device_set;

    Device cpu(DeviceType::CPU);
    Device cuda0(DeviceType::CUDA, 0);
    Device cuda1(DeviceType::CUDA, 1);

    device_set.insert(cpu);
    device_set.insert(cuda0);
    device_set.insert(cuda1);

    EXPECT_EQ(device_set.size(), 3);
    EXPECT_TRUE(device_set.find(cpu) != device_set.end());
    EXPECT_TRUE(device_set.find(cuda0) != device_set.end());
}

TEST(DeviceTest, CanBeUsedInMap) {
    std::map<Device, std::string> device_map;

    Device cpu(DeviceType::CPU);
    Device cuda(DeviceType::CUDA);

    device_map[cpu] = "CPU Device";
    device_map[cuda] = "CUDA Device";

    EXPECT_EQ(device_map.size(), 2);
    EXPECT_EQ(device_map[cpu], "CPU Device");
    EXPECT_EQ(device_map[cuda], "CUDA Device");
}
