#include <iostream>

#include "common/device.h"
#include "common/dtypes.h"
#include "common/logger.h"
#include "common/memory/utils.h"
#include "common/registry/allocator_registry.h"
#include "common/tensor/tensor.h"

#ifdef USE_CUDA
#include "gpu/kernel.h"
#endif

using loom::AllocatorRegistry;
using loom::Device;
using loom::DeviceType;
using loom::DType;
using loom::Tensor;
using loom::memory::MemoryUtils;

int main() {
    Logger::setMinLogLevel(LogLevel::TRACE);
    Logger::setLogOutput(LogOutput::BOTH);

    auto& logger = Logger::getInstance("System");
    logger.info("Starting Loom Neural Network Project");

    logger.info("SIMD alignment: {}", MemoryUtils::detectSIMDAlignment());
    logger.info("Cache line size: {}", MemoryUtils::getCacheLineSize());
    logger.info("Page size: {}", MemoryUtils::getPageSize());
    logger.info("Default alignment: {}", MemoryUtils::getDefaultAlignment());

#ifdef USE_CUDA
    logger.info("CUDA Enabled");
    // call_cuda_kernel();
#else
    auto& logger_cpu = Logger::getInstance("CPU");
    logger_cpu.info("Running on CPU");
#endif

    const auto exists = AllocatorRegistry::exists(Device(DeviceType::CPU));
    logger.info("Allocator registry initialized. Allocator exists {}", exists);

    // Create a 2D tensor
    Tensor tensor({2, 3}, loom::DType::FLOAT32, Device(DeviceType::CPU));

    // Use accessor for fast, typed element access
    auto acc = tensor.accessor<float, 2>();

    // Fill with values using natural array syntax
    for (size_t i = 0; i < acc.size(); ++i) {
        for (size_t j = 0; j < acc[i].size(); ++j) {
            acc[i][j] = static_cast<float>(i * 10 + j);  // 0, 1, 2, 10, 11, 12
        }
    }

    tensor.print("Tensor filled using accessor");

    // Reshape and access as 3D
    auto reshaped = tensor.reshape({1, 2, 3});
    auto acc3d = reshaped.accessor<float, 3>();
    acc3d[0][1][1] = 100.0f;  // Modify element [0][1][1]
    reshaped.print("Reshaped 3D Tensor (modified [0][1][1] to 100)");

    // Test reductions
    auto t = Tensor::full({2, 3}, 2.0f, loom::DType::FLOAT32, Device(DeviceType::CPU));
    t.print("Tensor for reductions");
    logger.info("sum() = {}", t.sum().item());
    t.sum(0).print("sum(dim=0) - reduce rows");
    t.sum(1).print("sum(dim=1) - reduce cols");
    logger.info("mean() = {}", t.mean().item());
    logger.info("max() = {}", t.max().item());
    logger.info("min() = {}", t.min().item());

    // Test broadcasting
    logger.info("--- Broadcasting Test ---");
    auto a = Tensor::full({4, 1}, 10.0f, loom::DType::FLOAT32, Device(DeviceType::CPU));
    auto b = Tensor::full({3}, 1.0f, loom::DType::FLOAT32, Device(DeviceType::CPU));
    auto b_acc = b.accessor<float, 1>();
    b_acc[0] = 1.0f;
    b_acc[1] = 2.0f;
    b_acc[2] = 3.0f;

    a.print("a [4,1]");
    b.print("b [3]");
    auto c = a + b;  // Should broadcast to [4, 3]
    c.print("a + b (broadcast to [4,3])");

    const auto allocator = AllocatorRegistry::get(Device(DeviceType::CPU));
    logger.info("Allocator: {}", allocator->device().toString());

    const auto exists2 = AllocatorRegistry::exists(Device(DeviceType::CPU));
    logger.info("Allocator exists {}", exists2);
    Logger::shutdown();
    return 0;
}
