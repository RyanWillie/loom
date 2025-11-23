#include <iostream>

#include "common/device.h"
#include "common/logger.h"
#include "common/memory/utils.h"
#include "common/registry/allocator_registry.h"

#ifdef USE_CUDA
#include "gpu/kernel.h"
#endif

using loom::AllocatorRegistry;
using loom::Device;
using loom::DeviceType;
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

    const auto allocator = AllocatorRegistry::get(Device(DeviceType::CPU));
    logger.info("Allocator: {}", allocator->device().toString());

    const auto exists2 = AllocatorRegistry::exists(Device(DeviceType::CPU));
    logger.info("Allocator exists {}", exists2);
    Logger::shutdown();
    return 0;
}
