#pragma once
#include <map>
#include <memory>
#include <mutex>

#include "common/device.h"
#include "common/memory/allocator.h"

namespace loom {

class AllocatorRegistry {
  public:
    AllocatorRegistry() = delete;

    // Get an allocator for a given device
    [[nodiscard]] static std::shared_ptr<Allocator> get(const loom::Device& device);

    // Check if an allocator exists for a given device
    [[nodiscard]] static bool exists(const loom::Device& device);

    // Set an allocator for a given device
    static void set(const loom::Device& device, std::shared_ptr<Allocator> allocator);

    // Remove all allocators from the registry
    static void clear();

  private:
    static std::map<loom::Device, std::shared_ptr<Allocator>> mAllocators;
    static std::mutex mMutex;
};
}  // namespace loom