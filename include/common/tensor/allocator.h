#pragma once

#include "common/device.h"

namespace loom {

class Allocator {
  public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* data) = 0;

    [[nodiscard]] virtual Device device() const = 0;
};
}  // namespace loom