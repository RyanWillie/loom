#pragma once

#include "common/tensor/tensor.h"

namespace loom {

struct Sample {
    Tensor input;
    Tensor target;
};

class Dataset {
  public:
    virtual ~Dataset() = default;

    [[nodiscard]] virtual size_t size() const = 0;
    [[nodiscard]] virtual Sample get(size_t index) const = 0;
};

}  // namespace loom