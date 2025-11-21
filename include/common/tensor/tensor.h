#pragma once
#include <memory>
#include <vector>

#include "common/device.h"
#include "common/dtypes.h"
#include "common/tensor/storage.h"

class Tensor {
  public:
    Tensor(const std::vector<size_t>& shape, const loom::DType dtype = loom::DType::FLOAT32,
           const loom::Device& device = loom::Device(loom::DeviceType::CPU));
    ~Tensor();

    [[nodiscard]] const std::vector<size_t>& shape() const;
    [[nodiscard]] const std::vector<size_t>& stride() const;
    [[nodiscard]] size_t offset() const;
    [[nodiscard]] loom::DType dtype() const;
    [[nodiscard]] loom::Device device() const;

    // Basic Arithmetic Operations with Tensor
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // Basic Arithmetic Operations with Scalar
    Tensor operator+(const double scalar) const;
    Tensor operator-(const double scalar) const;
    Tensor operator*(const double scalar) const;
    Tensor operator/(const double scalar) const;

    //

  private:
    std::shared_ptr<Storage> mStorage;
    std::vector<size_t> mShape;   // Shape of tensor
    std::vector<size_t> mStride;  // Stride of tensor
    size_t mOffset;               // Offset of tensor

    loom::DType mDType;    // Data type of tensor
    loom::Device mDevice;  // Device location of tensor
};