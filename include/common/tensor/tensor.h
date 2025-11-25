#pragma once
#include <memory>
#include <vector>

#include "common/device.h"
#include "common/dtypes.h"
#include "common/tensor/storage.h"
namespace loom {
class Tensor {
  public:
    Tensor(const std::vector<size_t>& shape, const loom::DType dtype = loom::DType::FLOAT32,
           const loom::Device& device = loom::Device(loom::DeviceType::CPU));
    ~Tensor() = default;

    // Shallow copy of the tensor
    Tensor(const Tensor& other) = default;
    // Copy assignment operator
    Tensor& operator=(const Tensor& other) = default;
    // Move constructor
    Tensor(Tensor&& other) noexcept = default;
    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Common Initialization Functions
    static Tensor zeros(const std::vector<size_t>& shape,
                        const loom::DType dtype = loom::DType::FLOAT32,
                        const loom::Device& device = loom::Device(loom::DeviceType::CPU));
    static Tensor ones(const std::vector<size_t>& shape,
                       const loom::DType dtype = loom::DType::FLOAT32,
                       const loom::Device& device = loom::Device(loom::DeviceType::CPU));
    static Tensor rand(const std::vector<size_t>& shape,
                       const loom::DType dtype = loom::DType::FLOAT32,
                       const loom::Device& device = loom::Device(loom::DeviceType::CPU));
    static Tensor randn(const std::vector<size_t>& shape,
                        const loom::DType dtype = loom::DType::FLOAT32,
                        const loom::Device& device = loom::Device(loom::DeviceType::CPU));
    static Tensor full(const std::vector<size_t>& shape, const double value,
                       const loom::DType dtype = loom::DType::FLOAT32,
                       const loom::Device& device = loom::Device(loom::DeviceType::CPU));

    // In place initialization functions
    Tensor& zero();
    Tensor& one();
    Tensor& rand();
    Tensor& fill(const double value);
    Tensor& uniform(const double min, const double max);

    // Basic Accessors
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

    // Matmul Operations
    Tensor matmul(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;

    // Reduction Operations
    Tensor sum(const int dim, const bool keepdim = false) const;
    Tensor mean(const int dim, const bool keepdim = false) const;
    Tensor max(const int dim, const bool keepdim = false) const;
    Tensor min(const int dim, const bool keepdim = false) const;

    Tensor sum() const;
    Tensor mean() const;
    Tensor max() const;
    Tensor min() const;

    // Helper Functions
    [[nodiscard]] size_t numel() const;
    [[nodiscard]] size_t size(const size_t dim) const;
    [[nodiscard]] size_t ndim() const;
    [[nodiscard]] bool isContiguous() const;

    // View operations
    Tensor reshape(const std::vector<size_t>& shape) const;
    Tensor permute(const std::vector<int>& axes) const;
    Tensor transpose() const;                    // Swap last 2 dims (matrix transpose)
    Tensor transpose(int dim0, int dim1) const;  // Swap any 2 dims
    Tensor flatten() const;
    Tensor squeeze() const;
    Tensor squeeze(const int dim) const;
    Tensor unsqueeze(const int dim) const;

    Tensor clone() const;
    Tensor contiguous() const;
    Tensor toDevice(const Device& device) const;

    Tensor slice(const int dim, const size_t start, const size_t end) const;
    Tensor operator[](const std::vector<size_t>& indices) const;

  private:
    std::shared_ptr<Storage> mStorage;
    std::vector<size_t> mShape;
    std::vector<size_t> mStride;
    size_t mOffset;

    static std::vector<size_t> calculateStride(const std::vector<size_t>& shape);
    static size_t calculateOffset(const std::vector<size_t>& shape,
                                  const std::vector<size_t>& stride);

    // Private constructor for internal use only
    Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape, std::vector<size_t> stride,
           size_t offset);
};
}  // namespace loom