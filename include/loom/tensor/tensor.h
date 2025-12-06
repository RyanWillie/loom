#pragma once
#include <memory>
#include <vector>

#include "loom/autograd/autograd_meta.h"
#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/tensor/accessor.h"
#include "loom/tensor/storage.h"

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

    // Set global random seed for reproducibility
    static void manualSeed(uint64_t seed);
    static Tensor full(const std::vector<size_t>& shape, const double value,
                       const loom::DType dtype = loom::DType::FLOAT32,
                       const loom::Device& device = loom::Device(loom::DeviceType::CPU));

    // In place initialization functions
    Tensor& zero();
    Tensor& one();
    Tensor& rand();
    Tensor& randn();
    Tensor& uniform(const double min, const double max);
    Tensor& fill(const double value);

    // Basic Accessors
    [[nodiscard]] const std::vector<size_t>& shape() const;
    [[nodiscard]] const std::vector<size_t>& stride() const;
    [[nodiscard]] size_t offset() const;
    [[nodiscard]] loom::DType dtype() const;
    [[nodiscard]] loom::Device device() const;

    // Visualization
    void print(const std::string& name = "Tensor") const;

    // Basic Arithmetic Operations with Tensor
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // In place arithmetic operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Basic Arithmetic Operations with Scalar
    Tensor operator+(const double scalar) const;
    Tensor operator-(const double scalar) const;
    Tensor operator*(const double scalar) const;
    Tensor operator/(const double scalar) const;

    // In place arithmetic operations with scalar
    Tensor& operator+=(const double scalar);
    Tensor& operator-=(const double scalar);
    Tensor& operator*=(const double scalar);
    Tensor& operator/=(const double scalar);

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
    Tensor var(const int dim, const bool keepdim = false) const;
    Tensor var() const;

    // Activation Functions
    Tensor relu() const;

    // Helper Functions
    [[nodiscard]] size_t numel() const;
    [[nodiscard]] size_t size(const size_t dim) const;
    [[nodiscard]] size_t ndim() const;
    [[nodiscard]] bool isContiguous() const;

    // ========================================================================
    // Autograd Methods
    // ========================================================================

    // Enable/disable gradient tracking for this tensor
    Tensor& requiresGrad(bool requires_grad);

    // Check if tensor requires gradient computation
    [[nodiscard]] bool requiresGrad() const;

    // Check if tensor is a leaf (user-created, not result of operation)
    [[nodiscard]] bool isLeaf() const;

    // Get accumulated gradient (nullptr if no gradient computed yet)
    [[nodiscard]] std::shared_ptr<Tensor> grad() const;

    // Accumulate gradient (handles initialization automatically)
    // Used internally by backward engine to add gradients from multiple paths
    void accumulateGrad(const Tensor& g);

    // Compute gradients for this tensor and all predecessors in the graph
    // grad: gradient of loss with respect to this tensor (defaults to ones)
    void backward(const Tensor& grad = Tensor::ones({1}));

    // Zero out accumulated gradients
    void zeroGrad();

    // Set the gradient function (used internally by operations)
    void setGradFn(std::shared_ptr<loom::autograd::Node> grad_fn);

    // Get the gradient function (nullptr for leaf tensors)
    [[nodiscard]] std::shared_ptr<loom::autograd::Node> gradFn() const;

    // Version tracking for in-place operation detection
    [[nodiscard]] uint64_t version() const;
    void bumpVersion();

    // Extract scalar value from single-element tensor
    [[nodiscard]] double item() const;

    // View operations
    Tensor reshape(const std::vector<size_t>& shape) const;
    Tensor permute(const std::vector<int>& axes) const;
    Tensor transpose() const;                    // Swap last 2 dims (matrix transpose)
    Tensor transpose(int dim0, int dim1) const;  // Swap any 2 dims
    Tensor flatten() const;
    Tensor squeeze() const;
    Tensor squeeze(int dim) const;
    Tensor unsqueeze(int dim) const;

    Tensor clone() const;
    Tensor contiguous() const;
    Tensor toDevice(const Device& device) const;

    Tensor slice(int dim, const size_t start, const size_t end) const;
    Tensor operator[](const std::vector<size_t>& indices) const;

    // ========================================================================
    // Typed Accessors - Zero-overhead element access for performance-critical code
    // ========================================================================

    // Get a typed accessor for fast element access
    // Usage: auto acc = tensor.accessor<float, 2>();
    //        acc[i][j] = 1.0f;  // Fast, typed access
    // T: The element type (must match tensor's dtype)
    // N: Number of dimensions (must match tensor's ndim)
    template <typename T, size_t N>
    TensorAccessor<T, N> accessor() {
        // Validate dimensions
        if (ndim() != N) {
            throw std::runtime_error("Accessor dimension mismatch: tensor has " +
                                     std::to_string(ndim()) + " dimensions, but accessor has " +
                                     std::to_string(N));
        }

        // Validate type matches dtype
        constexpr DType expected_dtype = dtype_traits<T>::value;
        if (dtype() != expected_dtype) {
            throw std::runtime_error("Accessor type mismatch: tensor has dtype " +
                                     std::string(name(dtype())) + ", but accessor requested " +
                                     std::string(name(expected_dtype)));
        }

        // Get typed pointer with offset applied
        T* data_ptr = static_cast<T*>(mStorage->data().get()) + mOffset;

        return TensorAccessor<T, N>(data_ptr, mStride.data(), mShape.data());
    }

    // Const version for read-only access
    // Returns TensorAccessor<const T, N> which prevents modification
    template <typename T, size_t N>
    TensorAccessor<const T, N> accessor() const {
        // Validate dimensions
        if (ndim() != N) {
            throw std::runtime_error("Accessor dimension mismatch: tensor has " +
                                     std::to_string(ndim()) + " dimensions, but accessor has " +
                                     std::to_string(N));
        }

        // Validate type matches dtype
        constexpr DType expected_dtype = dtype_traits<T>::value;
        if (dtype() != expected_dtype) {
            throw std::runtime_error("Accessor type mismatch: tensor has dtype " +
                                     std::string(name(dtype())) + ", but accessor requested " +
                                     std::string(name(expected_dtype)));
        }

        // Get typed pointer with offset applied
        const T* data_ptr = static_cast<const T*>(mStorage->data().get()) + mOffset;

        return TensorAccessor<const T, N>(data_ptr, mStride.data(), mShape.data());
    }

  private:
    std::shared_ptr<Storage> mStorage;
    std::vector<size_t> mShape;
    std::vector<size_t> mStride;
    size_t mOffset;

    // AUTOGRAD: Optional gradient tracking metadata (nullptr when autograd disabled)
    std::shared_ptr<loom::autograd::AutogradMeta> mAutogradMeta;

    static std::vector<size_t> calculateStride(const std::vector<size_t>& shape);
    static size_t calculateOffset(const std::vector<size_t>& indices,
                                  const std::vector<size_t>& stride);

    // Broadcasting helpers
    static std::vector<size_t> broadcastShape(const std::vector<size_t>& a,
                                              const std::vector<size_t>& b);
    static std::vector<size_t> broadcastStrides(const std::vector<size_t>& original_shape,
                                                const std::vector<size_t>& original_stride,
                                                const std::vector<size_t>& target_shape);

    // Broadcast binary operation with lambda
    template <typename Op>
    Tensor broadcastOp(const Tensor& other, Op op) const {
        if (other.dtype() != dtype()) {
            throw std::runtime_error("DType mismatch for arithmetic operation");
        }

        auto out_shape = broadcastShape(mShape, other.mShape);
        auto stride_a = broadcastStrides(mShape, mStride, out_shape);
        auto stride_b = broadcastStrides(other.mShape, other.mStride, out_shape);

        Tensor result = Tensor::zeros(out_shape, dtype(), device());
        size_t out_numel = result.numel();

        dispatchByDType(dtype(), result.mStorage->data().get(), out_numel,
                        [&](auto* dst_ptr, size_t n) {
                            using T = std::remove_pointer_t<decltype(dst_ptr)>;
                            const T* src_a = static_cast<const T*>(mStorage->data().get());
                            const T* src_b = static_cast<const T*>(other.mStorage->data().get());

                            for (size_t out_linear = 0; out_linear < n; ++out_linear) {
                                // Convert linear index to multi-dimensional index
                                std::vector<size_t> out_idx(out_shape.size());
                                size_t temp = out_linear;
                                for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                    out_idx[i] = temp % out_shape[i];
                                    temp /= out_shape[i];
                                }

                                // Calculate offsets using broadcast strides
                                size_t offset_a = mOffset;
                                size_t offset_b = other.mOffset;
                                for (size_t i = 0; i < out_shape.size(); ++i) {
                                    offset_a += out_idx[i] * stride_a[i];
                                    offset_b += out_idx[i] * stride_b[i];
                                }

                                dst_ptr[out_linear] = op(src_a[offset_a], src_b[offset_b]);
                            }
                        });

        return result;
    }

    // Private constructor for internal use only
    Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape, std::vector<size_t> stride,
           size_t offset);
};
}  // namespace loom