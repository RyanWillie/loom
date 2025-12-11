#include "loom/tensor/tensor.h"

#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include "loom/autograd/engine.h"
#include "loom/autograd/no_grad.h"
#include "loom/autograd/nodes/activation_ops.h"
#include "loom/autograd/nodes/binary_ops.h"
#include "loom/autograd/nodes/matmul_ops.h"
#include "loom/autograd/nodes/reduction_ops.h"
#include "loom/autograd/nodes/view_ops.h"
#include "loom/logger.h"
#include "loom/tensor/tensor_iterator.h"

namespace loom {

// Global random engine for reproducible random number generation
namespace {
std::mt19937& getRandomEngine() {
    static std::mt19937 engine(std::random_device{}());
    return engine;
}
}  // namespace

void Tensor::manualSeed(uint64_t seed) {
    getRandomEngine().seed(static_cast<std::mt19937::result_type>(seed));
}

Tensor::Tensor(const std::vector<size_t>& shape, const loom::DType dtype,
               const loom::Device& device)
    : mShape(shape), mStride(Tensor::calculateStride(shape)), mOffset(0) {
    // Calculate total number of elements
    size_t numElements = 1;
    for (size_t dim : shape) {
        numElements *= dim;
    }
    // Allocate storage for all elements
    mStorage = std::make_shared<Storage>(numElements, dtype, device);
}

Tensor::Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape,
               std::vector<size_t> stride, size_t offset)
    : mStorage(storage), mShape(shape), mStride(stride), mOffset(offset) {}

Tensor Tensor::zeros(const std::vector<size_t>& shape, const loom::DType dtype,
                     const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to 0
    tensor.zero();
    return tensor;
}

Tensor& Tensor::zero() {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    // Fast path: row-major contiguous view can be memset starting at the view base.
    if (isContiguous()) {
        auto* base = static_cast<unsigned char*>(mStorage->data().get());
        std::memset(base + mOffset * sizeOf(dtype()), 0, n * sizeOf(dtype()));
    } else {
        dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                        [&](auto* base, size_t) {
                            using T = std::remove_pointer_t<decltype(base)>;
                            TensorIterator it(mShape, mStride, mOffset);
                            while (it.hasNext()) {
                                base[it.offset()] = static_cast<T>(0);
                                it.next();
                            }
                        });
    }

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor Tensor::ones(const std::vector<size_t>& shape, const loom::DType dtype,
                    const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to 1
    tensor.one();
    return tensor;
}

Tensor& Tensor::one() {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        if (isContiguous()) {
            std::fill_n(base + mOffset, n, static_cast<T>(1));
        } else {
            forEachOffset(mShape, mStride, mOffset,
                          [&](size_t off) { base[off] = static_cast<T>(1); });
        }
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor Tensor::rand(const std::vector<size_t>& shape, const loom::DType dtype,
                    const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to random
    tensor.rand();
    return tensor;
}

Tensor& Tensor::rand() {
    auto& gen = getRandomEngine();
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<T>(dis(gen));
            }
        } else {
            forEachOffset(mShape, mStride, mOffset,
                          [&](size_t off) { base[off] = static_cast<T>(dis(gen)); });
        }
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor Tensor::randn(const std::vector<size_t>& shape, const loom::DType dtype,
                     const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to random
    tensor.randn();
    return tensor;
}

Tensor& Tensor::randn() {
    auto& gen = getRandomEngine();
    std::normal_distribution<double> dis(0.0, 1.0);

    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<T>(dis(gen));
            }
        } else {
            forEachOffset(mShape, mStride, mOffset,
                          [&](size_t off) { base[off] = static_cast<T>(dis(gen)); });
        }
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::uniform(const double min, const double max) {
    auto& gen = getRandomEngine();
    std::uniform_real_distribution<double> dis(min, max);

    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<T>(dis(gen));
            }
        } else {
            forEachOffset(mShape, mStride, mOffset,
                          [&](size_t off) { base[off] = static_cast<T>(dis(gen)); });
        }
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor Tensor::full(const std::vector<size_t>& shape, const double value, const loom::DType dtype,
                    const loom::Device& device) {
    Tensor tensor(shape, dtype, device);
    // Set all elements to value
    tensor.fill(value);
    return tensor;
}

Tensor& Tensor::fill(const double value) {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T v = static_cast<T>(value);
        if (isContiguous()) {
            std::fill_n(base + mOffset, n, v);
        } else {
            forEachOffset(mShape, mStride, mOffset, [&](size_t off) { base[off] = v; });
        }
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor Tensor::clone() const {
    // Deep copy of the *logical tensor* (respects shape/stride/offset), not the entire storage.
    Tensor result(mShape, dtype(), device());
    const size_t n = numel();
    if (n == 0) {
        return result;
    }

    dispatchByDType(dtype(), result.mStorage->data().get(), result.mStorage->size(),
                    [&](auto* dst_base, size_t) {
                        using T = std::remove_pointer_t<decltype(dst_base)>;
                        const T* src_base = static_cast<const T*>(mStorage->data().get());

                        if (isContiguous()) {
                            std::memcpy(dst_base, src_base + mOffset, n * sizeof(T));
                            return;
                        }

                        size_t di = 0;
                        forEachOffset(mShape, mStride, mOffset,
                                      [&](size_t off) { dst_base[di++] = src_base[off]; });
                    });
    return result;
}

std::vector<size_t> Tensor::calculateStride(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }

    std::vector<size_t> stride(shape.size());
    size_t acc = 1;

    // Calculate strides in row-major order (C-style)
    // For shape [2, 3, 4], strides are [12, 4, 1]
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        stride[i] = acc;
        acc *= shape[i];
    }
    return stride;
}

size_t Tensor::calculateOffset(const std::vector<size_t>& indices,
                               const std::vector<size_t>& stride) {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * stride[i];
    }
    return offset;
}

std::vector<size_t> Tensor::broadcastShape(const std::vector<size_t>& a,
                                           const std::vector<size_t>& b) {
    size_t max_ndim = std::max(a.size(), b.size());
    std::vector<size_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        // Get dimensions from right (or 1 if dimension doesn't exist)
        size_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dim_a == dim_b) {
            result[max_ndim - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            result[max_ndim - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            result[max_ndim - 1 - i] = dim_a;
        } else {
            throw std::runtime_error("Shapes not broadcastable: dimension " +
                                     std::to_string(max_ndim - 1 - i) + " has sizes " +
                                     std::to_string(dim_a) + " and " + std::to_string(dim_b));
        }
    }
    return result;
}

std::vector<size_t> Tensor::broadcastStrides(const std::vector<size_t>& original_shape,
                                             const std::vector<size_t>& original_stride,
                                             const std::vector<size_t>& target_shape) {
    // Create strides for broadcasting: stride is 0 for broadcast dimensions
    std::vector<size_t> result(target_shape.size(), 0);

    size_t shape_offset = target_shape.size() - original_shape.size();

    for (size_t i = 0; i < original_shape.size(); ++i) {
        size_t target_idx = shape_offset + i;
        if (original_shape[i] == target_shape[target_idx]) {
            // Same size: use original stride
            result[target_idx] = original_stride[i];
        } else if (original_shape[i] == 1) {
            // Broadcast dimension: stride = 0 (repeat the value)
            result[target_idx] = 0;
        } else {
            throw std::runtime_error("Cannot broadcast shape");
        }
    }
    // Dimensions prepended (shape_offset) remain 0, which is correct for broadcasting

    return result;
}

const std::vector<size_t>& Tensor::shape() const {
    return mShape;
}

const std::vector<size_t>& Tensor::stride() const {
    return mStride;
}

size_t Tensor::offset() const {
    return mOffset;
}

loom::DType Tensor::dtype() const {
    return mStorage->dtype();
}

loom::Device Tensor::device() const {
    return mStorage->device();
}

size_t Tensor::numel() const {
    if (mShape.empty())
        return 0;
    size_t total = 1;
    for (size_t dim : mShape) {
        total *= dim;
    }
    return total;
}

size_t Tensor::size(const size_t dim) const {
    return mShape[dim];
}

size_t Tensor::ndim() const {
    return mShape.size();
}

bool Tensor::isContiguous() const {
    // Edge case: empty or single element tensors are always contiguous
    if (mShape.empty() || numel() <= 1) {
        return true;
    }

    // A tensor is contiguous (row-major) if strides match the canonical row-major layout.
    // Note: offset does NOT affect contiguity; it only changes the base element.
    std::vector<size_t> expectedStride = calculateStride(mShape);
    return mStride == expectedStride;
}

double Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() only works on single-element tensors, got " +
                                 std::to_string(numel()) + " elements");
    }

    double result = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), 1,
                    [&](auto* ptr, size_t) { result = static_cast<double>(ptr[mOffset]); });
    return result;
}

Tensor Tensor::contiguous() const {
    // If already contiguous, return shallow copy
    if (isContiguous()) {
        return *this;
    }

    // Create new contiguous tensor
    Tensor result(mShape, dtype(), device());

    // Copy data element by element, respecting strides
    dispatchByDType(dtype(), result.mStorage->data().get(), numel(),
                    [this](auto* dst_ptr, size_t total_elements) {
                        using T = std::remove_pointer_t<decltype(dst_ptr)>;
                        const T* src_ptr = static_cast<const T*>(mStorage->data().get());

                        // For contiguous result, just iterate linearly
                        size_t dst_idx = 0;

                        // Need to iterate through all multi-dimensional indices
                        std::vector<size_t> indices(mShape.size(), 0);

                        for (size_t i = 0; i < total_elements; ++i) {
                            // Calculate source offset using strides
                            size_t src_offset = mOffset;
                            for (size_t dim = 0; dim < indices.size(); ++dim) {
                                src_offset += indices[dim] * mStride[dim];
                            }

                            // Copy element
                            dst_ptr[dst_idx++] = src_ptr[src_offset];

                            // Increment multi-dimensional index
                            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                                indices[dim]++;
                                if (indices[dim] < mShape[dim]) {
                                    break;  // No carry
                                }
                                indices[dim] = 0;  // Carry to next dimension
                            }
                        }
                    });

    return result;
}

Tensor Tensor::toDevice(const Device& device) const {
    // If same device, return shallow copy
    if (device.type() == this->device().type()) {
        return *this;
    }

    // Different device - would need to copy data across devices
    // For now, throw an error for non-CPU devices
    throw std::runtime_error("Device transfer not yet implemented for non-CPU devices");
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for addition");
    }

    // Forward pass - compute the result (stride-aware; never mutates inputs)
    Tensor result = [&]() {
        if (mShape != other.mShape) {
            return broadcastOp(other, [](auto a, auto b) { return a + b; });
        }

        Tensor out = Tensor::zeros(mShape, dtype(), device());
        const size_t n = out.numel();
        if (n == 0) {
            return out;
        }

        dispatchByDType(dtype(), out.mStorage->data().get(), out.mStorage->size(),
                        [&](auto* out_base, size_t) {
                            using T = std::remove_pointer_t<decltype(out_base)>;
                            const T* a_base = static_cast<const T*>(mStorage->data().get());
                            const T* b_base = static_cast<const T*>(other.mStorage->data().get());

                            if (isContiguous() && other.isContiguous()) {
                                const T* ap = a_base + mOffset;
                                const T* bp = b_base + other.mOffset;
                                for (size_t i = 0; i < n; ++i) {
                                    out_base[i] = ap[i] + bp[i];
                                }
                                return;
                            }

                            size_t di = 0;
                            forEachOffset2(mShape, mStride, mOffset, other.mStride, other.mOffset,
                                           [&](size_t ao, size_t bo) {
                                               out_base[di++] = a_base[ao] + b_base[bo];
                                           });
                        });

        return out;
    }();

    if ((!requiresGrad() && !other.requiresGrad()) || autograd::NoGradMode::isEnabled()) {
        return result;
    }

    // Build computation graph for backward pass
    auto add_node = std::make_shared<autograd::AddBackward>(mShape, other.mShape);

    std::vector<std::shared_ptr<autograd::Node>> next_fns;
    if (gradFn())
        next_fns.push_back(gradFn());
    if (other.gradFn())
        next_fns.push_back(other.gradFn());
    add_node->setNextFunctions(next_fns);
    add_node->setInputTensors({std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)});

    result.setGradFn(add_node);
    result.requiresGrad(true);
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for subtraction");
    }

    // Forward pass - compute the result (stride-aware; never mutates inputs)
    Tensor result = [&]() {
        if (mShape != other.mShape) {
            return broadcastOp(other, [](auto a, auto b) { return a - b; });
        }

        Tensor out = Tensor::zeros(mShape, dtype(), device());
        const size_t n = out.numel();
        if (n == 0) {
            return out;
        }

        dispatchByDType(dtype(), out.mStorage->data().get(), out.mStorage->size(),
                        [&](auto* out_base, size_t) {
                            using T = std::remove_pointer_t<decltype(out_base)>;
                            const T* a_base = static_cast<const T*>(mStorage->data().get());
                            const T* b_base = static_cast<const T*>(other.mStorage->data().get());

                            if (isContiguous() && other.isContiguous()) {
                                const T* ap = a_base + mOffset;
                                const T* bp = b_base + other.mOffset;
                                for (size_t i = 0; i < n; ++i) {
                                    out_base[i] = ap[i] - bp[i];
                                }
                                return;
                            }

                            size_t di = 0;
                            forEachOffset2(mShape, mStride, mOffset, other.mStride, other.mOffset,
                                           [&](size_t ao, size_t bo) {
                                               out_base[di++] = a_base[ao] - b_base[bo];
                                           });
                        });

        return out;
    }();

    if ((!requiresGrad() && !other.requiresGrad()) || autograd::NoGradMode::isEnabled()) {
        return result;
    }

    // Build computation graph for backward pass
    auto sub_node = std::make_shared<autograd::SubBackward>(mShape, other.mShape);

    std::vector<std::shared_ptr<autograd::Node>> next_fns;
    if (gradFn())
        next_fns.push_back(gradFn());
    if (other.gradFn())
        next_fns.push_back(other.gradFn());
    sub_node->setNextFunctions(next_fns);
    sub_node->setInputTensors({std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)});

    result.setGradFn(sub_node);
    result.requiresGrad(true);
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for multiplication");
    }

    // Forward pass - compute the result (stride-aware; never mutates inputs)
    Tensor result = [&]() {
        if (mShape != other.mShape) {
            return broadcastOp(other, [](auto a, auto b) { return a * b; });
        }

        Tensor out = Tensor::zeros(mShape, dtype(), device());
        const size_t n = out.numel();
        if (n == 0) {
            return out;
        }

        dispatchByDType(dtype(), out.mStorage->data().get(), out.mStorage->size(),
                        [&](auto* out_base, size_t) {
                            using T = std::remove_pointer_t<decltype(out_base)>;
                            const T* a_base = static_cast<const T*>(mStorage->data().get());
                            const T* b_base = static_cast<const T*>(other.mStorage->data().get());

                            if (isContiguous() && other.isContiguous()) {
                                const T* ap = a_base + mOffset;
                                const T* bp = b_base + other.mOffset;
                                for (size_t i = 0; i < n; ++i) {
                                    out_base[i] = ap[i] * bp[i];
                                }
                                return;
                            }

                            size_t di = 0;
                            forEachOffset2(mShape, mStride, mOffset, other.mStride, other.mOffset,
                                           [&](size_t ao, size_t bo) {
                                               out_base[di++] = a_base[ao] * b_base[bo];
                                           });
                        });

        return out;
    }();

    if ((!requiresGrad() && !other.requiresGrad()) || autograd::NoGradMode::isEnabled()) {
        return result;
    }

    // Build computation graph for backward pass (saves input values for gradient)
    auto mul_node = std::make_shared<autograd::MulBackward>(*this, other, mShape, other.mShape);

    std::vector<std::shared_ptr<autograd::Node>> next_fns;
    if (gradFn())
        next_fns.push_back(gradFn());
    if (other.gradFn())
        next_fns.push_back(other.gradFn());
    mul_node->setNextFunctions(next_fns);
    mul_node->setInputTensors({std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)});

    result.setGradFn(mul_node);
    result.requiresGrad(true);
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for division");
    }

    // Forward pass - compute the result (stride-aware; never mutates inputs)
    Tensor result = [&]() {
        if (mShape != other.mShape) {
            return broadcastOp(other, [](auto a, auto b) { return a / b; });
        }

        Tensor out = Tensor::zeros(mShape, dtype(), device());
        const size_t n = out.numel();
        if (n == 0) {
            return out;
        }

        dispatchByDType(dtype(), out.mStorage->data().get(), out.mStorage->size(),
                        [&](auto* out_base, size_t) {
                            using T = std::remove_pointer_t<decltype(out_base)>;
                            const T* a_base = static_cast<const T*>(mStorage->data().get());
                            const T* b_base = static_cast<const T*>(other.mStorage->data().get());

                            if (isContiguous() && other.isContiguous()) {
                                const T* ap = a_base + mOffset;
                                const T* bp = b_base + other.mOffset;
                                for (size_t i = 0; i < n; ++i) {
                                    out_base[i] = ap[i] / bp[i];
                                }
                                return;
                            }

                            size_t di = 0;
                            forEachOffset2(mShape, mStride, mOffset, other.mStride, other.mOffset,
                                           [&](size_t ao, size_t bo) {
                                               out_base[di++] = a_base[ao] / b_base[bo];
                                           });
                        });

        return out;
    }();

    if ((!requiresGrad() && !other.requiresGrad()) || autograd::NoGradMode::isEnabled()) {
        return result;
    }

    // Build computation graph for backward pass (saves input values for gradient)
    auto div_node = std::make_shared<autograd::DivBackward>(*this, other, mShape, other.mShape);

    std::vector<std::shared_ptr<autograd::Node>> next_fns;
    if (gradFn())
        next_fns.push_back(gradFn());
    if (other.gradFn())
        next_fns.push_back(other.gradFn());
    div_node->setNextFunctions(next_fns);
    div_node->setInputTensors({std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)});

    result.setGradFn(div_node);
    result.requiresGrad(true);
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for addition");
    }

    // If RHS aliases LHS storage but has a different view mapping, materialize RHS first
    // to avoid order-dependent reads.
    const bool rhs_aliases = (mStorage.get() == other.mStorage.get());
    const bool rhs_mapping_differs = (mOffset != other.mOffset) || (mStride != other.mStride);
    Tensor rhs = (rhs_aliases && rhs_mapping_differs) ? other.clone() : other;

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T* rhs_base = static_cast<const T*>(rhs.mStorage->data().get());

        const size_t n = numel();
        if (n == 0) {
            return;
        }

        if (isContiguous() && rhs.isContiguous()) {
            T* lhs_ptr = base + mOffset;
            const T* rhs_ptr = rhs_base + rhs.mOffset;
            for (size_t i = 0; i < n; ++i) {
                lhs_ptr[i] += rhs_ptr[i];
            }
            return;
        }

        forEachOffset2(mShape, mStride, mOffset, rhs.mStride, rhs.mOffset,
                       [&](size_t off_l, size_t off_r) { base[off_l] += rhs_base[off_r]; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for subtraction");
    }

    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for subtraction");
    }

    const bool rhs_aliases = (mStorage.get() == other.mStorage.get());
    const bool rhs_mapping_differs = (mOffset != other.mOffset) || (mStride != other.mStride);
    Tensor rhs = (rhs_aliases && rhs_mapping_differs) ? other.clone() : other;

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T* rhs_base = static_cast<const T*>(rhs.mStorage->data().get());

        const size_t n = numel();
        if (n == 0) {
            return;
        }

        if (isContiguous() && rhs.isContiguous()) {
            T* lhs_ptr = base + mOffset;
            const T* rhs_ptr = rhs_base + rhs.mOffset;
            for (size_t i = 0; i < n; ++i) {
                lhs_ptr[i] -= rhs_ptr[i];
            }
            return;
        }

        forEachOffset2(mShape, mStride, mOffset, rhs.mStride, rhs.mOffset,
                       [&](size_t off_l, size_t off_r) { base[off_l] -= rhs_base[off_r]; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for multiplication");
    }
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for multiplication");
    }

    const bool rhs_aliases = (mStorage.get() == other.mStorage.get());
    const bool rhs_mapping_differs = (mOffset != other.mOffset) || (mStride != other.mStride);
    Tensor rhs = (rhs_aliases && rhs_mapping_differs) ? other.clone() : other;

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T* rhs_base = static_cast<const T*>(rhs.mStorage->data().get());

        const size_t n = numel();
        if (n == 0) {
            return;
        }

        if (isContiguous() && rhs.isContiguous()) {
            T* lhs_ptr = base + mOffset;
            const T* rhs_ptr = rhs_base + rhs.mOffset;
            for (size_t i = 0; i < n; ++i) {
                lhs_ptr[i] *= rhs_ptr[i];
            }
            return;
        }

        forEachOffset2(mShape, mStride, mOffset, rhs.mStride, rhs.mOffset,
                       [&](size_t off_l, size_t off_r) { base[off_l] *= rhs_base[off_r]; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (other.shape() != shape()) {
        throw std::runtime_error("Shape mismatch for division");
    }
    if (other.dtype() != dtype()) {
        throw std::runtime_error("DType mismatch for division");
    }

    const bool rhs_aliases = (mStorage.get() == other.mStorage.get());
    const bool rhs_mapping_differs = (mOffset != other.mOffset) || (mStride != other.mStride);
    Tensor rhs = (rhs_aliases && rhs_mapping_differs) ? other.clone() : other;

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T* rhs_base = static_cast<const T*>(rhs.mStorage->data().get());

        const size_t n = numel();
        if (n == 0) {
            return;
        }

        if (isContiguous() && rhs.isContiguous()) {
            T* lhs_ptr = base + mOffset;
            const T* rhs_ptr = rhs_base + rhs.mOffset;
            for (size_t i = 0; i < n; ++i) {
                lhs_ptr[i] /= rhs_ptr[i];
            }
            return;
        }

        forEachOffset2(mShape, mStride, mOffset, rhs.mStride, rhs.mOffset,
                       [&](size_t off_l, size_t off_r) { base[off_l] /= rhs_base[off_r]; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator+=(const double scalar) {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T v = static_cast<T>(scalar);

        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] += v;
            }
            return;
        }

        forEachOffset(mShape, mStride, mOffset, [&](size_t off) { base[off] += v; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator-=(const double scalar) {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T v = static_cast<T>(scalar);

        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] -= v;
            }
            return;
        }

        forEachOffset(mShape, mStride, mOffset, [&](size_t off) { base[off] -= v; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator*=(const double scalar) {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T v = static_cast<T>(scalar);

        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] *= v;
            }
            return;
        }

        forEachOffset(mShape, mStride, mOffset, [&](size_t off) { base[off] *= v; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor& Tensor::operator/=(const double scalar) {
    const size_t n = numel();
    if (n == 0) {
        return *this;
    }

    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* base, size_t) {
        using T = std::remove_pointer_t<decltype(base)>;
        const T v = static_cast<T>(scalar);

        if (isContiguous()) {
            T* ptr = base + mOffset;
            for (size_t i = 0; i < n; ++i) {
                ptr[i] /= v;
            }
            return;
        }

        forEachOffset(mShape, mStride, mOffset, [&](size_t off) { base[off] /= v; });
    });

    // Increment version for in-place operation detection
    bumpVersion();

    return *this;
}

Tensor Tensor::operator+(const double scalar) const {
    Tensor result = Tensor::zeros(mShape, dtype(), device());
    const size_t n = numel();
    if (n == 0) {
        return result;
    }

    dispatchByDType(dtype(), result.mStorage->data().get(), result.mStorage->size(),
                    [&](auto* out_base, size_t) {
                        using T = std::remove_pointer_t<decltype(out_base)>;
                        const T* src_base = static_cast<const T*>(mStorage->data().get());
                        const T v = static_cast<T>(scalar);

                        if (isContiguous()) {
                            const T* sp = src_base + mOffset;
                            for (size_t i = 0; i < n; ++i) {
                                out_base[i] = sp[i] + v;
                            }
                            return;
                        }

                        size_t di = 0;
                        forEachOffset(mShape, mStride, mOffset,
                                      [&](size_t off) { out_base[di++] = src_base[off] + v; });
                    });

    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto add_node = std::make_shared<autograd::ScalarAddBackward>();

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn()) {
            next_fns.push_back(gradFn());
        }
        add_node->setNextFunctions(next_fns);
        add_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(add_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::operator-(const double scalar) const {
    Tensor result = Tensor::zeros(mShape, dtype(), device());
    const size_t n = numel();
    if (n == 0) {
        return result;
    }

    dispatchByDType(dtype(), result.mStorage->data().get(), result.mStorage->size(),
                    [&](auto* out_base, size_t) {
                        using T = std::remove_pointer_t<decltype(out_base)>;
                        const T* src_base = static_cast<const T*>(mStorage->data().get());
                        const T v = static_cast<T>(scalar);

                        if (isContiguous()) {
                            const T* sp = src_base + mOffset;
                            for (size_t i = 0; i < n; ++i) {
                                out_base[i] = sp[i] - v;
                            }
                            return;
                        }

                        size_t di = 0;
                        forEachOffset(mShape, mStride, mOffset,
                                      [&](size_t off) { out_base[di++] = src_base[off] - v; });
                    });

    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto sub_node = std::make_shared<autograd::ScalarSubBackward>();

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn()) {
            next_fns.push_back(gradFn());
        }
        sub_node->setNextFunctions(next_fns);
        sub_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(sub_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::operator*(const double scalar) const {
    Tensor result = Tensor::zeros(mShape, dtype(), device());
    const size_t n = numel();
    if (n == 0) {
        return result;
    }

    dispatchByDType(dtype(), result.mStorage->data().get(), result.mStorage->size(),
                    [&](auto* out_base, size_t) {
                        using T = std::remove_pointer_t<decltype(out_base)>;
                        const T* src_base = static_cast<const T*>(mStorage->data().get());
                        const T v = static_cast<T>(scalar);

                        if (isContiguous()) {
                            const T* sp = src_base + mOffset;
                            for (size_t i = 0; i < n; ++i) {
                                out_base[i] = sp[i] * v;
                            }
                            return;
                        }

                        size_t di = 0;
                        forEachOffset(mShape, mStride, mOffset,
                                      [&](size_t off) { out_base[di++] = src_base[off] * v; });
                    });

    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto mul_node = std::make_shared<autograd::ScalarMulBackward>(scalar);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn()) {
            next_fns.push_back(gradFn());
        }
        mul_node->setNextFunctions(next_fns);
        mul_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(mul_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::operator/(const double scalar) const {
    Tensor result = Tensor::zeros(mShape, dtype(), device());
    const size_t n = numel();
    if (n == 0) {
        return result;
    }

    dispatchByDType(dtype(), result.mStorage->data().get(), result.mStorage->size(),
                    [&](auto* out_base, size_t) {
                        using T = std::remove_pointer_t<decltype(out_base)>;
                        const T* src_base = static_cast<const T*>(mStorage->data().get());
                        const T v = static_cast<T>(scalar);

                        if (isContiguous()) {
                            const T* sp = src_base + mOffset;
                            for (size_t i = 0; i < n; ++i) {
                                out_base[i] = sp[i] / v;
                            }
                            return;
                        }

                        size_t di = 0;
                        forEachOffset(mShape, mStride, mOffset,
                                      [&](size_t off) { out_base[di++] = src_base[off] / v; });
                    });

    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto div_node = std::make_shared<autograd::ScalarDivBackward>(scalar);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn()) {
            next_fns.push_back(gradFn());
        }
        div_node->setNextFunctions(next_fns);
        div_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(div_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::flatten() const {
    return reshape({numel()});
}

Tensor Tensor::reshape(const std::vector<size_t>& shape) const {
    if (!isContiguous()) {
        throw std::runtime_error("Tensor is not contiguous");
    }

    // Check new shape has same number of elements as original
    size_t new_numel = 1;
    for (size_t dim : shape) {
        new_numel *= dim;
    }
    if (new_numel != numel()) {
        throw std::runtime_error("Invalid shape for reshape");
    }

    // Calculate new stride
    std::vector<size_t> new_stride = calculateStride(shape);

    Tensor result(mStorage, shape, new_stride, mOffset);

    // Autograd: attach backward function for reshape
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto reshape_node = std::make_shared<autograd::ReshapeBackward>(mShape);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        reshape_node->setNextFunctions(next_fns);
        reshape_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(reshape_node);
        result.requiresGrad(true);
    }

    return result;
}

// Remove single-element dimensions
Tensor Tensor::squeeze() const {
    std::vector<size_t> new_shape;
    std::vector<size_t> new_stride;
    for (size_t i = 0; i < mShape.size(); ++i) {
        if (mShape[i] != 1) {
            new_shape.push_back(mShape[i]);
            new_stride.push_back(mStride[i]);
        }
    }

    // Edge case: if all dimensions are 1, return a 1D tensor
    if (new_shape.empty()) {
        return reshape({1});
    }

    return {mStorage, new_shape, new_stride, mOffset};
}

Tensor Tensor::squeeze(int dim) const {
    int original_dim = dim;
    if (dim < 0) {
        dim += static_cast<int>(mShape.size());
    }
    if (dim < 0 || dim >= static_cast<int>(mShape.size())) {
        throw std::runtime_error("Dimension out of range for squeeze");
    }

    if (mShape[static_cast<size_t>(dim)] != 1) {
        throw std::runtime_error("Dimension is not 1 for squeeze");
    }
    // Calculate new stride
    std::vector<size_t> new_shape;
    std::vector<size_t> new_stride;
    for (size_t i = 0; i < mShape.size(); ++i) {
        if (i != static_cast<size_t>(dim)) {
            new_shape.push_back(mShape[i]);
            new_stride.push_back(mStride[i]);
        }
    }

    if (new_shape.empty()) {
        new_shape = {1};
        new_stride = {1};
    }

    Tensor result(mStorage, new_shape, new_stride, mOffset);

    // Autograd: attach backward function for squeeze
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto squeeze_node = std::make_shared<autograd::SqueezeBackward>(original_dim);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        squeeze_node->setNextFunctions(next_fns);
        squeeze_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(squeeze_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::unsqueeze(int dim) const {
    int original_dim = dim;

    // 1. Handle negative indexing
    int ndim_new = static_cast<int>(mShape.size()) + 1;
    if (dim < 0) {
        dim += ndim_new;
    }

    // 2. Validate dimension (can be 0 to ndim inclusive!)
    if (dim < 0 || dim > static_cast<int>(mShape.size())) {
        throw std::runtime_error("Dimension out of range");
    }

    // 3. Insert size-1 dimension at position
    std::vector<size_t> new_shape = mShape;
    new_shape.insert(new_shape.begin() + dim, 1);

    // 4. Insert stride (use stride of next dimension, or 1 if last)
    std::vector<size_t> new_stride = mStride;
    size_t stride_value = (dim < static_cast<int>(mStride.size())) ? mStride[dim] : 1;
    new_stride.insert(new_stride.begin() + dim, stride_value);

    Tensor result(mStorage, new_shape, new_stride, mOffset);

    // Autograd: attach backward function for unsqueeze
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto unsqueeze_node = std::make_shared<autograd::UnsqueezeBackward>(original_dim);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        unsqueeze_node->setNextFunctions(next_fns);
        unsqueeze_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(unsqueeze_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::transpose() const {
    if (mShape.size() < 2) {
        throw std::runtime_error("Tensor must have at least 2 dimensions for transpose");
    }

    int dim0 = static_cast<int>(mShape.size()) - 2;
    int dim1 = static_cast<int>(mShape.size()) - 1;

    std::vector<size_t> new_shape = mShape;
    std::vector<size_t> new_stride = mStride;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_stride[dim0], new_stride[dim1]);

    Tensor result(mStorage, new_shape, new_stride, mOffset);

    // Autograd: attach backward function for transpose
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto transpose_node = std::make_shared<autograd::TransposeBackward>();

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        transpose_node->setNextFunctions(next_fns);
        transpose_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(transpose_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 == dim1) {
        return *this;
    }

    if (dim0 < 0) {
        dim0 += static_cast<int>(mShape.size());
    }
    if (dim1 < 0) {
        dim1 += static_cast<int>(mShape.size());
    }

    if (dim0 < 0 || dim0 >= static_cast<int>(mShape.size()) || dim1 < 0 ||
        dim1 >= static_cast<int>(mShape.size())) {
        throw std::runtime_error("Dimension out of range for transpose");
    }

    std::vector<size_t> new_shape = mShape;
    std::vector<size_t> new_stride = mStride;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_stride[dim0], new_stride[dim1]);

    Tensor result(mStorage, new_shape, new_stride, mOffset);

    // Autograd: attach backward function for transpose
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto transpose_node = std::make_shared<autograd::TransposeBackward>(dim0, dim1);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        transpose_node->setNextFunctions(next_fns);
        transpose_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(transpose_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    // 1. Validate dims has correct size
    if (dims.size() != mShape.size()) {
        throw std::runtime_error("Permute: dims must have same length as tensor dimensions");
    }

    // 2. Check each dimension appears exactly once
    std::vector<bool> seen(dims.size(), false);
    for (int dim : dims) {
        int d = dim;
        // Handle negative indexing
        if (d < 0) {
            d += static_cast<int>(mShape.size());
        }

        // Validate range and uniqueness
        if (d < 0 || d >= static_cast<int>(mShape.size()) || seen[d]) {
            throw std::runtime_error(
                "Invalid permutation: each dimension must appear exactly once");
        }
        seen[d] = true;
    }

    // 3. Build new shape and stride based on permutation
    std::vector<size_t> new_shape(dims.size());
    std::vector<size_t> new_stride(dims.size());

    for (size_t i = 0; i < dims.size(); ++i) {
        int d = dims[i];
        if (d < 0) {
            d += static_cast<int>(mShape.size());
        }

        new_shape[i] = mShape[d];
        new_stride[i] = mStride[d];
    }

    Tensor result = Tensor(mStorage, new_shape, new_stride, mOffset);

    // Autograd: attach backward function for permute
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto permute_node = std::make_shared<autograd::PermuteBackward>(dims);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        permute_node->setNextFunctions(next_fns);
        permute_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(permute_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::slice(int dim, const size_t start, const size_t end) const {
    // Save original dim for autograd before handling negative indexing
    int original_dim = dim;

    // 1. Handle negative dim (like Python: -1 = last dim)
    if (dim < 0) {
        dim += static_cast<int>(mShape.size());
    }

    // 2. Validate dim is in range [0, ndim)
    if (dim < 0 || dim >= static_cast<int>(mShape.size())) {
        throw std::out_of_range("Dimension out of range for slice");
    }

    // 3. Validate start < end && end <= shape[dim]
    if (start >= end || end > mShape[static_cast<size_t>(dim)]) {
        throw std::out_of_range("Slice start and end indices are invalid");
    }

    // 4. Build new shape (same as mShape, but dim has size end-start)
    std::vector<size_t> new_shape = mShape;
    new_shape[static_cast<size_t>(dim)] = end - start;
    std::vector<size_t> new_stride = mStride;

    // 5. Calculate new offset: mOffset + start * mStride[dim]
    size_t new_offset = mOffset + start * mStride[static_cast<size_t>(dim)];

    // 6. Create result tensor with same storage, new shape, same stride, new offset
    Tensor result = Tensor(mStorage, new_shape, new_stride, new_offset);

    // Autograd: attach backward function for slice
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto slice_node =
            std::make_shared<autograd::SliceBackward>(mShape, original_dim, start, end);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        slice_node->setNextFunctions(next_fns);
        slice_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(slice_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::sum() const {
    // Sum all visible elements in the tensor view (respects stride/offset)
    double sum = 0.0;
    size_t total_elements = numel();

    dispatchByDType(dtype(), mStorage->data().get(), total_elements, [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        // Iterate through visible elements using multi-dimensional indexing
        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            // Calculate actual storage offset using strides
            size_t offset = mOffset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * mStride[dim];
            }

            sum += static_cast<double>(ptr[offset]);

            // Increment multi-dimensional index (row-major order)
            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < mShape[dim]) {
                    break;  // No carry needed
                }
                indices[dim] = 0;  // Carry to next dimension
            }
        }
    });
    Tensor result = Tensor::full({1}, sum, dtype(), device());

    // Autograd: attach backward function for sum
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto sum_node = std::make_shared<autograd::SumBackward>(mShape);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        sum_node->setNextFunctions(next_fns);
        sum_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(sum_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for sum");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
            // else: skip this dimension
        } else {
            out_shape.push_back(mShape[i]);
        }
    }

    // Handle edge case: reducing to scalar
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());

    // 4. Get total number of output elements
    size_t out_numel = result.numel();

    // 5. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            // Insert the reduced dimension back
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;  // Will iterate over this
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Sum over the reduced dimension
                            T acc = 0;
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;

                                // Calculate input offset using strides
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }

                                acc += src_ptr[in_offset];
                            }

                            dst_ptr[out_linear] = acc;
                        }
                    });

    // Autograd: attach backward function for sum
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto sum_node = std::make_shared<autograd::SumBackward>(mShape);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        sum_node->setNextFunctions(next_fns);
        sum_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(sum_node);
        result.requiresGrad(true);
    }

    return result;
}

// ============================================================================
// Mean Reductions
// ============================================================================

Tensor Tensor::mean() const {
    // Compute mean of all visible elements in the tensor view (respects stride/offset)
    double total = 0.0;
    size_t total_elements = numel();

    dispatchByDType(dtype(), mStorage->data().get(), total_elements, [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        // Iterate through visible elements using multi-dimensional indexing
        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            // Calculate actual storage offset using strides
            size_t offset = mOffset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * mStride[dim];
            }

            total += static_cast<double>(ptr[offset]);

            // Increment multi-dimensional index
            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < mShape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    });
    Tensor result = Tensor::full({1}, total / static_cast<double>(numel()), dtype(), device());

    // Autograd: attach backward function for mean
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto mean_node = std::make_shared<autograd::MeanBackward>(mShape, numel());

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        mean_node->setNextFunctions(next_fns);
        mean_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(mean_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for mean");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
        } else {
            out_shape.push_back(mShape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());
    size_t out_numel = result.numel();

    // 4. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Sum over the reduced dimension, then divide
                            T acc = 0;
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }
                                acc += src_ptr[in_offset];
                            }

                            dst_ptr[out_linear] = acc / static_cast<T>(reduce_size);
                        }
                    });

    // Autograd: attach backward function for mean
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto mean_node = std::make_shared<autograd::MeanBackward>(mShape, reduce_size);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        mean_node->setNextFunctions(next_fns);
        mean_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(mean_node);
        result.requiresGrad(true);
    }

    return result;
}

// ============================================================================
// Activation Functions
// ============================================================================

Tensor Tensor::relu() const {
    // ReLU: y = max(0, x) - element-wise
    Tensor result = Tensor::zeros(mShape, dtype(), device());

    // Apply ReLU to all elements
    Tensor input_flat = contiguous().flatten();
    Tensor result_flat = result.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto result_acc = result_flat.accessor<float, 1>();

    size_t n = input_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        result_acc[i] = input_acc[i] > 0.0f ? input_acc[i] : 0.0f;
    }

    // Autograd: attach backward function for ReLU
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto relu_node = std::make_shared<autograd::ReLUBackward>(*this);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        relu_node->setNextFunctions(next_fns);
        relu_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(relu_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::sigmoid() const {
    // Sigmoid: y = 1 / (1 + exp(-x)) - element-wise
    // Numerical stability: for large |x|, use different formulas
    //   If x >= 0:  sigmoid(x) = 1 / (1 + exp(-x))
    //   If x < 0:   sigmoid(x) = exp(x) / (1 + exp(x))
    Tensor result = Tensor::zeros(mShape, dtype(), device());

    Tensor input_flat = contiguous().flatten();
    Tensor result_flat = result.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto result_acc = result_flat.accessor<float, 1>();

    size_t n = input_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        float x = input_acc[i];
        if (x >= 0.0f) {
            result_acc[i] = 1.0f / (1.0f + std::exp(-x));
        } else {
            float exp_x = std::exp(x);
            result_acc[i] = exp_x / (1.0f + exp_x);
        }
    }

    // Autograd: attach backward function for Sigmoid
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto sigmoid_node = std::make_shared<autograd::SigmoidBackward>(result);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        sigmoid_node->setNextFunctions(next_fns);
        sigmoid_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(sigmoid_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::tanh() const {
    // Tanh: y = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) - element-wise
    // Using std::tanh for numerical stability
    Tensor result = Tensor::zeros(mShape, dtype(), device());

    Tensor input_flat = contiguous().flatten();
    Tensor result_flat = result.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto result_acc = result_flat.accessor<float, 1>();

    size_t n = input_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        result_acc[i] = std::tanh(input_acc[i]);
    }

    // Autograd: attach backward function for Tanh
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto tanh_node = std::make_shared<autograd::TanhBackward>(result);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        tanh_node->setNextFunctions(next_fns);
        tanh_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(tanh_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::softmax(int dim) const {
    // Softmax: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x))) along dimension
    // Using log-sum-exp trick for numerical stability

    // Handle negative dimension indexing
    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim += static_cast<int>(ndim());
    }

    // Compute max along dimension (keepdim=true for broadcasting)
    Tensor x_max = max(actual_dim, true);

    // Subtract max for numerical stability: x_shifted = x - max(x)
    Tensor x_shifted = *this - x_max;

    // Compute exp(x_shifted)
    Tensor exp_x = x_shifted.exp();

    // Sum along dimension (keepdim=true for broadcasting)
    Tensor exp_sum = exp_x.sum(actual_dim, true);

    // Compute softmax: exp_x / exp_sum (broadcasting)
    Tensor result = exp_x / exp_sum;

    // Autograd: attach backward function for Softmax
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto softmax_node = std::make_shared<autograd::SoftmaxBackward>(result, actual_dim);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        softmax_node->setNextFunctions(next_fns);
        softmax_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(softmax_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::logSoftmax(int dim) const {
    // LogSoftmax: y_i = x_i - max(x) - log(sum(exp(x_j - max(x)))) along dimension
    // More numerically stable than log(softmax(x))

    // Handle negative dimension indexing
    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim += static_cast<int>(ndim());
    }

    // Compute max along dimension (keepdim=true for broadcasting)
    Tensor x_max = max(actual_dim, true);

    // Subtract max for numerical stability: x_shifted = x - max(x)
    Tensor x_shifted = *this - x_max;

    // Compute exp(x_shifted)
    Tensor exp_x = x_shifted.exp();

    // Sum along dimension (keepdim=true for broadcasting)
    Tensor exp_sum = exp_x.sum(actual_dim, true);

    // Compute log(sum(exp(x_shifted)))
    Tensor log_sum_exp = exp_sum.log();

    // Compute log_softmax: x_shifted - log_sum_exp
    Tensor result = x_shifted - log_sum_exp;

    // Autograd: attach backward function for LogSoftmax
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto log_softmax_node = std::make_shared<autograd::LogSoftmaxBackward>(result, actual_dim);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        log_softmax_node->setNextFunctions(next_fns);
        log_softmax_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(log_softmax_node);
        result.requiresGrad(true);
    }

    return result;
}

// ============================================================================
// Element-wise Math Operations
// ============================================================================

Tensor Tensor::exp() const {
    // Exp: y = e^x - element-wise
    Tensor result = Tensor::zeros(mShape, dtype(), device());

    // Apply exp to all elements
    Tensor input_flat = contiguous().flatten();
    Tensor result_flat = result.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto result_acc = result_flat.accessor<float, 1>();

    size_t n = input_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        result_acc[i] = std::exp(input_acc[i]);
    }

    // Autograd: attach backward function for exp
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto exp_node = std::make_shared<autograd::ExpBackward>(result);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        exp_node->setNextFunctions(next_fns);
        exp_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(exp_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::log() const {
    // Log: y = ln(x) - element-wise natural logarithm
    Tensor result = Tensor::zeros(mShape, dtype(), device());

    // Apply log to all elements
    Tensor input_flat = contiguous().flatten();
    Tensor result_flat = result.flatten();

    auto input_acc = input_flat.accessor<float, 1>();
    auto result_acc = result_flat.accessor<float, 1>();

    size_t n = input_flat.numel();
    for (size_t i = 0; i < n; ++i) {
        result_acc[i] = std::log(input_acc[i]);
    }

    // Autograd: attach backward function for log
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto log_node = std::make_shared<autograd::LogBackward>(*this);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        log_node->setNextFunctions(next_fns);
        log_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(log_node);
        result.requiresGrad(true);
    }

    return result;
}

// ============================================================================
// Variance Reductions
// ============================================================================

Tensor Tensor::var() const {
    // Compute variance of visible elements: E[(X - μ)²] = E[X²] - μ²
    // Using two-pass algorithm for numerical stability (respects stride/offset)
    size_t total_elements = numel();

    // First pass: compute mean
    double mean_val = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), total_elements, [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            size_t offset = mOffset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * mStride[dim];
            }

            mean_val += static_cast<double>(ptr[offset]);

            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < mShape[dim])
                    break;
                indices[dim] = 0;
            }
        }
    });
    mean_val /= static_cast<double>(total_elements);

    // Second pass: compute mean of squared deviations
    double var_val = 0.0;
    dispatchByDType(dtype(), mStorage->data().get(), total_elements, [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            size_t offset = mOffset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * mStride[dim];
            }

            double diff = static_cast<double>(ptr[offset]) - mean_val;
            var_val += diff * diff;

            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < mShape[dim])
                    break;
                indices[dim] = 0;
            }
        }
    });
    var_val /= static_cast<double>(total_elements);

    return Tensor::full({1}, var_val, dtype(), device());
}

Tensor Tensor::var(int dim, bool keepdim) const {
    // For now, compute using tensor operations: ((x - mean) ** 2).mean(dim)
    // This is less efficient but reuses existing code

    Tensor mean_val = mean(dim, true);  // keepdim=true for broadcasting
    Tensor centered = *this - mean_val;
    Tensor squared = centered * centered;
    return squared.mean(dim, keepdim);
}

// ============================================================================
// Max Reductions
// ============================================================================

Tensor Tensor::max() const {
    // Find maximum of visible elements in the tensor view (respects stride/offset)
    double max_val = std::numeric_limits<double>::lowest();
    size_t total_elements = numel();

    dispatchByDType(dtype(), mStorage->data().get(), total_elements, [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        // Iterate through visible elements using multi-dimensional indexing
        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            // Calculate actual storage offset using strides
            size_t offset = mOffset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * mStride[dim];
            }

            double val = static_cast<double>(ptr[offset]);
            if (val > max_val)
                max_val = val;

            // Increment multi-dimensional index
            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < mShape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    });
    Tensor result = Tensor::full({1}, max_val, dtype(), device());

    // Attach autograd node if input requires gradients
    // For global max, treat it as max along all dimensions (flattened)
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto max_node = std::make_shared<autograd::MaxBackward>(this->contiguous().flatten(),
                                                                result, -1, false);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        max_node->setNextFunctions(next_fns);
        max_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(max_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::max(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for max");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
        } else {
            out_shape.push_back(mShape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());
    size_t out_numel = result.numel();

    // 4. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Find max over the reduced dimension
                            T acc = std::numeric_limits<T>::lowest();
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }
                                if (src_ptr[in_offset] > acc) {
                                    acc = src_ptr[in_offset];
                                }
                            }

                            dst_ptr[out_linear] = acc;
                        }
                    });

    // Attach autograd node if input requires gradients
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto max_node = std::make_shared<autograd::MaxBackward>(*this, result, dim, keepdim);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        max_node->setNextFunctions(next_fns);
        max_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(max_node);
        result.requiresGrad(true);
    }

    return result;
}

// ============================================================================
// Min Reductions
// ============================================================================

Tensor Tensor::min() const {
    // Find minimum of visible elements in the tensor view (respects stride/offset)
    double min_val = std::numeric_limits<double>::max();
    size_t total_elements = numel();

    dispatchByDType(dtype(), mStorage->data().get(), total_elements, [&](auto* ptr, size_t n) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        // Iterate through visible elements using multi-dimensional indexing
        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            // Calculate actual storage offset using strides
            size_t offset = mOffset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * mStride[dim];
            }

            double val = static_cast<double>(ptr[offset]);
            if (val < min_val)
                min_val = val;

            // Increment multi-dimensional index
            for (int dim = static_cast<int>(indices.size()) - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < mShape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    });
    Tensor result = Tensor::full({1}, min_val, dtype(), device());

    // Autograd: attach backward function for min (Note: scalar min picks arbitrary element)
    // Min of all elements doesn't have a well-defined gradient for all inputs,
    // only for the minimum value position. We use MinBackward with dim=-1 to handle this.
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto min_node = std::make_shared<autograd::MinBackward>(*this, result, -1, true);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        min_node->setNextFunctions(next_fns);
        min_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(min_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::min(int dim, bool keepdim) const {
    // 1. Handle negative dim & validate
    if (dim < 0)
        dim += static_cast<int>(ndim());
    if (dim < 0 || dim >= static_cast<int>(ndim())) {
        throw std::out_of_range("Dimension out of range for min");
    }
    size_t reduce_dim = static_cast<size_t>(dim);
    size_t reduce_size = mShape[reduce_dim];

    // 2. Build output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == reduce_dim) {
            if (keepdim)
                out_shape.push_back(1);
        } else {
            out_shape.push_back(mShape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // 3. Create output tensor
    Tensor result = Tensor::zeros(out_shape, dtype(), device());
    size_t out_numel = result.numel();

    // 4. Iterate over all output positions
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(),
                    [&, this](auto* src_ptr, size_t) {
                        using T = std::remove_pointer_t<decltype(src_ptr)>;
                        T* dst_ptr = static_cast<T*>(result.mStorage->data().get());

                        for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                            // Convert linear output index to multi-dimensional index
                            std::vector<size_t> out_idx(out_shape.size());
                            size_t temp = out_linear;
                            for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
                                out_idx[i] = temp % out_shape[i];
                                temp /= out_shape[i];
                            }

                            // Build input index from output index
                            std::vector<size_t> in_idx(ndim());
                            size_t out_pos = 0;
                            for (size_t i = 0; i < ndim(); ++i) {
                                if (i == reduce_dim) {
                                    in_idx[i] = 0;
                                } else {
                                    in_idx[i] = out_idx[out_pos++];
                                }
                            }

                            // Find min over the reduced dimension
                            T acc = std::numeric_limits<T>::max();
                            for (size_t r = 0; r < reduce_size; ++r) {
                                in_idx[reduce_dim] = r;
                                size_t in_offset = mOffset;
                                for (size_t i = 0; i < ndim(); ++i) {
                                    in_offset += in_idx[i] * mStride[i];
                                }
                                if (src_ptr[in_offset] < acc) {
                                    acc = src_ptr[in_offset];
                                }
                            }

                            dst_ptr[out_linear] = acc;
                        }
                    });

    // Autograd: attach backward function for min
    if (requiresGrad() && !autograd::NoGradMode::isEnabled()) {
        auto min_node = std::make_shared<autograd::MinBackward>(*this, result, dim, keepdim);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        min_node->setNextFunctions(next_fns);
        min_node->setInputTensors({std::make_shared<Tensor>(*this)});

        result.setGradFn(min_node);
        result.requiresGrad(true);
    }

    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    if (dtype() != other.dtype()) {
        throw std::runtime_error("Dot product requires matching dtypes");
    }

    if (ndim() != 1 || other.ndim() != 1) {
        throw std::runtime_error("Dot product requires 1D tensors");
    }
    if (mShape[0] != other.mShape[0]) {
        throw std::runtime_error("Dot product requires matching sizes");
    }

    // Dot product respecting stride/offset for 1D tensors
    double result = 0.0;
    size_t n = mShape[0];

    dispatchByDType(dtype(), mStorage->data().get(), n, [&](auto* ptr, size_t) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* other_ptr = static_cast<const T*>(other.mStorage->data().get());

        // Use strides to access elements (handles views like slices)
        size_t stride_a = mStride[0];
        size_t stride_b = other.mStride[0];
        size_t offset_a = mOffset;
        size_t offset_b = other.mOffset;

        for (size_t i = 0; i < n; ++i) {
            result += static_cast<double>(ptr[offset_a + i * stride_a]) *
                      static_cast<double>(other_ptr[offset_b + i * stride_b]);
        }
    });

    Tensor output = Tensor::full({1}, result, dtype(), device());

    // Autograd: attach backward function for dot product
    if ((requiresGrad() || other.requiresGrad()) && !autograd::NoGradMode::isEnabled()) {
        auto dot_node = std::make_shared<autograd::DotBackward>(*this, other);

        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        if (other.gradFn())
            next_fns.push_back(other.gradFn());
        dot_node->setNextFunctions(next_fns);
        dot_node->setInputTensors(
            {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)});

        output.setGradFn(dot_node);
        output.requiresGrad(true);
    }

    return output;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // 1. Validate: both must be 2D
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }

    // 2. Validate: inner dimensions must match [M, K] × [K, N]
    if (mShape[1] != other.mShape[0]) {
        throw std::runtime_error("matmul: inner dimensions must match. Got [" +
                                 std::to_string(mShape[0]) + ", " + std::to_string(mShape[1]) +
                                 "] × [" + std::to_string(other.mShape[0]) + ", " +
                                 std::to_string(other.mShape[1]) + "]");
    }

    // 3. Validate: same dtype
    if (dtype() != other.dtype()) {
        throw std::runtime_error("matmul requires matching dtypes");
    }

    size_t M = mShape[0];
    size_t K = mShape[1];
    size_t N = other.mShape[1];

    // 4. Create output tensor [M, N]
    Tensor result = Tensor::zeros({M, N}, dtype(), device());

    // 5. Compute: triple nested loop (naïve O(M×N×K) implementation)
    // Uses strides to support non-contiguous tensors (e.g., transposed views)
    dispatchByDType(dtype(), result.mStorage->data().get(), M * N, [&](auto* c_ptr, size_t) {
        using T = std::remove_pointer_t<decltype(c_ptr)>;
        const T* a_ptr = static_cast<const T*>(mStorage->data().get()) + mOffset;
        const T* b_ptr = static_cast<const T*>(other.mStorage->data().get()) + other.mOffset;

        // Get strides for proper indexing (handles transposed/non-contiguous tensors)
        const size_t a_stride0 = mStride[0];
        const size_t a_stride1 = mStride[1];
        const size_t b_stride0 = other.mStride[0];
        const size_t b_stride1 = other.mStride[1];

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    // A[i,k] × B[k,j] using strides
                    sum +=
                        a_ptr[i * a_stride0 + k * a_stride1] * b_ptr[k * b_stride0 + j * b_stride1];
                }
                c_ptr[i * N + j] = sum;
            }
        }
    });

    // AUTOGRAD: Attach backward function for matmul
    if ((requiresGrad() || other.requiresGrad()) && !autograd::NoGradMode::isEnabled()) {
        // Create MatmulBackward node
        auto matmul_node = std::make_shared<autograd::MatmulBackward>(*this, other);

        // Set up next_functions (predecessor nodes for graph traversal)
        std::vector<std::shared_ptr<autograd::Node>> next_fns;
        if (gradFn())
            next_fns.push_back(gradFn());
        if (other.gradFn())
            next_fns.push_back(other.gradFn());
        matmul_node->setNextFunctions(next_fns);

        // Set up inputTensors (for gradient accumulation)
        matmul_node->setInputTensors(
            {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)});

        // Attach to result
        result.setGradFn(matmul_node);
        result.requiresGrad(true);
    }

    return result;
}

// ============================================================================
// Autograd Methods
// ============================================================================

Tensor& Tensor::requiresGrad(bool requires_grad) {
    if (requires_grad) {
        if (!mAutogradMeta) {
            mAutogradMeta = std::make_shared<autograd::AutogradMeta>();
        }
        mAutogradMeta->requiresGrad = true;
    } else {
        if (mAutogradMeta) {
            mAutogradMeta->requiresGrad = false;
        }
    }
    return *this;
}

bool Tensor::requiresGrad() const {
    if (!mAutogradMeta) {
        return false;
    }
    return mAutogradMeta->requiresGrad;
}

bool Tensor::isLeaf() const {
    if (!mAutogradMeta) {
        return true;
    }
    return mAutogradMeta->isLeaf;
}

std::shared_ptr<Tensor> Tensor::grad() const {
    if (!mAutogradMeta) {
        return nullptr;
    }
    return mAutogradMeta->grad;
}

void Tensor::zeroGrad() {
    if (!mAutogradMeta || mAutogradMeta->grad == nullptr) {
        return;
    }
    mAutogradMeta->grad->zero();
}

void Tensor::setGradFn(std::shared_ptr<loom::autograd::Node> grad_fn) {
    if (!mAutogradMeta) {
        requiresGrad(true);
    }
    mAutogradMeta->gradFn = grad_fn;
    mAutogradMeta->isLeaf = false;
    return;
}

std::shared_ptr<loom::autograd::Node> Tensor::gradFn() const {
    if (!mAutogradMeta) {
        return nullptr;
    }
    return mAutogradMeta->gradFn;
}

uint64_t Tensor::version() const {
    if (!mAutogradMeta) {
        return 0;  // No autograd metadata, version doesn't matter
    }
    return mAutogradMeta->version;
}

void Tensor::bumpVersion() {
    if (mAutogradMeta) {
        mAutogradMeta->version++;
    }
}

void Tensor::accumulateGrad(const Tensor& g) {
    // Lazy initialize autograd metadata if needed
    if (!mAutogradMeta) {
        mAutogradMeta = std::make_shared<autograd::AutogradMeta>();
    }

    // Initialize gradient tensor on first accumulation
    if (!mAutogradMeta->grad) {
        mAutogradMeta->grad = std::make_shared<Tensor>(Tensor::zeros(mShape, dtype(), device()));
    }

    // Accumulate gradient (supports multiple backward paths)
    *mAutogradMeta->grad += g;
}

void Tensor::backward(const Tensor& grad) {
    autograd::Engine::backward(*this, grad);
}

// ============================================================================
// Visualization
// ============================================================================

void Tensor::print(const std::string& name) const {
    auto& logger = Logger::getInstance("Tensor");

    // Build shape string
    std::ostringstream shape_str;
    shape_str << "[";
    for (size_t i = 0; i < mShape.size(); ++i) {
        shape_str << mShape[i];
        if (i < mShape.size() - 1)
            shape_str << ", ";
    }
    shape_str << "]";

    // Build stride string
    std::ostringstream stride_str;
    stride_str << "[";
    for (size_t i = 0; i < mStride.size(); ++i) {
        stride_str << mStride[i];
        if (i < mStride.size() - 1)
            stride_str << ", ";
    }
    stride_str << "]";

    // Print metadata
    logger.info("========================================");
    logger.info("{}", name);
    logger.info("========================================");
    logger.info("Shape:      {}", shape_str.str());
    logger.info("Stride:     {}", stride_str.str());
    logger.info("Offset:     {}", mOffset);
    logger.info("DType:      {}", loom::name(dtype()));
    logger.info("Device:     {}", device().toString());
    logger.info("Elements:   {}", numel());
    logger.info("Contiguous: {}", isContiguous() ? "Yes" : "No");
    logger.info("----------------------------------------");

    // Limit output for large tensors
    const size_t MAX_ELEMENTS = 100;
    bool truncated = numel() > MAX_ELEMENTS;
    size_t elements_to_print = truncated ? MAX_ELEMENTS : numel();

    if (numel() == 0) {
        logger.info("Data: (empty)");
        logger.info("========================================");
        return;
    }

    // Print data based on dimensionality
    dispatchByDType(dtype(), mStorage->data().get(), mStorage->size(), [&](auto* ptr, size_t) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T* data = static_cast<const T*>(ptr);

        std::ostringstream data_str;
        data_str << std::fixed << std::setprecision(4);

        if (ndim() == 1) {
            // 1D: Print as simple list
            data_str << "[";
            for (size_t i = 0; i < elements_to_print; ++i) {
                data_str << data[mOffset + i * mStride[0]];
                if (i < elements_to_print - 1)
                    data_str << ", ";
            }
            if (truncated)
                data_str << ", ...";
            data_str << "]";
            logger.info("Data: {}", data_str.str());

        } else if (ndim() == 2) {
            // 2D: Print as matrix
            logger.info("Data:");
            size_t rows = mShape[0];
            size_t cols = mShape[1];

            for (size_t i = 0; i < rows && i * cols < elements_to_print; ++i) {
                data_str.str("");
                data_str << "  [";
                size_t max_cols = std::min(cols, (elements_to_print - i * cols));
                for (size_t j = 0; j < max_cols; ++j) {
                    size_t idx = mOffset + i * mStride[0] + j * mStride[1];
                    data_str << std::setw(8) << data[idx];
                    if (j < max_cols - 1)
                        data_str << ", ";
                }
                if (truncated && max_cols < cols)
                    data_str << ", ...";
                data_str << "]";
                logger.info("{}", data_str.str());
            }
            if (truncated && rows * cols > elements_to_print) {
                logger.info("  ...");
            }

        } else if (ndim() == 3) {
            // 3D: Print as separate matrices
            logger.info("Data:");
            size_t depth = mShape[0];
            size_t rows = mShape[1];
            size_t cols = mShape[2];

            for (size_t d = 0; d < depth; ++d) {
                logger.info("  Matrix [{}]:", d);
                for (size_t i = 0; i < rows; ++i) {
                    data_str.str("");
                    data_str << "    [";
                    for (size_t j = 0; j < cols; ++j) {
                        size_t idx = mOffset + d * mStride[0] + i * mStride[1] + j * mStride[2];
                        data_str << std::setw(8) << data[idx];
                        if (j < cols - 1)
                            data_str << ", ";
                    }
                    data_str << "]";
                    logger.info("{}", data_str.str());
                }
            }

        } else {
            // 4D+: Print flattened view with indices
            logger.info("Data (first {} elements, shape too complex for matrix view):",
                        elements_to_print);
            data_str << "[";
            for (size_t i = 0; i < elements_to_print; ++i) {
                // For now, just print in storage order (contiguous assumption)
                data_str << data[mOffset + i];
                if (i < elements_to_print - 1)
                    data_str << ", ";
                // Add newline every 10 elements for readability
                if ((i + 1) % 10 == 0 && i < elements_to_print - 1) {
                    data_str << "\n ";
                }
            }
            if (truncated)
                data_str << ", ...";
            data_str << "]";
            logger.info("{}", data_str.str());
        }
    });

    logger.info("========================================");
}

}  // namespace loom
