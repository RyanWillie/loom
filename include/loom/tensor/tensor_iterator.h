#pragma once
#include <cstddef>
#include <vector>

namespace loom {

/// @brief Lightweight iterator for traversing tensor elements in logical order
/// while respecting physical layout (stride/offset).
///
/// TensorIterator abstracts the complexity of multi-dimensional indexing,
/// allowing operations to iterate over tensor elements regardless of their
/// memory layout (contiguous, transposed, sliced, etc.).
///
/// Usage:
/// @code
///   TensorIterator it(tensor.shape(), tensor.stride(), tensor.offset());
///   while (it.hasNext()) {
///       T value = data[it.offset()];
///       // ... process value ...
///       it.next();
///   }
/// @endcode
class TensorIterator {
  public:
    /// @brief Construct iterator for a tensor view
    /// @param shape Logical shape of the tensor
    /// @param stride Stride for each dimension (in elements)
    /// @param base_offset Starting offset in storage (in elements)
    TensorIterator(const std::vector<size_t>& shape, const std::vector<size_t>& stride,
                   size_t base_offset);

    /// @brief Get current storage offset (in elements from storage base)
    /// @return Offset suitable for indexing into typed array (e.g., float* base[offset()])
    [[nodiscard]] inline size_t offset() const { return mCurrentOffset; }

    /// @brief Check if there are more elements to iterate
    /// @return true if more elements remain, false if iteration is complete
    [[nodiscard]] inline bool hasNext() const { return mPosition < mTotalElements; }

    /// @brief Advance to next element
    /// Updates internal state to point to the next logical element.
    /// Behavior is undefined if hasNext() is false.
    inline void next();

    /// @brief Get current linear position in iteration (0-indexed)
    /// @return Linear index of current element (0 to numel-1)
    [[nodiscard]] inline size_t position() const { return mPosition; }

    /// @brief Get total number of elements being iterated
    /// @return Total elements (product of shape dimensions)
    [[nodiscard]] inline size_t totalElements() const { return mTotalElements; }

    /// @brief Reset iterator to beginning
    void reset();

  private:
    std::vector<size_t> mIndices;  // Current multi-dimensional index
    const std::vector<size_t>& mShape;
    const std::vector<size_t>& mStride;
    const size_t mBaseOffset;
    size_t mCurrentOffset;  // Current offset in storage
    size_t mPosition;       // Linear position (0 to numel-1)
    size_t mTotalElements;  // Total number of elements

    // Helper to compute total elements
    static size_t computeNumel(const std::vector<size_t>& shape);
};

// ============================================================================
// Inline Implementations (for performance)
// ============================================================================

inline void TensorIterator::next() {
    // Move to next position
    ++mPosition;

    // If we've exhausted all elements, stop
    if (mPosition >= mTotalElements) {
        return;
    }

    // Increment multi-dimensional index (row-major order)
    // Start from the last dimension (fastest-changing)
    for (int dim = static_cast<int>(mIndices.size()) - 1; dim >= 0; --dim) {
        ++mIndices[dim];

        // If no overflow in this dimension, update offset and we're done
        if (mIndices[dim] < mShape[dim]) {
            mCurrentOffset += mStride[dim];
            return;
        }

        // Overflow: reset this dimension and carry to next
        // Subtract the offset we accumulated in this dimension
        mCurrentOffset -= mIndices[dim] * mStride[dim];
        mIndices[dim] = 0;
    }
}

}  // namespace loom
