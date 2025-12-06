#pragma once

#include <cstddef>
#include <cstdint>

#include "loom/logger.h"

namespace loom {
namespace memory {

class MemoryUtils {
  public:
    MemoryUtils() = delete;

    /**
     * @brief Detect optimal SIMD alignment based on CPU capabilities
     *
     * Returns alignment required for SIMD operations:
     * - 64 bytes for AVX-512
     * - 32 bytes for AVX/AVX2
     * - 16 bytes for SSE or ARM NEON
     */
    [[nodiscard]] static size_t detectSIMDAlignment();

    /**
     * @brief Get system cache line size
     *
     * Returns the L1 data cache line size, typically 64 bytes on modern CPUs.
     * Useful for preventing false sharing in concurrent code.
     */
    [[nodiscard]] static size_t getCacheLineSize();

    /**
     * @brief Get system page size
     *
     * Returns the memory page size, typically 4096 bytes.
     * Useful for large allocations and memory mapping.
     */
    [[nodiscard]] static size_t getPageSize();

    /**
     * @brief Get default alignment for general allocations
     *
     * Returns a sensible default alignment that balances:
     * - SIMD requirements
     * - Cache line alignment
     * - Common use cases
     */
    [[nodiscard]] static size_t getDefaultAlignment();

    /**
     * @brief Round size up to multiple of alignment
     *
     * @param size Original size in bytes
     * @param alignment Required alignment (must be power of 2)
     * @return Size rounded up to nearest multiple of alignment
     */
    [[nodiscard]] static inline size_t alignSize(size_t size, size_t alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    }

    /**
     * @brief Check if pointer is aligned
     *
     * @param ptr Pointer to check
     * @param alignment Required alignment
     * @return true if ptr is aligned to specified boundary
     */
    [[nodiscard]] static inline bool isAligned(const void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }

  private:
    [[nodiscard]] static Logger& getLogger();
};
}  // namespace memory
}  // namespace loom
