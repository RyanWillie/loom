#include "loom/memory/utils.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "loom/logger.h"

#ifdef __linux__
#include <unistd.h>
#endif

namespace loom {
namespace memory {

Logger& MemoryUtils::getLogger() {
    static Logger& logger = Logger::getInstance("MemoryUtils");
    return logger;
}

size_t MemoryUtils::detectSIMDAlignment() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (__builtin_cpu_supports("avx512f")) {
        getLogger().trace("AVX-512 supported");
        return 64;
    }
    if (__builtin_cpu_supports("avx") || __builtin_cpu_supports("avx2")) {
        getLogger().trace("AVX/AVX2 supported");
        return 32;
    }
    if (__builtin_cpu_supports("sse")) {
        getLogger().trace("SSE supported");
        return 16;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    getLogger().trace("ARM64 supported");
    return 16;
#endif
    getLogger().trace("No SIMD supported");
    return alignof(std::max_align_t);
}

size_t MemoryUtils::getCacheLineSize() {
// Stub implementation
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
    long size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (size > 0) {
        getLogger().trace("Cache line size: {}", size);
        return static_cast<size_t>(size);
    }
#elif defined(_WIN32)
    getLogger().trace("Windows supported");
    return 64;
#else
    getLogger().trace("Unsupported platform");
    return 64;
#endif

    getLogger().warning("Failed to get cache line size");
    return 64;
}

size_t MemoryUtils::getPageSize() {
#ifdef _SC_PAGESIZE
    long size = sysconf(_SC_PAGESIZE);
    if (size > 0) {
        getLogger().trace("Page size: {}", size);
        return static_cast<size_t>(size);
    }
#elif defined(_WIN32)
    getLogger().trace("Windows supported");
    return 64;
#endif
    getLogger().warning("Failed to get page size");
    return 4096;
}

size_t MemoryUtils::getDefaultAlignment() {
    // Return the maximum of the SIMD alignment and the cache line size
    size_t alignment = detectSIMDAlignment();
    size_t cache_line_size = getCacheLineSize();
    const size_t result = std::max(alignment, cache_line_size);
    getLogger().trace("Default alignment: {}", result);
    return result;
}

}  // namespace memory
}  // namespace loom
