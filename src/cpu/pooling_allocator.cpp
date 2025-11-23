#include "cpu/pooling_allocator.h"

#include <cstdlib>

#include "common/logger.h"

namespace loom {

// Define the static sentinel for zero-size allocations
char PoolingAllocator::sZeroSizeSentinel = 0;

/**
 * @brief Header stored before each allocated block
 *
 * Memory layout:
 * [BlockHeader][padding...][void* back_ptr][user data (aligned to requested alignment)]
 *  ^                        ^               ^
 *  |                        |               returned to user (aligned!)
 *  raw_ptr             points to raw_ptr
 *
 * The header stores the size and offset to the user data.
 * A back-pointer is stored immediately before the user data to enable O(1) deallocation.
 * We allocate extra space to ensure both the back-pointer and user data are properly aligned.
 */
struct BlockHeader {
    size_t size;    // Size of user data (aligned)
    size_t offset;  // Offset from header to user data
};

PoolingAllocator::PoolingAllocator(const PoolingAllocatorConfig& config) : mConfig(config) {
    if (config.alignment == 0) {
        mConfig.alignment = memory::MemoryUtils::getDefaultAlignment();
    }
}

PoolingAllocator::~PoolingAllocator() {
    //! Deallocate all pooled blocks
    for (auto& [size, blocks] : mFreeBlocks) {
        for (void* ptr : blocks) {
            std::free(ptr);  // ptr points to BlockHeader, which is what we allocated
        }
    }
    mFreeBlocks.clear();
}

void* PoolingAllocator::allocate(size_t bytes) {
    //! Special case: zero-size allocation
    //! Return a unique sentinel pointer that can be safely deallocated
    //! This matches behavior of modern tensor libraries (PyTorch, NumPy)
    if (bytes == 0) {
        return &sZeroSizeSentinel;
    }

    //! Round up to the nearest multiple of the alignment
    size_t aligned_bytes = memory::MemoryUtils::alignSize(bytes, mConfig.alignment);

    //! Try to find exact size match in pool (O(1) hash lookup)
    auto it = mFreeBlocks.find(aligned_bytes);

    if (it != mFreeBlocks.end() && !it->second.empty()) {
        //! Found in pool - reuse it!
        void* header_ptr = it->second.back();
        it->second.pop_back();

        //! Calculate user data pointer from header metadata
        BlockHeader* header = static_cast<BlockHeader*>(header_ptr);
        void* user_ptr = static_cast<char*>(header_ptr) + header->offset;

        //! Back-pointer is already set from when we first allocated this block
        //! (It doesn't change, so we don't need to rewrite it)

        return user_ptr;
    }

    //! Pool miss - allocate from system
    //! Calculate required space:
    //! We need: sizeof(BlockHeader) + padding + sizeof(void*) + aligned_bytes
    //! The padding ensures user data is aligned to mConfig.alignment
    //! We also store a back-pointer (void*) just before the user data
    size_t header_size = sizeof(BlockHeader);
    size_t pointer_size = sizeof(void*);
    size_t max_total_size = header_size + mConfig.alignment + pointer_size + aligned_bytes;

    //! Allocate from system (this gives us aligned memory at the start)
    void* raw_ptr = AllocateBlock(max_total_size);

    //! Place header at the start
    BlockHeader* header = static_cast<BlockHeader*>(raw_ptr);

    //! Calculate where user data should start (aligned to mConfig.alignment)
    //! We need space for a back-pointer before the user data
    uintptr_t after_header = reinterpret_cast<uintptr_t>(raw_ptr) + header_size;
    uintptr_t user_data_addr =
        memory::MemoryUtils::alignSize(after_header + pointer_size, mConfig.alignment);
    size_t offset = user_data_addr - reinterpret_cast<uintptr_t>(raw_ptr);

    //! Write metadata into header
    header->size = aligned_bytes;
    header->offset = offset;

    //! Store back-pointer to header just before user data
    void** back_ptr = reinterpret_cast<void**>(user_data_addr - pointer_size);
    *back_ptr = raw_ptr;

    //! Return pointer to user data (now properly aligned!)
    return reinterpret_cast<void*>(user_data_addr);
}

void PoolingAllocator::deallocate(void* user_ptr) {
    //! Safety check
    if (user_ptr == nullptr) {
        throw std::runtime_error("Cannot deallocate nullptr");
    }

    //! Check if this is the zero-size sentinel
    if (user_ptr == &sZeroSizeSentinel) {
        // Nothing to deallocate for zero-size allocation
        return;
    }

    //! Get header by reading the back-pointer stored before user data
    void** back_ptr = static_cast<void**>(user_ptr) - 1;
    void* header_ptr = *back_ptr;
    BlockHeader* header = static_cast<BlockHeader*>(header_ptr);
    size_t size = header->size;

    //! Add back to pool (store the header pointer, not user pointer)
    mFreeBlocks[size].push_back(header_ptr);
}

Device PoolingAllocator::device() const {
    return Device(DeviceType::CPU);
}

void* PoolingAllocator::AllocateBlock(size_t bytes) {
    //! Ensure bytes is a multiple of alignment for aligned_alloc
    size_t aligned_bytes = memory::MemoryUtils::alignSize(bytes, mConfig.alignment);

    void* ptr = std::aligned_alloc(mConfig.alignment, aligned_bytes);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    return ptr;
}
}  // namespace loom