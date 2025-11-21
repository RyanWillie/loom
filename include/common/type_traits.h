#pragma once
#include <concepts>
#include <type_traits>

#include "common/dtypes.h"

namespace loom {

// ============================================================================
// Type to DType Mapping
// ============================================================================
// This maps C++ types to our DType enum at compile-time
// Example: dtype_traits<float>::value == DType::FLOAT32

template <typename T>
struct dtype_traits;  // Forward declaration - no definition means unsupported types fail

// Specializations for each supported type
template <>
struct dtype_traits<float> {
    static constexpr DType value = DType::FLOAT32;
};

template <>
struct dtype_traits<double> {
    static constexpr DType value = DType::FLOAT64;
};

template <>
struct dtype_traits<int8_t> {
    static constexpr DType value = DType::INT8;
};

template <>
struct dtype_traits<int16_t> {
    static constexpr DType value = DType::INT16;
};

template <>
struct dtype_traits<int32_t> {
    static constexpr DType value = DType::INT32;
};

template <>
struct dtype_traits<int64_t> {
    static constexpr DType value = DType::INT64;
};

template <>
struct dtype_traits<uint8_t> {
    static constexpr DType value = DType::UINT8;
};

template <>
struct dtype_traits<uint16_t> {
    static constexpr DType value = DType::UINT16;
};

template <>
struct dtype_traits<uint32_t> {
    static constexpr DType value = DType::UINT32;
};

template <>
struct dtype_traits<uint64_t> {
    static constexpr DType value = DType::UINT64;
};

// ============================================================================
// Concepts - Compile-Time Type Constraints
// ============================================================================

// Concept: Type T must have a corresponding DType mapping
// This prevents using unsupported types like std::string, pointers, etc.
template <typename T>
concept HasDType = requires {
    // T must have a dtype_traits specialization with a 'value' member
    { dtype_traits<T>::value } -> std::convertible_to<DType>;
};

// Concept: Type must be suitable for tensor storage
// Requirements:
// 1. Must be trivially copyable (safe for memcpy)
// 2. Must not be const (we need to modify data)
// 3. Must not be a reference
// 4. Must have a DType mapping
template <typename T>
concept TensorStorageType = HasDType<T> && std::is_trivially_copyable_v<T> &&
    (!std::is_const_v<T>)&&(!std::is_reference_v<T>);

}  // namespace loom
