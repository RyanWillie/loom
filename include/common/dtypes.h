#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// ============================================================================
// Data Type Enumeration
// ============================================================================
namespace loom {

enum class DType {
    FLOAT32,
    FLOAT64,
    FLOAT16,   // Half precision (IEEE 754-2008)
    BFLOAT16,  // Brain float (truncated FP32)
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
};

// ============================================================================
// Utility Functions
// ============================================================================

inline std::size_t sizeOf(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32:
            return sizeof(float);
        case DType::FLOAT64:
            return sizeof(double);
        case DType::FLOAT16:
        case DType::BFLOAT16:
            return 2;  // Both are 16-bit types
        case DType::INT8:
        case DType::UINT8:
            return 1;
        case DType::INT16:
        case DType::UINT16:
            return 2;
        case DType::INT32:
        case DType::UINT32:
            return 4;
        case DType::INT64:
        case DType::UINT64:
            return 8;
        default:
            throw std::runtime_error("Unknown DType");
    }
}

inline const char* name(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32:
            return "float32";
        case DType::FLOAT64:
            return "float64";
        case DType::FLOAT16:
            return "float16";
        case DType::BFLOAT16:
            return "bfloat16";
        case DType::INT8:
            return "int8";
        case DType::INT16:
            return "int16";
        case DType::INT32:
            return "int32";
        case DType::INT64:
            return "int64";
        case DType::UINT8:
            return "uint8";
        case DType::UINT16:
            return "uint16";
        case DType::UINT32:
            return "uint32";
        case DType::UINT64:
            return "uint64";
        default:
            return "unknown";
    }
}

inline bool isInteger(DType dtype) {
    return dtype == DType::INT8 || dtype == DType::INT16 || dtype == DType::INT32 ||
           dtype == DType::INT64 || dtype == DType::UINT8 || dtype == DType::UINT16 ||
           dtype == DType::UINT32 || dtype == DType::UINT64;
}

inline bool isFloatingPoint(DType dtype) {
    return dtype == DType::FLOAT32 || dtype == DType::FLOAT64 || dtype == DType::FLOAT16 ||
           dtype == DType::BFLOAT16;
}

inline bool isUnsigned(DType dtype) {
    return dtype == DType::UINT8 || dtype == DType::UINT16 || dtype == DType::UINT32 ||
           dtype == DType::UINT64;
}

inline bool isSigned(DType dtype) {
    return isInteger(dtype) && !isUnsigned(dtype);
}

}  // namespace loom