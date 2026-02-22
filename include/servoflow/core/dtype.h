// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace sf {

enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int8    = 3,
    Int4    = 4,
    Int32   = 5,
    Bool    = 6,
    Unknown = 255,
};

constexpr size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int8:    return 1;
        case DType::Int32:   return 4;
        case DType::Bool:    return 1;
        // Int4 is sub-byte; callers must handle packing explicitly
        case DType::Int4:    return 1;  // 2 values per byte
        default: return 0;
    }
}

constexpr bool is_floating_point(DType dt) {
    return dt == DType::Float32 || dt == DType::Float16 || dt == DType::BFloat16;
}

constexpr bool is_integer(DType dt) {
    return dt == DType::Int8 || dt == DType::Int4 || dt == DType::Int32;
}

constexpr std::string_view dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32:  return "float32";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int8:     return "int8";
        case DType::Int4:     return "int4";
        case DType::Int32:    return "int32";
        case DType::Bool:     return "bool";
        default:              return "unknown";
    }
}

inline DType dtype_from_string(std::string_view s) {
    if (s == "float32" || s == "F32") return DType::Float32;
    if (s == "float16" || s == "F16") return DType::Float16;
    if (s == "bfloat16" || s == "BF16") return DType::BFloat16;
    if (s == "int8"    || s == "I8")  return DType::Int8;
    if (s == "int4"    || s == "I4")  return DType::Int4;
    if (s == "int32"   || s == "I32") return DType::Int32;
    if (s == "bool"    || s == "BOOL") return DType::Bool;
    return DType::Unknown;
}

}  // namespace sf
