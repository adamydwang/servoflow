// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

namespace sf {

enum class DeviceType : uint8_t {
    CPU    = 0,
    CUDA   = 1,
    ROCm   = 2,
    Metal  = 3,
};

struct Device {
    DeviceType type  = DeviceType::CPU;
    int32_t    index = 0;

    constexpr Device() = default;
    constexpr Device(DeviceType t, int32_t idx = 0) : type(t), index(idx) {}

    constexpr bool is_cpu()  const { return type == DeviceType::CPU;  }
    constexpr bool is_cuda() const { return type == DeviceType::CUDA; }
    constexpr bool is_rocm() const { return type == DeviceType::ROCm; }
    constexpr bool is_gpu()  const { return is_cuda() || is_rocm() || type == DeviceType::Metal; }

    constexpr bool operator==(const Device& o) const {
        return type == o.type && index == o.index;
    }
    constexpr bool operator!=(const Device& o) const { return !(*this == o); }

    std::string str() const {
        switch (type) {
            case DeviceType::CPU:   return "cpu";
            case DeviceType::CUDA:  return "cuda:" + std::to_string(index);
            case DeviceType::ROCm:  return "rocm:" + std::to_string(index);
            case DeviceType::Metal: return "metal:" + std::to_string(index);
        }
        return "unknown";
    }

    // Parse "cpu", "cuda:0", "cuda", "rocm:1", etc.
    static Device parse(const std::string& s) {
        if (s == "cpu") return Device(DeviceType::CPU);
        auto colon = s.find(':');
        std::string type_str = (colon == std::string::npos) ? s : s.substr(0, colon);
        int idx = (colon == std::string::npos) ? 0 : std::stoi(s.substr(colon + 1));
        if (type_str == "cuda")  return Device(DeviceType::CUDA,  idx);
        if (type_str == "rocm")  return Device(DeviceType::ROCm,  idx);
        if (type_str == "metal") return Device(DeviceType::Metal, idx);
        throw std::invalid_argument("Unknown device: " + s);
    }
};

// Convenience constants
constexpr Device kCPU       = Device(DeviceType::CPU);
constexpr Device kCUDA0     = Device(DeviceType::CUDA, 0);

}  // namespace sf
