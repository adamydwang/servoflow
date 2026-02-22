// SPDX-License-Identifier: Apache-2.0
#include "servoflow/backend/backend.h"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace sf {

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry reg;
    return reg;
}

void BackendRegistry::register_backend(DeviceType type, Factory factory) {
    factories_[static_cast<uint8_t>(type)] = std::move(factory);
}

BackendPtr BackendRegistry::get(Device device) {
    std::string key = device.str();
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    auto fit = factories_.find(static_cast<uint8_t>(device.type));
    if (fit == factories_.end())
        throw std::runtime_error("No backend registered for device: " + device.str());

    auto backend = fit->second(device.index);
    cache_[key]  = backend;
    return backend;
}

bool BackendRegistry::has(DeviceType type) const {
    return factories_.count(static_cast<uint8_t>(type)) > 0;
}

BackendPtr get_backend(Device device) {
    return BackendRegistry::instance().get(device);
}

BackendPtr get_backend(DeviceType type, int index) {
    return get_backend(Device(type, index));
}

}  // namespace sf
