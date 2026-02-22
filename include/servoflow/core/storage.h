// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include "servoflow/core/device.h"

namespace sf {

// Raw byte storage with a custom deleter so the caller controls how memory
// was allocated (CUDA device memory, pinned host memory, plain malloc, etc.).
// Storage is ref-counted via shared_ptr; Tensor holds a shared_ptr<Storage>.
// Multiple Tensor views may share the same Storage.
class Storage {
public:
    using Deleter = std::function<void(void*)>;

    // Takes ownership of ptr; deleter is called when refcount reaches zero.
    Storage(void* ptr, size_t bytes, Device device, Deleter deleter)
        : ptr_(ptr), bytes_(bytes), device_(device), deleter_(std::move(deleter)) {}

    ~Storage() {
        if (ptr_ && deleter_) deleter_(ptr_);
    }

    // Non-copyable, non-movable (shared_ptr handles lifetime).
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    void*        data()       { return ptr_; }
    const void*  data() const { return ptr_; }
    size_t       bytes()  const { return bytes_; }
    const Device& device() const { return device_; }

private:
    void*   ptr_     = nullptr;
    size_t  bytes_   = 0;
    Device  device_;
    Deleter deleter_;
};

using StoragePtr = std::shared_ptr<Storage>;

}  // namespace sf
