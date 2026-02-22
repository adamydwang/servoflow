// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "servoflow/core/device.h"
#include "servoflow/core/dtype.h"
#include "servoflow/core/shape.h"
#include "servoflow/core/storage.h"

namespace sf {

// Tensor is a lightweight handle (pointer + metadata) into a shared Storage.
// Creating views (slice, permute) is O(1) and zero-copy.
// All actual memory operations are dispatched through the Backend.
class Tensor {
public:
    Tensor() = default;

    // Construct from existing storage (used by Backend allocators).
    Tensor(StoragePtr storage, Shape shape, DType dtype,
           std::vector<int64_t> strides = {}, int64_t byte_offset = 0)
        : storage_(std::move(storage)),
          shape_(std::move(shape)),
          dtype_(dtype),
          strides_(std::move(strides)),
          byte_offset_(byte_offset)
    {
        if (strides_.empty()) strides_ = contiguous_strides(shape_, dtype_size(dtype_));
    }

    // ── Metadata accessors ─────────────────────────────────────────────────
    const Shape&              shape()       const { return shape_;  }
    DType                     dtype()       const { return dtype_;  }
    const Device&             device()      const { return storage_->device(); }
    int64_t                   ndim()        const { return shape_.ndim(); }
    int64_t                   numel()       const { return shape_.numel(); }
    size_t                    nbytes()      const { return shape_.nbytes(dtype_size(dtype_)); }
    const std::vector<int64_t>& strides()  const { return strides_; }
    int64_t                   byte_offset() const { return byte_offset_; }

    bool is_valid()       const { return storage_ != nullptr; }
    bool is_contiguous()  const { return strides_ == contiguous_strides(shape_, dtype_size(dtype_)); }
    bool is_cuda()        const { return device().is_cuda(); }
    bool is_cpu()         const { return device().is_cpu(); }

    // ── Raw data access ────────────────────────────────────────────────────
    // For typed access after confirming dtype. Prefer these over void*.
    template<typename T>
    T* data_ptr() {
        return reinterpret_cast<T*>(static_cast<char*>(storage_->data()) + byte_offset_);
    }

    template<typename T>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(static_cast<const char*>(storage_->data()) + byte_offset_);
    }

    void* raw_data_ptr() {
        return static_cast<char*>(storage_->data()) + byte_offset_;
    }

    const void* raw_data_ptr() const {
        return static_cast<const char*>(storage_->data()) + byte_offset_;
    }

    // ── View operations (zero-copy) ────────────────────────────────────────
    // Return a new Tensor sharing the same Storage with different metadata.
    Tensor view(Shape new_shape) const {
        if (new_shape.numel() != shape_.numel())
            throw std::invalid_argument("Tensor::view: element count mismatch");
        if (!is_contiguous())
            throw std::runtime_error("Tensor::view: tensor must be contiguous");
        return Tensor(storage_, new_shape, dtype_, {}, byte_offset_);
    }

    // Squeeze/unsqueeze a single dimension.
    Tensor unsqueeze(int64_t dim) const {
        auto dims = to_vec(shape_);
        if (dim < 0) dim += static_cast<int64_t>(dims.size()) + 1;
        dims.insert(dims.begin() + dim, 1);
        return view(Shape(dims));
    }

    Tensor squeeze(int64_t dim) const {
        auto dims = to_vec(shape_);
        if (dim < 0) dim += static_cast<int64_t>(dims.size());
        if (dims[dim] != 1)
            throw std::invalid_argument("Tensor::squeeze: dim size must be 1");
        dims.erase(dims.begin() + dim);
        return view(Shape(dims));
    }

    // Slice along dim 0 (contiguous tensors only for now).
    Tensor slice(int64_t start, int64_t end) const {
        if (!is_contiguous())
            throw std::runtime_error("Tensor::slice: tensor must be contiguous");
        if (ndim() < 1)
            throw std::runtime_error("Tensor::slice: scalar tensor");
        auto dims = to_vec(shape_);
        int64_t dim_size = dims[0];
        if (start < 0) start += dim_size;
        if (end   < 0) end   += dim_size;
        if (start < 0 || end > dim_size || start >= end)
            throw std::out_of_range("Tensor::slice: invalid range");
        dims[0] = end - start;
        int64_t elem_stride = 1;
        for (size_t i = 1; i < shape_.ndim(); ++i) elem_stride *= shape_[i];
        int64_t new_offset = byte_offset_ + start * elem_stride
                             * static_cast<int64_t>(dtype_size(dtype_));
        return Tensor(storage_, Shape(dims), dtype_, {}, new_offset);
    }

    const StoragePtr& storage() const { return storage_; }

private:
    static std::vector<int64_t> contiguous_strides(const Shape& shape, size_t elem_bytes) {
        std::vector<int64_t> s(shape.ndim());
        if (shape.ndim() == 0) return s;
        s[shape.ndim() - 1] = static_cast<int64_t>(elem_bytes);
        for (int i = static_cast<int>(shape.ndim()) - 2; i >= 0; --i)
            s[i] = s[i + 1] * shape[i + 1];
        return s;
    }

    static std::vector<int64_t> to_vec(const Shape& s) {
        return std::vector<int64_t>(s.begin(), s.end());
    }

    StoragePtr           storage_;
    Shape                shape_;
    DType                dtype_      = DType::Float32;
    std::vector<int64_t> strides_;
    int64_t              byte_offset_ = 0;
};

}  // namespace sf
