// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace sf {

// Fixed-capacity shape to avoid heap allocation in the common case (≤8 dims).
class Shape {
public:
    static constexpr size_t kMaxDims = 8;

    Shape() : ndim_(0) { dims_.fill(0); }

    Shape(std::initializer_list<int64_t> dims) {
        if (dims.size() > kMaxDims)
            throw std::invalid_argument("Shape: too many dimensions");
        ndim_ = dims.size();
        dims_.fill(0);
        size_t i = 0;
        for (auto d : dims) dims_[i++] = d;
    }

    explicit Shape(const std::vector<int64_t>& dims) {
        if (dims.size() > kMaxDims)
            throw std::invalid_argument("Shape: too many dimensions");
        ndim_ = dims.size();
        dims_.fill(0);
        for (size_t i = 0; i < ndim_; ++i) dims_[i] = dims[i];
    }

    size_t  ndim()  const { return ndim_; }
    bool    empty() const { return ndim_ == 0; }

    int64_t operator[](size_t i) const {
        if (i >= ndim_) throw std::out_of_range("Shape: index out of range");
        return dims_[i];
    }

    int64_t numel() const {
        if (ndim_ == 0) return 0;
        int64_t n = 1;
        for (size_t i = 0; i < ndim_; ++i) n *= dims_[i];
        return n;
    }

    // Number of bytes needed given element size.
    size_t nbytes(size_t elem_size) const {
        return static_cast<size_t>(numel()) * elem_size;
    }

    bool operator==(const Shape& o) const {
        if (ndim_ != o.ndim_) return false;
        for (size_t i = 0; i < ndim_; ++i)
            if (dims_[i] != o.dims_[i]) return false;
        return true;
    }
    bool operator!=(const Shape& o) const { return !(*this == o); }

    std::string str() const {
        std::string s = "[";
        for (size_t i = 0; i < ndim_; ++i) {
            s += std::to_string(dims_[i]);
            if (i + 1 < ndim_) s += ", ";
        }
        return s + "]";
    }

    // Iterators for range-for.
    const int64_t* begin() const { return dims_.data(); }
    const int64_t* end()   const { return dims_.data() + ndim_; }

private:
    std::array<int64_t, kMaxDims> dims_;
    size_t ndim_;
};

}  // namespace sf
