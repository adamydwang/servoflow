// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/tensor.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sf {
namespace cuda_ops {

// ─────────────────────────────────────────────────────────────────────────────
// Fused LayerNorm kernel (Welford online algorithm for numerical stability).
//
// One warp per row. For hidden_size ≤ 1024 we fit entirely in registers;
// for larger sizes we loop over tiles.
// Input shape: [*, hidden_size]  (treated as 2D: [rows, hidden_size])
// ─────────────────────────────────────────────────────────────────────────────

template<typename T, int kBlockSize = 256>
__global__ void layer_norm_kernel_impl(const T* __restrict__ x,
                                        const T* __restrict__ gamma,
                                        const T* __restrict__ beta,
                                        T* __restrict__ out,
                                        int64_t rows, int64_t cols,
                                        float eps) {
    // Each block handles one row.
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const T* row_x   = x   + row * cols;
    T*       row_out = out + row * cols;

    // ── Compute mean and variance via Welford ──────────────────────────────
    float mean = 0.f, M2 = 0.f;
    int   count = 0;

    for (int64_t col = threadIdx.x; col < cols; col += kBlockSize) {
        float val = static_cast<float>(row_x[col]);
        ++count;
        float delta = val - mean;
        mean += delta / count;
        M2   += delta * (val - mean);
    }

    // Warp-level reduce (only correct for kBlockSize == 32 or with shared mem).
    // For simplicity, use shared memory reduction.
    __shared__ float s_mean[kBlockSize], s_m2[kBlockSize];
    __shared__ int   s_count[kBlockSize];

    s_mean [threadIdx.x] = mean  * count;   // store sum, not mean
    s_m2   [threadIdx.x] = M2;
    s_count[threadIdx.x] = count;
    __syncthreads();

    // Tree reduction.
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a_sum = s_mean[threadIdx.x],  b_sum = s_mean[threadIdx.x + stride];
            int   a_cnt = s_count[threadIdx.x], b_cnt = s_count[threadIdx.x + stride];
            float a_m2  = s_m2[threadIdx.x],    b_m2  = s_m2[threadIdx.x + stride];
            int   total = a_cnt + b_cnt;
            float delta = b_sum / b_cnt - a_sum / a_cnt;
            s_mean [threadIdx.x] = a_sum + b_sum;
            s_m2   [threadIdx.x] = a_m2 + b_m2 + delta * delta
                                   * static_cast<float>(a_cnt) * b_cnt / total;
            s_count[threadIdx.x] = total;
        }
        __syncthreads();
    }

    float final_mean = s_mean[0] / static_cast<float>(cols);
    float final_var  = s_m2[0]  / static_cast<float>(cols);
    float inv_std    = rsqrtf(final_var + eps);

    // ── Normalise and apply affine transform ──────────────────────────────
    for (int64_t col = threadIdx.x; col < cols; col += kBlockSize) {
        float norm_val = (static_cast<float>(row_x[col]) - final_mean) * inv_std;
        row_out[col] = static_cast<T>(
            norm_val * static_cast<float>(gamma[col])
            + static_cast<float>(beta[col]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused RMSNorm kernel.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T, int kBlockSize = 256>
__global__ void rms_norm_kernel_impl(const T* __restrict__ x,
                                      const T* __restrict__ gamma,
                                      T* __restrict__ out,
                                      int64_t rows, int64_t cols,
                                      float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const T* row_x   = x   + row * cols;
    T*       row_out = out + row * cols;

    // ── Compute sum of squares ─────────────────────────────────────────────
    float ss = 0.f;
    for (int64_t col = threadIdx.x; col < cols; col += kBlockSize) {
        float v = static_cast<float>(row_x[col]);
        ss += v * v;
    }

    __shared__ float s_ss[kBlockSize];
    s_ss[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_ss[threadIdx.x] += s_ss[threadIdx.x + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(s_ss[0] / static_cast<float>(cols) + eps);

    for (int64_t col = threadIdx.x; col < cols; col += kBlockSize) {
        float norm_val = static_cast<float>(row_x[col]) * inv_rms;
        row_out[col]   = static_cast<T>(norm_val * static_cast<float>(gamma[col]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side launchers
// ─────────────────────────────────────────────────────────────────────────────
inline void layer_norm_kernel(const Tensor& x, const Tensor& gamma,
                               const Tensor& beta, Tensor& out,
                               float eps, cudaStream_t stream) {
    int64_t cols = x.shape()[x.ndim() - 1];
    int64_t rows = x.numel() / cols;

    auto launch = [&]<typename T>(T*) {
        constexpr int kBS = 256;
        layer_norm_kernel_impl<T, kBS><<<rows, kBS, 0, stream>>>(
            x.data_ptr<T>(), gamma.data_ptr<T>(), beta.data_ptr<T>(),
            out.data_ptr<T>(), rows, cols, eps);
    };

    switch (x.dtype()) {
        case DType::Float32:  launch((float*)nullptr);           break;
        case DType::Float16:  launch((__half*)nullptr);          break;
        case DType::BFloat16: launch((__nv_bfloat16*)nullptr);   break;
        default: throw std::runtime_error("layer_norm: unsupported dtype");
    }
}

inline void rms_norm_kernel(const Tensor& x, const Tensor& gamma,
                             Tensor& out, float eps, cudaStream_t stream) {
    int64_t cols = x.shape()[x.ndim() - 1];
    int64_t rows = x.numel() / cols;

    auto launch = [&]<typename T>(T*) {
        constexpr int kBS = 256;
        rms_norm_kernel_impl<T, kBS><<<rows, kBS, 0, stream>>>(
            x.data_ptr<T>(), gamma.data_ptr<T>(),
            out.data_ptr<T>(), rows, cols, eps);
    };

    switch (x.dtype()) {
        case DType::Float32:  launch((float*)nullptr);           break;
        case DType::Float16:  launch((__half*)nullptr);          break;
        case DType::BFloat16: launch((__nv_bfloat16*)nullptr);   break;
        default: throw std::runtime_error("rms_norm: unsupported dtype");
    }
}

}  // namespace cuda_ops
}  // namespace sf
