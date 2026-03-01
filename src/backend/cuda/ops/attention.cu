// SPDX-License-Identifier: Apache-2.0
#include "attention.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cmath>

// Conditionally include FlashAttention headers.
// FlashAttention is fetched as an external dependency via CMake.
// If SF_USE_FLASH_ATTN is not defined we fall back to our own implementation.
#ifdef SF_USE_FLASH_ATTN
#  include <flash_attn/flash_api.h>
#endif

namespace sf {
namespace cuda_ops {

#ifdef SF_USE_FLASH_ATTN

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention v2 path (preferred for Ampere+ GPUs).
// FlashAttention expects inputs in [batch, seq, heads, head_dim] layout (BSHD).
// Our layout is [B, H, S, D].
// We use stride support in FlashAttention to handle [B, H, S, D] directly.
// ─────────────────────────────────────────────────────────────────────────────
static void flash_attn_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V,
                                Tensor& out, const Tensor* /*mask*/,
                                float scale, bool is_causal,
                                void* workspace, size_t workspace_size,
                                cudaStream_t stream) {
    int64_t B  = Q.shape()[0];
    int64_t H  = Q.shape()[1];
    int64_t Sq = Q.shape()[2];
    int64_t D  = Q.shape()[3];
    int64_t H_k = K.shape()[1];
    int64_t Sk = K.shape()[2];

    if (scale == 0.f) scale = 1.f / sqrtf(static_cast<float>(D));

    bool is_fp16 = (Q.dtype() == DType::Float16);
    bool is_bf16 = (Q.dtype() == DType::BFloat16);
    if (!is_fp16 && !is_bf16)
        throw std::runtime_error("FlashAttention requires fp16 or bf16 input");

    size_t lse_bytes = static_cast<size_t>(B) * H * Sq * sizeof(float);
    
    // Allocate softmax_lse buffer.
    // FlashAttention requires this buffer for backward pass, but also uses it
    // internally in forward pass.
    // Add padding to be safe against tiled access.
    // For D=64, kBlockN=256. If Sk=128, it reads 256, so we need extra padding.
    // 128KB padding is sufficient.
    size_t padding = 128 * 1024;
    size_t required = lse_bytes + padding;

    if (!workspace || workspace_size < required) {
        throw std::runtime_error("FlashAttention workspace too small");
    }
    
    void *softmax_lse = workspace;
    cudaMemsetAsync(softmax_lse, 0, required, stream);
    
    // ServoFlow layout is [B, H, S, D].
    // FlashAttention supports this via strides.
    // We pass is_BSHD=false to indicate [B, H, S, D] layout.
    flash_attn::mha_fwd(
        Q.raw_data_ptr(), K.raw_data_ptr(), V.raw_data_ptr(),
        out.raw_data_ptr(),
        softmax_lse,
        B, Sq, Sk, H, H_k, D,
        scale, is_causal,
        /*window_size_left=*/-1, /*window_size_right=*/is_causal ? 0 : -1,
        /*softcap=*/0.f,
        /*is_bf16=*/is_bf16,
        stream,
        /*is_BSHD=*/false);
}
#endif  // SF_USE_FLASH_ATTN

// ─────────────────────────────────────────────────────────────────────────────
// Fallback: memory-efficient attention with online softmax (single-pass).
// Operates on [B, H, S, D] layout in fp32 for correctness.
// Not intended as final performance path — FlashAttention should be used
// on CUDA hardware; this exists as a correctness reference and CPU fallback.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void naive_attention_kernel(const T* __restrict__ Q,
                                        const T* __restrict__ K,
                                        const T* __restrict__ V,
                                        T* __restrict__ out,
                                        int64_t Sq, int64_t Sk,
                                        int64_t H,  int64_t D,
                                        float scale, bool is_causal) {
    // Grid: [B, H, Sq], thread handles one query position.
    int64_t b    = blockIdx.z;
    int64_t h    = blockIdx.y;
    int64_t q    = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Sq) return;

    int64_t stride_bh = H * Sq * D;
    int64_t stride_h  = Sq * D;

    const T* Qbh = Q + b * stride_bh + h * stride_h + q * D;
    const T* Kbh = K + b * stride_bh + h * (Sk * D);
    const T* Vbh = V + b * stride_bh + h * (Sk * D);
    T*       Obh = out + b * stride_bh + h * stride_h + q * D;

    // Compute attention scores and accumulate in fp32.
    float m = -1e30f;
    float sum = 0.f;
    float acc[128] = {};  // head_dim ≤ 128

    for (int64_t k = 0; k < Sk; ++k) {
        if (is_causal && k > q) break;

        float dot = 0.f;
        for (int64_t d = 0; d < D; ++d)
            dot += static_cast<float>(Qbh[d]) * static_cast<float>(Kbh[k * D + d]);
        dot *= scale;

        // Online softmax (numerically stable).
        float new_m = fmaxf(m, dot);
        float exp_old = expf(m - new_m);
        float exp_new = expf(dot - new_m);

        for (int64_t d = 0; d < D; ++d)
            acc[d] = acc[d] * exp_old + exp_new * static_cast<float>(Vbh[k * D + d]);

        sum = sum * exp_old + exp_new;
        m   = new_m;
    }

    float inv_sum = 1.f / sum;
    for (int64_t d = 0; d < D; ++d)
        Obh[d] = static_cast<T>(acc[d] * inv_sum);
}

static void fallback_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                               Tensor& out, float scale, bool is_causal,
                               cudaStream_t stream) {
    int64_t B  = Q.shape()[0];
    int64_t H  = Q.shape()[1];
    int64_t Sq = Q.shape()[2];
    int64_t Sk = K.shape()[2];
    int64_t D  = Q.shape()[3];

    if (scale == 0.f) scale = 1.f / sqrtf(static_cast<float>(D));

    // One thread per query token; grid covers [B, H, Sq].
    constexpr int kTX = 32;
    dim3 grid(static_cast<unsigned>((Sq + kTX - 1) / kTX),
               static_cast<unsigned>(H),
               static_cast<unsigned>(B));
    dim3 block(kTX);

    switch (Q.dtype()) {
        case DType::Float32:
            naive_attention_kernel<float><<<grid, block, 0, stream>>>(
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                out.data_ptr<float>(),
                Sq, Sk, H, D, scale, is_causal);
            break;
        case DType::Float16:
            naive_attention_kernel<half><<<grid, block, 0, stream>>>(
                Q.data_ptr<half>(), K.data_ptr<half>(), V.data_ptr<half>(),
                out.data_ptr<half>(),
                Sq, Sk, H, D, scale, is_causal);
            break;
        case DType::BFloat16:
            naive_attention_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                Q.data_ptr<__nv_bfloat16>(), K.data_ptr<__nv_bfloat16>(), V.data_ptr<__nv_bfloat16>(),
                out.data_ptr<__nv_bfloat16>(),
                Sq, Sk, H, D, scale, is_causal);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for fallback attention");
    }
}

void flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                     Tensor& out,
                     const Tensor* mask,
                     float scale, bool is_causal,
                     void* workspace, size_t workspace_size,
                     cudaStream_t stream) {
#ifdef SF_USE_FLASH_ATTN
    // Use FlashAttention if possible (fp16/bf16).
    if (Q.dtype() == DType::Float16 || Q.dtype() == DType::BFloat16) {
        flash_attn_dispatch(Q, K, V, out, mask, scale, is_causal, workspace, workspace_size, stream);
        return;
    }
#endif
    fallback_attention(Q, K, V, out, scale, is_causal, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unpack QKV
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void unpack_qkv_kernel_impl(
    const T* __restrict__ qkv,
    T* __restrict__ q,
    T* __restrict__ k,
    T* __restrict__ v,
    int64_t B, int64_t S, int64_t H, int64_t D)
{
    // Grid covers output elements: B * H * S * D
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = B * H * S * D;
    if (idx >= total) return;

    // Decode output index (layout: B, H, S, D)
    int64_t d = idx % D;
    int64_t temp = idx / D;
    int64_t s = temp % S;
    temp /= S;
    int64_t h = temp % H;
    int64_t b = temp / H;

    // Input layout: [B, S, 3 * H * D]
    // Inner dim 3*H*D is packed as [Q_all_heads, K_all_heads, V_all_heads]
    // where each block is [H * D].
    // Within H*D, layout is [h * D + d].
    
    int64_t D_model = H * D;
    int64_t in_offset_base = b * (S * 3 * D_model) + s * (3 * D_model);
    
    // Q
    q[idx] = qkv[in_offset_base + 0 * D_model + h * D + d];
    // K
    k[idx] = qkv[in_offset_base + 1 * D_model + h * D + d];
    // V
    v[idx] = qkv[in_offset_base + 2 * D_model + h * D + d];
}

void unpack_qkv_kernel(const Tensor& qkv, int64_t num_heads, int64_t head_dim,
                       Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    int64_t B = q.shape()[0];
    int64_t H = num_heads;
    int64_t S = q.shape()[2];
    int64_t D = head_dim;
    
    int64_t total = B * H * S * D;
    int block = 256;
    int grid = (total + block - 1) / block;

    if (qkv.dtype() == DType::Float16) {
        unpack_qkv_kernel_impl<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(qkv.raw_data_ptr()),
            static_cast<half*>(q.raw_data_ptr()),
            static_cast<half*>(k.raw_data_ptr()),
            static_cast<half*>(v.raw_data_ptr()),
            B, S, H, D);
    } else if (qkv.dtype() == DType::Float32) {
        unpack_qkv_kernel_impl<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(qkv.raw_data_ptr()),
            static_cast<float*>(q.raw_data_ptr()),
            static_cast<float*>(k.raw_data_ptr()),
            static_cast<float*>(v.raw_data_ptr()),
            B, S, H, D);
    } else {
         throw std::runtime_error("unpack_qkv: unsupported dtype");
    }
}

} // namespace cuda_ops
} // namespace sf
