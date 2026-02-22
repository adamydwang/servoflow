// SPDX-License-Identifier: Apache-2.0
#pragma once

// Attention dispatch: routes to FlashAttention when available,
// falls back to a memory-efficient manual implementation otherwise.

#include "servoflow/core/tensor.h"
#include <cuda_runtime.h>

namespace sf {
namespace cuda_ops {

// flash_attention:
//   Q, K, V: [batch, heads, seq_len, head_dim]
//   out:     [batch, heads, seq_len, head_dim]  (pre-allocated)
//   mask:    optional [batch, 1, seq_q, seq_k] additive mask (−inf for masked)
//   scale:   if 0, defaults to 1/sqrt(head_dim)
//   is_causal: apply lower-triangular causal mask
void flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                     Tensor& out,
                     const Tensor* mask,
                     float scale, bool is_causal,
                     cudaStream_t stream);

}  // namespace cuda_ops
}  // namespace sf
