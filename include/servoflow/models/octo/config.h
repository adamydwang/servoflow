// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/dtype.h"
#include <cstdint>
#include <string>

namespace sf {
namespace octo {

// ─────────────────────────────────────────────────────────────────────────────
// OctoConfig
// Based on Octo: An Open-Source Generalist Robot Policy
// ─────────────────────────────────────────────────────────────────────────────
struct OctoConfig {
    // ── Transformer Backbone ──────────────────────────────────────────────
    int64_t hidden_dim       = 384;    // Small: 384, Base: 768
    int64_t num_layers       = 12;     // Small: 12, Base: 12?
    int64_t num_heads        = 6;      // Small: 6, Base: 12
    int64_t head_dim         = 64;     // Typically 64
    int64_t intermediate_dim = 1536;   // 4 * hidden_dim
    float   dropout          = 0.1f;
    bool    use_bias         = true;
    
    // ── Vision Encoder (ViT / ResNet) ─────────────────────────────────────
    // Octo typically uses a pretrained ViT or ResNet-50.
    // For ServoFlow, we assume image embeddings are pre-computed or we implement
    // the patch embedder.
    int64_t patch_size       = 16;
    int64_t image_size       = 256;
    int64_t num_patches      = 256;    // (256/16)^2
    int64_t image_embed_dim  = 768;    // e.g. CLIP ViT-B/16 or similar

    // ── Diffusion Head ────────────────────────────────────────────────────
    int64_t action_dim       = 7;      // 7-DoF default
    int64_t diffusion_steps  = 100;    // DDPM/DDIM steps
    int64_t action_horizon   = 4;      // Chunk size (typically 4 or 8)
    int64_t head_hidden_dim  = 256;    // MLP hidden dim
    int64_t num_head_layers  = 3;      // MLP depth

    // ── Tokenization ──────────────────────────────────────────────────────
    int64_t task_token_dim   = 768;    // T5-base embedding dim
    int64_t obs_token_dim    = 384;    // Project inputs to this dim

    // ── Compute ───────────────────────────────────────────────────────────
    DType compute_dtype = DType::Float16;

    static OctoConfig from_json(const std::string& path);
};

}  // namespace octo
}  // namespace sf
