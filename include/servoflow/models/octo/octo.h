// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/engine/inference_engine.h"
#include "servoflow/models/octo/config.h"
#include <memory>
#include <vector>

namespace sf {
namespace octo {

// ─────────────────────────────────────────────────────────────────────────────
// OctoBlock
// Standard Pre-LN Transformer Block with Self-Attention and MLP.
// Optionally uses FiLM or Cross-Attention for conditioning (but typically
// Octo concatenates tokens).
// ─────────────────────────────────────────────────────────────────────────────
class OctoBlock {
public:
    explicit OctoBlock(const OctoConfig& config, BackendPtr backend);
    
    // forward: [B, S, D] -> [B, S, D]
    Tensor forward(const Tensor& x, 
                   const Tensor& mask, // Attention mask
                   BackendPtr backend, 
                   StreamHandle stream);

private:
    OctoConfig cfg_;
    // Self Attention
    Tensor qkv_weight_;
    Tensor qkv_bias_;
    Tensor proj_weight_;
    Tensor proj_bias_;
    
    // MLP
    Tensor fc1_weight_;
    Tensor fc1_bias_;
    Tensor fc2_weight_;
    Tensor fc2_bias_;
    
    // Norms
    Tensor norm1_weight_;
    Tensor norm1_bias_;
    Tensor norm2_weight_;
    Tensor norm2_bias_;
};

// ─────────────────────────────────────────────────────────────────────────────
// OctoDiffusionHead
// MLP-based Diffusion Head that predicts noise or action.
// Input: [B, Horizon, D] (action tokens or embeddings) + Time embedding
// Output: [B, Horizon, ActionDim]
// ─────────────────────────────────────────────────────────────────────────────
class OctoDiffusionHead {
public:
    explicit OctoDiffusionHead(const OctoConfig& config, BackendPtr backend);

    // forward: Predict noise/action from latent features and time
    // x: [B, Horizon, D] (features from transformer readout)
    // t: [B] (timestep)
    // context: [B, D_ctx] (conditioning from transformer)
    Tensor forward(const Tensor& x, 
                   const Tensor& t, 
                   const Tensor& context,
                   BackendPtr backend, 
                   StreamHandle stream);
                   
private:
    OctoConfig cfg_;
    // MLP layers (typically 3)
    Tensor fc1_weight_, fc1_bias_;
    Tensor fc2_weight_, fc2_bias_;
    Tensor fc3_weight_, fc3_bias_;
    
    // Time embedding projection
    Tensor time_embed_proj_;
};

// ─────────────────────────────────────────────────────────────────────────────
// OctoModel
// Implements the full Octo policy:
// 1. Tokenize inputs (Images, Proprio, Task)
// 2. Transformer Backbone
// 3. Readout Heads -> Action Latents
// 4. Diffusion Head -> Denoised Action
// ─────────────────────────────────────────────────────────────────────────────
class OctoModel : public IVLAModel {
public:
    explicit OctoModel(OctoConfig config, BackendPtr backend);

    // IVLAModel Interface
    Tensor encode_condition(const VLAInput& input,
                            BackendPtr backend,
                            StreamHandle stream) override;

    void denoise_step(const Tensor& x_t, float t,
                      const Tensor& condition,
                      Tensor& velocity,
                      BackendPtr backend,
                      StreamHandle stream) override;

    Tensor decode_action(const Tensor& raw,
                         BackendPtr backend,
                         StreamHandle stream) override;

    int64_t action_dim()   const override { return cfg_.action_dim; }
    int64_t action_horizon() const override { return cfg_.action_horizon; }
    DType   dtype()         const override { return cfg_.compute_dtype; }

private:
    OctoConfig cfg_;
    BackendPtr backend_;

    // ── Components ──
    std::vector<OctoBlock> blocks_;
    std::unique_ptr<OctoDiffusionHead> head_;
    
    // Embeddings
    Tensor pos_embed_;        // Learnable positional embeddings
    Tensor time_embed_;       // Sinusoidal or learnable time embeddings
    
    // Input Projections
    Tensor img_proj_;         // Project image features to obs_token_dim
    Tensor text_proj_;        // Project text features to obs_token_dim
    Tensor proprio_proj_;     // Project proprioception to obs_token_dim
    
    // Readout
    Tensor action_readout_;   // Project transformer output to diffusion input dim
};

// Factory
std::shared_ptr<OctoModel> load_octo(const std::string& checkpoint_dir, 
                                     BackendPtr backend, 
                                     const Device& device);

}  // namespace octo
}  // namespace sf
