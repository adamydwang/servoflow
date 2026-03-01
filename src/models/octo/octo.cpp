// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/octo/octo.h"
#include "servoflow/backend/backend.h"
#include <cmath>
#include <iostream>

namespace sf {
namespace octo {

// ─────────────────────────────────────────────────────────────────────────────
// OctoBlock Implementation
// ─────────────────────────────────────────────────────────────────────────────
OctoBlock::OctoBlock(const OctoConfig& config, BackendPtr backend) : cfg_(config) {
    int64_t d_model = cfg_.hidden_dim;
    int64_t d_ff = cfg_.intermediate_dim;

    // Self-Attention: Q, K, V
    // Note: Octo likely uses combined QKV or separate. Let's assume combined for efficiency.
    // [3 * d_model, d_model]
    qkv_weight_ = backend->alloc({3 * d_model, d_model}, cfg_.compute_dtype);
    qkv_bias_   = backend->alloc({3 * d_model}, cfg_.compute_dtype);
    
    // Proj
    proj_weight_ = backend->alloc({d_model, d_model}, cfg_.compute_dtype);
    proj_bias_   = backend->alloc({d_model}, cfg_.compute_dtype);

    // MLP
    fc1_weight_ = backend->alloc({d_ff, d_model}, cfg_.compute_dtype);
    fc1_bias_   = backend->alloc({d_ff}, cfg_.compute_dtype);
    fc2_weight_ = backend->alloc({d_model, d_ff}, cfg_.compute_dtype);
    fc2_bias_   = backend->alloc({d_model}, cfg_.compute_dtype);

    // Norms (LayerNorm)
    norm1_weight_ = backend->alloc({d_model}, cfg_.compute_dtype);
    norm1_bias_   = backend->alloc({d_model}, cfg_.compute_dtype);
    norm2_weight_ = backend->alloc({d_model}, cfg_.compute_dtype);
    norm2_bias_   = backend->alloc({d_model}, cfg_.compute_dtype);
    
    // Initialize (dummy for now, real weights loaded later)
    backend->fill(norm1_weight_, 1.0f);
    backend->fill(norm2_weight_, 1.0f);
    backend->fill(norm1_bias_, 0.0f);
    backend->fill(norm2_bias_, 0.0f);
}

Tensor OctoBlock::forward(const Tensor& x, const Tensor& mask, BackendPtr backend, StreamHandle stream) {
    int64_t B = x.shape()[0];
    int64_t S = x.shape()[1];
    int64_t D = x.shape()[2];

    // Flatten x to [B*S, D] for GEMM/LayerNorm
    Tensor x_flat = x.view({B * S, D});

    // 1. Pre-Norm
    Tensor norm1 = backend->alloc({B * S, D}, x.dtype());
    backend->layer_norm(x_flat, norm1_weight_, norm1_bias_, norm1, 1e-5f, stream);

    // 2. Self-Attention
    // QKV Projection
    // x: [B, S, D] -> [B, S, 3*D]
    Tensor qkv_flat = backend->alloc({B * S, 3 * D}, x.dtype());
    // qkv = norm1 @ qkv_weight.T + bias
    backend->gemm_bias_act(norm1, qkv_weight_, qkv_bias_, qkv_flat, 
                           IBackend::ActivationType::None, 
                           1.0f, 0.0f, false, true, stream);

    // Reshape for Attention
    Tensor qkv = qkv_flat.view({B, S, 3 * D});

    // Split Q, K, V
    Tensor q = backend->alloc({B, cfg_.num_heads, S, cfg_.head_dim}, x.dtype());
    Tensor k = backend->alloc({B, cfg_.num_heads, S, cfg_.head_dim}, x.dtype());
    Tensor v = backend->alloc({B, cfg_.num_heads, S, cfg_.head_dim}, x.dtype());
    
    backend->unpack_qkv(qkv, cfg_.num_heads, cfg_.head_dim, q, k, v, stream);
    
    // Run Attention
    Tensor attn_val = backend->alloc({B, cfg_.num_heads, S, cfg_.head_dim}, x.dtype());
    backend->attention(q, k, v, attn_val, &mask, 0.0f, false, stream);
    
    // Project Output
    // [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
    Tensor attn_permuted = backend->alloc({B, S, cfg_.num_heads, cfg_.head_dim}, x.dtype());
    backend->permute(attn_val, attn_permuted, {0, 2, 1, 3}, stream);
    
    // View as [B, S, D] -> [B*S, D]
    Tensor attn_out_flat = attn_permuted.view({B * S, D});
    
    // proj = attn_out @ proj_weight.T + bias
    Tensor proj_flat = backend->alloc({B * S, D}, x.dtype());
    backend->gemm_bias_act(attn_out_flat, proj_weight_, proj_bias_, proj_flat,
                           IBackend::ActivationType::None,
                           1.0f, 0.0f, false, true, stream);

    // Residual 1
    // x = x + proj
    backend->add(x_flat, proj_flat, const_cast<Tensor&>(x_flat), stream);

    // 3. MLP
    // Pre-Norm 2
    Tensor norm2 = backend->alloc({B * S, D}, x.dtype());
    backend->layer_norm(x_flat, norm2_weight_, norm2_bias_, norm2, 1e-5f, stream);
    
    // FC1
    Tensor hidden = backend->alloc({B * S, cfg_.intermediate_dim}, x.dtype());
    backend->gemm_bias_act(norm2, fc1_weight_, fc1_bias_, hidden,
                           IBackend::ActivationType::GELU,
                           1.0f, 0.0f, false, true, stream);
                           
    // FC2
    Tensor mlp_out = backend->alloc({B * S, D}, x.dtype());
    backend->gemm_bias_act(hidden, fc2_weight_, fc2_bias_, mlp_out,
                           IBackend::ActivationType::None,
                           1.0f, 0.0f, false, true, stream);
                           
    // Residual 2
    backend->add(x_flat, mlp_out, const_cast<Tensor&>(x_flat), stream);
    
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// OctoDiffusionHead Implementation
// ─────────────────────────────────────────────────────────────────────────────
OctoDiffusionHead::OctoDiffusionHead(const OctoConfig& config, BackendPtr backend) : cfg_(config) {
    int64_t action_dim = cfg_.action_dim * cfg_.action_horizon;
    int64_t time_dim = 128; // Default sinusoidal
    int64_t context_dim = cfg_.hidden_dim; // Transformer output
    int64_t hidden_dim = cfg_.head_hidden_dim;
    
    // Layer 1
    fc1_weight_ = backend->alloc({hidden_dim, action_dim + time_dim + context_dim}, cfg_.compute_dtype);
    fc1_bias_   = backend->alloc({hidden_dim}, cfg_.compute_dtype);
    
    // Layer 2
    fc2_weight_ = backend->alloc({hidden_dim, hidden_dim}, cfg_.compute_dtype);
    fc2_bias_   = backend->alloc({hidden_dim}, cfg_.compute_dtype);
    
    // Layer 3 (Output)
    fc3_weight_ = backend->alloc({action_dim, hidden_dim}, cfg_.compute_dtype);
    fc3_bias_   = backend->alloc({action_dim}, cfg_.compute_dtype);
}

Tensor OctoDiffusionHead::forward(const Tensor& x, const Tensor& t, const Tensor& context, BackendPtr backend, StreamHandle stream) {
    // 1. Time Embedding (Placeholder)
    int64_t B = x.shape()[0];
    Tensor t_embed = backend->alloc({B, 128}, x.dtype());
    backend->fill(t_embed, 0.1f, stream);
    
    // 2. Flatten x
    Tensor x_flat = x.view({B, x.shape()[1] * x.shape()[2]});
    
    // Flatten context if needed
    Tensor context_flat = context.view({B, context.numel() / B});
    
    // 3. Concat
    std::vector<Tensor> inputs = {x_flat, t_embed, context_flat};
    Tensor input = backend->alloc({B, x_flat.shape()[1] + 128 + context_flat.shape()[1]}, x.dtype());
    backend->cat(inputs, input, 1, stream);
    
    // 4. MLP
    // FC1
    Tensor h1 = backend->alloc({B, cfg_.head_hidden_dim}, x.dtype());
    backend->gemm_bias_act(input, fc1_weight_, fc1_bias_, h1, IBackend::ActivationType::GELU, 1.0f, 0.0f, false, true, stream);
    
    // FC2
    Tensor h2 = backend->alloc({B, cfg_.head_hidden_dim}, x.dtype());
    backend->gemm_bias_act(h1, fc2_weight_, fc2_bias_, h2, IBackend::ActivationType::GELU, 1.0f, 0.0f, false, true, stream);
    
    // Output
    Tensor out_flat = backend->alloc({B, x_flat.shape()[1]}, x.dtype());
    backend->gemm_bias_act(h2, fc3_weight_, fc3_bias_, out_flat, IBackend::ActivationType::None, 1.0f, 0.0f, false, true, stream);
    
    // Reshape back
    return out_flat.view(x.shape());
}

// ─────────────────────────────────────────────────────────────────────────────
// OctoModel Implementation
// ─────────────────────────────────────────────────────────────────────────────
OctoModel::OctoModel(OctoConfig config, BackendPtr backend) 
    : cfg_(config), backend_(backend) {
    
    // Create Blocks
    for(int i=0; i<cfg_.num_layers; ++i) {
        blocks_.emplace_back(cfg_, backend);
    }
    
    // Create Head
    head_ = std::make_unique<OctoDiffusionHead>(cfg_, backend);
    
    // Allocate embeddings
    // ...
}

Tensor OctoModel::encode_condition(const VLAInput& input, BackendPtr backend, StreamHandle stream) {
    // This is where the heavy lifting happens:
    // 1. Tokenize Images (ViT) -> Not implemented yet (need ViT module)
    // 2. Tokenize Text -> Not implemented yet (need T5)
    // 3. Run Transformer Backbone
    
    // For benchmark/placeholder: Create dummy tokens
    int64_t B = 1;
    int64_t seq_len = 64; // Typical sequence length
    Tensor x = backend->alloc({B, seq_len, cfg_.hidden_dim}, cfg_.compute_dtype);
    backend->fill(x, 0.1f, stream);
    
    // Create dummy mask
    Tensor mask = backend->alloc({B, 1, seq_len, seq_len}, cfg_.compute_dtype); 
    backend->fill(mask, 0.0f, stream);
    
    // Run Blocks
    for(auto& block : blocks_) {
        // x is modified in-place
        block.forward(x, mask, backend, stream);
    }
    
    // Readout (simulate extraction of 1 token)
    Tensor readout = backend->alloc({B, 1, cfg_.hidden_dim}, cfg_.compute_dtype);
    backend->fill(readout, 0.1f, stream);
    
    // Return latent [B, 1, D]
    return readout;
}

void OctoModel::denoise_step(const Tensor& x_t, float t, const Tensor& condition,
                             Tensor& velocity, BackendPtr backend, StreamHandle stream) {
    // Octo expects t as Tensor [B] (or scalar broadcasted)
    int64_t B = x_t.shape()[0];
    Tensor t_tensor = backend->alloc({B}, DType::Float32); // Assuming time is float
    backend->fill(t_tensor, t, stream);
    
    // Call Head
    Tensor noise_pred = head_->forward(x_t, t_tensor, condition, backend, stream);
    
    // Copy result to velocity (in-place output)
    backend->copy(velocity, noise_pred, stream);
}

Tensor OctoModel::decode_action(const Tensor& raw, BackendPtr backend, StreamHandle stream) {
    // Un-normalize. For now identity.
    Tensor out = backend->alloc(raw.shape(), raw.dtype());
    backend->copy(out, raw, stream);
    return out;
}

std::shared_ptr<OctoModel> load_octo(const std::string& checkpoint_dir, 
                                     BackendPtr backend, 
                                     const Device& device) {
    // Load config from json
    OctoConfig config; // Default for now
    // TODO: Load from json
    
    auto model = std::make_shared<OctoModel>(config, backend);
    
    // Load weights (safetensors)
    // ...
    
    return model;
}

OctoConfig OctoConfig::from_json(const std::string& path) {
    return OctoConfig();
}
}
}
