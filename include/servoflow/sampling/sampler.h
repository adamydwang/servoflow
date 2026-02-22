// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/tensor.h"
#include "servoflow/backend/backend.h"
#include <functional>
#include <vector>

namespace sf {

// Denoising function type:
//   (noisy_action, timestep, condition_cache) → predicted_velocity
// The model fills this in; the sampler calls it at each step.
using DenoiseFn = std::function<void(
    const Tensor& x_t,           // noisy action [B, T_action, action_dim]
    float         t,             // timestep in [0, 1]
    const Tensor& condition,     // pre-computed condition embedding [B, S, D]
    Tensor&       velocity_out,  // output: predicted velocity
    StreamHandle  stream
)>;

// Noise schedule for ODE / SDE samplers.
struct Schedule {
    int   num_steps    = 10;    // number of denoising steps
    float t_start      = 1.f;  // start time (pure noise)
    float t_end        = 0.f;  // end time (clean action)

    // Returns uniformly spaced timesteps from t_start to t_end.
    std::vector<float> linspace() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// ISampler: interface for all sampling strategies.
// ─────────────────────────────────────────────────────────────────────────────
class ISampler {
public:
    virtual ~ISampler() = default;

    // Sample clean action given a noisy start and a denoising function.
    // x_init: initial noise [B, T_action, action_dim] — caller provides this.
    // condition: pre-computed condition (image/language tokens) [B, S, D].
    // Returns the denoised action tensor.
    virtual Tensor sample(const Tensor& x_init,
                          const Tensor& condition,
                          const DenoiseFn& denoise_fn,
                          const Schedule& schedule,
                          BackendPtr backend,
                          StreamHandle stream) = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// FlowMatchingSampler: Euler ODE solver for Rectified Flow / Flow Matching.
//
// Used by RDT-1B (and π0). The ODE is:
//   dx/dt = v_θ(x_t, t)
// Euler update: x_{t-Δt} = x_t - Δt * v_θ(x_t, t)
//
// Supports an optional CUDA Graph capture of the inner loop for maximum
// throughput when the sequence length and action dim are fixed.
// ─────────────────────────────────────────────────────────────────────────────
class FlowMatchingSampler : public ISampler {
public:
    // If use_cuda_graph=true the sampler captures the denoising loop into a
    // CUDA Graph on first call and replays it on subsequent calls.
    // IMPORTANT: CUDA Graph capture requires that tensor addresses are stable
    // across calls — the InferenceEngine must guarantee this.
    explicit FlowMatchingSampler(bool use_cuda_graph = true);

    Tensor sample(const Tensor& x_init,
                  const Tensor& condition,
                  const DenoiseFn& denoise_fn,
                  const Schedule& schedule,
                  BackendPtr backend,
                  StreamHandle stream) override;

private:
    bool use_cuda_graph_ = true;
    bool graph_captured_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// DDIMSampler: deterministic DDIM for DDPM-trained models.
// ─────────────────────────────────────────────────────────────────────────────
class DDIMSampler : public ISampler {
public:
    Tensor sample(const Tensor& x_init,
                  const Tensor& condition,
                  const DenoiseFn& denoise_fn,
                  const Schedule& schedule,
                  BackendPtr backend,
                  StreamHandle stream) override;
};

}  // namespace sf
