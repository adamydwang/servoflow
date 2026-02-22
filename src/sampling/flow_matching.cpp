// SPDX-License-Identifier: Apache-2.0
#include "servoflow/sampling/sampler.h"

#include <stdexcept>
#include <vector>
#include <cmath>

namespace sf {

std::vector<float> Schedule::linspace() const {
    std::vector<float> ts(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        float alpha = static_cast<float>(i) / num_steps;
        ts[i] = t_start + alpha * (t_end - t_start);
    }
    return ts;
}

// ─────────────────────────────────────────────────────────────────────────────
// FlowMatchingSampler
// ─────────────────────────────────────────────────────────────────────────────
FlowMatchingSampler::FlowMatchingSampler(bool use_cuda_graph)
    : use_cuda_graph_(use_cuda_graph) {}

Tensor FlowMatchingSampler::sample(const Tensor& x_init,
                                    const Tensor& condition,
                                    const DenoiseFn& denoise_fn,
                                    const Schedule& schedule,
                                    BackendPtr backend,
                                    StreamHandle stream) {
    // Allocate working buffers — same shape as x_init.
    Tensor x_t    = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    Tensor vel    = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    Tensor x_next = backend->alloc(x_init.shape(), x_init.dtype(), stream);

    // Initialise x_t = x_init (copy, not alias, so graph capture sees stable ptrs).
    backend->copy(x_t, x_init, stream);

    auto timesteps = schedule.linspace();  // [t_start, ..., t_end], length = N+1

    // ── Optionally capture the denoising loop into a CUDA Graph ───────────
    // Graph capture: the first call runs the loop "live" while CUDA records
    // all kernel launches; subsequent calls replay the graph instantly,
    // eliminating CPU overhead from the N-step loop (~microseconds per step
    // saved, adds up at 50 Hz with N=10 steps).
    //
    // Requirements for graph capture:
    //   1. All tensor addresses must be stable (pre-allocated by InferenceEngine).
    //   2. No host-side branches inside the loop that depend on device data.
    //   3. denoise_fn must not synchronize the stream.
    //
    if (use_cuda_graph_ && !graph_captured_) {
        backend->graph_begin_capture(stream);
    }

    for (int step = 0; step < schedule.num_steps; ++step) {
        float t     = timesteps[step];
        float t_next = timesteps[step + 1];
        float dt    = t_next - t;  // negative (going from noise to data)

        // Predict velocity: vel = v_θ(x_t, t, condition)
        denoise_fn(x_t, t, condition, vel, stream);

        // Euler update: x_next = x_t + dt * vel
        // Implemented as: x_next = x_t + scale(vel, dt)
        backend->scale(vel, dt, x_next, stream);
        backend->add(x_t, x_next, x_t, stream);
    }

    if (use_cuda_graph_ && !graph_captured_) {
        backend->graph_end_capture(stream);
        graph_captured_ = true;
    } else if (use_cuda_graph_ && graph_captured_) {
        // Replay: reset x_t from x_init first (done before capture replay).
        backend->copy(x_t, x_init, stream);
        backend->graph_launch(stream);
    }

    return x_t;
}

// ─────────────────────────────────────────────────────────────────────────────
// DDIMSampler
// ─────────────────────────────────────────────────────────────────────────────
Tensor DDIMSampler::sample(const Tensor& x_init,
                            const Tensor& condition,
                            const DenoiseFn& denoise_fn,
                            const Schedule& schedule,
                            BackendPtr backend,
                            StreamHandle stream) {
    Tensor x_t  = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    Tensor eps  = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    Tensor tmp  = backend->alloc(x_init.shape(), x_init.dtype(), stream);

    backend->copy(x_t, x_init, stream);
    auto timesteps = schedule.linspace();

    for (int step = 0; step < schedule.num_steps; ++step) {
        float t      = timesteps[step];
        float t_prev = timesteps[step + 1];

        float alpha_t      = 1.f - t;
        float alpha_t_prev = 1.f - t_prev;

        // Predict noise (model output is treated as noise prediction here).
        denoise_fn(x_t, t, condition, eps, stream);

        // DDIM deterministic update:
        //   x_0_pred = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        //   x_prev   = sqrt(alpha_t_prev) * x_0_pred + sqrt(1-alpha_t_prev) * eps
        float sqrt_alpha_t      = sqrtf(alpha_t);
        float sqrt_1m_alpha_t   = sqrtf(1.f - alpha_t);
        float sqrt_alpha_t_prev = sqrtf(alpha_t_prev);
        float sqrt_1m_alpha_tp  = sqrtf(1.f - alpha_t_prev);

        // x_0_pred = (x_t - sqrt_1m_alpha_t * eps) / sqrt_alpha_t
        backend->scale(eps, -sqrt_1m_alpha_t, tmp, stream);
        backend->add(x_t, tmp, tmp, stream);           // tmp = x_t - sqrt(1-a)*eps
        backend->scale(tmp, 1.f / sqrt_alpha_t, tmp, stream);  // tmp = x_0_pred

        // x_prev = sqrt_alpha_t_prev * x_0_pred + sqrt_1m_alpha_tp * eps
        backend->scale(tmp, sqrt_alpha_t_prev, x_t, stream);
        backend->scale(eps, sqrt_1m_alpha_tp, tmp, stream);
        backend->add(x_t, tmp, x_t, stream);
    }

    return x_t;
}

}  // namespace sf
