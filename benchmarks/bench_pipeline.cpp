// SPDX-License-Identifier: Apache-2.0
// End-to-end pipeline benchmark: measures ServoFlow denoising loop latency
// and compares it against a reference (run python comparison separately).
//
// This benchmark does NOT load real RDT-1B weights — it uses a stub model
// that runs the same tensor operations to measure framework overhead.
// Use tools/convert/hf_to_servoflow.py to convert real weights and run
// benchmarks/scripts/run_comparison.py for the full diffusers comparison.
#include "servoflow/engine/inference_engine.h"
#include "servoflow/sampling/sampler.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

using namespace sf;

// ── Stub model: runs realistic tensor ops without real weights ────────────────
class StubVLAModel : public IVLAModel {
public:
    explicit StubVLAModel(BackendPtr backend, int64_t action_dim = 14,
                          int64_t action_horizon = 64)
        : backend_(backend), action_dim_(action_dim),
          action_horizon_(action_horizon) {
        // Pre-allocate dummy weights for stub ops.
        cond_embed_    = backend->alloc(Shape({1, 512, 1024}), DType::Float16);
        dit_weight_    = backend->alloc(Shape({1024, 1024}),   DType::Float16);
        action_weight_ = backend->alloc(Shape({action_dim_, 1024}), DType::Float16);
        StreamHandle s = backend->create_stream();
        backend->fill(cond_embed_,    0.01f, s);
        backend->fill(dit_weight_,    0.01f, s);
        backend->fill(action_weight_, 0.01f, s);
        backend->sync_stream(s);
        backend->destroy_stream(s);
    }

    Tensor encode_condition(const VLAInput& /*input*/, BackendPtr be,
                            StreamHandle stream) override {
        // Stub: return pre-allocated condition (no real vision encode).
        (void)be; (void)stream;
        return cond_embed_;
    }

    void denoise_step(const Tensor& x_t, float /*t*/,
                      const Tensor& /*condition*/,
                      Tensor& velocity,
                      BackendPtr be, StreamHandle stream) override {
        // Stub DiT step: one Linear + GELU + Linear.
        // Shape: x_t [1, T, action_dim] → project to [1, T, 1024] → back.
        Tensor hidden = be->alloc(Shape({1, action_horizon_, 1024}),
                                  DType::Float16, stream);
        // x_t → hidden
        be->gemm(x_t.view({action_horizon_, action_dim_}),
                 dit_weight_,
                 hidden.view({action_horizon_, 1024}),
                 1.f, 0.f, false, true, stream);
        be->gelu(hidden, hidden, stream);
        // hidden → velocity
        be->gemm(hidden.view({action_horizon_, 1024}),
                 action_weight_,
                 velocity.view({action_horizon_, action_dim_}),
                 1.f, 0.f, false, true, stream);
    }

    Tensor decode_action(const Tensor& raw, BackendPtr /*be*/,
                         StreamHandle /*stream*/) override {
        return raw;
    }

    int64_t action_dim()     const override { return action_dim_;     }
    int64_t action_horizon() const override { return action_horizon_; }
    DType   dtype()          const override { return DType::Float16;  }

private:
    BackendPtr backend_;
    int64_t    action_dim_;
    int64_t    action_horizon_;
    Tensor     cond_embed_;
    Tensor     dit_weight_;
    Tensor     action_weight_;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int num_steps = 10;  // default denoising steps
    if (argc > 1) num_steps = std::atoi(argv[1]);

    std::printf("=== ServoFlow Pipeline Benchmark ===\n");
    std::printf("Denoising steps: %d\n\n", num_steps);

    BackendPtr backend = get_backend(kCUDA0);

    EngineConfig cfg;
    cfg.device           = kCUDA0;
    cfg.compute_dtype    = DType::Float16;
    cfg.num_denoise_steps = num_steps;
    cfg.cache_condition  = true;
    cfg.use_cuda_graph   = true;

    auto model   = std::make_shared<StubVLAModel>(backend);
    auto sampler = std::make_shared<FlowMatchingSampler>(cfg.use_cuda_graph);

    InferenceEngine engine(model, sampler, cfg);

    // Dummy input (no real images/tokens needed for stub).
    VLAInput input;
    input.images        = {};
    input.language_tokens = backend->alloc(Shape({1, 32}), DType::Int32);
    input.robot_state   = backend->alloc(Shape({1, 14}),   DType::Float32);

    // Warm-up.
    constexpr int kWarmup = 5;
    constexpr int kIters  = 50;

    std::printf("Warming up (%d iters)...\n", kWarmup);
    for (int i = 0; i < kWarmup; ++i) engine.infer(input);

    std::printf("Benchmarking (%d iters)...\n", kIters);
    std::vector<double> latencies;
    latencies.reserve(kIters);

    for (int i = 0; i < kIters; ++i) {
        auto out = engine.infer(input);
        latencies.push_back(out.latency_ms);
    }

    // Statistics.
    double sum = 0, min_v = latencies[0], max_v = latencies[0];
    for (double v : latencies) {
        sum   += v;
        min_v  = std::min(min_v, v);
        max_v  = std::max(max_v, v);
    }
    double mean = sum / kIters;

    double freq_hz = 1000.0 / mean;
    std::printf("\nResults:\n");
    std::printf("  Mean latency:  %.2f ms  (%.1f Hz)\n", mean, freq_hz);
    std::printf("  Min  latency:  %.2f ms\n", min_v);
    std::printf("  Max  latency:  %.2f ms\n", max_v);
    std::printf("  50Hz target:   %s\n", mean < 20.0 ? "PASS ✓" : "FAIL (need more optimisation)");
    std::printf("=====================================\n");

    return 0;
}
