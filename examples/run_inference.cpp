// SPDX-License-Identifier: Apache-2.0
// ServoFlow — minimal inference example.
//
// Demonstrates how to set up an InferenceEngine with a stub model and run
// a fixed-frequency control loop at ~50 Hz.
//
// Usage:
//   ./run_inference [num_steps]
//
// For a real deployment replace StubModel with an RDT1B (or other) instance
// and call engine.load_weights("/path/to/checkpoint").

#include "servoflow/engine/inference_engine.h"
#include "servoflow/sampling/sampler.h"
#include "servoflow/backend/backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

using namespace sf;

// ── Stub model (replace with RDT1B for real use) ──────────────────────────────
class StubModel : public IVLAModel {
public:
    Tensor encode_condition(const VLAInput&, BackendPtr b, StreamHandle s) override {
        auto c = b->alloc({1, 64, 512}, DType::Float16, s);
        b->fill(c, 0.f, s);
        return c;
    }
    void denoise_step(const Tensor&, float, const Tensor&,
                      Tensor& vel, BackendPtr b, StreamHandle s) override {
        b->fill(vel, 0.f, s);
    }
    Tensor decode_action(const Tensor& raw, BackendPtr, StreamHandle) override {
        return raw;
    }
    int64_t action_dim()     const override { return 7; }
    int64_t action_horizon() const override { return 16; }
    DType   dtype()          const override { return DType::Float16; }
};

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    const int total_steps = (argc > 1) ? std::atoi(argv[1]) : 10;

    // Build engine.
    auto model   = std::make_shared<StubModel>();
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/true);

    EngineConfig cfg;
    cfg.num_denoise_steps = 10;
    cfg.cache_condition   = true;
    cfg.use_cuda_graph    = true;
    cfg.pinned_output     = true;

    InferenceEngine engine(model, sampler, cfg);
    auto backend = engine.backend();

    // Build a dummy observation (camera image + language tokens + robot state).
    VLAInput obs;
    auto img = backend->alloc({1, 3, 224, 224}, DType::Float32);
    backend->fill(img, 0.f);
    obs.images.push_back(img);
    obs.language_tokens = backend->alloc({1, 32}, DType::Int32);
    obs.robot_state     = backend->alloc({1, 7},  DType::Float32);
    backend->fill(obs.robot_state, 0.f);

    // Simulate a 50 Hz control loop.
    const double target_hz  = 50.0;
    const auto   period_us  = std::chrono::microseconds(
                                  static_cast<long>(1e6 / target_hz));

    std::printf("ServoFlow — running %d steps at %.0f Hz target\n",
                total_steps, target_hz);
    std::printf("%-6s  %-10s  %-10s\n", "step", "latency_ms", "actions[0]");

    double total_latency_ms = 0.0;

    for (int step = 0; step < total_steps; ++step) {
        auto t0 = std::chrono::steady_clock::now();

        // Mark a new camera frame every 5 control steps to exercise
        // the condition cache (images assumed to arrive at ~10 Hz).
        if (step % 5 == 0) {
            engine.mark_new_frame(static_cast<uint64_t>(step / 5 + 1));
        }

        VLAOutput out = engine.infer(obs);
        total_latency_ms += out.latency_ms;

        // Read back first action value (fp16 → fp32 on CPU).
        float a0 = 0.f;
        if (out.actions.dtype() == DType::Float16) {
            // Reinterpret the raw bytes as uint16_t and convert manually.
            const uint16_t raw = *out.actions.data_ptr<uint16_t>();
            // Simple fp16→fp32 via memcpy trick (requires host pointer).
            uint32_t bits = (static_cast<uint32_t>(raw & 0x8000) << 16)
                          | (static_cast<uint32_t>((raw >> 10) & 0x1f) << 23)
                          | (static_cast<uint32_t>(raw & 0x3ff) << 13);
            std::memcpy(&a0, &bits, 4);
        } else {
            a0 = *out.actions.data_ptr<float>();
        }

        std::printf("%-6d  %-10.3f  %-10.6f\n", step, out.latency_ms, a0);

        // Sleep to maintain target frequency.
        auto elapsed = std::chrono::steady_clock::now() - t0;
        if (elapsed < period_us) {
            std::this_thread::sleep_for(period_us - elapsed);
        }
    }

    std::printf("\nAverage latency: %.3f ms  (%.1f Hz achievable)\n",
                total_latency_ms / total_steps,
                1000.0 / (total_latency_ms / total_steps));

    return 0;
}
