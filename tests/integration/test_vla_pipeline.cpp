// SPDX-License-Identifier: Apache-2.0
// Integration test: full InferenceEngine pipeline with a stub model.
// Requires CUDA; validates the condition-cache and denoising loop without
// real model weights.
#include <gtest/gtest.h>

#include "servoflow/engine/inference_engine.h"
#include "servoflow/sampling/sampler.h"
#include "servoflow/backend/backend.h"

#include <memory>

namespace sf {

// ── Stub model ────────────────────────────────────────────────────────────────
// Satisfies IVLAModel without real weights.  All outputs are zeroed tensors of
// the correct shape so the sampler can run a full denoising loop.
class StubVLAModel : public IVLAModel {
public:
    explicit StubVLAModel(int64_t action_dim    = 7,
                          int64_t action_horizon = 16)
        : action_dim_(action_dim), action_horizon_(action_horizon) {}

    Tensor encode_condition(const VLAInput& /*input*/,
                            BackendPtr backend,
                            StreamHandle stream) override {
        auto cond = backend->alloc({1, 64, 512}, DType::Float16, stream);
        backend->fill(cond, 0.f, stream);
        return cond;
    }

    void denoise_step(const Tensor& /*x_t*/, float /*t*/,
                      const Tensor& /*condition*/,
                      Tensor& velocity,
                      BackendPtr backend,
                      StreamHandle stream) override {
        backend->fill(velocity, 0.f, stream);
    }

    Tensor decode_action(const Tensor& raw,
                         BackendPtr /*backend*/,
                         StreamHandle /*stream*/) override {
        return raw;
    }

    int64_t action_dim()      const override { return action_dim_; }
    int64_t action_horizon()  const override { return action_horizon_; }
    DType   dtype()           const override { return DType::Float16; }

private:
    int64_t action_dim_;
    int64_t action_horizon_;
};

// ── Helper: minimal VLAInput on CPU (engine copies to GPU internally) ─────────
static VLAInput make_dummy_input(BackendPtr backend) {
    VLAInput inp;
    auto img = backend->alloc({1, 3, 224, 224}, DType::Float32);
    backend->fill(img, 0.f);
    inp.images.push_back(img);
    inp.language_tokens = backend->alloc({1, 32}, DType::Int32);
    inp.robot_state     = backend->alloc({1, 7},  DType::Float32);
    backend->fill(inp.robot_state, 0.f);
    return inp;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(VLAPipeline, SingleInference) {
    auto model   = std::make_shared<StubVLAModel>();
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/false);

    EngineConfig cfg;
    cfg.num_denoise_steps = 2;
    cfg.use_cuda_graph    = false;
    cfg.cache_condition   = false;

    InferenceEngine engine(model, sampler, cfg);
    VLAInput inp = make_dummy_input(engine.backend());
    VLAOutput out = engine.infer(inp);

    ASSERT_TRUE(out.actions.is_valid());
    EXPECT_GT(out.latency_ms, 0.0);

    const Shape& s = out.actions.shape();
    ASSERT_EQ(s.ndim(), 3);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[1], model->action_horizon());
    EXPECT_EQ(s[2], model->action_dim());
}

TEST(VLAPipeline, ConditionCacheHit) {
    auto model   = std::make_shared<StubVLAModel>();
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/false);

    EngineConfig cfg;
    cfg.num_denoise_steps = 2;
    cfg.use_cuda_graph    = false;
    cfg.cache_condition   = true;

    InferenceEngine engine(model, sampler, cfg);
    VLAInput inp = make_dummy_input(engine.backend());

    engine.mark_new_frame(1);
    VLAOutput out1 = engine.infer(inp);

    // Same frame id → cache hit, no re-encode.
    VLAOutput out2 = engine.infer(inp);

    EXPECT_TRUE(out1.actions.is_valid());
    EXPECT_TRUE(out2.actions.is_valid());
}

TEST(VLAPipeline, ConditionCacheInvalidatedOnNewFrame) {
    auto model   = std::make_shared<StubVLAModel>();
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/false);

    EngineConfig cfg;
    cfg.num_denoise_steps = 2;
    cfg.use_cuda_graph    = false;
    cfg.cache_condition   = true;

    InferenceEngine engine(model, sampler, cfg);
    VLAInput inp = make_dummy_input(engine.backend());

    engine.mark_new_frame(1);
    engine.infer(inp);

    // New frame id → cache invalidated, encode runs again.
    engine.mark_new_frame(2);
    VLAOutput out = engine.infer(inp);
    EXPECT_TRUE(out.actions.is_valid());
}

TEST(VLAPipeline, ExplicitCacheInvalidation) {
    auto model   = std::make_shared<StubVLAModel>();
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/false);

    EngineConfig cfg;
    cfg.num_denoise_steps = 2;
    cfg.use_cuda_graph    = false;
    cfg.cache_condition   = true;

    InferenceEngine engine(model, sampler, cfg);
    VLAInput inp = make_dummy_input(engine.backend());

    engine.mark_new_frame(1);
    engine.infer(inp);

    engine.invalidate_condition_cache();
    VLAOutput out = engine.infer(inp);
    EXPECT_TRUE(out.actions.is_valid());
}

}  // namespace sf
