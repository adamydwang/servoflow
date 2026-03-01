#include <gtest/gtest.h>
#include "servoflow/models/octo/octo.h"
#include "servoflow/backend/backend.h"
#include <vector>

using namespace sf;
using namespace sf::octo;

TEST(OctoTest, BasicFlow) {
    if (!BackendRegistry::instance().has(DeviceType::CUDA)) {
        GTEST_SKIP() << "CUDA backend not available";
    }
    auto backend = get_backend(DeviceType::CUDA, 0);

    OctoConfig config;
    config.hidden_dim = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.intermediate_dim = 128;
    config.num_layers = 1;
    config.action_dim = 7;
    config.action_horizon = 4;
    config.head_hidden_dim = 32;

    OctoModel model(config, backend);

    // 1. Encode Condition
    VLAInput input; // Empty for now
    // Create dummy condition manually since encode_condition is placeholder
    Tensor context = backend->alloc({1, 1, config.hidden_dim}, DType::Float16);
    backend->fill(context, 0.1f);

    // 2. Forward
    Tensor x_t = backend->alloc({1, config.action_horizon, config.action_dim}, DType::Float16);
    Tensor velocity = backend->alloc({1, config.action_horizon, config.action_dim}, DType::Float16);
    float t = 0.5f;
    
    backend->fill(x_t, 0.5f);

    // Context is tensor, but denoise_step expects condition tensor.
    // In our test we set cond.data["context"], but denoise_step signature is:
    // void denoise_step(..., const Tensor& condition, ...)
    // So we should pass context directly.
    
    model.denoise_step(x_t, t, context, velocity, backend, nullptr);
    backend->sync_device();

    EXPECT_EQ(velocity.shape(), x_t.shape());
    EXPECT_EQ(velocity.dtype(), DType::Float16);
}
