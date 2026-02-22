// SPDX-License-Identifier: Apache-2.0
// Tests the sampler math on CPU (no CUDA required).
#include <gtest/gtest.h>
#include "servoflow/sampling/sampler.h"

using namespace sf;

TEST(Schedule, LinspaceCorrect) {
    Schedule s;
    s.num_steps = 4;
    s.t_start   = 1.f;
    s.t_end     = 0.f;

    auto ts = s.linspace();
    ASSERT_EQ(ts.size(), 5u);
    EXPECT_FLOAT_EQ(ts[0], 1.f);
    EXPECT_FLOAT_EQ(ts[4], 0.f);
    // Uniformly spaced.
    for (size_t i = 1; i < ts.size(); ++i)
        EXPECT_NEAR(ts[i - 1] - ts[i], 0.25f, 1e-6f);
}

TEST(Schedule, SingleStep) {
    Schedule s;
    s.num_steps = 1;
    s.t_start   = 1.f;
    s.t_end     = 0.f;
    auto ts = s.linspace();
    ASSERT_EQ(ts.size(), 2u);
    EXPECT_FLOAT_EQ(ts[0], 1.f);
    EXPECT_FLOAT_EQ(ts[1], 0.f);
}
