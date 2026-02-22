// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "servoflow/core/dtype.h"

using namespace sf;

TEST(DType, SizeCorrect) {
    EXPECT_EQ(dtype_size(DType::Float32),  4u);
    EXPECT_EQ(dtype_size(DType::Float16),  2u);
    EXPECT_EQ(dtype_size(DType::BFloat16), 2u);
    EXPECT_EQ(dtype_size(DType::Int8),     1u);
    EXPECT_EQ(dtype_size(DType::Int32),    4u);
}

TEST(DType, IsFloatingPoint) {
    EXPECT_TRUE(is_floating_point(DType::Float32));
    EXPECT_TRUE(is_floating_point(DType::Float16));
    EXPECT_TRUE(is_floating_point(DType::BFloat16));
    EXPECT_FALSE(is_floating_point(DType::Int8));
    EXPECT_FALSE(is_floating_point(DType::Int32));
}

TEST(DType, RoundTripString) {
    EXPECT_EQ(dtype_from_string(dtype_name(DType::Float32)),  DType::Float32);
    EXPECT_EQ(dtype_from_string(dtype_name(DType::Float16)),  DType::Float16);
    EXPECT_EQ(dtype_from_string(dtype_name(DType::BFloat16)), DType::BFloat16);
    EXPECT_EQ(dtype_from_string(dtype_name(DType::Int8)),     DType::Int8);
    EXPECT_EQ(dtype_from_string("F32"),  DType::Float32);
    EXPECT_EQ(dtype_from_string("BF16"), DType::BFloat16);
}

TEST(DType, UnknownString) {
    EXPECT_EQ(dtype_from_string("garbage"), DType::Unknown);
}
