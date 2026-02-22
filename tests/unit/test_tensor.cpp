// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "servoflow/core/tensor.h"

using namespace sf;

// Helper: create a CPU tensor backed by a plain malloc buffer.
static Tensor make_cpu_tensor(Shape shape, DType dtype) {
    size_t bytes = shape.nbytes(dtype_size(dtype));
    void*  ptr   = std::malloc(bytes);
    auto   del   = [](void* p) { std::free(p); };
    auto   stor  = std::make_shared<Storage>(ptr, bytes, kCPU, del);
    return Tensor(std::move(stor), shape, dtype);
}

TEST(Tensor, BasicMetadata) {
    auto t = make_cpu_tensor({2, 3, 4}, DType::Float32);
    EXPECT_EQ(t.ndim(),  3);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.nbytes(), 96u);
    EXPECT_TRUE(t.is_cpu());
    EXPECT_FALSE(t.is_cuda());
    EXPECT_TRUE(t.is_contiguous());
}

TEST(Tensor, DataAccess) {
    auto t = make_cpu_tensor({4}, DType::Float32);
    float* p = t.data_ptr<float>();
    for (int i = 0; i < 4; ++i) p[i] = static_cast<float>(i);
    EXPECT_FLOAT_EQ(t.data_ptr<float>()[2], 2.f);
}

TEST(Tensor, ViewSharesStorage) {
    auto t = make_cpu_tensor({6}, DType::Float32);
    float* p = t.data_ptr<float>();
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i);

    auto v = t.view({2, 3});
    EXPECT_EQ(v.shape()[0], 2);
    EXPECT_EQ(v.shape()[1], 3);
    // Same underlying storage.
    EXPECT_EQ(v.raw_data_ptr(), t.raw_data_ptr());
}

TEST(Tensor, ViewNumelMismatch) {
    auto t = make_cpu_tensor({6}, DType::Float32);
    EXPECT_THROW(t.view({2, 4}), std::invalid_argument);
}

TEST(Tensor, Slice) {
    auto t = make_cpu_tensor({4}, DType::Float32);
    float* p = t.data_ptr<float>();
    for (int i = 0; i < 4; ++i) p[i] = static_cast<float>(i);

    auto s = t.slice(1, 3);
    EXPECT_EQ(s.numel(), 2);
    EXPECT_FLOAT_EQ(s.data_ptr<float>()[0], 1.f);
    EXPECT_FLOAT_EQ(s.data_ptr<float>()[1], 2.f);
}

TEST(Tensor, Unsqueeze) {
    auto t = make_cpu_tensor({3, 4}, DType::Float16);
    auto u = t.unsqueeze(0);
    EXPECT_EQ(u.ndim(), 3);
    EXPECT_EQ(u.shape()[0], 1);
    EXPECT_EQ(u.shape()[1], 3);
    EXPECT_EQ(u.shape()[2], 4);
}

TEST(Tensor, InvalidConstruct) {
    // Default tensor should be invalid.
    Tensor t;
    EXPECT_FALSE(t.is_valid());
}
