// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "servoflow/core/shape.h"

using namespace sf;

TEST(Shape, DefaultEmpty) {
    Shape s;
    EXPECT_EQ(s.ndim(), 0u);
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.numel(), 0);
}

TEST(Shape, InitializerList) {
    Shape s{2, 3, 4};
    EXPECT_EQ(s.ndim(), 3u);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 4);
    EXPECT_EQ(s.numel(), 24);
}

TEST(Shape, NBytes) {
    Shape s{4, 16};
    EXPECT_EQ(s.nbytes(4), 256u);   // float32
    EXPECT_EQ(s.nbytes(2), 128u);   // float16
}

TEST(Shape, Equality) {
    EXPECT_EQ((Shape{1, 2, 3}), (Shape{1, 2, 3}));
    EXPECT_NE((Shape{1, 2, 3}), (Shape{1, 2, 4}));
    EXPECT_NE((Shape{1, 2}),    (Shape{1, 2, 3}));
}

TEST(Shape, TooManyDims) {
    EXPECT_THROW(
        (Shape{1, 2, 3, 4, 5, 6, 7, 8, 9}),
        std::invalid_argument);
}

TEST(Shape, StringRepresentation) {
    EXPECT_EQ((Shape{2, 3}).str(), "[2, 3]");
}

TEST(Shape, RangeFor) {
    Shape s{1, 2, 3};
    std::vector<int64_t> dims(s.begin(), s.end());
    EXPECT_EQ(dims, (std::vector<int64_t>{1, 2, 3}));
}
