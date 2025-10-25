#include <gtest/gtest.h>
#include "tensor.h"
#include <vector>

using namespace matrix_library;

TEST(MatrixUsageTest, BasicTensorOperations)
{
    Tensor<int> t({2, 2});
    EXPECT_EQ(t.size(), 4);

    EXPECT_EQ((t[{0, 0}]), 5);
    EXPECT_EQ((t[{0, 1}]), 5);
    EXPECT_EQ((t[{1, 0}]), 5);
    EXPECT_EQ((t[{1, 1}]), 5);

    t.set_data(std::vector<int>{1, 2, 3, 4});
    EXPECT_EQ((t[{0, 0}]), 1);
    EXPECT_EQ((t[{0, 1}]), 2);
    EXPECT_EQ((t[{1, 0}]), 3);
    EXPECT_EQ((t[{1, 1}]), 4);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
