#include <torch/torch.h>
#include <controller_network/architecture.h>

#include <gtest/gtest.h>

TEST(ArchitectureImpl, forward) {

        auto batch_size = 1;
        auto row = 1;
        auto col = 12;
        auto cha = 1;
        auto x1 = torch::ones({batch_size, cha, row, col});
        auto x2 = torch::ones({batch_size, cha, 1, 6});
        auto x3 = torch::ones({batch_size, cha, 1, 6});
        int in_features = row * col * cha;
        int out_features = 6;

        Architecture architecture{in_features, out_features};

        auto y = architecture->forward(x1, x2, x3);

        ASSERT_EQ((std::vector<int64_t>{out_features, batch_size}), y.sizes());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}