#include <torch/torch.h>
#include <controller_network/architecture.h>
#include <controller_network/custom_dataset.h>
#include <gtest/gtest.h>

TEST(ArchitectureImpl, forward)
{

        auto batch_size = 1;
        auto row = 1;
        auto col = 24;
        auto cha = 1;
        auto x = torch::ones({batch_size, cha, row, col});
        std::cout << x << '\n';
        // auto x1 = torch::ones({batch_size, cha, row, col});
        // auto x2 = torch::ones({batch_size, cha, 1, 6});
        // auto x3 = torch::ones({batch_size, cha, 1, 6});
        int in_features = row * col * cha;
        int out_features = 6;

        Architecture architecture{in_features, out_features};

        auto y = architecture->forward(x);

        ASSERT_EQ((std::vector<int64_t>{out_features, batch_size}), y.sizes());
}

TEST(CustomDateset, forward)
{
        torch::manual_seed(1);
        std::vector<std::ifstream> ifss{};

        auto ds = CustomDataset{ifss}.map(torch::data::transforms::Stack<>());

        constexpr int BATCH_SIZE = 32;

        auto data_loader = torch::data::make_data_loader(
            std::move(ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));

        for (auto &batch : *data_loader)
        {
                auto batch_size = batch.data.size(0);
                ASSERT_EQ(batch.data.sizes(), (std::vector<std::int64_t>{32, 1, 1, 24}));
                ASSERT_EQ(BATCH_SIZE, batch_size);
                for (auto i = 0; i < batch_size; ++i)
                {
                        const auto &target = batch.target[i];
                        const auto &data = batch.data[i];

                        ASSERT_EQ(1, batch.data[i].size(0));
                        ASSERT_EQ(1, batch.data[i].size(1));
                        ASSERT_EQ(24, batch.data[i].size(2));

                        ASSERT_EQ(batch.target.sizes(), (std::vector<std::int64_t>{32, 6, 1}));
                }
        }
}

int main(int argc, char **argv)
{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}