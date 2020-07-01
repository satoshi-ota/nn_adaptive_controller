#include <torch/torch.h>
#include <controller_network/architecture.h>
#include <controller_network/custom_dataset.h>
#include <gtest/gtest.h>

#include <ros/ros.h>

#include "neural_adaptive_controller/neural_adaptive_controller.h"

// TEST(ArchitectureImpl, forward)
// {

//         auto batch_size = 1;
//         auto row = 1;
//         auto col = 24;
//         auto cha = 1;
//         auto x = torch::ones({batch_size, cha, row, col});
//         // std::cout << x << '\n';
//         // auto x1 = torch::ones({batch_size, cha, row, col});
//         // auto x2 = torch::ones({batch_size, cha, 1, 6});
//         // auto x3 = torch::ones({batch_size, cha, 1, 6});
//         int in_features = row * col * cha;
//         int out_features = 8;

//         Architecture architecture{in_features, out_features};

//         auto y = architecture->forward(x);

//         ASSERT_EQ((std::vector<int64_t>{batch_size, out_features}), y.sizes());
// }

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

                        ASSERT_EQ(batch.target.sizes(), (std::vector<std::int64_t>{32, 1, 8}));
                }
        }
}

TEST(Training, forward)
{
        constexpr double LEARNING_RATE{0.0005};
        int kNumberOfEpochs = 10;

        EXPECT_TRUE(torch::cuda::is_available());
        torch::DeviceType device_type{};
        if (torch::cuda::is_available())
        {
                std::cout << "CUDA available! Training on GPU." << std::endl;
                device_type = torch::kCUDA;
        }
        else
        {
                std::cout << "Training on CPU." << std::endl;
                device_type = torch::kCPU;
        }
        torch::Device device{device_type};

        Architecture model(100, 100);
        model->to(device);

        torch::optim::Adam optimizer{
            model->parameters(),
            torch::optim::AdamOptions(LEARNING_RATE)};

        torch::manual_seed(1);
        std::vector<std::ifstream> ifss{};

        auto ds = CustomDataset{ifss}.map(torch::data::transforms::Stack<>());

        constexpr int BATCH_SIZE = 256;

        auto data_loader = torch::data::make_data_loader(
            std::move(ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));
        for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
        {
                for (auto &batch : *data_loader)
                {
                        auto batch_size = batch.data.size(0);
                        ASSERT_EQ(batch.data.sizes(), (std::vector<std::int64_t>{256, 1, 1, 24}));
                        ASSERT_EQ(BATCH_SIZE, batch_size);

                        auto data = batch.data.to(device);
                        auto targets = batch.target.to(device);
                        optimizer.zero_grad();

                        auto output = model->forward(data);

                        auto loss = torch::mse_loss(output, targets);
                        float loss_val = loss.item<float>();

                        loss.backward();
                        optimizer.step();

                        std::cout << "Loss: " << loss_val << std::endl;
                }
        }

        // torch::save(model, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/model.pt");
        // torch::save(optimizer, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/opt.pt");
}

TEST(Adaptation, outputLayer)
{
        constexpr double LEARNING_RATE{0.0005};
        int kNumberOfEpochs = 10;

        EXPECT_TRUE(torch::cuda::is_available());
        torch::DeviceType device_type{};
        if (torch::cuda::is_available())
        {
                std::cout << "CUDA available! Training on GPU." << std::endl;
                device_type = torch::kCUDA;
        }
        else
        {
                std::cout << "Training on CPU." << std::endl;
                device_type = torch::kCPU;
        }
        torch::Device device{device_type};

        Architecture model(100, 100);
        model->to(device);
        // model->updateMode(BACKPROP);
        // model->updateMode(ADAPTATION);

        torch::optim::Adam optimizer{
            model->parameters(),
            torch::optim::AdamOptions(LEARNING_RATE)};

        // torch::load(model, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/model.pt");
        // torch::load(optimizer, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/opt.pt");

        torch::manual_seed(1);
        std::vector<std::ifstream> ifss{};

        auto ds = CustomDataset{ifss}.map(torch::data::transforms::Stack<>());

        constexpr int BATCH_SIZE = 256;

        auto data_loader = torch::data::make_data_loader(
            std::move(ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));
        for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
        {
                for (auto &batch : *data_loader)
                {
                        auto batch_size = batch.data.size(0);
                        ASSERT_EQ(batch.data.sizes(), (std::vector<std::int64_t>{256, 1, 1, 24}));
                        ASSERT_EQ(BATCH_SIZE, batch_size);

                        auto data = batch.data.to(device);
                        auto targets = batch.target.to(device);
                        optimizer.zero_grad();

                        auto output = model->forward(data);

                        auto loss = torch::mse_loss(output, targets);
                        float loss_val = loss.item<float>();

                        loss.backward();
                        optimizer.step();

                        std::cout << "Loss: " << loss_val << std::endl;
                }
        }
}

// TEST(NeuralAdaptiveController, neural_adaptive_controller)
// {
//         ros::NodeHandle nh;
//         ros::NodeHandle private_nh("~");
//         neural_adaptive_controller::NeuralAdaptiveController neural_adaptive_controller(nh, private_nh);
// }

int main(int argc, char **argv)
{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}