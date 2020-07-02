#include <torch/torch.h>
#include <controller_network/architecture.h>
#include <controller_network/custom_dataset.h>
#include <gtest/gtest.h>

#include <ros/ros.h>

#include "neural_adaptive_controller/neural_adaptive_controller.h"

TEST(ArchitectureImpl, forward)
{
        torch::manual_seed(1);
        std::vector<std::ifstream> ifss{};

        auto ds = CustomDataset{ifss}.map(torch::data::transforms::Stack<>());

        constexpr int BATCH_SIZE = 32;

        auto data_loader = torch::data::make_data_loader(
            std::move(ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));
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
                ASSERT_EQ(batch.data.sizes(), (std::vector<std::int64_t>{32, 1, 1, 12}));
                ASSERT_EQ(BATCH_SIZE, batch_size);
                for (auto i = 0; i < batch_size; ++i)
                {
                        const auto &target = batch.target[i];
                        const auto &data = batch.data[i];

                        ASSERT_EQ(1, batch.data[i].size(0));
                        ASSERT_EQ(1, batch.data[i].size(1));
                        ASSERT_EQ(12, batch.data[i].size(2));

                        ASSERT_EQ(batch.target.sizes(), (std::vector<std::int64_t>{32, 1, 3}));
                }
        }
}

TEST(Training, forward)
{
        constexpr double LEARNING_RATE{0.000005};
        int kNumberOfEpochs = 5;

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

        // auto ds = CustomDataset{ifss}.map(torch::data::transforms::Stack<>());
        auto ds = CustomDataset{ifss}.map(torch::data::transforms::Normalize<>(0.0, 0.0001)).map(torch::data::transforms::Stack<>());

        constexpr int BATCH_SIZE = 256;

        auto data_loader = torch::data::make_data_loader(
            std::move(ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));

        for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
        {
                for (auto &batch : *data_loader)
                {
                        auto batch_size = batch.data.size(0);
                        ASSERT_EQ(batch.data.sizes(), (std::vector<std::int64_t>{BATCH_SIZE, 1, 1, 12}));
                        ASSERT_EQ(BATCH_SIZE, batch_size);

                        auto data = batch.data.to(device);
                        auto targets = batch.target.to(device);
                        optimizer.zero_grad();

                        // std::cout << data << '\n';
                        // std::cout << targets << '\n';

                        auto output = model->forward(data);
                        // std::cout << output << '\n';

                        auto loss = torch::mse_loss(output, targets);
                        float loss_val = loss.item<float>();

                        loss.backward();
                        optimizer.step();

                        std::cout << "Loss: " << loss_val << std::endl;
                }
        }

        torch::save(model, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/model.pt");
        torch::save(optimizer, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/opt.pt");
}

// TEST(Test, outputLayer)
// {
//         constexpr double LEARNING_RATE{0.000005};
//         int kNumberOfEpochs = 5;

//         EXPECT_TRUE(torch::cuda::is_available());
//         torch::DeviceType device_type{};
//         if (torch::cuda::is_available())
//         {
//                 std::cout << "CUDA available! Training on GPU." << std::endl;
//                 device_type = torch::kCUDA;
//         }
//         else
//         {
//                 std::cout << "Training on CPU." << std::endl;
//                 device_type = torch::kCPU;
//         }
//         torch::Device device{device_type};

//         Architecture model(100, 100);
//         model->to(device);
//         // model->updateMode(BACKPROP);
//         // model->updateMode(ADAPTATION);

//         torch::optim::Adam optimizer{
//             model->parameters(),
//             torch::optim::AdamOptions(LEARNING_RATE)};

//         torch::load(model, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/model.pt");
//         // torch::load(optimizer, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/opt.pt");

//         torch::manual_seed(1);
//         std::vector<std::ifstream> ifss{};

//         auto ds = CustomDataset{ifss}.map(torch::data::transforms::Stack<>());
//         // auto ds = CustomDataset{ifss}.map(torch::data::transforms::Normalize<>(0, 0.01)).map(torch::data::transforms::Stack<>());

//         constexpr int BATCH_SIZE = 32;

//         auto data_loader = torch::data::make_data_loader(
//             std::move(ds),
//             torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));
//         for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
//         {
//                 for (auto &batch : *data_loader)
//                 {
//                         auto batch_size = batch.data.size(0);
//                         ASSERT_EQ(batch.data.sizes(), (std::vector<std::int64_t>{BATCH_SIZE, 1, 1, 12}));
//                         ASSERT_EQ(BATCH_SIZE, batch_size);

//                         auto data = batch.data.to(device);
//                         auto targets = batch.target.to(device);
//                         // optimizer.zero_grad();

//                         auto output = model->forward(data);

//                         // std::cout << output << '\n';

//                         auto loss = torch::mse_loss(output, targets);
//                         float loss_val = loss.item<float>();

//                         // loss.backward();
//                         // optimizer.step();

//                         std::cout << "Test Loss: " << loss_val << std::endl;
//                 }
//         }
// }

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