#include <ros/ros.h>
#include <torch/torch.h>
#include <controller_network/architecture.h>
#include <controller_network/custom_dataset.h>
#include <gtest/gtest.h>
#include "neural_adaptive_controller/neural_adaptive_controller.h"

TEST(NeuralAdaptiveController, neural_adaptive_controller)
{
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    // neural_adaptive_controller::NeuralAdaptiveController neural_adaptive_controller(nh, private_nh);

    EXPECT_TRUE(true);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}