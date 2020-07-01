#include <ros/ros.h>

#include "neural_adaptive_controller/neural_adaptive_controller.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "neural_adaptive_controller_node");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    neural_adaptive_controller::NeuralAdaptiveController neural_adaptive_controller(nh, private_nh);

    ros::spin();

    return 0;
}