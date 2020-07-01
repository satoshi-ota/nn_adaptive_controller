#include "controller_network/dataset_gen_node.h"

using namespace message_filters;
namespace jetrov_control
{

    DatasetGenNode::DatasetGenNode(
        const ros::NodeHandle &nh, const ros::NodeHandle &private_nh)
        : nh_(nh)
    {
        odom_sub_.subscribe(nh_, "/pelican/payload/odom", 1);
        rotor_sub_1_.subscribe(nh_, "/pelican1/command/motor_speed", 1);
        rotor_sub_2_.subscribe(nh_, "/pelican2/command/motor_speed", 1);
        imu_sub_.subscribe(nh_, "/pelican/payload/imu/data", 1);

        sync_.reset(new Sync(MySyncPolicy(10), odom_sub_, rotor_sub_1_, rotor_sub_2_, imu_sub_));
        sync_->registerCallback(boost::bind(&DatasetGenNode::StatusCB, this, _1, _2, _3, _4));

        std::string filename = "data.csv";

        ofs_.open("/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/data.csv");

        std::cout << "writing " << filename << "..." << std::endl;
    }

    DatasetGenNode::~DatasetGenNode() {}

    void DatasetGenNode::StatusCB(const nav_msgs::OdometryConstPtr &msg_1,
                                  const mav_msgs::ActuatorsConstPtr &msg_2,
                                  const mav_msgs::ActuatorsConstPtr &msg_3,
                                  const sensor_msgs::ImuConstPtr &msg_4)
    {
        ROS_INFO_ONCE("Got first data! Generate csv");

        geometry_msgs::Pose pose = msg_1->pose.pose;

        double roll, pitch, yaw;
        tf::Quaternion quat;
        quaternionMsgToTF(msg_1->pose.pose.orientation, quat);
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

        geometry_msgs::Twist twist = msg_1->twist.twist;

        ofs_ << pose.position.x << ", " << pose.position.y << ", " << pose.position.z << ", "
             << roll << ", " << pitch << ", " << yaw << ", "
             << twist.linear.x << ", " << twist.linear.y << ", " << twist.linear.z << ", "
             << twist.angular.x << ", " << twist.angular.y << ", " << twist.angular.z << ", "
             << msg_4->linear_acceleration.x << ", " << msg_4->linear_acceleration.y << ", " << msg_4->linear_acceleration.z << ", "
             << 0.0 << ", " << 0.0 << ", " << 0.0 << ", "
             << twist.linear.x << ", " << twist.linear.y << ", " << twist.linear.z << ", "
             << twist.angular.x << ", " << twist.angular.y << ", " << twist.angular.z << ", "
             << msg_2->angular_velocities[0] / 838 << ", " << msg_2->angular_velocities[1] / 838 << ", " << msg_2->angular_velocities[2] / 838 << ", " << msg_2->angular_velocities[3] / 838 << ", "
             << msg_3->angular_velocities[0] / 838 << ", " << msg_3->angular_velocities[1] / 838 << ", " << msg_3->angular_velocities[2] / 838 << ", " << msg_3->angular_velocities[3] / 838 << std::endl;
    }

} //namespace jetrov_control

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dataset_gen_node");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    jetrov_control::DatasetGenNode dataset_gen_node(nh, private_nh);

    ros::spin();

    return 0;
}