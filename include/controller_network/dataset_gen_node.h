#ifndef NN_ADAPTIVE_CONTROLLER_DATASET_GEN_NODE_H
#define NN_ADAPTIVE_CONTROLLER_DATASET_GEN_NODE_H

#include <fstream>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/Actuators.h>
#include <mav_msgs/AttitudeThrust.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <sensor_msgs/Imu.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf/transform_broadcaster.h>

namespace jetrov_control
{

    class DatasetGenNode
    {
    public:
        DatasetGenNode(const ros::NodeHandle &nh, const ros::NodeHandle &private_nh);
        ~DatasetGenNode();

    private:
        //general
        ros::NodeHandle nh_;
        ros::NodeHandle private_nh_;

        std::ofstream ofs_;

        //message_filter
        message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
        message_filters::Subscriber<mav_msgs::Actuators> rotor_sub_1_;
        message_filters::Subscriber<mav_msgs::Actuators> rotor_sub_2_;
        message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;

        typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, mav_msgs::Actuators, mav_msgs::Actuators, sensor_msgs::Imu> MySyncPolicy;
        typedef message_filters::Synchronizer<MySyncPolicy> Sync;
        boost::shared_ptr<Sync> sync_;

    private:
        void StatusCB(const nav_msgs::OdometryConstPtr &msg_1,
                      const mav_msgs::ActuatorsConstPtr &msg_2,
                      const mav_msgs::ActuatorsConstPtr &msg_3,
                      const sensor_msgs::ImuConstPtr &msg_4);
    };

} //namespace jetrov_control

#endif //NN_ADAPTIVE_CONTROLLER_DATASET_GEN_NODE_H