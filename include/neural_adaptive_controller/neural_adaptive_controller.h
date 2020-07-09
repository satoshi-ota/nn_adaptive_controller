#ifndef NEURAL_ADAPTIVE_CONTROLLER_H
#define NEURAL_ADAPTIVE_CONTROLLER_H

#include <boost/bind.hpp>
#include <Eigen/Eigen>
#include <stdio.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <mav_msgs/Actuators.h>
#include <mav_msgs/AttitudeThrust.h>
#include <mav_msgs/common.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <mav_msgs/conversions.h>
#include <nav_msgs/Odometry.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <torch/torch.h>
#include <controller_network/architecture.h>
#include <controller_network/custom_dataset.h>

#include "neural_adaptive_controller/common.h"

#include <controller_network/matplotlibcpp.h>

namespace plt = matplotlibcpp;

// #include "neural_adaptive_controller/parameters.h"
// #include <lee_controller/lee_controller_co_trans.h>
// using namespace rotors_control;
// namespace rotors_control
// {
namespace neural_adaptive_controller
{
    // Default values for the lee position controller and the Asctec Firefly.
    static const Eigen::Vector3d kDefaultPositionGain = Eigen::Vector3d(6, 6, 6);
    static const Eigen::Vector3d kDefaultVelocityGain = Eigen::Vector3d(4.7, 4.7, 4.7);
    static const Eigen::Vector3d kDefaultAttitudeGain = Eigen::Vector3d(3, 3, 0.035);
    static const Eigen::Vector3d kDefaultAngularRateGain = Eigen::Vector3d(0.52, 0.52, 0.025);

    class NeuralAdaptiveControllerParameters
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        NeuralAdaptiveControllerParameters()
            : position_gain_(kDefaultPositionGain),
              velocity_gain_(kDefaultVelocityGain),
              attitude_gain_(kDefaultAttitudeGain),
              angular_rate_gain_(kDefaultAngularRateGain)
        {
            calculateAllocationMatrix(rotor_configuration_, &allocation_matrix_);
        }

        Eigen::Matrix4Xd allocation_matrix_, global_allocation_matrix_;
        Eigen::Vector3d position_gain_;
        Eigen::Vector3d velocity_gain_;
        Eigen::Vector3d attitude_gain_;
        Eigen::Vector3d angular_rate_gain_;
        RotorConfiguration rotor_configuration_;
    };

    class NeuralAdaptiveController
    {
    public:
        NeuralAdaptiveController(const ros::NodeHandle &nh, const ros::NodeHandle &private_nh);
        ~NeuralAdaptiveController();

        void InitializeParams();

        void InitializeParameters();
        void CalculateRotorVelocities(const Eigen::Vector3d &acceleration, const Eigen::VectorXd &nn_input, Eigen::VectorXd *rotor_velocities);
        void CalculateNNInput(const Eigen::Vector3d &x_n,
                              const Eigen::Vector3d &v_n,
                              const Eigen::Vector3d &x_e,
                              const Eigen::Vector3d &v_e,
                              Eigen::VectorXd *nn_input);

        void SetOdometry(const EigenOdometry &odometry);
        void SetTrajectoryPoint(const mav_msgs::EigenTrajectoryPoint &command_trajectory);

        NeuralAdaptiveControllerParameters controller_parameters_;
        VehicleParameters vehicle_parameters_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        ros::NodeHandle nh_;
        ros::NodeHandle private_nh_;

        Architecture model_;

        // LeeControllerCoTrans lee_controller_co_trans_;

        std::string namespace_;

        // subscribers
        ros::Subscriber cmd_trajectory_sub_;
        ros::Subscriber cmd_multi_dof_joint_trajectory_sub_;
        ros::Subscriber cmd_pose_sub_;
        ros::Subscriber odometry_sub_;

        ros::Publisher motor_velocity_reference_pub1_;
        ros::Publisher motor_velocity_reference_pub2_;

        mav_msgs::EigenTrajectoryPointDeque commands_;
        std::deque<ros::Duration> command_waiting_times_;
        ros::Timer command_timer_;

        bool initialized_params_;
        bool controller_active_;

        Eigen::Vector3d normalized_attitude_gain_;
        Eigen::Vector3d normalized_angular_rate_gain_;
        Eigen::MatrixX4d angular_acc_to_rotor_velocities_;

        Eigen::MatrixXd allocate_rotor_velocities_;
        Eigen::MatrixXd force_torque_mapping_;

        Eigen::MatrixXd W_;

        mav_msgs::EigenTrajectoryPoint command_trajectory_;
        EigenOdometry odometry_;

        std::vector<float> pitch_, pitch_err_, rate_y_, rate_err_, torque_s_, torque_y_;

        void ComputePosAtt(Eigen::Vector3d *acceleration, Eigen::Vector3d *posatt_now, Eigen::Vector3d *posatt_err, Eigen::Vector3d *rate_now,
                           Eigen::Vector3d *rate_err);

        void TimedCommandCallback(const ros::TimerEvent &e);

        void MultiDofJointTrajectoryCallback(
            const trajectory_msgs::MultiDOFJointTrajectoryConstPtr &trajectory_reference_msg);

        void CommandPoseCallback(
            const geometry_msgs::PoseStampedConstPtr &pose_msg);

        void OdometryCallback(const nav_msgs::OdometryConstPtr &odometry_msg);
    };
} // namespace neural_adaptive_controller
// } // namespace rotors_control

#endif // NEURAL_ADAPTIVE_CONTROLLER_H