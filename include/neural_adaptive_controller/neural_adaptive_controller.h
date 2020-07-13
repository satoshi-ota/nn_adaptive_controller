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

#include "neural_adaptive_controller/common.h"

#include <controller_network/matplotlibcpp.h>

namespace plt = matplotlibcpp;

namespace neural_adaptive_controller
{
    // Default values for the lee position controller and the Asctec Firefly.
    static const Eigen::Vector3d kDefaultPositionGain = Eigen::Vector3d(6, 6, 6);
    static const Eigen::Vector3d kDefaultVelocityGain = Eigen::Vector3d(4.7, 4.7, 4.7);
    static const Eigen::Vector3d kDefaultAttitudeGain = Eigen::Vector3d(3, 3, 0.035);
    static const Eigen::Vector3d kDefaultAngularRateGain = Eigen::Vector3d(0.52, 0.52, 0.025);

    static const Eigen::Vector3d kDefaultSigmaAbs = Eigen::Vector3d(0.10, 0.30, 0.0);

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
        void CalculateRotorVelocities(Eigen::VectorXd *rotor_velocities);

        void SetOdometry(const EigenOdometry &odometry);
        void SetTrajectoryPoint(const mav_msgs::EigenTrajectoryPoint &command_trajectory);

        NeuralAdaptiveControllerParameters controller_parameters_;
        VehicleParameters vehicle_parameters_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        ros::NodeHandle nh_;
        ros::NodeHandle private_nh_;

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

        Eigen::Vector3d angular_velocity_pred_;
        Eigen::Vector3d last_LPF_;
        Eigen::Vector3d last_angle_error_;
        Eigen::Matrix3d R_ref_;

        mav_msgs::EigenTrajectoryPoint command_trajectory_;
        EigenOdometry odometry_;

        std::vector<float> pitch_command_, roll_command_;
        std::vector<float> pitch_, roll_;
        std::vector<float> pitch_ref_, roll_ref_;
        std::vector<float> torque_x_, torque_y_;
        std::vector<float> angular_acceleration_roll_, angular_acceleration_pitch_;
        std::vector<float> sigma_roll_, sigma_pitch_;
        std::vector<float> error_roll_, error_pitch_;

        void adaptation(const Eigen::Vector3d &angle_error, Eigen::Vector3d *sigma);

        Eigen::Vector3d lowPassFilter(const Eigen::Vector3d &raw);

        void predReferenceOutput(const Eigen::Vector3d &input, const Eigen::Matrix3d &R_des, Eigen::Matrix3d *R_ref);

        void TimedCommandCallback(const ros::TimerEvent &e);

        void MultiDofJointTrajectoryCallback(
            const trajectory_msgs::MultiDOFJointTrajectoryConstPtr &trajectory_reference_msg);

        void CommandPoseCallback(
            const geometry_msgs::PoseStampedConstPtr &pose_msg);

        void OdometryCallback(const nav_msgs::OdometryConstPtr &odometry_msg);

        void ComputeDesiredAngularAcc(const Eigen::Vector3d &acceleration,
                                      Eigen::Matrix3d *R_des,
                                      Eigen::Vector3d *angular_acceleration);
        void ComputeDesiredAcceleration(Eigen::Vector3d *acceleration) const;
    };
} // namespace neural_adaptive_controller
// } // namespace rotors_control

#endif // NEURAL_ADAPTIVE_CONTROLLER_H