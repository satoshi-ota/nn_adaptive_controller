#include <ros/ros.h>
#include <mav_msgs/default_topics.h>
#include "neural_adaptive_controller/parameters_ros.h"
#include "neural_adaptive_controller/neural_adaptive_controller.h"

// #include "rotors_control/parameters_ros.h"
#define PRINT_MAT(X) std::cout << #X << ":\n"    \
                               << X << std::endl \
                               << std::endl

// namespace rotors_control
// {
namespace neural_adaptive_controller
{

    NeuralAdaptiveController::NeuralAdaptiveController(
        const ros::NodeHandle &nh, const ros::NodeHandle &private_nh)
        : nh_(nh),
          private_nh_(private_nh),
          angular_velocity_pred_(Eigen::Vector3d::Zero()),
          last_LPF_(Eigen::Vector3d::Zero()),
          last_angle_error_(Eigen::Vector3d::Zero()),
          R_ref_(Eigen::Matrix3d::Identity()),
          last_angle_ref_(Eigen::Vector3d::Zero())
    {
        InitializeParams();

        cmd_pose_sub_ = nh_.subscribe(
            mav_msgs::default_topics::COMMAND_POSE, 1,
            &NeuralAdaptiveController::CommandPoseCallback, this);

        cmd_multi_dof_joint_trajectory_sub_ = nh_.subscribe(
            "/pelican/command/trajectory", 1,
            &NeuralAdaptiveController::MultiDofJointTrajectoryCallback, this);

        odometry_sub_ = nh_.subscribe("pelican/payload/odom", 1,
                                      &NeuralAdaptiveController::OdometryCallback, this);

        motor_velocity_reference_pub1_ = nh_.advertise<mav_msgs::Actuators>(
            "/pelican1/neural_adaptive_controller/command/motor_speed", 1);

        motor_velocity_reference_pub2_ = nh_.advertise<mav_msgs::Actuators>(
            "/pelican2/neural_adaptive_controller/command/motor_speed", 1);

        command_timer_ = nh_.createTimer(ros::Duration(0), &NeuralAdaptiveController::TimedCommandCallback, this,
                                         true, false);
    }

    NeuralAdaptiveController::~NeuralAdaptiveController()
    {
        plt::figure_size(1200, 780);

        plt::subplot(3, 1, 1);
        plt::named_plot("roll_command", roll_command_);
        plt::named_plot("roll", roll_);
        plt::named_plot("roll_reference", roll_ref_);
        // plt::named_plot("torque_roll", torque_roll_);
        // plt::named_plot("angular_acceleration_roll", angular_acceleration_roll_);
        plt::named_plot("sigma_roll", sigma_roll_);
        plt::named_plot("error_roll", error_roll_);
        plt::title("roll");
        plt::legend();

        plt::subplot(3, 1, 2);
        plt::named_plot("pitch", pitch_);
        plt::named_plot("pitch_reference", pitch_ref_);
        plt::title("pitch");
        plt::legend();

        plt::subplot(3, 1, 3);
        plt::named_plot("pitch_command", pitch_command_);
        plt::named_plot("torque_pitch", torque_pitch_);
        plt::named_plot("sigma_pitch", sigma_pitch_);
        plt::title("pitch");
        plt::legend();

        plt::save("/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/pitch.png");
    }

    void NeuralAdaptiveController::InitializeParams()
    {
        // Read parameters from rosparam.
        GetRosParameter(private_nh_, "position_gain/x",
                        controller_parameters_.position_gain_.x(),
                        &controller_parameters_.position_gain_.x());
        GetRosParameter(private_nh_, "position_gain/y",
                        controller_parameters_.position_gain_.y(),
                        &controller_parameters_.position_gain_.y());
        GetRosParameter(private_nh_, "position_gain/z",
                        controller_parameters_.position_gain_.z(),
                        &controller_parameters_.position_gain_.z());
        GetRosParameter(private_nh_, "velocity_gain/x",
                        controller_parameters_.velocity_gain_.x(),
                        &controller_parameters_.velocity_gain_.x());
        GetRosParameter(private_nh_, "velocity_gain/y",
                        controller_parameters_.velocity_gain_.y(),
                        &controller_parameters_.velocity_gain_.y());
        GetRosParameter(private_nh_, "velocity_gain/z",
                        controller_parameters_.velocity_gain_.z(),
                        &controller_parameters_.velocity_gain_.z());
        GetRosParameter(private_nh_, "attitude_gain/x",
                        controller_parameters_.attitude_gain_.x(),
                        &controller_parameters_.attitude_gain_.x());
        GetRosParameter(private_nh_, "attitude_gain/y",
                        controller_parameters_.attitude_gain_.y(),
                        &controller_parameters_.attitude_gain_.y());
        GetRosParameter(private_nh_, "attitude_gain/z",
                        controller_parameters_.attitude_gain_.z(),
                        &controller_parameters_.attitude_gain_.z());
        GetRosParameter(private_nh_, "angular_rate_gain/x",
                        controller_parameters_.angular_rate_gain_.x(),
                        &controller_parameters_.angular_rate_gain_.x());
        GetRosParameter(private_nh_, "angular_rate_gain/y",
                        controller_parameters_.angular_rate_gain_.y(),
                        &controller_parameters_.angular_rate_gain_.y());
        GetRosParameter(private_nh_, "angular_rate_gain/z",
                        controller_parameters_.angular_rate_gain_.z(),
                        &controller_parameters_.angular_rate_gain_.z());
        GetVehicleParameters(private_nh_, &vehicle_parameters_);
        // InitializeParameters();

        calculateAllocationMatrix(vehicle_parameters_.rotor_configuration_, &(controller_parameters_.allocation_matrix_));
        // To make the tuning independent of the inertia matrix we divide here.
        normalized_attitude_gain_ = controller_parameters_.attitude_gain_.transpose() * vehicle_parameters_.inertia_.inverse();
        // To make the tuning independent of the inertia matrix we divide here.
        normalized_angular_rate_gain_ = controller_parameters_.angular_rate_gain_.transpose() * vehicle_parameters_.inertia_.inverse();

        Eigen::Matrix4d I;
        I.setZero();
        I.block<3, 3>(0, 0) = vehicle_parameters_.inertia_;
        I(3, 3) = 1;

        Eigen::Matrix4d A1, A2;
        A1 << 1.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

        A2 << 1.0, 0.0, 0.0, 0.0,
            -0.5, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

        controller_parameters_.global_allocation_matrix_.resize(4, 8);
        controller_parameters_.global_allocation_matrix_.topLeftCorner(4, 4) = A1 * controller_parameters_.allocation_matrix_;
        controller_parameters_.global_allocation_matrix_.topRightCorner(4, 4) = A2 * controller_parameters_.allocation_matrix_;

        angular_acc_to_rotor_velocities_.resize(vehicle_parameters_.rotor_configuration_.rotors.size() * 2, 4);
        // Calculate the pseude-inverse A^{ \dagger} and then multiply by the inertia matrix I.
        // A^{ \dagger} = A^T*(A*A^T)^{-1}
        angular_acc_to_rotor_velocities_ = controller_parameters_.global_allocation_matrix_.transpose() * (controller_parameters_.global_allocation_matrix_ * controller_parameters_.global_allocation_matrix_.transpose()).inverse() * I;
        initialized_params_ = true;

        roll_command_.clear();
        roll_.clear();
        roll_ref_.clear();
        torque_roll_.clear();
        pitch_command_.clear();
        pitch_.clear();
        pitch_ref_.clear();
        torque_pitch_.clear();
        angular_acceleration_roll_.clear();
        angular_acceleration_pitch_.clear();
        sigma_roll_.clear();
        sigma_pitch_.clear();
        error_roll_.clear();
        error_pitch_.clear();
    }

    void NeuralAdaptiveController::CommandPoseCallback(
        const geometry_msgs::PoseStampedConstPtr &pose_msg)
    {
        // Clear all pending commands.
        command_timer_.stop();
        commands_.clear();
        command_waiting_times_.clear();

        ROS_INFO_ONCE("NeuralAdaptiveController got first command.");

        mav_msgs::EigenTrajectoryPoint eigen_reference;
        mav_msgs::eigenTrajectoryPointFromPoseMsg(*pose_msg, &eigen_reference);
        commands_.push_front(eigen_reference);

        SetTrajectoryPoint(commands_.front());
        commands_.pop_front();
    }

    void NeuralAdaptiveController::MultiDofJointTrajectoryCallback(
        const trajectory_msgs::MultiDOFJointTrajectoryConstPtr &msg)
    {
        // Clear all pending commands.
        command_timer_.stop();
        commands_.clear();
        command_waiting_times_.clear();

        const size_t n_commands = msg->points.size();

        if (n_commands < 1)
        {
            ROS_WARN_STREAM("Got MultiDOFJointTrajectory message, but message has no points.");
            return;
        }

        mav_msgs::EigenTrajectoryPoint eigen_reference;
        mav_msgs::eigenTrajectoryPointFromMsg(msg->points.front(), &eigen_reference);
        commands_.push_front(eigen_reference);

        for (size_t i = 1; i < n_commands; ++i)
        {
            const trajectory_msgs::MultiDOFJointTrajectoryPoint &reference_before = msg->points[i - 1];
            const trajectory_msgs::MultiDOFJointTrajectoryPoint &current_reference = msg->points[i];

            mav_msgs::eigenTrajectoryPointFromMsg(current_reference, &eigen_reference);

            commands_.push_back(eigen_reference);
            command_waiting_times_.push_back(current_reference.time_from_start - reference_before.time_from_start);
        }

        // We can trigger the first command immediately.
        SetTrajectoryPoint(commands_.front());
        commands_.pop_front();

        if (n_commands > 1)
        {
            command_timer_.setPeriod(command_waiting_times_.front());
            command_waiting_times_.pop_front();
            command_timer_.start();
        }
    }

    void NeuralAdaptiveController::TimedCommandCallback(const ros::TimerEvent &e)
    {
        if (commands_.empty())
        {
            ROS_WARN("Commands empty, this should not happen here");
            return;
        }

        // const mav_msgs::EigenTrajectoryPoint eigen_reference = commands_.front();
        SetTrajectoryPoint(commands_.front());
        commands_.pop_front();
        command_timer_.stop();
        if (!command_waiting_times_.empty())
        {
            command_timer_.setPeriod(command_waiting_times_.front());
            command_waiting_times_.pop_front();
            command_timer_.start();
        }
    }

    void NeuralAdaptiveController::OdometryCallback(const nav_msgs::OdometryConstPtr &odometry_msg)
    {

        ROS_INFO_ONCE("NeuralAdaptiveController got first odometry message.");

        EigenOdometry odometry;
        eigenOdometryFromMsg(odometry_msg, &odometry);
        SetOdometry(odometry);

        Eigen::VectorXd ref_rotor_velocities;
        CalculateRotorVelocities(&ref_rotor_velocities);
        // Todo(ffurrer): Do this in the conversions header.
        mav_msgs::ActuatorsPtr actuator_msg(new mav_msgs::Actuators);

        actuator_msg->angular_velocities.clear();
        for (int i = 0; i < 4; i++)
            actuator_msg->angular_velocities.push_back(ref_rotor_velocities[i]);
        actuator_msg->header.stamp = odometry_msg->header.stamp;

        motor_velocity_reference_pub1_.publish(actuator_msg);

        actuator_msg->angular_velocities.clear();
        for (int i = 4; i < 8; i++)
            actuator_msg->angular_velocities.push_back(ref_rotor_velocities[i]);
        actuator_msg->header.stamp = odometry_msg->header.stamp;

        motor_velocity_reference_pub2_.publish(actuator_msg);
    }

    void NeuralAdaptiveController::SetOdometry(const EigenOdometry &odometry)
    {
        odometry_ = odometry;
    }

    void NeuralAdaptiveController::SetTrajectoryPoint(
        const mav_msgs::EigenTrajectoryPoint &command_trajectory)
    {
        command_trajectory_ = command_trajectory;
        controller_active_ = true;
    }

    void NeuralAdaptiveController::CalculateRotorVelocities(Eigen::VectorXd *rotor_velocities)
    {
        rotor_velocities->resize(vehicle_parameters_.rotor_configuration_.rotors.size() * 2);
        // Return 0 velocities on all rotors, until the first command is received.
        if (!controller_active_)
        {
            *rotor_velocities = Eigen::VectorXd::Zero(rotor_velocities->rows());
            return;
        }

        Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
        Eigen::Matrix3d angle_error_matrix = 0.5 * (R_ref_.transpose() * R - R.transpose() * R_ref_);
        Eigen::Vector3d angle_error;
        vectorFromSkewMatrix(angle_error_matrix, &angle_error);

        Eigen::Vector3d sigma;
        adaptation(-1.0 * angle_error, &sigma);
        sigma = lowPassFilter(sigma);

        Eigen::Vector3d acceleration;
        ComputeDesiredAcceleration(&acceleration);

        Eigen::Matrix3d R_des;
        Eigen::Vector3d angular_acceleration;
        ComputeDesiredAngularAcc(acceleration, &R_des, &angular_acceleration);

        double u_thrust = -2.5 * acceleration.dot(odometry_.orientation.toRotationMatrix().col(2));

        Eigen::Vector4d angular_thrust;
        angular_thrust.block<3, 1>(0, 0) = angular_acceleration - sigma;
        angular_thrust(3) = u_thrust;

        *rotor_velocities = angular_acc_to_rotor_velocities_ * angular_thrust;
        *rotor_velocities = rotor_velocities->cwiseMax(Eigen::VectorXd::Zero(rotor_velocities->rows()));
        *rotor_velocities = rotor_velocities->cwiseSqrt();

        torque_roll_.push_back(angular_thrust(0));
        torque_pitch_.push_back(angular_thrust(1));

        angular_acceleration_roll_.push_back(angular_acceleration(0));
        angular_acceleration_pitch_.push_back(angular_acceleration(1));

        sigma_roll_.push_back(sigma(0));
        sigma_pitch_.push_back(sigma(1));

        error_roll_.push_back(angle_error(0));
        error_pitch_.push_back(angle_error(1));

        roll_command_.push_back(angular_acceleration(0));
        pitch_command_.push_back(angular_acceleration(1));

        predReferenceOutput(angular_acceleration + sigma, R_des, &R_ref_);
    }

    void NeuralAdaptiveController::ComputeDesiredAcceleration(Eigen::Vector3d *acceleration) const
    {
        assert(acceleration);

        Eigen::Vector3d position_error;
        position_error = odometry_.position - command_trajectory_.position_W;

        // Transform velocity to world frame.
        const Eigen::Matrix3d R_W_I = odometry_.orientation.toRotationMatrix();
        Eigen::Vector3d velocity_W = R_W_I * odometry_.velocity;
        Eigen::Vector3d velocity_error;
        velocity_error = velocity_W - command_trajectory_.velocity_W;

        Eigen::Vector3d e_3(Eigen::Vector3d::UnitZ());

        *acceleration = (position_error.cwiseProduct(controller_parameters_.position_gain_) + velocity_error.cwiseProduct(controller_parameters_.velocity_gain_)) / vehicle_parameters_.mass_ - vehicle_parameters_.gravity_ * e_3 - command_trajectory_.acceleration_W;
    }

    void NeuralAdaptiveController::ComputeDesiredAngularAcc(const Eigen::Vector3d &acceleration,
                                                            Eigen::Matrix3d *R_des,
                                                            Eigen::Vector3d *angular_acceleration)
    {
        assert(R_des);

        Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();

        // Get the desired rotation matrix.
        Eigen::Vector3d b1_des;
        double yaw = command_trajectory_.getYaw();
        b1_des << cos(yaw), sin(yaw), 0;

        Eigen::Vector3d b3_des;
        b3_des = -acceleration / acceleration.norm();

        Eigen::Vector3d b2_des;
        b2_des = b3_des.cross(b1_des);
        b2_des.normalize();

        R_des->col(0) = b2_des.cross(b3_des);
        R_des->col(1) = b2_des;
        R_des->col(2) = b3_des;

        Eigen::Matrix3d angle_error_matrix = 0.5 * (R_des->transpose() * R - R.transpose() * (*R_des));
        Eigen::Vector3d angle_error;
        vectorFromSkewMatrix(angle_error_matrix, &angle_error);

        Eigen::Vector3d angular_rate_des(Eigen::Vector3d::Zero());
        angular_rate_des[2] = command_trajectory_.getYawRate();

        Eigen::Vector3d angular_rate_error = odometry_.angular_velocity - R_des->transpose() * R * angular_rate_des;

        *angular_acceleration = -1 * angle_error.cwiseProduct(normalized_attitude_gain_) - angular_rate_error.cwiseProduct(normalized_angular_rate_gain_) + odometry_.angular_velocity.cross(odometry_.angular_velocity);

        Eigen::Vector3d angle_des;
        mav_msgs::vectorFromRotationMatrix(*R_des, &angle_des);

        // roll_command_.push_back(angle_des(0));
        // pitch_command_.push_back(angle_des(1));
    }

    void NeuralAdaptiveController::adaptation(const Eigen::Vector3d &angle_error, Eigen::Vector3d *sigma)
    {
        double dt = 0.1;
        double gamma = 50.0;

        last_angle_error_ += angle_error * dt;
        last_angle_error_ = last_angle_error_.cwiseMax(-1.0 * kDefaultSigmaAbs);
        last_angle_error_ = last_angle_error_.cwiseMin(kDefaultSigmaAbs);

        *sigma = -gamma * last_angle_error_;
    }

    Eigen::Vector3d NeuralAdaptiveController::lowPassFilter(const Eigen::Vector3d &raw)
    {
        double k = 0.1;
        Eigen::Vector3d LPF = (1 - k) * last_LPF_ + k * raw;
        last_LPF_ = LPF;

        return LPF;
    }

    void NeuralAdaptiveController::predReferenceOutput(const Eigen::Vector3d &input, const Eigen::Matrix3d &R_des, Eigen::Matrix3d *R_ref)
    {
        assert(R_ref);

        double dt = 0.1;

        Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();

        Eigen::Vector3d angle_des;
        mav_msgs::vectorFromRotationMatrix(R_des, &angle_des);

        roll_.push_back(angle_des(0));
        pitch_.push_back(angle_des(1));

        Eigen::Matrix3d angle_error_matrix = 0.5 * (R_des.transpose() * R - R.transpose() * R_des);
        Eigen::Vector3d angle_error;
        vectorFromSkewMatrix(angle_error_matrix, &angle_error);

        Eigen::Matrix3d A_ref;
        A_ref << -5.0, 0.0, 0.0,
            0.0, -5.0, 0.0,
            0.0, 0.0, -5.0;

        Eigen::Vector3d angle_ref;
        angle_ref = angle_des + 1.0 * (Eigen::Matrix3d::Identity() + A_ref * dt) * angle_error;

        mav_msgs::matrixFromRotationVector(angle_ref, R_ref);

        roll_ref_.push_back(angle_ref.x());
        pitch_ref_.push_back(angle_ref.y());
    }

} // namespace neural_adaptive_controller
// } // namespace rotors_control
