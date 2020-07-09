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
          model_(Architecture(100, 100))
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

        plt::subplot(1, 1, 1);
        plt::plot(pitch_);
        plt::plot(rate_y_);
        plt::plot(pitch_err_);
        plt::plot(rate_err_);
        plt::plot(torque_s_);
        plt::plot(torque_y_);
        plt::named_plot("pitch", pitch_);
        plt::named_plot("rate", rate_y_);
        plt::named_plot("pitch_err", pitch_err_);
        plt::named_plot("rate_err", rate_err_);
        plt::named_plot("torque_s", torque_s_);
        plt::named_plot("torque", torque_y_);

        plt::title("pitch");
        plt::legend();
        plt::save("/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/pitch.png");
    }

    void NeuralAdaptiveController::InitializeParams()
    {
        constexpr double LEARNING_RATE{0.0005};
        int kNumberOfEpochs = 10;

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

        // torch::load(model_, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/model.pt");

        // Architecture model(100, 100);
        model_->to(device);

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

        // vehicle_parameters_.inertia_ << 0.8, 0.0, 0.0,
        //     0.0, 0.025, 0.0,
        //     0.0, 0.0, 0.6;

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

        PRINT_MAT(controller_parameters_.allocation_matrix_);
        controller_parameters_.global_allocation_matrix_.resize(4, 8);
        controller_parameters_.global_allocation_matrix_.topLeftCorner(4, 4) = A1 * controller_parameters_.allocation_matrix_;
        controller_parameters_.global_allocation_matrix_.topRightCorner(4, 4) = A2 * controller_parameters_.allocation_matrix_;
        // ROS_INFO("GOOD1");

        angular_acc_to_rotor_velocities_.resize(vehicle_parameters_.rotor_configuration_.rotors.size() * 2, 4);
        // Calculate the pseude-inverse A^{ \dagger} and then multiply by the inertia matrix I.
        // A^{ \dagger} = A^T*(A*A^T)^{-1}
        angular_acc_to_rotor_velocities_ = controller_parameters_.global_allocation_matrix_.transpose() * (controller_parameters_.global_allocation_matrix_ * controller_parameters_.global_allocation_matrix_.transpose()).inverse() * I;
        initialized_params_ = true;
        // PRINT_MAT(angular_acc_to_rotor_velocities_);
        // allocate_rotor_velocities_ = Eigen::MatrixXd::Zero(6, 8);
        // allocate_rotor_velocities_.topLeftCorner(4, 8) = controller_parameters_.global_allocation_matrix_;

        W_ = 0.0001 * Eigen::MatrixXd::Random(54, 3);
        pitch_.clear();
        pitch_err_.clear();
        rate_y_.clear();
        rate_err_.clear();
        torque_s_.clear();
        torque_y_.clear();
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
        // ROS_INFO("GOOD2");

        Eigen::Vector3d acceleration;

        Eigen::Vector3d att_now, att_err;
        Eigen::Vector3d rate_now, rate_err;

        ComputePosAtt(&acceleration, &att_now, &att_err, &rate_now, &rate_err);

        Eigen::VectorXd nn_input;
        CalculateNNInput(att_now, rate_now, att_err, rate_err, &nn_input);
        // ROS_INFO("GOOD5");
        Eigen::VectorXd ref_rotor_velocities;
        CalculateRotorVelocities(acceleration, nn_input, &ref_rotor_velocities);
        // ROS_INFO("GOOD6");
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

    void NeuralAdaptiveController::CalculateNNInput(const Eigen::Vector3d &x_n,
                                                    const Eigen::Vector3d &v_n,
                                                    const Eigen::Vector3d &x_e,
                                                    const Eigen::Vector3d &v_e,
                                                    Eigen::VectorXd *nn_input)
    {
        assert(nn_input);

        nn_input->resize(12);

        Eigen::Vector3d a_d = Eigen::Vector3d::Zero();
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, 3);
        // R << 7.5, 1, 0,
        //     1, 11, 0,
        //     0, 0.75, 1;
        R << 5.0, 0, 0,
            0, 5.0, 0,
            0, 0, 0;

        Eigen::Vector3d v_r = (v_n - v_e) + R * x_e;
        Eigen::Vector3d a_r = a_d + R * v_e;

        Eigen::VectorXd v_in;
        v_in.resize(12);
        v_in << x_n, v_r, v_n, a_r;

        *nn_input = v_in;
    }

    void NeuralAdaptiveController::CalculateRotorVelocities(const Eigen::Vector3d &acceleration, const Eigen::VectorXd &input, Eigen::VectorXd *rotor_velocities)
    {
        assert(input.size() == 12);

        rotor_velocities->resize(vehicle_parameters_.rotor_configuration_.rotors.size() * 2);
        // Return 0 velocities on all rotors, until the first command is received.
        // if (!controller_active_)
        // {
        //     *rotor_velocities = Eigen::VectorXd::Zero(rotor_velocities->rows());
        //     return;
        // }

        PRINT_MAT(input);

        Eigen::VectorXd z = Eigen::VectorXd::Zero(9);
        z << -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0;

        Eigen::VectorXd phi = Eigen::VectorXd::Zero(9);
        Eigen::VectorXd Phi = Eigen::VectorXd::Zero(z.size() * 6);

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < z.size(); j++)
            {
                phi(j) = std::exp(-(input(i) - z(j)) * (input(i) - z(j)) / 0.0001);
            }

            Phi.block(i * z.size(), 0, z.size(), 1) = phi;
        }

        // PRINT_MAT(Phi);

        Eigen::MatrixXd P = Eigen::MatrixXd::Identity(54, 54);

        Eigen::Vector3d output_torque;

        output_torque = W_.transpose() * Phi;

        PRINT_MAT(output_torque);

        // ROS_INFO("CalculateRotorVelocities3");

        Eigen::Vector3d s = Eigen::Vector3d::Zero();
        Eigen::Vector3d v_n = input.block(3, 0, 3, 1);
        Eigen::Vector3d v_r = input.block(6, 0, 3, 1);

        // PRINT_MAT(v_n);
        // PRINT_MAT(v_r);

        s = -v_n + v_r;

        Eigen::Vector3d Rs = s;
        PRINT_MAT(Rs);

        Eigen::Vector3d Rs_sgn;
        // ROS_INFO("CalculateRotorVelocities4");

        double alpha = 10;
        for (unsigned int i = 0; i < Rs.size(); ++i)
            Rs_sgn(i) = tanh(alpha * Rs(i));

        // PRINT_MAT(rot_speed);
        // PRINT_MAT(Rs);
        // PRINT_MAT(Rs_sgn);

        W_ += -0.0000001 * P * Phi * s.transpose();
        // PRINT_MAT(W_);

        double thrust = -3.0 * acceleration.dot(odometry_.orientation.toRotationMatrix().col(2));

        Eigen::Vector4d angular_thrust;
        // angular_thrust.block<3, 1>(0, 0) = - 0.01*Rs - 0.01*Rs_sgn;
        angular_thrust.block<3, 1>(0, 0) = output_torque - Rs - Rs_sgn;
        // angular_thrust.block<3, 1>(0, 0) = Eigen::Vector3d::Zero();
        angular_thrust(3) = thrust;

        torque_s_.push_back(Rs(1));
        torque_y_.push_back(angular_thrust(1));

        PRINT_MAT(angular_thrust);

        *rotor_velocities = angular_acc_to_rotor_velocities_ * angular_thrust;
        *rotor_velocities = rotor_velocities->cwiseMax(Eigen::VectorXd::Zero(rotor_velocities->rows()));
        *rotor_velocities = rotor_velocities->cwiseSqrt();
    }

    void NeuralAdaptiveController::ComputePosAtt(Eigen::Vector3d *acceleration,
                                                 Eigen::Vector3d *att_now,
                                                 Eigen::Vector3d *att_err,
                                                 Eigen::Vector3d *rate_now,
                                                 Eigen::Vector3d *rate_err)
    {
        assert(att_now);
        assert(att_err);

        Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();

        Eigen::Vector3d b1_des;
        double yaw = command_trajectory_.getYaw();
        b1_des << cos(yaw), sin(yaw), 0;

        Eigen::Vector3d position_error;
        position_error = odometry_.position - command_trajectory_.position_W;

        // Transform velocity to world frame.
        const Eigen::Matrix3d R_W_I = odometry_.orientation.toRotationMatrix();
        Eigen::Vector3d velocity_W = R_W_I * odometry_.velocity;
        Eigen::Vector3d velocity_error;
        velocity_error = velocity_W - command_trajectory_.velocity_W;

        Eigen::Vector3d e_3(Eigen::Vector3d::UnitZ());

        *acceleration = (position_error.cwiseProduct(controller_parameters_.position_gain_) + velocity_error.cwiseProduct(controller_parameters_.velocity_gain_)) / 3.0 - vehicle_parameters_.gravity_ * e_3 - command_trajectory_.acceleration_W;
        Eigen::Vector3d b3_des;
        b3_des = -*acceleration / acceleration->norm();

        Eigen::Vector3d b2_des;
        b2_des = b3_des.cross(b1_des);
        b2_des.normalize();

        Eigen::Matrix3d R_des;
        R_des.col(0) = b2_des.cross(b3_des);
        R_des.col(1) = b2_des;
        R_des.col(2) = b3_des;

        // Angle error according to lee et al.
        Eigen::Matrix3d angle_error_matrix = 0.5 * (R_des.transpose() * R - R.transpose() * R_des);
        Eigen::Vector3d angle_error;
        vectorFromSkewMatrix(angle_error_matrix, &angle_error);

        *att_err = -angle_error;

        // TODO(burrimi) include angular rate references at some point.
        Eigen::Vector3d angular_rate_des(Eigen::Vector3d::Zero());
        angular_rate_des[2] = command_trajectory_.getYawRate();

        Eigen::Vector3d angular_rate_error = odometry_.angular_velocity - R_des.transpose() * R * angular_rate_des;

        *rate_now = odometry_.angular_velocity;
        *rate_err = angular_rate_error;

        Eigen::Vector3d euler;
        mav_msgs::getEulerAnglesFromQuaternion(odometry_.orientation, &euler);

        *att_now = euler;
        // ROS_INFO("ComputePosAtt1");
        pitch_.push_back(euler(1));
        rate_y_.push_back(odometry_.angular_velocity(1));
        pitch_err_.push_back(angle_error(1));
        rate_err_.push_back(angular_rate_error(1));
    }

} // namespace neural_adaptive_controller
// } // namespace rotors_control
