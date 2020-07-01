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
            "/pelican1/command/motor_speed", 1);

        motor_velocity_reference_pub2_ = nh_.advertise<mav_msgs::Actuators>(
            "/pelican2/command/motor_speed", 1);

        command_timer_ = nh_.createTimer(ros::Duration(0), &NeuralAdaptiveController::TimedCommandCallback, this,
                                         true, false);
    }

    NeuralAdaptiveController::~NeuralAdaptiveController() {}

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

        torch::load(model_, "/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/model.pt");

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
        // ROS_INFO("GOOD");

        angular_acc_to_rotor_velocities_.resize(vehicle_parameters_.rotor_configuration_.rotors.size() * 2, 4);
        // Calculate the pseude-inverse A^{ \dagger} and then multiply by the inertia matrix I.
        // A^{ \dagger} = A^T*(A*A^T)^{-1}
        angular_acc_to_rotor_velocities_ = controller_parameters_.global_allocation_matrix_.transpose() * (controller_parameters_.global_allocation_matrix_ * controller_parameters_.global_allocation_matrix_.transpose()).inverse() * I;
        initialized_params_ = true;
        // PRINT_MAT(angular_acc_to_rotor_velocities_);
        allocate_rotor_velocities_ = Eigen::MatrixXd::Zero(6, 8);
        allocate_rotor_velocities_.topLeftCorner(4, 8) = controller_parameters_.global_allocation_matrix_;
    }

    void NeuralAdaptiveController::CommandPoseCallback(
        const geometry_msgs::PoseStampedConstPtr &pose_msg)
    {
        // Clear all pending commands.
        command_timer_.stop();
        commands_.clear();
        command_waiting_times_.clear();

        ROS_INFO_ONCE("LeePositionController got first command.");

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

        ROS_INFO_ONCE("LeePositionController got first odometry message.");

        EigenOdometry odometry;
        eigenOdometryFromMsg(odometry_msg, &odometry);
        SetOdometry(odometry);

        // Eigen::Vector3d x_error;
        // position_error = odometry_.position - command_trajectory_.position_W;

        Eigen::VectorXd posatt_now, posatt_err;
        ComputePosAtt(&posatt_now, &posatt_err);

        Eigen::VectorXd velrate_now, velrate_err;
        ComputeVelRate(&velrate_now, &velrate_err);

        Eigen::VectorXd nn_input;
        CalculateNNInput(posatt_now, velrate_now, posatt_err, velrate_err, &nn_input);

        Eigen::VectorXd ref_rotor_velocities;
        CalculateRotorVelocities(nn_input, &ref_rotor_velocities);

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

    void NeuralAdaptiveController::CalculateNNInput(const Eigen::VectorXd &x_d,
                                                    const Eigen::VectorXd &v_d,
                                                    const Eigen::VectorXd &x_e,
                                                    const Eigen::VectorXd &v_e,
                                                    Eigen::VectorXd *nn_input)
    {
        assert(nn_input);
        // assert(initialized_params_);

        nn_input->resize(6 * 4);

        Eigen::VectorXd a_d = Eigen::VectorXd::Zero(6);
        Eigen::VectorXd R = Eigen::MatrixXd::Zero(6, 6);

        Eigen::VectorXd v_r = v_d - R * x_e;
        Eigen::VectorXd a_r = a_d - R * v_e;

        Eigen::VectorXd v_in(x_d.cols() + v_d.cols() + v_r.cols() + a_r.cols());
        v_in << x_d, v_d, v_r, a_r;

        *nn_input = v_in;
    }

    void NeuralAdaptiveController::CalculateRotorVelocities(const Eigen::VectorXd &input, Eigen::VectorXd *rotor_velocities)
    {
        assert(input.size() == 24);

        std::vector<float> in_vec, out_vec;

        in_vec.clear();
        out_vec.clear();

        for (int i = 0; i < 24; i++)
        {
            in_vec.push_back(input(i));
        }

        auto data = torch::from_blob(in_vec.data(), {1, 1, static_cast<unsigned int>(in_vec.size())}).clone();
        auto output = model_->forward(data);

        Eigen::VectorXd rot_speed = Eigen::VectorXd::Zero(8);

        for (int i = 0; i < 8; i++)
        {
            rot_speed(i) = output[0][i].item<int>();
        }

        Eigen::VectorXd s = Eigen::VectorXd::Zero(6);
        Eigen::VectorXd v_d = input.block(6, 0, 6, 1);
        Eigen::VectorXd v_e = input.block(12, 0, 6, 1);

        s = v_d - v_e;

        Eigen::VectorXd Rs = allocate_rotor_velocities_.transpose() * s;

        Eigen::VectorXd Rs_sgn;

        double alpha = 10;
        for (unsigned int i = 0; i < Rs.size(); ++i)
            Rs_sgn(i) = tanh(alpha * Rs(i));

        *rotor_velocities = rot_speed - Rs - Rs_sgn;
        *rotor_velocities = rotor_velocities->cwiseMax(Eigen::VectorXd::Zero(rotor_velocities->rows()));
        *rotor_velocities = rotor_velocities->cwiseSqrt();
    }

    void NeuralAdaptiveController::ComputeVelRate(Eigen::VectorXd *velrate_now,
                                                  Eigen::VectorXd *velrate_err) const
    {
        assert(velrate_now);
        assert(velrate_err);

        // Transform velocity to world frame.
        const Eigen::Matrix3d R_W_I = odometry_.orientation.toRotationMatrix();
        Eigen::Vector3d velocity_W = R_W_I * odometry_.velocity;
        Eigen::Vector3d vel_err;
        vel_err = velocity_W - command_trajectory_.velocity_W;

        *velrate_now << velocity_W, odometry_.angular_velocity;

        Eigen::Vector3d rate = odometry_.angular_velocity;
        Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
        Eigen::Vector3d rate_err = command_trajectory_.angular_velocity_W - R.transpose() * rate;

        *velrate_err << vel_err, rate_err;
    }

    void NeuralAdaptiveController::ComputePosAtt(Eigen::VectorXd *posatt_now,
                                                 Eigen::VectorXd *posatt_err) const
    {
        assert(posatt_now);
        assert(posatt_err);

        Eigen::Vector3d pos_err;
        pos_err = odometry_.position - command_trajectory_.position_W;

        Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
        Eigen::Vector3d euler = R.eulerAngles(0, 1, 2);

        posatt_now->resize(6);
        *posatt_now << command_trajectory_.position_W, euler;

        // Get the desired rotation matrix.
        Eigen::Vector3d b1_des;
        double yaw = command_trajectory_.getYaw();
        b1_des << cos(yaw), sin(yaw), 0;

        Eigen::Vector3d e_3(Eigen::Vector3d::UnitZ());
        Eigen::Vector3d position_gain(0.1, 0.1, 0.1);
        Eigen::Vector3d acceleration = pos_err.cwiseProduct(position_gain) - 9.81 * e_3 - command_trajectory_.acceleration_W;

        Eigen::Vector3d b3_des;
        b3_des = -acceleration / acceleration.norm();

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

        *posatt_err << pos_err, angle_error;
    }

} // namespace neural_adaptive_controller
// } // namespace rotors_control
