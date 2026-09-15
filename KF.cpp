// kalman_node_simple.cpp
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "kalman_filter_demo/kalman_filter.hpp"

using namespace std::chrono_literals;

class KalmanNode : public rclcpp::Node
{
public:
  static constexpr int NX = 2;
  static constexpr int NZ = 1;
  static constexpr int NU = 0;
  using KF = kf::KalmanFilter<NX, NZ, NU>;

  KalmanNode() : Node("kalman_filter")
  {
    sigma_z_ = declare_parameter<double>("sigma_z", 1.0);
    sigma_a_ = declare_parameter<double>("sigma_a", 0.5);
    init_p_  = declare_parameter<double>("init_position", 0.0);
    init_v_  = declare_parameter<double>("init_velocity", 0.0);
    init_P_  = declare_parameter<double>("init_covariance", 10.0);

    KF::ObsMat H;
    H << 1.0, 0.0;
    KF::MeasMat R;
    R << sigma_z_ * sigma_z_;
    KF::StateVec x0;
    x0 << init_p_, init_v_;
    KF::StateMat P0 = KF::StateMat::Identity() * init_P_;
    kf_.setH(H).setR(R).setX0(x0).setP0(P0).setJosephForm(true);

    sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "measurement", rclcpp::SensorDataQoS(),
      std::bind(&KalmanNode::onMeasurement, this, std::placeholders::_1));
    pub_est_  = create_publisher<geometry_msgs::msg::PointStamped>("estimate", 10);
    pub_diag_ = create_publisher<std_msgs::msg::Float64MultiArray>("innovation", 10);

    RCLCPP_INFO(get_logger(), "Kalman filter node started");
  }

private:
  void rebuildModel(double dt)
  {
    KF::StateMat F;
    F << 1.0, dt,
         0.0, 1.0;

    const double sa2 = sigma_a_ * sigma_a_;
    KF::StateMat Q;
    Q << sa2 * std::pow(dt, 4) / 4.0, sa2 * std::pow(dt, 3) / 2.0,
         sa2 * std::pow(dt, 3) / 2.0, sa2 * std::pow(dt, 2);

    kf_.setF(F).setQ(Q);
  }

  void onMeasurement(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    double dt = 0.0;
    if (last_stamp_.nanoseconds() > 0) {
      dt = (rclcpp::Time(msg->header.stamp) - last_stamp_).seconds();
      if (dt <= 0.0) return;
    } else {
      dt = 1e-3;
    }
    last_stamp_ = rclcpp::Time(msg->header.stamp);

    rebuildModel(dt);
    kf_.predict();

    KF::MeasVec z;
    z << msg->point.x;
    kf_.update(z);

    geometry_msgs::msg::PointStamped out;
    out.header = msg->header;
    out.point.x = kf_.x()(0);
    pub_est_->publish(out);

    std_msgs::msg::Float64MultiArray diag;
    diag.data = {kf_.innovation(z)(0), kf_.P()(0, 0), kf_.P()(1, 1), kf_.x()(1)};
    pub_diag_->publish(diag);
  }

  double sigma_z_{1.0};
  double sigma_a_{0.5};
  double init_p_{0.0};
  double init_v_{0.0};
  double init_P_{10.0};

  KF kf_;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_est_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_diag_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanNode>());
  rclcpp::shutdown();
  return 0;
}