
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "kalman_filter_demo/kalman_filter.hpp"

using namespace std::chrono_literals;//便于使用 100ms、1s 等时间单位

class KalmanNode : public rclcpp::Node
{
public:
  //维度
  static constexpr int NX = 2;
  static constexpr int NZ = 1;
  static constexpr int NU = 0;
  using KF = kf::KalmanFilter<NX, NZ, NU>;

  KalmanNode()
  : Node("kalman_filter")
  {
    
    // 把这些量做成参数
    sigma_z_ = declare_parameter<double>("sigma_z", 1.0);   // 观测噪声位置标准差，只需定义·一次
    sigma_a_ = declare_parameter<double>("sigma_a", 0.5);   // 过程噪声加速度标准差
    init_p_  = declare_parameter<double>("init_position", 0.0);
    init_v_  = declare_parameter<double>("init_velocity", 0.0);
    init_P_  = declare_parameter<double>("init_covariance", 10.0);//初始方差，越大表示对初始状态越不确定
    const std::string frame = declare_parameter<std::string>("frame_id", "map");

    //  2) 初始化滤波器
    KF::ObsMat H;                       // 只观测位置
    H << 1.0, 0.0;

    KF::MeasMat R;                      // 观测噪声协方差
    R << sigma_z_ * sigma_z_;

    KF::StateVec x0;
    x0 << init_p_, init_v_;

    KF::StateMat P0 = KF::StateMat::Identity() * init_P_;

    kf_.setH(H).setR(R).setX0(x0).setP0(P0).setJosephForm(true);

    sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "measurement", rclcpp::SensorDataQoS(),
      std::bind(&KalmanNode::onMeasurement, this, std::placeholders::_1));

    pub_est_ = create_publisher<geometry_msgs::msg::PointStamped>("estimate", 10);
    pub_diag_ = create_publisher<std_msgs::msg::Float64MultiArray>("innovation", 10);

    RCLCPP_INFO(get_logger(),
      "卡尔曼滤波节点已启动 | sigma_z=%.3f sigma_a=%.3f dt=自适应 | 订阅: measurement",
      sigma_z_, sigma_a_);
  }

private:
  
  void rebuildModel(double dt)
  {
    KF::StateMat F;//运动模型
    F << 1.0, dt,
         0.0, 1.0;
    
    const double sa2 = sigma_a_ * sigma_a_;
    KF::StateMat Q;
    Q << sa2 * std::pow(dt, 4) / 4.0, sa2 * std::pow(dt, 3) / 2.0,
         sa2 * std::pow(dt, 3) / 2.0, sa2 * std::pow(dt, 2);
        //运动噪声协方差矩阵 Q 是根据连续时间白噪声加速度模型离散化得到的，反映了在时间间隔 dt 内由于加速度噪声引起的状态不确定性，要时刻更新 Q 以适应不同的 dt。
    kf_.setF(F).setQ(Q);
  }

  void onMeasurement(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    // ---- 计算真实时间间隔 dt ----
    double dt = 0.0;
    if (last_stamp_.nanoseconds() > 0) {
      dt = (rclcpp::Time(msg->header.stamp) - last_stamp_).seconds();

      // 时间戳异常保护：倒退、重复、跨度过大都不适合直接用于滤波
      if (dt <= 0.0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "时间戳未推进（dt=%.6f），跳过本帧以避免重复更新", dt);
        return;
      }
      if (dt > 1.0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "时间间隔过大（dt=%.3f s），滤波器可能需要重新初始化", dt);
      }
    } else {
      // 第一帧：只初始化时间戳，用很小的 dt 让 predict 近似恒等
      dt = 1e-3;
      RCLCPP_INFO(get_logger(), "收到首帧观测，初始化时间基准");
    }
    last_stamp_ = rclcpp::Time(msg->header.stamp);

    rebuildModel(dt);
    kf_.predict();

    //更新观测 
    KF::MeasVec z;
    z << msg->point.x;
    kf_.update(z);

    const auto & x = kf_.x();

    geometry_msgs::msg::PointStamped out;
    out.header = msg->header;      // 沿用输入时间戳，方便下游做时间对齐
    out.point.x = x(0);            // 位置估计
    out.point.y = 0.0;
    out.point.z = 0.0;
    pub_est_->publish(out);
    //      调参时最有用的是「新息」：它应该像零均值白噪声。
    std_msgs::msg::Float64MultiArray diag;
    diag.data = {
      kf_.innovation(z)(0),      // 新息 y = z - H*x
      kf_.P()(0, 0),           
      kf_.P()(1, 1),             
      x(1)                       
    };
    pub_diag_->publish(diag);

    RCLCPP_DEBUG(get_logger(), "dt=%.4f  z=%.3f  p=%.3f  v=%.3f", dt, z(0), x(0), x(1));
  }

  double sigma_z_{1.0};
  double sigma_a_{0.5};
  double init_p_{0.0};
  double init_v_{0.0};
  double init_P_{10.0};

  KF kf_;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};

  // ---- 通信接口 ----
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