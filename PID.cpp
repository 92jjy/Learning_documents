
#include <iostream>

class PID {
public:
    // 构造函数：三个增益 + 输出限幅
    PID(double kp, double ki, double kd, double u_max)
        : kp_(kp), ki_(ki), kd_(kd), u_max_(u_max) {}

    // 每步调用：传入目标值 target、当前值 current、时间间隔 dt
    double update(double target, double current, double dt) {
        double error = target - current;          // 当前误差

        integral_ += error * dt;                  // 积分累积
        double derivative = (error - prev_error_) / dt;  // 误差变化率
        prev_error_ = error;

        double u = kp_ * error                       // P 项
                 + ki_ * integral_                   // I 项
                 + kd_ * derivative;                 // D 项

        // 输出限幅
        if (u >  u_max_) u =  u_max_;
        if (u < -u_max_) u = -u_max_;

        return u;
    }

    void reset() { integral_ = 0.0; prev_error_ = 0.0; }

private:
    double kp_, ki_, kd_;      // 三个增益
    double u_max_;             // 输出限幅
    double integral_{0.0};     // 积分累积
    double prev_error_{0.0};   // 上一步误差
};

int main() {
    PID pid(2.0, 0.5, 0.1, 10.0);   // kp=2, ki=0.5, kd=0.1, 限幅 10

    double current = 0.0;           // 当前状态
    double target  = 5.0;           // 目标
    double dt      = 0.01;          // 采样周期 10ms

    for (int k = 0; k < 500; ++k) {
        double u = pid.update(target, current, dt);
        current += u * dt;          // 一阶系统：dx/dt = u

        if (k % 50 == 0) {
            std::cout << "t=" << k * dt
                      << "  current=" << current
                      << "  u=" << u << "\n";
        }
    }
}