// mpc_simple.cpp
// 编译：g++ -std=c++17 -O2 mpc_simple.cpp -o mpc -I /path/to/eigen
// 运行：./mpc
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;

constexpr int NX = 2;      // 状态：[p, v]
constexpr int NU = 1;      // 控制：[a]
constexpr int N  = 10;     // 预测时域
constexpr double DT = 0.1; // 采样周期

class MPC {
public:
    MPC() {
        // 1. 状态转移：p' = p + v*dt, v' = v + a*dt
        A_ << 1.0, DT,
              0.0, 1.0;
        B_ << 0.5 * DT * DT,
              DT;

        // 2. 代价权重
        Q_ << 10.0, 0.0,
              0.0,  1.0;   // 位置比速度重要
        R_ << 0.1;         // 控制代价
        Qf_ = Q_ * 10.0;   // 终端代价更大

        // 3. 构造预测矩阵
        buildPrediction();
    }

    // 求解：给定当前状态和目标，返回最优控制序列
    VectorXd solve(const VectorXd& x0, const VectorXd& x_ref) {
        // 参考轨迹：每步都跟踪同一目标
        VectorXd Xref(N * NX);
        for (int k = 0; k < N; ++k) Xref.segment<NX>(k * NX) = x_ref;

        // 无约束 QP 闭式解：U = -H^{-1} g
        MatrixXd H = B_bar_.transpose() * Q_bar_ * B_bar_ + R_bar_;
        VectorXd g = B_bar_.transpose() * Q_bar_ * (A_bar_ * x0 - Xref);
        return -H.ldlt().solve(g);
    }

    // 仿真：用模型推进一步
    VectorXd step(const VectorXd& x, const VectorXd& u) {
        return A_ * x + B_ * u;
    }

private:
    void buildPrediction() {
        A_bar_.setZero(N * NX, NX);
        B_bar_.setZero(N * NX, N * NU);

        MatrixXd Ap = A_;
        for (int k = 0; k < N; ++k) {
            A_bar_.block<NX, NX>(k * NX, 0) = Ap;

            MatrixXd Aj = MatrixXd::Identity(NX, NX);
            for (int j = k; j >= 0; --j) {
                B_bar_.block<NX, NU>(k * NX, j * NU) = Aj * B_;
                Aj = Aj * A_;
            }
            Ap = Ap * A_;
        }

        Q_bar_.setZero(N * NX, N * NX);
        for (int k = 0; k < N - 1; ++k)
            Q_bar_.block<NX, NX>(k * NX, k * NX) = Q_;
        Q_bar_.block<NX, NX>((N - 1) * NX, (N - 1) * NX) = Qf_;

        R_bar_.setZero(N * NU, N * NU);
        for (int k = 0; k < N; ++k)
            R_bar_.block<NU, NU>(k * NU, k * NU) = R_;
    }

    MatrixXd A_, B_, Q_, R_, Qf_;
    MatrixXd A_bar_, B_bar_, Q_bar_, R_bar_;
};

int main() {
    MPC mpc;

    VectorXd x(NX), x_ref(NX);
    x     << 0.0, 0.0;
    x_ref << 5.0, 0.0;

    for (int k = 0; k < 50; ++k) {
        VectorXd U  = mpc.solve(x, x_ref);
        VectorXd u0 = U.head(NU);           // 只执行第一步

        std::cout << "step " << k
                  << "  p=" << x(0)
                  << "  v=" << x(1)
                  << "  a=" << u0(0) << "\n";

        x = mpc.step(x, u0);
    }
}