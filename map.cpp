// occ_grid.cpp
// 编译：g++ -std=c++17 -O2 occ_grid.cpp -o occ_grid
// 运行：./occ_grid
#include <bits/stdc++.h>
using namespace std;

// ===== 参数 =====
const int    W = 30;              // 栅格宽（列数）
const int    H = 15;              // 栅格高（行数）
const double P_OCC   = 0.7;       // 传感器命中时，该格为占据的概率
const double P_FREE  = 0.3;       // 传感器穿过时，该格为占据的概率
const double P_PRIOR = 0.5;       // 先验：未知，占据=空闲=0.5
const double L_MIN   = -3.0;      // 对数几率下限（防止过度自信）
const double L_MAX   =  3.0;      // 对数几率上限

// ===== 全局数据 =====
double L_occ, L_free, L_prior;            // 三种情形的对数几率
vector<vector<double>> grid;              // 每个格子的对数几率
vector<vector<int>>    truth;             // 真实地图（仅仿真用）

// 概率 -> 对数几率
double prob2logodds(double p) { return log(p / (1.0 - p)); }

// 对数几率 -> 概率
double logodds2prob(double l) {
    if (l >  30) return 1.0;
    if (l < -30) return 0.0;
    return 1.0 / (1.0 + exp(-l));
}

// 初始化
void init() {
    L_occ   = prob2logodds(P_OCC);
    L_free  = prob2logodds(P_FREE);
    L_prior = prob2logodds(P_PRIOR);
    grid.assign(H, vector<double>(W, L_prior));
    truth.assign(H, vector<int>(W, 0));
}

// 仿真世界：四周墙壁 + 一个方柱
void buildWorld() {
    for (int x = 0; x < W; ++x) { truth[0][x] = 1; truth[H-1][x] = 1; }
    for (int y = 0; y < H; ++y) { truth[y][0] = 1; truth[y][W-1] = 1; }
    for (int y = 5; y <= 8; ++y)
        for (int x = 12; x <= 15; ++x)
            truth[y][x] = 1;
}

// 单条激光：从 (sx, sy) 沿 angle 发射，最大射程 maxRange
void updateBeam(double sx, double sy, double angle, double maxRange) {
    double step = 0.2;                        // 沿光束走的步长
    double dx = cos(angle) * step;
    double dy = sin(angle) * step;
    double x = sx, y = sy;

    while (true) {
        x += dx;
        y += dy;

        int gx = (int)floor(x);
        int gy = (int)floor(y);

        if (gx < 0 || gx >= W || gy < 0 || gy >= H) return;

        if (truth[gy][gx] == 1) {             // 命中障碍
            grid[gy][gx] += L_occ;
            if (grid[gy][gx] > L_MAX) grid[gy][gx] = L_MAX;
            return;
        }

        grid[gy][gx] += L_free;               // 穿过：标记空闲
        if (grid[gy][gx] < L_MIN) grid[gy][gx] = L_MIN;

        double dist = sqrt((x - sx) * (x - sx) + (y - sy) * (y - sy));
        if (dist > maxRange) return;
    }
}

// 一帧扫描：360° 均匀发射 nBeams 条
void updateScan(double sx, double sy, int nBeams, double maxRange) {
    for (int i = 0; i < nBeams; ++i) {
        double angle = 2.0 * M_PI * i / nBeams;
        updateBeam(sx, sy, angle, maxRange);
    }
}

// 打印栅格
void printGrid() {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double p = logodds2prob(grid[y][x]);
            char c;
            if (p > 0.7)      c = '#';
            else if (p < 0.3) c = ' ';
            else              c = '.';
            cout << c;
        }
        cout << '\n';
    }
    cout << '\n';
}

int main() {
    init();
    buildWorld();

    double sx = W / 2.0;
    double sy = H / 2.0;
    int    nBeams   = 180;
    double maxRange = 30.0;

    cout << "=== 真实世界 ===\n";
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) cout << (truth[y][x] ? '#' : ' ');
        cout << '\n';
    }
    cout << '\n';

    cout << "=== 初始栅格（全未知） ===\n";
    printGrid();

    for (int k = 0; k < 5; ++k) updateScan(sx, sy, nBeams, maxRange);

    cout << "=== 扫描 5 次后 ===\n";
    printGrid();

    return 0;
}