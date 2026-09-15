#include <bits/stdc++.h>
using namespace std;

const int W = 10, H = 10;
int  grid[H][W];        // 0 通行，1 障碍
int  g[H][W];           // 起点到各点的实际代价
int  par[H][W];         
bool done[H][W];      
int  dx[4] = {0, 0, -1, 1};
int  dy[4] = {-1, 1, 0, 0};

int h(int x, int y, int gx, int gy) {
    return abs(x - gx) + abs(y - gy);        // 曼哈顿距离
}

vector<pair<int,int>> aStar(int sx, int sy, int gx, int gy) {
    memset(g, 0x3f, sizeof(g));              // g 初始化为"无穷大"
    memset(par, -1, sizeof(par));
    memset(done, 0, sizeof(done));


    priority_queue<pair<int,pair<int,int>>,
                   vector<pair<int,pair<int,int>>>,
                   greater<>> open;

    g[sy][sx] = 0;
    open.push({h(sx,sy,gx,gy), {sx, sy}});   // 起点 f = h

    while (!open.empty()) {
        auto [f, p] = open.top(); open.pop();
        auto [x, y] = p;

        if (done[y][x]) continue;           
        done[y][x] = true;

        if (x == gx && y == gy) break;       // 到达终点

        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            if (grid[ny][nx] == 1) continue;             // 越界或障碍

            int ng = g[y][x] + 1;                        // 4 邻域代价 1
            if (ng < g[ny][nx]) {                        // 松弛
                g[ny][nx]   = ng;
                par[ny][nx] = y * W + x;
                open.push({ng + h(nx,ny,gx,gy), {nx, ny}});
            }
        }
    }

    // 回溯路径
    vector<pair<int,int>> path;
    if (!done[gy][gx]) return path;          // 不可达
    int cx = gx, cy = gy;
    while (cx != -1) {
        path.push_back({cx, cy});
        int p = par[cy][cx];
        if (p == -1) break;
        cx = p % W; cy = p / W;
    }
    reverse(path.begin(), path.end());
    return path;
}
//全局数组、启发式、初始化、入队、主循环、松弛、回溯