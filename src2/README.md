# HOPE APA Planner — C++ 部署

将 HOPE 强化学习泊车模型部署为 C++ 可执行程序。

## 目录结构

```
src2/
├── CMakeLists.txt              # 构建脚本
├── README.md
├── export_torchscript.py       # Python 导出脚本（在 src/ 目录下执行）
├── include/
│   └── apa_planner.h           # APAPlanner 类头文件
├── src/
│   ├── apa_planner.cpp         # APAPlanner 实现
│   └── main.cpp                # 演示程序（虚拟输入测试）
└── model/
    └── actor_traced.pt         # 导出后的 TorchScript 模型（需执行步骤一生成）
```

## 步骤一：导出 TorchScript 模型

在 `src/` 目录下执行：

```bash
cd src
python ../src2/export_torchscript.py ./log/exp/sac_20260319_092327/SAC_best.pt \
    --output ../src2/model/actor_traced.pt
```

## 步骤二：下载 LibTorch

从 https://pytorch.org/get-started/locally/ 下载 **LibTorch** (CPU 版)，解压到 `src2/libtorch/`：

```bash
cd src2
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

## 步骤三：编译

```bash
cd src2
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make -j$(nproc)
```

## 步骤四：运行

```bash
./apa_demo ../model/actor_traced.pt 10
```

## 集成到自己的项目

```cpp
#include "apa_planner.h"

APAPlanner planner("actor_traced.pt", /*deterministic=*/true);

// 每个控制周期调用一次
auto result = planner.run(lidar_data, target_pose, camera_img, action_mask);
// result.actual_steer → 转向角 (rad), [-0.75, 0.75]
// result.actual_speed → 速度 (m/s),   [-2.5,  2.5]
```

## 输入输出说明

| 输入 | 形状 | 说明 |
|------|------|------|
| lidar | (120,) | 激光雷达距离值 |
| target | (5,) | [x, y, θ, cosθ, sinθ] 目标位姿 |
| img | (3,64,64) | 通道优先的摄像头图像 |
| action_mask | (42,) | 动作可行性掩码 |

| 输出 | 说明 |
|------|------|
| actual_steer | 转向角 (rad) |
| actual_speed | 速度 (m/s) |
| action_idx | 离散动作索引 |
