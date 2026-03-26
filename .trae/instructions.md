# HOPE 项目说明

## 项目概览

**HOPE** (Hybrid Policy Path Planner) 是一个基于**强化学习**的**混合策略自动泊车路径规划器**，发表于论文 [HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios](https://arxiv.org/abs/2405.20579)。

### 核心创新

HOPE 提出了一种新颖的混合策略框架，将：
- **强化学习智能体** (SAC/PPO) 用于处理复杂环境感知和决策
- **Reeds-Shepp 几何曲线** 用于精确的末端路径规划

通过这种混合设计，HOPE 在多样化泊车场景中相比纯规则基算法和纯强化学习方法都取得了更高的规划成功率。

### 关键技术特点

1. **多模态观测融合**：激光雷达 + 摄像头图像 + 动作掩码 + 目标位姿
2. **Transformer 注意力机制**：融合多模态环境信息
3. **动作掩码引导探索**：提前过滤不可行动作，提高样本效率
4. **混合决策机制**：当存在可行的 Reeds-Shepp 路径时优先走几何规划，否则由 RL 决策
5. **课程学习调度**：从易到难渐进训练，覆盖 Normal → Complex → Extreme → DLP 真实数据集

---

## 项目架构

```
HOPE/
├── src/                    # Python 训练与评估（主版本）
│   ├── env/               # Gym 环境实现
│   ├── model/             # 网络结构与 RL 算法
│   │   ├── agent/         # PPOAgent, SACAgent, ParkingAgent
│   │   ├── ckpt/          # 预训练权重
│   │   └── ...            # 网络、回放缓冲区等
│   ├── train/             # 训练入口脚本
│   ├── evaluation/        # 评估入口脚本
│   └── configs.py         # 全局配置（唯一配置源）
│
├── src2/                  # C++ 部署版本
│   ├── include/           # 头文件
│   ├── src/               # 实现代码
│   ├── model/             # 导出后的 TorchScript 模型
│   └── CMakeLists.txt     # 构建脚本
│
├── assets/                # 项目文档图片
├── data/                  # 数据文件
└── ...
```

---

## 核心组件详解

### 环境模块 (src/env/)

| 文件 | 功能说明 |
|------|---------|
| [car_parking_base.py](file:///home/hyh/coder/HOPE/src/env/car_parking_base.py) | 底层 Gym 环境核心，负责物理仿真、碰撞检测、到达判定、奖励计算、渲染 |
| [env_wrapper.py](file:///home/hyh/coder/HOPE/src/env/env_wrapper.py) | Gym 接口适配，负责奖励整形、动作缩放、观测预处理 |
| [vehicle.py](file:///home/hyh/coder/HOPE/src/env/vehicle.py) | 车辆动力学模型（自行车模型），状态传播 |
| [map_base.py](file:///home/hyh/coder/HOPE/src/env/map_base.py) | 地图基类，定义起点/终点/障碍物接口 |
| [parking_map_normal.py](file:///home/hyh/coder/HOPE/src/env/parking_map_normal.py) | 随机生成不同难度的泊车场景（Normal/Complex/Extreme） |
| [parking_map_dlp.py](file:///home/hyh/coder/HOPE/src/env/parking_map_dlp.py) | 加载 DLP 真实数据集场景 |
| [lidar_simulator.py](file:///home/hyh/coder/HOPE/src/env/lidar_simulator.py) | 激光雷达仿真器，120 个光束扫描障碍物 |
| [reeds_shepp.py](file:///home/hyh/coder/HOPE/src/env/reeds_shepp.py) | Reeds-Shepp 最短路径计算 |
| [observation_processor.py](file:///home/hyh/coder/HOPE/src/env/observation_processor.py) | 图像观测下采样和预处理 |

**重要设计点：**
- 观测空间是 Dict 格式：`{'lidar', 'target', 'img', 'action_mask'}`
- `step` 返回 `(observation, reward_info, status, info)`，status 是枚举：`CONTINUE/ARRIVED/COLLIDED/OUTBOUND/OUTTIME`
- 当靠近目标且存在无碰撞 RS 路径时，`info['path_to_dest']` 会携带路径对象

### 模型模块 (src/model/)

| 文件 | 功能说明 |
|------|---------|
| [network.py](file:///home/hyh/coder/HOPE/src/model/network.py) | 网络基础组件，`MultiObsEmbedding` 多模态嵌入，VAE/CAE 图像编码器 |
| [attention.py](file:///home/hyh/coder/HOPE/src/model/attention.py) | Transformer 注意力网络，融合多模态特征 |
| [action_mask.py](file:///home/hyh/coder/HOPE/src/model/action_mask.py) | 预计算 42 个离散动作的可行性掩码 |
| [replay_memory.py](file:///home/hyh/coder/HOPE/src/model/replay_memory.py) | 经验回放缓冲区 |
| [state_norm.py](file:///home/hyh/coder/HOPE/src/model/state_norm.py) | 状态归一化 |
| [agent_base.py](file:///home/home/hyh/coder/HOPE/src/model/agent_base.py) | RL 智能体基类，定义接口 |
| [agent/ppo_agent.py](file:///home/hyh/coder/HOPE/src/model/agent/ppo_agent.py) | PPO 算法实现 |
| [agent/sac_agent.py](file:///home/hyh/coder/HOPE/src/model/agent/sac_agent.py) | SAC 算法实现 |
| [agent/parking_agent.py](file:///home/hyh/coder/HOPE/src/model/agent/parking_agent.py) | **混合策略调度器**，融合 RL + RS |

**混合调度机制（ParkingAgent）：**
- 若 `RsPlanner` 持有有效路径 → 执行 RS 动作（几何规划模式）
- 否则 → 交由底层 RL 智能体决策（学习模式）
- 通过 `__getattr__` 代理属性访问，外部无需区分

**预训练权重 (src/model/ckpt/)：**
- `autoencoder.pt`：预训练图像自动编码器（训练 RL 时冻结）
- `HOPE_PPO.pt`：PPO 训练的 HOPE 模型
- `HOPE_SAC0.pt`, `HOPE_SAC1.pt`：两个 SAC 训练的 HOPE 模型

### 训练与评估

| 文件 | 功能说明 |
|------|---------|
| [train/train_HOPE_sac.py](file:///home/hyh/coder/HOPE/src/train/train_HOPE_sac.py) | SAC 训练入口，支持课程学习调度 |
| [train/train_HOPE_ppo.py](file:///home/hyh/coder/HOPE/src/train/train_HOPE_ppo.py) | PPO 训练入口 |
| [evaluation/eval_mix_scene.py](file:///home/hyh/coder/HOPE/src/evaluation/eval_mix_scene.py) | 在所有难度级别上评估模型 |
| [evaluation/eval_utils.py](file:///home/hyh/coder/HOPE/src/evaluation/eval_utils.py) | 评估工具函数 |

**课程学习调度器：**
- `SceneChoose`：四场景（Normal/Complex/Extreme/dlp）自适应调度
- `DlpCaseChoose`：DLP 数据集 248 个场景难例重采样
- 策略：热身期均匀采样，之后优先选择成功率低的场景

### C++ 部署 (src2/)

这是一个独立的 C++ 部署版本，使用 LibTorch 推理训练好的模型：

- [apa_planner.h](file:///home/hyh/coder/HOPE/src2/include/apa_planner.h)：API 头文件
- [apa_planner.cpp](file:///home/hyh/coder/HOPE/src2/src/apa_planner.cpp)：核心实现
- [main.cpp](file:///home/hyh/coder/HOPE/src2/src/main.cpp)：演示程序，支持场景可视化
- [bicycle_model.h](file:///home/hyh/coder/HOPE/src2/include/bicycle_model.h)：自行车运动学模型
- [scene_loader.h](file:///home/hyh/coder/HOPE/src2/include/scene_loader.h)：JSON 场景加载
- [visualizer.h](file:///home/hyh/coder/HOPE/src2/include/visualizer.h)：轨迹可视化

**使用方法：**
```bash
# 1. 导出 TorchScript 模型（在 src/ 目录执行）
python ../src2/export_torchscript.py ./log/exp/sac_xxx/SAC_best.pt --output ../src2/model/actor_traced.pt

# 2. 下载 LibTorch 并解压到 src2/libtorch/

# 3. 编译
cd src2 && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make -j$(nproc)

# 4. 运行
./apa_demo ../model/actor_traced.pt 10
```

---

## 配置与超参数

所有配置都在 [src/configs.py](file:///home/hyh/coder/HOPE/src/configs.py)，这是唯一配置源，禁止硬编码已有参数。

### 车辆参数

| 参数 | 值 | 说明 |
|------|-----|------|
| WHEEL_BASE | 2.8 m | 轴距 |
| LENGTH | 4.69 m | 车长 |
| WIDTH | 1.94 m | 车宽 |
| VALID_STEER | [-0.75, 0.75] rad | 转向角范围 |
| VALID_SPEED | [-2.5, 2.5] m/s | 速度范围 |

### 观测空间

- `lidar`: (120,) - 120 束激光雷达距离
- `target`: (5,) - [距离, cos(相对方位角), sin(相对方位角), cos(目标朝向差), sin(目标朝向差)]
- `img`: (3, 64, 64) - 自车视角摄像头图像
- `action_mask`: (42,) - 42 个离散动作的可行性掩码

**动作空间**（连续）：`[转向角, 速度] ∈ [-0.75, 0.75] × [-2.5, 2.5]`

### 网络配置

```python
ACTOR_CONFIGS = {
    'n_modal': 4,              # 模态数 = lidar(1) + target(1) + img(1) + action_mask(1)
    'embed_size': 128,         # 每个模态嵌入维度
    'hidden_size': 256,        # 融合后隐藏层维度
    'attention_configs': {
        'depth': 1,            # Transformer 层数
        'heads': 8,            # 注意力头数
        'dim_head': 32,        # 每个头维度
    }
}
```

### 训练超参数

| 参数 | SAC | PPO | 说明 |
|------|-----|-----|------|
| GAMMA | 0.98 | 0.98 | 折扣因子 |
| BATCH_SIZE | 65536 | 65536 | 回放批次大小 |
| LR | 5e-6 | 5e-6 | 学习率 |
| TAU | 0.1 | - | SAC 目标网络软更新系数 |
| MAX_TRAIN_STEP | 1e6 | 1e6 | 最大训练步数 |

### 奖励配置

权重定义在 `REWARD_WEIGHT`：

```python
REWARD_WEIGHT = OrderedDict({
    'time_cost': 1,          # 时间惩罚
    'rs_dist_reward': 0,     # RS 距离奖励（默认关闭）
    'dist_reward': 5,        # 距离奖励（欧氏距离减少量）
    'angle_reward': 0,       # 朝向奖励（默认关闭）
    'box_union_reward': 10,  # IoU 奖励（车体与目标车位交并比）
})
```

终止奖励：
- `ARRIVED` (成功到达): +50
- `COLLIDED` (碰撞): -50
- `OUTBOUND` (出界): -50
- `OUTTIME` (超时): -1

---

## 技术栈

### Python 依赖

```
numpy
shapely          # 几何计算
pygame           # 渲染
gym              # 环境接口
heapdict         # 优先队列
opencv-python    # 图像处理
scipy            # 科学计算
tensorboard      # 日志
tqdm             # 进度条
matplotlib       # 绘图
einops           # 张量操作
torch            # PyTorch 深度学习框架
```

### C++ 依赖

- LibTorch (PyTorch C++ API)
- OpenCV (图像处理)
- nlohmann/json (JSON 解析)
- CMake 3.18+
- C++17

---

## 快速开始

### 环境配置

```bash
conda create -n HOPE python==3.8
conda activate HOPE
pip install -r requirements.txt
# 从 https://pytorch.org/ 安装对应 CUDA 版本的 PyTorch
```

### 运行预训练智能体

```bash
cd src
python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True
```

### 训练 HOPE

```bash
cd src
# SAC 训练（推荐）
python ./train/train_HOPE_sac.py

# 或 PPO 训练
python ./train/train_HOPE_ppo.py
```

**重要提示**：所有训练和评估脚本**必须在 `src/` 目录下执行**，因为使用相对导入。

---

## 开发规范

1. **设备处理**：始终使用 `configs.device`，禁止硬编码 `'cuda'` 或 `'cpu'`
2. **观测字典键**：固定为 `'lidar'`、`'target'`、`'img'`、`'action_mask'`
3. **回合终止状态**：使用 `ParkingStatus.XXX` 枚举判断
4. **环境重置**：DLP 场景需要传入 `case_id` 和 `data_dir`
5. **权重初始化**：使用正交初始化 `torch.nn.init.orthogonal_`
6. **模型保存加载**：使用 `agent.save()` / `agent.load()` 接口
7. **日志输出**：TensorBoard 写入 `log/<run_name>/tb/`

---

## 常见问题

1. **ImportError**：通常因为从项目根目录而不是 `src/` 目录运行脚本
2. **pygame 显示错误**：无头服务器训练时设置 `--visualize false`
3. **DLP 数据缺失**：DLP 数据集不随仓库提供，需自行下载
4. **内存占用大**：SAC 的 `batch_size=65536` 是累积多条回合后才更新，属于正常设计

---

## 论文引用

```bibtex
@article{jiang2024hope,
  title={HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios},
  author={Jiang, Mingyang and Li, Yueyuan and Zhang, Songan and Chen, Siyuan and Wang, Chunxiang and Yang, Ming},
  journal={arXiv preprint arXiv:2405.20579},
  year={2024}
}
```

---

## 项目信息

- **项目主页**：https://github.com/jiamiya/HOPE
- **论文链接**：https://arxiv.org/abs/2405.20579
- **演示视频**：https://www.youtube.com/watch?v=62w9qhjIuRI
- **最后更新**：2026-03-26
