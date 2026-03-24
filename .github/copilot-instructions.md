# HOPE：工作区说明
HOPE 是一个**基于强化学习的混合策略自动泊车路径规划器**。它将 Soft Actor-Critic（SAC）或 PPO 智能体与 Reeds-Shepp 几何路径融合，使用多模态观测（激光雷达 + 摄像头 + 动作掩码），并通过 Transformer 注意力网络进行处理。

## 语言使用
我们交流回答时，一律使用中文。在注释里使用中文，代码里使用英文。


## 自动更新提示
当发现当前规范有遗漏（如未覆盖新语法/框架规范），请自动在本文件补充对应的规范条目，并标注更新时间。

## 架构

```
env/          ← Gym 环境：地图、车辆动力学、激光雷达、碰撞检测
model/        ← 网络、智能体、回放缓冲区、动作掩码
  agent/      ← ParkingAgent（融合 RL + RS 规划器）、PPOAgent、SACAgent
  ckpt/       ← 预训练权重（autoencoder.pt、HOPE_SAC*.pt、HOPE_PPO.pt）
train/        ← 训练入口（train_HOPE_sac.py、train_HOPE_ppo.py）
evaluation/   ← 评估入口（eval_mix_scene.py）
configs.py    ← 所有超参数的唯一配置源
```

**核心组件关系：**
- `CarParkingWrapper`（env_wrapper.py）封装 `CarParking`（car_parking_base.py）——始终通过 Wrapper 交互，勿直接使用底层环境。
- `ParkingAgent`（model/agent/parking_agent.py）在 RL 智能体与 `RsPlanner` 之间进行委托调度，勿绕过它直接调用。
- `MultiObsEmbedding`（model/network.py）负责多模态融合，输入顺序固定为：`lidar`、`target`、`img`、`action_mask`。
- `ActionMask`（model/action_mask.py）预计算 42 个无碰撞离散动作（11 个转向角 × 2 个速度档）。

## 构建与环境

```bash
# 环境配置
conda create -n HOPE python==3.8
conda activate HOPE
pip install -r requirements.txt
# 从 https://pytorch.org/ 安装对应 CUDA 版本的 PyTorch

# 运行预训练智能体（需在 src/ 目录下执行）
cd src
python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True

# SAC 训练
python ./train/train_HOPE_sac.py

# PPO 训练
python ./train/train_HOPE_ppo.py
```

所有训练和评估脚本必须在 `src/` 目录下启动——路径均相对于该目录。

## 配置

**`src/configs.py` 是所有参数的唯一配置源**，禁止硬编码其中已有的值。

关键常量：
- `SEED = 42`——保证环境和智能体的可复现性
- `discrete_steer_num = 11`，`discrete_speed_num = 2` → 前进 22 + 后退 20 = **动作掩码维度 42**
- 观测空间为 **Dict**：`lidar`（120,）、`target`（5,）、`img`（3,64,64）、`action_mask`（42,）
- 动作空间（连续）：`[转向角, 速度]` ∈ `[-0.75, 0.75] × [-2.5, 2.5]` rad/m·s⁻¹
- 课程训练难度级别：`'Normal'` | `'Complex'` | `'Extreme'` | `'dlp'`

## 开发规范

- **设备处理：** 始终使用 `configs.device`（自动检测 CUDA），禁止硬编码 `'cuda'` 或 `'cpu'`。
- **观测字典键：** 均为小写字符串 `'lidar'`、`'target'`、`'img'`、`'action_mask'`。Wrapper 会在输入网络前将 `img` 转为通道优先格式 `(3, 64, 64)`。
- **回合终止状态** 是 `car_parking_base.py` 中的枚举：`CONTINUE`、`ARRIVED`、`COLLIDED`、`OUTBOUND`、`OUTTIME`。用 `status == ParkingStatus.ARRIVED` 判断成功。
- **环境重置接口：** `env.reset(case_id=None, data_dir=None, level='Normal')`——DLP 场景需传入 `case_id`（整数 0–247）和 `data_dir`。
- **Reeds-Shepp 规划器** 在 `ParkingAgent.act()` 内部自动激活（当存在有效 RS 路径时），智能体层不应直接调用。
- **模型检查点** 通过 `agent_base.py` 中定义的 `agent.save(path)` / `agent.load(path, device)` 进行保存和加载。
- **PPO 与 SAC 的识别** 在评估时通过检查检查点文件名是否含 `'PPO'` 来完成。
- **日志：** TensorBoard 日志写入 `log/<run_name>/tb/`；CSV 评估结果写入 `log/eval/<timestamp>/`。
- **正交权重初始化** 贯穿整个网络（`torch.nn.init.orthogonal_`），新增网络层时须保持此初始化方式。

## 常见陷阱

- 脚本使用**相对导入**（`from env.car_parking_base import ...`）——必须从 `src/` 目录运行，不能在仓库根目录执行。
- `pygame` 渲染需要显示服务器；无头服务器训练时请设置 `visualize=False`（或 `--visualize false`）。
- 训练开始前必须加载预训练的 `autoencoder.pt`——它在强化学习训练过程中**冻结**，不参与梯度更新。
- `ActionMask` 预计算量随 `discrete_steer_num × discrete_speed_num` 缩放；修改这两个参数后需重新生成掩码表。
- DLP 地图需要传入数据集路径参数（`data_dir`）；内置的 248 个场景不随仓库一起提供，需自行下载数据集。
- SAC 的 `batch_size`（65536）是跨多个回合累积后才触发一次更新，不要与单回合步数混淆。
