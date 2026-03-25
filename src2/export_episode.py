#!/usr/bin/env python3
"""
导出一个完整泊车 episode 的场景数据与动作序列为 JSON 文件。
用于 C++ 端加载后做轨迹计算和可视化。

用法（在 src/ 目录下执行）：
    python ../src2/export_episode.py ./model/ckpt/HOPE_SAC0.pt \
        --output ../src2/data/episode.json \
        --level Normal \
        --visualize false
"""

import sys
import os
import json
import argparse
import math

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import Status, VALID_SPEED
from configs import (
    SEED,
    ACTOR_CONFIGS,
    CRITIC_CONFIGS,
    WHEEL_BASE,
    FRONT_HANG,
    REAR_HANG,
    LENGTH,
    WIDTH,
    STEP_LENGTH,
    NUM_STEP,
)


def extract_obstacles(env):
    """从环境地图中提取所有障碍物的多边形顶点坐标。"""
    obstacles = []
    for obs_area in env.map.obstacles:
        coords = list(obs_area.shape.coords)
        obstacles.append([[float(x), float(y)] for x, y in coords])
    return obstacles


def extract_vehicle_box(state):
    """从车辆状态中提取车辆包围盒的多边形顶点坐标。"""
    box = state.create_box()
    return [[float(x), float(y)] for x, y in box.coords]


def run_episode(env, agent, case_id=None, level="Normal"):
    """运行一个完整 episode，记录每步的动作和状态。"""

    env.set_level(level)
    obs = env.reset(case_id)
    agent.reset()
    done = False
    steps = []

    # 记录初始状态
    init_state = env.vehicle.state
    start = [
        float(init_state.loc.x),
        float(init_state.loc.y),
        float(init_state.heading),
    ]

    # 目标位置
    dest_state = env.map.dest
    dest = [float(dest_state.loc.x), float(dest_state.loc.y), float(dest_state.heading)]

    # 目标车位多边形
    parking_lot = [[float(x), float(y)] for x, y in env.map.dest_box.coords]

    # 起始车辆包围盒
    start_box = [[float(x), float(y)] for x, y in env.map.start_box.coords]

    # 障碍物
    obstacles = extract_obstacles(env)

    step_idx = 0
    while not done:
        # 获取确定性动作（SAC 用 get_action）
        action, _ = agent.get_action(obs)

        # 记录 wrapper 输入的归一化动作 [-1, 1]
        raw_action = np.array(action, dtype=np.float32).copy()

        # 执行一步（wrapper 内部会 rescale 到真实物理量）
        next_obs, reward, done, info = env.step(action)

        # 记录 rescale 后的真实物理量
        curr_state = env.vehicle.state
        actual_steer = float(curr_state.steering)
        actual_speed = float(curr_state.speed)

        steps.append(
            {
                "step": step_idx,
                "raw_action": [float(raw_action[0]), float(raw_action[1])],
                "actual_steer": actual_steer,
                "actual_speed": actual_speed,
                "state": [
                    float(curr_state.loc.x),
                    float(curr_state.loc.y),
                    float(curr_state.heading),
                ],
                "vehicle_box": extract_vehicle_box(curr_state),
            }
        )

        # 设置 RS 路径
        if info.get("path_to_dest") is not None:
            agent.set_planner_path(info["path_to_dest"])

        obs = next_obs
        step_idx += 1

    final_status = info.get("status", None)
    status_name = final_status.name if final_status else "UNKNOWN"

    return {
        "start": start,
        "dest": dest,
        "start_box": start_box,
        "parking_lot": parking_lot,
        "obstacles": obstacles,
        "steps": steps,
        "status": status_name,
        "total_steps": step_idx,
        "vehicle_params": {
            "wheelbase": WHEEL_BASE,
            "front_hang": FRONT_HANG,
            "rear_hang": REAR_HANG,
            "length": LENGTH,
            "width": WIDTH,
            "step_length": STEP_LENGTH,
            "num_step": NUM_STEP,
            "mini_iter": 20,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="导出泊车 episode 场景数据为 JSON")
    parser.add_argument("ckpt_path", type=str, help="模型检查点路径")
    parser.add_argument(
        "--output",
        type=str,
        default="../src2/data/episode.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="Normal",
        choices=["Normal", "Complex", "Extrem"],
        help="场景难度级别",
    )
    parser.add_argument("--case_id", type=int, default=None, help="指定场景 ID（可选）")
    parser.add_argument(
        "--visualize", type=str, default="false", help="是否开启 pygame 可视化窗口"
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    visualize = args.visualize.lower() in ("true", "1", "yes")

    if visualize:
        raw_env = CarParking(fps=100, verbose=True)
    else:
        raw_env = CarParking(fps=100, verbose=True, render_mode="rgb_array")

    env = CarParkingWrapper(raw_env)
    env.set_level(args.level)

    # 根据检查点文件名判断智能体类型
    Agent_type = PPO if "ppo" in args.ckpt_path.lower() else SAC

    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": ACTOR_CONFIGS,
        "critic_layers": CRITIC_CONFIGS,
    }

    rl_agent = Agent_type(configs)
    rl_agent.load(args.ckpt_path, params_only=True)
    print(f"已加载模型: {args.ckpt_path}")

    step_ratio = (
        env.vehicle.kinetic_model.step_len
        * env.vehicle.kinetic_model.n_step
        * VALID_SPEED[1]
    )
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    with torch.no_grad():
        episode_data = run_episode(
            env, parking_agent, case_id=args.case_id, level=args.level
        )

    # 确保输出目录存在
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)

    print(f"Episode 导出完成: {output_path}")
    print(f"  状态: {episode_data['status']}")
    print(f"  总步数: {episode_data['total_steps']}")
    print(f"  障碍物数量: {len(episode_data['obstacles'])}")

    env.close()


if __name__ == "__main__":
    main()
