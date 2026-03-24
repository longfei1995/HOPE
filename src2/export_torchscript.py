#!/usr/bin/env python3
"""
将 HOPE SAC 的 actor 网络导出为 TorchScript 格式，用于 LibTorch C++ 推理。

用法（在 src/ 目录下执行）：
    python ../src2/export_torchscript.py ./log/exp/sac_20260319_092327/SAC_best.pt \
        --output ../src2/model/actor_traced.pt
"""

import sys
import os
import argparse

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn

from model.network import MultiObsEmbedding
from configs import ACTOR_CONFIGS, LIDAR_NUM, N_DISCRETE_ACTION


class ActorExportWrapper(nn.Module):
    """
    将状态归一化 + actor 网络 + log_std 打包为一个可导出的模块。
    输入为 4 个独立张量（而非 dict），兼容 torch.jit.trace。
    """

    def __init__(self, actor_net, log_std_tensor, state_normalize):
        super().__init__()

        # 复制 actor_net 的各子模块
        self.embed_lidar = actor_net.embed_lidar
        self.embed_tgt = actor_net.embed_tgt
        self.embed_am = actor_net.embed_am
        self.embed_img = actor_net.embed_img
        self.re_embed_img = actor_net.re_embed_img
        self.net = actor_net.net  # AttentionNetwork
        self.use_attention = actor_net.use_attention

        # 将归一化参数注册为 buffer（会随 TorchScript 一起保存）
        # 只有 lidar 和 target 需要归一化（update_modal 中 img 和 action_mask 为 False）
        lidar_mean = torch.as_tensor(
            state_normalize.state_mean.get(
                "lidar", np.zeros(LIDAR_NUM, dtype=np.float32)
            ),
            dtype=torch.float32,
        )
        lidar_std = torch.as_tensor(
            state_normalize.state_std.get(
                "lidar", np.ones(LIDAR_NUM, dtype=np.float32)
            ),
            dtype=torch.float32,
        )
        target_mean = torch.as_tensor(
            state_normalize.state_mean.get("target", np.zeros(5, dtype=np.float32)),
            dtype=torch.float32,
        )
        target_std = torch.as_tensor(
            state_normalize.state_std.get("target", np.ones(5, dtype=np.float32)),
            dtype=torch.float32,
        )

        self.register_buffer("lidar_mean", lidar_mean)
        self.register_buffer("lidar_std", lidar_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("log_std_buf", log_std_tensor.clone().detach().float())

    def forward(self, lidar, target, img, action_mask):
        """
        参数:
            lidar       : (batch, 120)
            target      : (batch, 5)
            img         : (batch, 3, 64, 64)
            action_mask : (batch, 42)
        返回:
            mean : (batch, 2)  动作均值，范围 [-1, 1]
            std  : (batch, 2)  动作标准差
        """
        # 状态归一化（只对 lidar 和 target）
        lidar_n = (lidar - self.lidar_mean) / (self.lidar_std + 1e-8)
        target_n = (target - self.target_mean) / (self.target_std + 1e-8)

        # 各模态嵌入（顺序与 MultiObsEmbedding.forward 一致）
        f_lidar = self.embed_lidar(lidar_n)
        f_target = self.embed_tgt(target_n)
        f_am = self.embed_am(action_mask)
        f_img, _ = self.embed_img(img)
        f_img = self.re_embed_img(f_img)

        # 多模态融合（attention 路径：stack → transformer）
        if self.use_attention:
            embed = torch.stack([f_lidar, f_target, f_am, f_img], dim=1)
        else:
            embed = torch.cat([f_lidar, f_target, f_am, f_img], dim=1)

        out = self.net(embed)
        # output_layer = Tanh（actor 配置 use_tanh_output=True）
        out = torch.tanh(out)
        mean = torch.clamp(out, -1.0, 1.0)

        log_std = self.log_std_buf.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std


def main():
    parser = argparse.ArgumentParser(description="导出 HOPE SAC Actor 为 TorchScript")
    parser.add_argument(
        "ckpt_path", type=str, help="SAC 检查点路径，如 ./log/exp/.../SAC_best.pt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../src2/model/actor_traced.pt",
        help="导出的 TorchScript 文件路径",
    )
    args = parser.parse_args()

    # 加载检查点
    print(f"加载检查点: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    # 构建 actor 网络并加载权重
    actor_net = MultiObsEmbedding(ACTOR_CONFIGS)
    actor_net.load_state_dict(checkpoint["actor_net"])
    actor_net.eval()

    # 提取 log_std 和 state_normalize
    log_std = checkpoint["log"]
    state_normalize = checkpoint["state_norm"]

    print(f"log_std = {log_std.detach().numpy().flatten()}")
    print(f"state_norm n_state = {state_normalize.n_state}")

    # 构建导出 wrapper
    wrapper = ActorExportWrapper(actor_net, log_std, state_normalize)
    wrapper.eval()

    # 创建示例输入用于 trace
    example_lidar = torch.randn(1, LIDAR_NUM)
    example_target = torch.randn(1, 5)
    example_img = torch.randn(1, 3, 64, 64)
    example_mask = torch.ones(1, N_DISCRETE_ACTION)

    # Trace
    print("正在 trace 模型...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (example_lidar, example_target, example_img, example_mask)
        )

    # 验证
    with torch.no_grad():
        mean_orig, std_orig = wrapper(
            example_lidar, example_target, example_img, example_mask
        )
        mean_traced, std_traced = traced(
            example_lidar, example_target, example_img, example_mask
        )
    max_diff = (mean_orig - mean_traced).abs().max().item()
    print(f"验证: 原始 vs traced 最大误差 = {max_diff:.2e}")

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    traced.save(args.output)
    print(f"已导出 TorchScript 模型到: {args.output}")
    print(
        f"模型输入: lidar(1,{LIDAR_NUM}), target(1,5), img(1,3,64,64), action_mask(1,{N_DISCRETE_ACTION})"
    )
    print(f"模型输出: mean(1,2), std(1,2)")


if __name__ == "__main__":
    main()
