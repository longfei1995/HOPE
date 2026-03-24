import sys
sys.path.append("..")   # 把上级目录加入搜索路径（适配不同工作目录）
sys.path.append(".")
import time
import os
from shutil import copyfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED,Status
from evaluation.eval_utils import eval
from configs import *


# ------------------------------------------------------------------------------
# 辅助类 1：SceneChoose —— 四种场景的课程学习调度器
#
# 训练时有四种难度场景：Normal < Complex < Extrem < dlp（真实数据集）
# 策略：前 200 episode 均匀采样；之后 50% 选"当前最差"场景，50% 均匀采样。
# 目的：让智能体把精力集中在还没学好的场景上，避免只刷简单场景。
# ------------------------------------------------------------------------------
class SceneChoose():
    def __init__(self) -> None:
        self.scene_types = {0:'Normal', 
                            1:'Complex',
                            2:'Extrem',
                            3:'dlp',
                            }
        # 各场景的目标成功率阈值，达到后不再优先选择该场景
        self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
        self.success_record = {}    # 记录每个场景的历史成功/失败（1/0）
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record = []      # 记录每个 episode 选择的场景 id
        self.history_horizon = 200  # 前 200 episode 为"热身期"，均匀采样
        
        
    def choose_case(self,):
        # 热身期结束前：优先补齐各场景出现次数（均匀分配）
        if len(self.scene_record) < self.history_horizon:
            scene_chosen = self._choose_case_uniform()
        else:
            # 热身期结束后：50% 选最差场景，50% 均匀
            if np.random.random() > 0.5:
                scene_chosen = self._choose_case_worst_perform()
            else:
                scene_chosen = self._choose_case_uniform()
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]
    
    def update_success_record(self, success:int):
        # 每个 episode 结束后由主循环调用，传入 1（成功）或 0（失败）
        self.success_record[self.scene_record[-1]].append(success)

    def _choose_case_uniform(self,):
        # 统计最近 history_horizon 个 episode 中各场景出现次数，选最少的那个
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i+1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)
    
    def _choose_case_worst_perform(self,):
        # 计算各场景近期成功率，距离目标越远（fail_rate 越大）被选中概率越高
        success_rate = []
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)    # 保证每个场景都有最低 1% 被选概率
        fail_rate = fail_rate/np.sum(fail_rate)     # 归一化为概率分布
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

# ------------------------------------------------------------------------------
# 辅助类 2：DlpCaseChoose —— DLP 数据集内 248 个场景的细粒度调度器
#
# 当 SceneChoose 选中 'dlp' 后，由这个类进一步决定用哪个具体场景（0~247）。
# 策略：20% 概率随机选；80% 概率优先选失败率高的场景（难例重采样）。
# ------------------------------------------------------------------------------
class DlpCaseChoose():
    def __init__(self) -> None:
        self.dlp_case_num = 248          # DLP 数据集共 248 个泊车场景
        self.case_record = []            # 历史场景选择记录
        self.case_success_rate = {}      # 每个场景 id → 成功/失败历史列表
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] = []
        self.horizon = 500               # 前 500 个 dlp episode 完全随机
    
    def choose_case(self,):
        # 热身期或 20% 概率：完全随机，保证探索覆盖
        if np.random.random()<0.2 or len(self.case_record)<self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        # 否则：按失败率加权采样，让智能体反复练习难场景
        success_rate = []
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)   # 尚未见过的场景视为成功率 0（最高优先级）
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = 1-np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)   # 每个场景至少 0.5% 被选概率
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)
    
    def update_success_record(self, success:int, case_id:int):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)


# ==============================================================================
# 程序入口（等价于 C++ 的 main()）
# 直接运行此脚本时执行；被其他文件 import 时不执行。
# ==============================================================================
if __name__=="__main__":

    # ── 第一步：命令行参数解析 ──────────────────────────────────────────────────
    # 可以在命令行用 --xxx 覆盖默认值，例如：
    #   python train_HOPE_sac.py --visualize false --train_episode 50000
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None) # 继续训练时传入已有权重路径
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')  # 图像编码器权重
    parser.add_argument('--train_episode', type=int, default=100000)  # 总训练回合数
    parser.add_argument('--eval_episode', type=int, default=2000)     # 训练结束后评估的回合数
    parser.add_argument('--verbose', type=lambda x: x.lower() != 'false', default=True)   # 是否打印训练日志
    parser.add_argument('--visualize', type=lambda x: x.lower() != 'false', default=True) # 是否开启 pygame 可视化
    args = parser.parse_args()

    verbose = args.verbose

    # ── 第二步：创建环境 ────────────────────────────────────────────────────────
    # CarParking 是底层 Gym 环境（物理仿真 + 渲染）
    # CarParkingWrapper 在其上封装：动作缩放、观测预处理、奖励整形
    # 无头服务器训练时设 visualize=False，避免 pygame 要求显示器
    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose,)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)          # 后续所有交互都通过 env（Wrapper）
    scene_chooser = SceneChoose()            # 四场景课程调度器
    dlp_case_chooser = DlpCaseChoose()       # DLP 248 场景细粒度调度器

    # ── 第三步：日志与保存路径配置 ────────────────────────────────────────────
    # 每次训练生成带时间戳的独立目录，避免覆盖历史实验
    # 目录结构：log/exp/sac_YYYYMMDD_HHMMSS/
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/sac_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)   # TensorBoard 写入器，记录损失/奖励/成功率等曲线
    # 将当前 configs.py 备份到日志目录，便于事后复现实验
    copyfile('./configs.py', save_path+'configs.txt')
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    # ── 第四步：随机种子固定（保证实验可复现）────────────────────────────────
    seed = SEED   # SEED=42，定义在 configs.py
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 第五步：构建智能体 ──────────────────────────────────────────────────────
    # 网络结构参数来自 configs.py（ACTOR_CONFIGS / CRITIC_CONFIGS），避免硬编码
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,                              # SAC 使用连续动作空间
        "observation_shape": env.observation_shape,     # Dict 观测：lidar/target/img/action_mask
        "action_dim": env.action_space.shape[0],        # 动作维度 = 2（转向角 + 速度）
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",                        # SAC 用高斯分布对动作采样
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    rl_agent = SAC(configs)                             # 纯 RL 智能体（SAC 算法）
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)    # 从检查点恢复，继续训练
        print('load pre-trained model!')
    img_encoder_checkpoint =  args.img_ckpt if USE_IMG else None
    if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
        # 加载预训练图像编码器（autoencoder），训练期间冻结（require_grad=False）
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)

    # RsPlanner：将 Reeds-Shepp 几何路径转换为离散动作序列
    # step_ratio 将路径长度换算为动作步数
    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    # ParkingAgent 是调度层：当存在有效 RS 路径时走规划器，否则走 RL
    parking_agent = ParkingAgent(rl_agent, rs_planner)


    # ── 第六步：训练统计变量初始化 ────────────────────────────────────────────
    reward_list = []            # 每个 episode 的总奖励（用于画图）
    reward_per_state_list = []  # 每个时间步的奖励（用于计算滑动均值）
    reward_info_list = []       # 各奖励分量的逐 episode 累加（便于分析）
    case_id_list = []           # 每个 episode 对应的场景 id（调试用）
    succ_record = []            # 全局成功/失败记录（1/0）
    total_step_num = 0          # 全局总步数（控制探索 → 学习的切换时机）
    best_success_rate = [0, 0, 0, 0]  # 各场景当前最优成功率（用于保存 best 模型）

    # ── 第七步：训练主循环（每次迭代 = 一个完整 episode）────────────────────
    for i in range(args.train_episode):
        # 7a. 课程调度：选场景类型，若是 dlp 再选具体场景编号
        scene_chosen = scene_chooser.choose_case()
        if scene_chosen == 'dlp':
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None
        # 重置环境，返回初始观测（Dict：lidar/target/img/action_mask）
        obs = env.reset(case_id, None, scene_chosen)
        parking_agent.reset()               # 清空 RS 规划器路径缓存
        case_id_list.append(env.map.case_id)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        xy = []
        # 7b. episode 内逐步交互循环
        while not done:
            step_num += 1
            total_step_num += 1

            # ── 动作选择：探索期 vs 学习期 ──────────────────────────────────
            # 经验回放池未填满时：随机动作（SAC 纯探索），但仍计算 log_prob 供存储
            # 经验回放池填满后 / 正在执行 RS 路径时：智能体自主决策
            if total_step_num <= parking_agent.configs.memory_size and not parking_agent.executing_rs:
                action = env.action_space.sample()                  # 随机探索
                log_prob = parking_agent.get_log_prob(obs, action)  # 仍需记录概率
            else:
                action, log_prob = parking_agent.get_action(obs)    # 策略网络决策

            # ── 环境交互 ────────────────────────────────────────────────────
            next_obs, reward, done, info = env.step(action)
            reward_info.append(list(info['reward_info'].values()))  # 记录各奖励分量
            total_reward += reward
            reward_per_state_list.append(reward)
            # 将 (s, a, r, done, log_π, s') 存入经验回放池
            parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs

            # ── SAC 网络更新（每 10 步更新一次，平衡速度与稳定性）──────────
            if total_step_num > parking_agent.configs.memory_size and total_step_num%10==0:
                actor_loss, critic_loss = parking_agent.update()    # 从回放池采样并梯度更新
                if total_step_num%200==0:
                    writer.add_scalar("actor_loss", actor_loss, i)
                    writer.add_scalar("critic_loss", critic_loss, i)
            
            # ── Reeds-Shepp 路径更新 ────────────────────────────────────────
            # 环境每步会检测是否能规划出到目标的 RS 路径；有效时传给 ParkingAgent
            # ParkingAgent 接管后会按 RS 路径执行，直到路径走完再切回 RL
            if info['path_to_dest'] is not None:
                parking_agent.set_planner_path(info['path_to_dest'])

            if done:
                # episode 结束：判断终止原因（ARRIVED/COLLIDED/OUTBOUND/OUTTIME）
                # 只有 ARRIVED 才算成功，其余均视为失败
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)      # 更新课程调度器
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(1, case_id)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(0, case_id)

        # 7c. episode 结束后：写 TensorBoard 日志
        # action_std：策略的探索幅度（越小说明策略越确定）
        # alpha：SAC 熵温度系数（越大越鼓励探索）
        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar("action_std0", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[0],i)
        writer.add_scalar("action_std1", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[1],i)
        writer.add_scalar("alpha", parking_agent.alpha.detach().cpu().numpy().reshape(-1)[0],i)
        for type_id in scene_chooser.scene_types:
            writer.add_scalar("success_rate_%s"%scene_chooser.scene_types[type_id],
                np.mean(scene_chooser.success_record[type_id][-100:]), i)
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record),'/',len(succ_record))
            print(parking_agent.log_std.detach().cpu().numpy().reshape(-1), parking_agent.alpha.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]),np.mean(parking_agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            for j in range(10):
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        # 7d. 保存模型策略
        # 策略一：所有场景成功率同时刷新历史最高时，保存 SAC_best.pt
        # best_success_rate 被 min(实际, 目标) 上限截断，避免超过目标后无限刷新
        # save best model
        for type_id in scene_chooser.scene_types:
            success_rate_normal = np.mean(scene_chooser.success_record[0][-100:])
            success_rate_complex = np.mean(scene_chooser.success_record[1][-100:])
            success_rate_extreme = np.mean(scene_chooser.success_record[2][-100:])
            success_rate_dlp = np.mean(scene_chooser.success_record[3][-100:])
        if success_rate_normal >= best_success_rate[0] and success_rate_complex >= best_success_rate[1] and\
            success_rate_extreme >= best_success_rate[2] and success_rate_dlp >= best_success_rate[3] and i>100:
            raw_best_success_rate = np.array([success_rate_normal, success_rate_complex, success_rate_extreme, success_rate_dlp])
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            parking_agent.save("%s/SAC_best.pt" % (save_path),params_only=True)
            f_best_log = open(save_path+'best.txt', 'w')
            f_best_log.write('epoch: %s, success rate: %s %s %s %s'%(i+1, raw_best_success_rate[0],
                                raw_best_success_rate[1], raw_best_success_rate[2], raw_best_success_rate[3]))
            f_best_log.close()

        if (i+1) % 2000 == 0:
            # 策略二：每 2000 episode 保存一次定期检查点（防止意外中断丢失进度）
            parking_agent.save("%s/SAC_%s.pt" % (save_path, i),params_only=True)
        

        if verbose and i%20==0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            f = plt.gcf()
            f.savefig('%s/reward.png'%save_path)
            f.clear()

    # ── 第八步：训练结束后的最终评估 ──────────────────────────────────────────
    # 依次在四种场景上各跑 eval_episode 个回合，结果写入 CSV
    # torch.no_grad() 关闭梯度计算，节省显存（纯推理，不需要反向传播）
    # choose_action=False 表示 SAC 使用随机采样动作（而非确定性均值）
    eval_episode = args.eval_episode
    choose_action = False
    with torch.no_grad():
        # eval on dlp
        env.set_level('dlp')
        log_path = save_path+'/dlp'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on extreme
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on complex
        env.set_level('Complex')
        log_path = save_path+'/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on normalize
        env.set_level('Normal')
        log_path = save_path+'/normalize'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

    env.close()