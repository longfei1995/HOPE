
class RsPlanner:
    """
    Reeds-Shepp 几何路径规划器。

    将一条 RS 路径（由曲线类型 L/S/R 和长度组成）转换为一系列
    离散动作序列，供车辆逐步执行。每个动作格式为 [转向, 步长]，
    其中步长被归一化到 [-1, 1] 范围，长段会被拆分为多个单位步。
    """

    def __init__(self, step_ratio: float) -> None:
        """
        初始化规划器。

        参数：
            step_ratio (float): 路径长度到动作步长的缩放比例，
                                用于将 RS 路径的实际长度映射为归一化步长。
        """
        self.route = None       # 当前存储的 RS 路径对象，None 表示无激活路径
        self.actions = []       # 待执行的动作队列，每项为 [转向, 步长]
        self.step_ratio = step_ratio  # 路径长度与动作步长之间的缩放因子

    def reset(self):
        """清空当前路径和动作队列，重置规划器到初始状态。"""
        self.route = None
        self.actions.clear()

    def set_rs_path(self, rs_path):
        """
        将一条 Reeds-Shepp 路径解析为可执行的离散动作序列。

        RS 路径由若干段组成，每段有曲线类型（L/S/R）和弧长。
        本方法将每段转换为 [转向值, 步长] 对，并将步长超过 1 的段
        拆分为多个单位步，确保每一步的步长绝对值不超过 1。

        参数：
            rs_path: Reeds-Shepp 路径对象，需包含 ctypes（曲线类型列表）
                     和 lengths（各段弧长列表）两个属性。
        """
        # 将 RS 曲线类型字母映射为转向值：左转=1，直行=0，右转=-1
        action_type = {'L': 1, 'S': 0, 'R': -1}

        self.route = rs_path
        step_ratio = self.step_ratio
        action_list = []

        # 遍历路径的每一段，生成对应的 [转向, 步长] 动作
        for i in range(len(rs_path.ctypes)):
            steer = action_type[rs_path.ctypes[i]]           # 转向值
            step_len = rs_path.lengths[i] / step_ratio       # 归一化步长
            action_list.append([steer, step_len])

        # 将步长超过单位 1 的动作拆分为多个单位步，过滤极小步长（< 1e-3）
        filtered_actions = []
        for action in action_list:
            action[0] *= 1  # 保持转向值不变（保留扩展空间）

            if abs(action[1]) < 1 and abs(action[1]) > 1e-3:
                # 步长在 (1e-3, 1) 范围内，直接加入队列
                filtered_actions.append(action)
            elif action[1] > 1:
                # 正向步长超过 1，拆分为多个步长为 1 的前进动作
                while action[1] > 1:
                    filtered_actions.append([action[0], 1])
                    action[1] -= 1
                if abs(action[1]) > 1e-3:
                    filtered_actions.append(action)  # 追加剩余不足 1 的部分
            elif action[1] < -1:
                # 负向步长（后退）超过 -1，拆分为多个步长为 -1 的后退动作
                while action[1] < -1:
                    filtered_actions.append([action[0], -1])
                    action[1] += 1
                if abs(action[1]) > 1e-3:
                    filtered_actions.append(action)  # 追加剩余不足 -1 的部分

        self.actions = filtered_actions

    def get_action(self):
        """
        从动作队列中取出并返回下一个待执行的动作。

        当队列清空时自动重置规划器（路径执行完毕）。

        返回：
            action (list): [转向值, 步长] 格式的动作。
        """
        action = self.actions.pop(0)  # 弹出队首动作（FIFO）
        # 队列为空且路径仍存在，说明路径已执行完毕，重置状态
        if len(self.actions) == 0 and self.route is not None:
            self.reset()
        return action


class ParkingAgent:
    """
    混合策略泊车智能体，融合 RL 神经网络策略与 Reeds-Shepp 几何规划器。

    决策优先级：
        - 若 RS 规划器持有有效路径（executing_rs=True），则优先执行 RS 动作；
        - 否则，将观测转交给底层 RL 智能体（SAC 或 PPO）做推断。

    对底层 RL 智能体的属性访问通过 __getattr__ 代理实现，
    使调用方无需区分 ParkingAgent 和原始 RL 智能体。
    """

    def __init__(self, rl_agent, planner=None) -> None:
        """
        初始化混合智能体。

        参数：
            rl_agent: 底层强化学习智能体（SACAgent 或 PPOAgent）。
            planner (RsPlanner, 可选): RS 几何路径规划器，为 None 时退化为纯 RL 模式。
        """
        self.agent = rl_agent    # 底层 RL 智能体（SAC/PPO）
        self.planner = planner   # RS 路径规划器（可为 None）

    def __getattr__(self, name: str):
        """
        属性代理：将未在本类定义的公有属性/方法转发给底层 RL 智能体。

        这样外部代码可以直接通过 ParkingAgent 访问 rl_agent 上的属性，
        例如 agent.memory、agent.policy 等，无需手动逐一封装。

        参数：
            name (str): 被访问的属性名。

        异常：
            AttributeError: 访问私有属性（以 '_' 开头）时抛出，
                            防止无限递归等异常行为。
        """
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)

    def reset(self):
        """回合开始时重置规划器状态，清空上一回合残留的 RS 路径和动作队列。"""
        if self.planner is not None:
            self.planner.reset()

    def set_planner_path(self, path=None, forced=False):
        """
        为 RS 规划器设置新的路径。

        参数：
            path: Reeds-Shepp 路径对象；为 None 时不做任何操作。
            forced (bool): 为 True 时强制覆盖当前正在执行的路径；
                           为 False（默认）时仅在规划器无激活路径时才设置。
        """
        if self.planner is None:
            return  # 无规划器，直接返回
        if path is not None and (forced or self.planner.route is None):
            self.planner.set_rs_path(path)

    @property
    def executing_rs(self):
        """
        只读属性：判断当前是否正在执行 RS 几何路径。

        返回：
            bool: True 表示规划器存在且持有有效路径（RS 模式），
                  False 表示应使用 RL 策略决策。
        """
        return not (self.planner is None or self.planner.route is None)

    def get_log_prob(self, obs, action):
        """
        计算给定观测和动作对应的对数概率（供 PPO 计算重要性采样比使用）。

        参数：
            obs (dict): 环境观测字典。
            action: 待评估的动作。

        返回：
            log_prob (Tensor): 该动作在当前策略下的对数概率。
        """
        return self.agent.get_log_prob(obs, action)

    def choose_action(self, obs):
        """
        训练阶段的动作选择接口（含探索噪声）。

        决策逻辑：
            - 若正在执行 RS 路径，则从规划器取出下一个几何动作，
              同时用 RL 智能体计算该动作的对数概率（用于 PPO 更新）；
            - 否则，直接调用 RL 智能体的 choose_action（含随机探索）。

        参数：
            obs (dict): 环境观测字典，键包括 'lidar'、'target'、'img'、'action_mask'。

        返回：
            action (list/np.array): [转向角, 速度] 格式的动作。
            log_prob (Tensor 或 其他): RL 智能体返回的附加信息（如对数概率）。
        """
        if not self.executing_rs:
            # RL 模式：由神经网络策略生成动作（含探索）
            return self.agent.choose_action(obs)
        else:
            # RS 模式：执行几何路径中的下一步动作
            action = self.planner.get_action()
            # 同步计算该 RS 动作在当前策略下的对数概率，用于训练
            log_prob = self.agent.get_log_prob(obs, action)
            return action, log_prob

    def get_action(self, obs):
        """
        评估/推断阶段的动作选择接口（无探索噪声，使用确定性策略）。

        决策逻辑与 choose_action 相同，但底层 RL 智能体采用均值动作，
        不加随机扰动，适用于评估和部署场景。

        参数：
            obs (dict): 环境观测字典，键包括 'lidar'、'target'、'img'、'action_mask'。

        返回：
            action (list/np.array): [转向角, 速度] 格式的动作。
            log_prob (Tensor 或 其他): RL 智能体返回的附加信息（如对数概率）。
        """
        if not self.executing_rs:
            # RL 模式：确定性推断，不加探索噪声
            return self.agent.get_action(obs)
        else:
            # RS 模式：执行几何路径中的下一步动作
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action)
            return action, log_prob

    def push_memory(self, experience):
        """
        将一条经验元组存入 RL 智能体的回放缓冲区。

        参数：
            experience: 经验元组，通常为 (obs, action, reward, next_obs, done)。
        """
        self.agent.push_memory(experience)

    def update(self):
        """
        触发 RL 智能体的一次参数更新（从回放缓冲区采样并反向传播）。

        返回：
            更新统计信息（如损失值），具体格式由底层智能体定义。
        """
        return self.agent.update()

    def save(self, *args, **kwargs):
        """
        保存 RL 智能体的模型权重到文件。

        参数透传给底层智能体的 save 方法（通常接受文件路径字符串）。
        """
        self.agent.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        """
        从文件加载 RL 智能体的模型权重。

        参数透传给底层智能体的 load 方法（通常接受文件路径和设备参数）。
        """
        self.agent.load(*args, **kwargs)