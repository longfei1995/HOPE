'''
CarParking 环境核心实现。

定位：
1) 这是底层 Gym 环境，负责物理推进、碰撞检测、到达判定、奖励计算与渲染。
2) 训练脚本通常通过 env_wrapper 与该类交互，但真实逻辑都在这里。

说明：
- 该环境的 step 第 3 个返回值是 Status 枚举，而不是布尔 done。
- 上层 Wrapper 会将 Status 转换为算法使用的 done/terminated 语义。
'''


import sys
sys.path.append("../")
from typing import Optional, Union
import math
from typing import OrderedDict
import random

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from heapdict import heapdict
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from env.vehicle import *
from env.map_base import *
from env.lidar_simulator import LidarSimlator
from env.parking_map_normal import ParkingMapNormal
from env.parking_map_dlp import ParkingMapDLP
import env.reeds_shepp as rsCurve
from env.observation_processor import Obs_Processor
from model.action_mask import ActionMask
from configs import *

class CarParking(gym.Env):

    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True, 
        use_lidar_observation: bool =USE_LIDAR,
        use_img_observation: bool=USE_IMG,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        super().__init__()

        # 运行与观测开关
        self.verbose = verbose
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.use_action_mask = use_action_mask

        # 渲染控制
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None
        self.level = MAP_LEVEL
        # target 向量维度: [相对距离, cos(theta), sin(theta), cos(phi), sin(phi)]
        self.tgt_repr_size = 5

        # 根据难度级别选择地图生成器
        if self.level in ['Normal', 'Complex', 'Extrem']:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()

        # 车辆、激光雷达仿真器
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)

        # 奖励相关状态（包含 box_union 的累计门控）
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0

        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
        ) # 连续动作: [steer, speed]
       
        # 观测空间是 Dict，按开关动态拼装
        self.observation_space = {}
        if self.use_action_mask:
            self.action_filter = ActionMask()
            self.observation_space['action_mask'] = spaces.Box(low=0, high=1, 
                shape=(N_DISCRETE_ACTION,), dtype=np.float64
            )
        if self.use_img_observation:
            self.img_processor = Obs_Processor()
            self.observation_space['img'] = spaces.Box(low=0, high=255, 
                shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
                self.img_processor.n_channels), dtype=np.uint8
            )
            self.raw_img_shape = (OBS_W, OBS_H, 3)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of 
            # parking lot in the polar coordinate of the ego car's view
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float64
            )
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1])
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )
    
    def set_level(self, level:str=None):
        # 动态切换场景难度；评估脚本会在多个 level 间切换
        if level is None:
            self.map = ParkingMapNormal()
            return
        self.level = level
        if self.level in ['Normal', 'Complex', 'Extrem',]:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()

    def reset(self, case_id: int = None, data_dir: str = None, level: str = None,) -> np.ndarray:
        # 每个 episode 重置奖励累计与时间步
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.t = 0.0

        if level is not None:
            self.set_level(level)

        # map.reset 负责生成起点/终点/障碍；vehicle.reset 负责车辆状态归位
        initial_state = self.map.reset(case_id, data_dir)
        self.vehicle.reset(initial_state)

        # 渲染坐标变换矩阵依赖当前地图边界，reset 后重算
        self.matrix = self.coord_transform_matrix()

        # 复用 step(None) 构造初始观测，确保 reset 与 step 的观测流程一致
        return self.step()[0]

    def coord_transform_matrix(self) -> list:
        """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
        """
        # 仿真坐标 -> 画布坐标的仿射矩阵 [a,b,d,e,xoff,yoff]
        # 其中这里只做等比缩放 + 平移，不做旋转/切变
        k = K
        bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
        by = 0.5 * (WIN_H - k * (self.map.ymax + self.map.ymin))
        self.k = k
        return [k, 0, 0, k, bx, by]

    def _coord_transform(self, object) -> list:
        transformed = affine_transform(object, self.matrix)
        return list(transformed.coords)

    def _detect_collision(self):
        # 车辆包围盒与任一障碍相交则碰撞
        for obstacle in self.map.obstacles:
            if self.vehicle.box.intersects(obstacle.shape):
                return True
        return False
    
    def _detect_outbound(self):
        # 超出地图边界判定
        x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y
        return x>self.map.xmax or x<self.map.xmin or y>self.map.ymax or y<self.map.ymin

    def _check_arrived(self):
        # 以车辆与目标车位区域重叠率作为到达标准（>95%）
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            return True
        return False
    
    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME

    def _check_status(self):
        # 终止优先级：碰撞 > 越界 > 到达 > 超时 > 继续
        # 顺序意味着同一步内若既碰撞又到达，会先记为碰撞。
        if self._detect_collision():
            return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

    def _get_reward(self, prev_state: State, curr_state: State):

        # 1) 时间惩罚：随时间递增，范围接近 (-1, 0)
        time_cost = - np.tanh(self.t / (10*TOLERANT_TIME))

        # 2) RS 距离奖励：基于 Reeds-Shepp 最短路径长度的改变量
        #    由 REWARD_WEIGHT['rs_dist_reward'] 控制是否启用
        if REWARD_WEIGHT['rs_dist_reward'] != 0:
            radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
            curr_rs_dist = rsCurve.calc_optimal_path(*curr_state.get_pos(), *self.map.dest.get_pos(), radius , 0.1).L
            prev_rs_dist = rsCurve.calc_optimal_path(*prev_state.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_norm_ratio = rsCurve.calc_optimal_path(*self.map.start.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_reward = math.exp(-curr_rs_dist/rs_dist_norm_ratio) - \
                math.exp(-prev_rs_dist/rs_dist_norm_ratio)
        else:
            rs_dist_reward = 0

        # 3) 欧氏距离与朝向奖励：均使用“前后状态差分”
        def get_angle_diff(angle1, angle2):
            # norm to 0 ~ pi/2
            angle_dif = math.acos(math.cos(angle1 - angle2)) # 0~pi
            return angle_dif if angle_dif<math.pi/2 else math.pi-angle_dif
        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)
        dist_norm_ratio = max(self.map.dest.loc.distance(self.map.start.loc),10)
        angle_norm_ratio = math.pi
        dist_reward = prev_dist_diff/dist_norm_ratio - dist_diff/dist_norm_ratio
        angle_reward = prev_angle_diff/angle_norm_ratio - angle_diff/angle_norm_ratio
        
        # 4) Box IoU 风格奖励：鼓励车体逐步覆盖目标车位
        #    这里使用累积门控，只有“比历史最好更好”时才给增量奖励，防抖动刷分。
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        box_union_reward = union_area/(2*dest_box.area - union_area)
        if box_union_reward < self.accum_arrive_reward:
            box_union_reward = 0 
        else:
            prev_arrive_reward = self.accum_arrive_reward
            self.accum_arrive_reward = box_union_reward
            box_union_reward -= prev_arrive_reward
        return [time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward]
        
    def get_reward(self, status, prev_state):
        # 仅在 CONTINUE 状态计算 shaped reward；终止状态奖励由上层策略处理
        reward_info = [0,0,0,0,0]
        if status == Status.CONTINUE:
            reward_info = self._get_reward(prev_state, self.vehicle.state)
        return reward_info

    def step(self, action:np.ndarray = None):
        '''
        Parameters:
        ----------
        `action`: `np.ndarray`

        Returns:
        ----------
        ``obsercation`` (Dict): 
            the observation of image based surroundings, lidar view and target representation.
            If `use_lidar_observation` is `True`, then `obsercation['img'] = None`.
            If `use_lidar_observation` is `False`, then `obsercation['lidar'] = None`. 

        ``reward_info`` (OrderedDict): different types of reward information, including:
                time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward
        `status` (`Status`): represent the state of vehicle, including:
                `CONTINUE`, `ARRIVED`, `COLLIDED`, `OUTBOUND`, `OUTTIME`
        `info` (`OrderedDict`): other information.
        '''
        # 注意：该接口返回 (observation, reward_info, status, info)
        # 与标准 Gym 的 reward/done 约定不同，Wrapper 会负责适配。
        assert self.vehicle is not None
        prev_state = self.vehicle.state
        collide = False
        arrive = False
        if action is not None:
            # 一个外部 action 会在底层做 NUM_STEP 次微步仿真，提高动力学稳定性
            for simu_step_num in range(NUM_STEP):
                prev_info = self.vehicle.step(action,step_time=1)
                if self._check_arrived():
                    arrive = True
                    break
                if self._detect_collision():
                    # 碰撞时回退到上一个可行状态，避免轨迹穿透障碍
                    if simu_step_num == 0:
                        collide = ENV_COLLIDE
                        self.vehicle.retreat(prev_info)
                    else:
                        self.vehicle.retreat(prev_info)
                    simu_step_num -= 1
                    break
            simu_step_num += 1
            # remove redundant trajectory
            if simu_step_num > 1:
                del self.vehicle.trajectory[-simu_step_num:-1]

        # 这里的 t 是环境“高层步”，不是 NUM_STEP 的微步计数
        self.t += 1
        observation = self.render(self.render_mode)
        if arrive:
            status = Status.ARRIVED
        else:
            status = Status.COLLIDED if collide else self._check_status()

        reward_list = self.get_reward(status, prev_state)
        reward_info = OrderedDict({'time_cost':reward_list[0],\
            'rs_dist_reward':reward_list[1],\
            'dist_reward':reward_list[2],\
            'angle_reward':reward_list[3],\
            'box_union_reward':reward_list[4],})

        info = OrderedDict({'reward_info':reward_info,
            'path_to_dest':None})

        # 当靠近目标且仍在继续状态时，尝试求解一条无碰 RS 路径。
        # 找到后交由上层 ParkingAgent 决定是否切换为规划器执行。
        if self.t > 1 and status==Status.CONTINUE\
            and self.vehicle.state.loc.distance(self.map.dest.loc)<RS_MAX_DIST:
            rs_path_to_dest = self.find_rs_path(status)
            if rs_path_to_dest is not None:
                info['path_to_dest'] = rs_path_to_dest

        return observation, reward_info, status, info

    def _render(self, surface: pygame.Surface):
        # 渲染顺序：背景 -> 障碍 -> 起终点 -> 车辆 -> 历史轨迹
        surface.fill(BG_COLOR)
        for obstacle in self.map.obstacles:
            pygame.draw.polygon(
                surface, OBSTACLE_COLOR, self._coord_transform(obstacle.shape))

        pygame.draw.polygon(
            surface, START_COLOR, self._coord_transform(self.map.start_box), width=1)
        pygame.draw.polygon(
            surface, DEST_COLOR, self._coord_transform(self.map.dest_box))
        
        pygame.draw.polygon(
            surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

        if RENDER_TRAJ and len(self.vehicle.trajectory) > 1:
            render_len = min(len(self.vehicle.trajectory), TRAJ_RENDER_LEN)
            for i in range(render_len):
                vehicle_box = self.vehicle.trajectory[-(render_len-i)].create_box()
                pygame.draw.polygon(
                    surface, TRAJ_COLORS[-(render_len-i)], self._coord_transform(vehicle_box))

    def _get_img_observation(self, surface: pygame.Surface):
        # 以车辆朝向对全局画面做逆向旋转，再裁剪 ego-centered 观测窗
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((WIN_W, WIN_H))
        rotate.blit(capture, capture.get_rect(center=old_center))
        
        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0])
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) \
            + (vehicle_center[1]-old_center[1])*np.sin(angle)
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) \
            + (vehicle_center[1]-old_center[1])*np.cos(angle)
        
        # 将车辆中心对齐到观测图像中心，得到稳定的自车视角
        observation = pygame.Surface((WIN_W, WIN_H))
    
        observation.fill(BG_COLOR)
        observation.blit(rotate, (int(-dx), int(-dy)))
        observation = observation.subsurface((
            (WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

    
        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.raw_img_shape)

        return observation

    def _process_img_observation(self, img):
        '''
        Process the img into channels of different information.

        Parameters
        ------
        img (np.ndarray): RGB image of shape (OBS_W, OBS_H, 3)

        Returns
        ------
        processed img (np.ndarray): shape (OBS_W//downsample_rate, OBS_H//downsample_rate, n_channels )
        '''
        processed_img = self.img_processor.process_img(img)
        return processed_img

    def _get_lidar_observation(self,):
        # 基于当前障碍轮廓做激光扫描
        obs_list = [obs.shape for obs in self.map.obstacles]
        lidar_view = self.lidar.get_observation(self.vehicle.state, obs_list)
        return lidar_view
    
    def _get_targt_repr(self,):
        # 目标表示向量：
        # [距离, 相对方位角 cos/sin, 目标朝向相对角 cos/sin]
        # 用 sin/cos 编码角度可避免角度在 +/-pi 处不连续。
        dest_pos = (self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading)
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def render(self, mode: str = "human"):
        assert mode in self.metadata["render_mode"]
        assert self.vehicle is not None

        if mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render(self.screen)
        # 固定输出字段，方便上层模型按键取值
        observation = {'img':None, 'lidar':None, 'target':None, 'action_mask':None}
        if self.use_img_observation:
            raw_observation = self._get_img_observation(self.screen)
            observation['img'] = self._process_img_observation(raw_observation)
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation()
        if self.use_action_mask:
            observation['action_mask'] = self.action_filter.get_steps(observation['lidar'])
        observation['target'] = self._get_targt_repr()
        pygame.display.update()
        self.clock.tick(self.fps)
        
        return observation

    def find_rs_path(self,status):
        '''
        Find collision-free RS path. 

        Returns:
            path (PATH): the related PATH object which is collision-free.
        '''
        startX, startY, startYaw = self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading
        goalX, goalY, goalYaw = self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading
        radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
        # 1) 枚举所有 RS 候选路径
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 0.1)

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None

        # 2) 以路径长度为代价放入优先队列（短路径优先）
        costQueue = heapdict()
        for path in reedsSheppPaths:
            costQueue[path] = path.L

        # 3) 从短到长检查碰撞，返回第一条可行路径
        #    额外约束：当长度超过最短路径 1.6 倍且已检查若干条后提前停止
        min_path_len = -1
        idx = 0
        while len(costQueue)!=0:
            idx += 1
            path = costQueue.popitem()[0]
            if min_path_len < 0:
                min_path_len = path.L
            if path.L > 1.6*min_path_len and idx > 2:
                break
            traj=[]
            traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
            traj_valid = self.is_traj_valid(traj)
            if traj_valid:
                return path
        return None
    
    def is_traj_valid(self, traj):
        # 向量化碰撞检测：
        # 将“每帧车辆四条边”与“附近障碍所有边”统一转为线段相交检测。
        # 这样可避免逐帧逐边 Python 循环，提高 RS 轨迹验证速度。
        car_coords1 = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords2 = np.array(VehicleBox.coords)[1:] # (4,2)
        car_coords_x1 = car_coords1[:,0].reshape(1,-1)
        car_coords_y1 = car_coords1[:,1].reshape(1,-1) # (1,4)
        car_coords_x2 = car_coords2[:,0].reshape(1,-1)
        car_coords_y2 = car_coords2[:,1].reshape(1,-1) # (1,4)
        vxs = np.array([t[0] for t in traj])
        vys = np.array([t[1] for t in traj])
        # 先做边界盒快速拒绝，越界直接失败
        if np.min(vxs) < self.map.xmin or np.max(vxs) > self.map.xmax \
        or np.min(vys) < self.map.ymin or np.max(vys) > self.map.ymax:
            return False
        vthetas = np.array([t[2] for t in traj])
        cos_theta = np.cos(vthetas).reshape(-1,1) # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1,1)
        vehicle_coords_x1 = cos_theta*car_coords_x1 - sin_theta*car_coords_y1 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y1 = sin_theta*car_coords_x1 + cos_theta*car_coords_y1 + vys.reshape(-1,1)
        vehicle_coords_x2 = cos_theta*car_coords_x2 - sin_theta*car_coords_y2 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y2 = sin_theta*car_coords_x2 + cos_theta*car_coords_y2 + vys.reshape(-1,1)
        vx1s = vehicle_coords_x1.reshape(-1,1)
        vx2s = vehicle_coords_x2.reshape(-1,1)
        vy1s = vehicle_coords_y1.reshape(-1,1)
        vy2s = vehicle_coords_y2.reshape(-1,1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1,1) # (4*t,1)
        b = (vx1s - vx2s).reshape(-1,1)
        c = (vy1s*vx2s - vx1s*vy2s).reshape(-1,1)
        
        # 仅收集车辆轨迹包围盒附近障碍，减少无关计算
        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x_max = np.max(vx1s) + 5
        x_min = np.min(vx1s) - 5
        y_max = np.max(vy1s) + 5
        y_min = np.min(vy1s) - 5

        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in self.map.obstacles:
            if isinstance(obst, Area):
                obst = obst.shape
            obst_coords = np.array(obst.coords) # (n+1,2)
            if (obst_coords[:,0] > x_max).all() or (obst_coords[:,0] < x_min).all()\
                or (obst_coords[:,1] > y_max).all() or (obst_coords[:,1] < y_min).all():
                continue
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return True
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # 线段相交核心：解两直线交点，再过滤“不在线段范围内”的假交点
        # 若任一有效交点存在，则判定轨迹碰撞。
        det = a*e - b*d # (4, E)
        parallel_line_pos = (det==0) # (4, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (4, E)
        raw_y = (c*d - a*f)/det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        collide_map_x[raw_x>np.maximum(x1s, x2s)] = 0
        collide_map_x[raw_x<np.minimum(x1s, x2s)] = 0
        collide_map_y[raw_y>np.maximum(y1s, y2s)] = 0
        collide_map_y[raw_y<np.minimum(y1s, y2s)] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(vx1s, vx2s)] = 0
        collide_map_x[raw_x<np.minimum(vx1s, vx2s)] = 0
        collide_map_y[raw_y>np.maximum(vy1s, vy2s)] = 0
        collide_map_y[raw_y<np.minimum(vy1s, vy2s)] = 0

        collide_map = collide_map_x*collide_map_y
        collide_map[parallel_line_pos] = 0
        collide = np.sum(collide_map) > 0

        if collide:
            return False
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()

