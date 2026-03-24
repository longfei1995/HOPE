"""
Reeds-Shepp 曲线规划模块
======================
实现了 Reeds-Shepp（RS）最短路径规划算法。

RS 曲线是能以最大曲率 maxc 行驶、同时允许前进与后退的车辆的最短路径集合。
每条路径由若干段基元组成，基元类型为：
  'S'（直线）、'L'（左转圆弧）、'R'（右转圆弧）。
路径长度的符号表示行驶方向：正数 → 前进，负数 → 后退。

参考文献：
  J.A. Reeds & L.A. Shepp, "Optimal paths for a car that goes both forwards
  and backwards", Pacific J. Math., 1990.

主要对外接口：
  calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc) → PATH
  calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc)    → list[PATH]
"""

import math
import numpy as np

# --------------------------------------------------------------------------- #
# 全局参数
# --------------------------------------------------------------------------- #
STEP_SIZE = 0.2  # 路径离散化步长（归一化坐标系下），单位与 1/maxc 一致
MAX_LENGTH = 1000.0  # 单条路径总长度上限，超过此值则丢弃（避免无意义的超长路径）
PI = math.pi


# --------------------------------------------------------------------------- #
# 路径数据结构
# --------------------------------------------------------------------------- #
class PATH:
    """存储一条 Reeds-Shepp 路径的完整信息。

    属性
    ----
    lengths    : list[float]  各段的有符号长度（正→前进，负→后退），归一化坐标系，单位 maxc·rad 或 maxc·m
    ctypes     : list[str]    各段的类型，每个元素为 'S'、'L' 或 'R'
    L          : float        路径总长度（各段绝对值之和），归一化坐标系
    x          : list[float]  全局坐标系下路径点的 x 坐标 [m]
    y          : list[float]  全局坐标系下路径点的 y 坐标 [m]
    yaw        : list[float]  全局坐标系下路径点的航向角 [rad]
    directions : list[int]    各路径点的行驶方向（1 = 前进，-1 = 后退）
    """

    def __init__(self, lengths, ctypes, L, x, y, yaw, directions):
        self.lengths = lengths  # 各段有符号长度（+ 前进，- 后退）[float]
        self.ctypes = ctypes  # 各段路径类型：'S'直线 / 'L'左弧 / 'R'右弧 [string]
        self.L = L  # 路径总长度（各段长度绝对值之和）[float]
        self.x = x  # 全局坐标系下各路径点的 x 坐标 [m]
        self.y = y  # 全局坐标系下各路径点的 y 坐标 [m]
        self.yaw = yaw  # 全局坐标系下各路径点的航向角 [rad]
        self.directions = directions  # 行驶方向（1 = 前进，-1 = 后退）


# --------------------------------------------------------------------------- #
# 对外主接口
# --------------------------------------------------------------------------- #


def calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    """计算从起点到终点的最短 Reeds-Shepp 路径。

    参数
    ----
    sx, sy, syaw : 起点位置 [m] 与航向角 [rad]
    gx, gy, gyaw : 终点位置 [m] 与航向角 [rad]
    maxc         : 车辆最大曲率 [1/m]（= 1 / 最小转弯半径）
    step_size    : 归一化坐标系下的离散步长（默认 STEP_SIZE）

    返回
    ----
    PATH : 总长度最短的那条 RS 路径对象
    """
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=step_size)

    # 遍历所有候选路径，找到总长度最小的那条
    minL = paths[0].L
    mini = 0

    for i in range(len(paths)):
        if paths[i].L <= minL:
            minL, mini = paths[i].L, i

    return paths[mini]


def calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    """枚举从起点到终点所有合法的 Reeds-Shepp 路径，并将坐标转换回全局系。

    流程
    ----
    1. 以起点为原点，将终点坐标投影到车辆局部坐标系，并用 maxc 归一化。
    2. 调用 generate_path 枚举所有 RS 路径族（SCS / CSC / CCC / CCCC / CCSC / CCSCC）。
    3. 对每条路径调用 generate_local_course 生成离散路径点（局部归一化系）。
    4. 将路径点旋转回全局坐标系，并将长度从归一化单位还原为实际米数。

    返回
    ----
    list[PATH] : 所有合法路径的列表（至少 1 条）
    """
    q0 = [sx, sy, syaw]  # 起点状态 [x, y, yaw]
    q1 = [gx, gy, gyaw]  # 终点状态 [x, y, yaw]

    # 在归一化局部坐标系中生成所有 RS 路径（仅含参数，无离散点）
    paths = generate_path(q0, q1, maxc)

    for path in paths:
        # 在局部归一化坐标系中生成离散路径点
        x, y, yaw, directions = generate_local_course(
            path.L, path.lengths, path.ctypes, maxc, step_size * maxc
        )

        # 将局部归一化坐标点旋转回全局坐标系（逆旋转 + 平移）
        path.x = [
            math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0]
            for (ix, iy) in zip(x, y)
        ]
        path.y = [
            -math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1]
            for (ix, iy) in zip(x, y)
        ]
        # 航向角叠加起点航向并归一化到 [-π, π]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        # 将归一化长度（maxc·m 或 maxc·rad）还原为实际长度 [m] 或 [rad]
        path.lengths = [l / maxc for l in path.lengths]
        path.L = path.L / maxc

    return paths


def set_path(paths, lengths, ctypes):
    """验证候选路径，如果合适则加入路径集合 paths。

    重复性检查：若集合中已存在类型相同且各段长度之差 ≤ 0.01 的路径，则跳过。
    长度上限检查：路径总长度超过 MAX_LENGTH 时丢弃。

    参数
    ----
    paths   : 当前路径集合（list[PATH]），会被原地修改后返回
    lengths : 各段有符号长度列表（归一化）
    ctypes  : 各段类型列表，元素为 'S'/'L'/'R'

    返回
    ----
    list[PATH] : 更新后的路径集合
    """
    new_path = PATH([], [], 0.0, [], [], [], [])
    new_path.ctypes = ctypes
    new_path.lengths = lengths

    # 检查是否已存在相同路径，避免重复添加
    for path_exist in paths:
        if path_exist.ctypes == new_path.ctypes:
            total_length_diff = 0.0
            for len_exist, len_new in zip(path_exist.lengths, new_path.lengths):
                total_length_diff += len_exist - len_new
            if abs(total_length_diff) <= 0.01:
                return paths  # 已存在相似路径，不再插入

    # 计算路径总长度（各段绝对值之和）
    new_path.L = sum([abs(i) for i in lengths])

    # 丢弃超长路径
    if new_path.L >= MAX_LENGTH:
        return paths

    assert new_path.L >= 0.001  # 路径不能退化为零长度
    paths.append(new_path)

    return paths


# --------------------------------------------------------------------------- #
# 基本路径基元：CSC 族（Curve-Straight-Curve）
# 以下函数均在以起点为原点、以 maxc 归一化的局部坐标系中工作。
# 输入 (x, y, phi) 为归一化后的终点相对位置与航向差。
# t, u, v 为各段的归一化有符号弧长（正→前进，负→后退）。
# 返回值：(flag, t, u, v)，flag=True 表示该基元存在合法解。
# --------------------------------------------------------------------------- #


def LSL(x, y, phi):
    """左弧–直线–左弧（Left-Straight-Left）基元求解。

    LSL 是最基本的 CSC 路径之一。先左转弧长 t，再直行 u，再左转 v。
    公式来自 RS 论文 8.1 节。

    参数：x, y 为归一化终点坐标；phi 为归一化航向差。
    返回：(True, t, u, v) 或 (False, 0, 0, 0)
    """
    # 将终点位置变换到以起点左转圆心为参考的极坐标
    u, t = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if t >= 0.0:  # t ≥ 0 保证第一段弧为前进方向
        v = M(phi - t)  # 第三段弧长，由航向差约束得出
        if v >= 0.0:  # v ≥ 0 保证第三段弧为前进方向
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LSR(x, y, phi):
    """左弧–直线–右弧（Left-Straight-Right）基元求解。

    先左转弧长 t，再直行 u，再右转 v。
    公式来自 RS 论文 8.2 节。

    返回：(True, t, u, v) 或 (False, 0, 0, 0)
    """
    u1, t1 = R(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1**2  # u1 此时是距离的平方

    if u1 >= 4.0:  # 两圆心距离足够大才有解
        u = math.sqrt(u1 - 4.0)  # 直线段实际长度（归一化）
        theta = math.atan2(2.0, u)  # 辅助角
        t = M(t1 + theta)  # 第一段弧长
        v = M(t - phi)  # 第三段弧长

        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRL(x, y, phi):
    """左弧–右弧–左弧（Left-Right-Left）基元求解（CCC 族基础）。

    三段全为圆弧，中间弧方向相反。
    公式来自 RS 论文 8.3 节。

    返回：(True, t, u, v) 或 (False, 0, 0, 0)，其中 u ≤ 0（后退右弧）
    """
    u1, t1 = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if u1 <= 4.0:  # 两圆心距离足够近才有解
        u = -2.0 * math.asin(0.25 * u1)  # 中间弧长（负值，表示后退）
        t = M(t1 + 0.5 * u + PI)  # 第一段弧长
        v = M(phi - t + u)  # 第三段弧长

        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


# --------------------------------------------------------------------------- #
# SCS 族（Straight-Curve-Straight）
# --------------------------------------------------------------------------- #


def SCS(x, y, phi, paths):
    """枚举所有直线–圆弧–直线（S-C-S）类型路径，并加入 paths 集合。

    通过对称变换同时涵盖 SLS（左弧居中）和 SRS（右弧居中）两种情况：
      SLS(x,  y,  phi) → 路径类型 ["S", "L", "S"]
      SLS(x, -y, -phi) → 路径类型 ["S", "R", "S"]（坐标翻转等价于右转）
    """
    flag, t, u, v = SLS(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "L", "S"])

    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])

    return paths


def SLS(x, y, phi):
    """直线–左弧–直线（Straight-Left-Straight）基元求解。

    要求 y > 0（在左侧）且 0 < phi < π，来保证几何构型合法。
    公式通过圆弧切线几何推导得出。

    参数：x, y 为归一化终点坐标；phi 为归一化航向差。
    返回：(True, t, u, v) 或 (False, 0, 0, 0)
    """
    phi = M(phi)  # 先将航向差归一化到 [-π, π]

    if y > 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x  # 圆弧切点的 x 坐标
        t = xd - math.tan(phi / 2.0)  # 第一段直线长度
        u = phi  # 圆弧转角（= 路径长度，归一化后相等）
        v = math.sqrt((x - xd) ** 2 + y**2) - math.tan(phi / 2.0)  # 第二段直线长度
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < PI * 0.99:
        # y < 0 时第三段直线方向取反（后退）
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd) ** 2 + y**2) - math.tan(phi / 2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0


# --------------------------------------------------------------------------- #
# CSC 族（Curve-Straight-Curve），三段路径
# --------------------------------------------------------------------------- #


def CSC(x, y, phi, paths):
    """枚举所有三段 曲线-直线-曲线（C-S-C）路径并加入 paths。

    利用坐标翻转对称性，从 LSL 和 LSR 两个基元派生出 8 种变体：
      LSL(x,  y,  phi)  → L-S-L（前进前进前进）
      LSL(-x, y, -phi)  → L-S-L（各段取反 → 整体后退）
      LSL(x, -y, -phi)  → R-S-R（左右对称）
      LSL(-x,-y,  phi)  → R-S-R（后退对称）
      LSR(x,  y,  phi)  → L-S-R
      LSR(-x, y, -phi)  → L-S-R（后退）
      LSR(x, -y, -phi)  → R-S-L
      LSR(-x,-y,  phi)  → R-S-L（后退）
    """
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "L"])

    # x 取反 → 终点在起点后方 → 整条路径各段长度取反（后退行驶）
    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "L"])

    # y 取反 → 左右镜像 → L 变 R
    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])

    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])

    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "R"])

    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "R"])

    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "L"])

    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "L"])

    return paths


# --------------------------------------------------------------------------- #
# CCC 族（Curve-Curve-Curve），三段全为圆弧
# --------------------------------------------------------------------------- #


def CCC(x, y, phi, paths):
    """枚举所有三段全圆弧（C-C-C）路径，基于 LRL 基元派生 8 种变体。

    前 4 种为正向（起点→终点），后 4 种为反向（终点→起点，通过反向坐标变换实现）：
      正向：LRL(x,  y,  phi) / 镜像 / 左右翻转 / 两者同时
      反向：先将坐标系变换到以终点朝向为参考，再求解，得到的 t,u,v 顺序倒置
    """
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "R", "L"])

    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "R", "L"])

    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "L", "R"])

    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "L", "R"])

    # 反向路径：将坐标系旋转到终点参考系
    # xb, yb 是在以终点朝向为 x 轴的坐标系中，起点相对于终点的位置
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    # 反向求解：LRL 的 t,u,v 顺序要倒置（v,u,t），因为路径是从终点到起点
    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["L", "R", "L"])

    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["L", "R", "L"])

    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "L", "R"])

    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "L", "R"])

    return paths


# --------------------------------------------------------------------------- #
# CCCC 族（四段全圆弧），需要辅助函数 calc_tauOmega / LRLRn / LRLRp
# --------------------------------------------------------------------------- #


def calc_tauOmega(u, v, xi, eta, phi):
    """计算四段 LRLR 路径中第一段和最后一段的弧长 tau 和 omega。

    用于 LRLRn 和 LRLRp 两个基元的辅助计算，根据几何约束方程联立求解。

    参数
    ----
    u, v   : 中间两段弧长（已知）
    xi, eta: 归一化坐标系中的辅助量（与终点位置有关）
    phi    : 航向差

    返回
    ----
    tau   : 第一段弧长（归一化）
    omega : 最后一段弧长（归一化）
    """
    delta = M(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1.0

    # 通过 atan2 获取辅助角 t1
    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    # 判断需要加 π 还是直接取值（取决于几何关系的凸凹性）
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0

    if t2 < 0:
        tau = M(t1 + PI)
    else:
        tau = M(t1)

    omega = M(tau - u + v - phi)

    return tau, omega


def LRLRn(x, y, phi):
    """四段 LRLR 路径基元，中间两弧反向（n = negative，中间弧向内收缩）。

    解满足 rho ≤ 1 的构型，对应两圆部分重叠的情况。
    公式来自 RS 论文 8.7 节（LRLRn 类型）。

    返回：(True, t, u, v) 或 (False, 0, 0, 0)
      t : 第一段弧长（L 弧，前进）
      u : 第二段弧长（R 弧，>=0 即前进，实际在 CCCC 中使用 -u 作为第三段）
      v : 第四段弧长（R 弧，≤0 即后退）
    """
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))

    if rho <= 1.0:
        u = math.acos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRLRp(x, y, phi):
    """四段 LRLR 路径基元，中间两弧同向（p = positive，两弧等长展开）。

    解满足 0 ≤ rho ≤ 1 且 u ≥ -π/2 的构型，中间两弧弧长相等（都为 u）。
    公式来自 RS 论文 8.8 节（LRLRp 类型）。

    返回：(True, t, u, v) 或 (False, 0, 0, 0)
      t : 第一段弧长（L 弧，前进）
      u : 第二/三段弧长（u ≤ 0，后退弧，两段等长）
      v : 第四段弧长（R 弧，前进）
    """
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0  # 归一化距离参数

    if 0.0 <= rho <= 1.0:  # 距离约束：需在合法范围内
        u = -math.acos(rho)  # 中间弧长（负值，表示后退）
        if u >= -0.5 * PI:  # 保证弧长不超过半圆
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCCC(x, y, phi, paths):
    """枚举所有四段全圆弧（C-C-C-C）路径，基于 LRLRn 和 LRLRp 两个基元派生 8 种变体。

    LRLRn 基元（中间两弧反向）：第三段弧长为 -u（与第二段方向相反）
      [t, u, -u, v] - 4段长度，中间两段互反
    LRLRp 基元（中间两弧同向）：第二、三段等长
      [t, u, u, v]  - 4段长度，中间两段等长
    每种基元通过坐标翻转（x→-x、y→-y）派生出 4 种对称变体。
    """
    # LRLRn 基元的 4 种变体
    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(-x, y, -phi)  # 后退等价
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(x, -y, -phi)  # 左右镜像 → 类型变为 RLRL
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRn(-x, -y, phi)  # 后退镜像
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["R", "L", "R", "L"])

    # LRLRp 基元的 4 种变体
    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["R", "L", "R", "L"])

    return paths


# --------------------------------------------------------------------------- #
# CCSC 族辅助基元：左弧-右弧-直线-弧（C-C-S-C，四段路径）
# --------------------------------------------------------------------------- #


def LRSR(x, y, phi):
    """左弧–右弧–直线–右弧（L-R-S-R）基元，CCSC 族的子类型之一。

    中间右弧固定为 -π/2（半圆），因此参数 t/u/v 对应：
      t : 第一段 L 弧长（前进，≥0）
      u : 直线段长度（≤0，后退直线）
      v : 最后 R 弧长（≤0，后退弧）

    返回：(True, t, u, v) 或 (False, 0, 0, 0)
    """
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(-eta, xi)  # 极坐标转换，获取辅助距离和角度

    if rho >= 2.0:  # 两圆心距离足以容纳 π/2 弧
        t = theta
        u = 2.0 - rho  # 直线段（负值表示后退）
        v = M(t + 0.5 * PI - phi)  # 末段弧长
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRSL(x, y, phi):
    """左弧–右弧–直线–左弧（L-R-S-L）基元，CCSC 族的子类型之一。

    中间右弧固定为 -π/2（半圆），参数 t/u/v 对应：
      t : 第一段 L 弧长（前进，≥0）
      u : 直线段长度（≤0，后退直线）
      v : 最后 L 弧长（≤0，后退弧）

    返回：(True, t, u, v) 或 (False, 0, 0, 0)
    """
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = R(xi, eta)  # 极坐标，获取辅助距离 rho 和角度 theta

    if rho >= 2.0:  # 两圆心距离需足够大
        r = math.sqrt(rho * rho - 4.0)  # 辅助量：r 对应直线段
        u = 2.0 - r  # 直线段长度（≤0）
        t = M(theta + math.atan2(r, -2.0))  # 第一段弧长
        v = M(phi - 0.5 * PI - t)  # 末段弧长
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


# --------------------------------------------------------------------------- #
# CCSC 族（四段路径：曲线-曲线-直线-曲线）
# --------------------------------------------------------------------------- #


def CCSC(x, y, phi, paths):
    """枚举所有四段 曲线-曲线-直线-曲线（C-C-S-C）路径，共 16 种变体。

    分两组：
      正向（8 种）：基于 LRSL / LRSR 基元，中间弧固定为 ±π/2，通过坐标翻转覆盖所有方向。
      反向（8 种）：将坐标系旋转到终点参考系后求解 LRSL / LRSR，t/v 对调得到反向路径。

    正向路径类型（第2段固定为 ±π/2）：
      LRSL → L-R-S-L  LRSR → L-R-S-R  （及镜像 R-L-S-R / R-L-S-L）
    反向路径类型（第3段固定为 ±π/2）：
      → L-S-R-L / R-S-L-R / R-S-R-L / L-S-L-R
    """
    # ---- 正向路径（基于 LRSL / LRSR 基元）----
    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(-x, y, -phi)  # 后退等价
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(x, -y, -phi)  # 左右镜像 → R-L-S-R
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(x, -y, -phi)  # 左右镜像 → R-L-S-L
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "L", "S", "L"])

    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "L", "S", "L"])

    # ---- 反向路径：将坐标系旋转到终点参考系后求解 ----
    # xb, yb 是以终点航向为 x 轴的坐标系中，起点相对于终点的位置
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    # 反向路径中 t 和 v 位置对调（CSRC 变为 CRSC → C-S-C-C 格式）
    flag, t, u, v = LRSL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["L", "S", "L", "R"])

    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["L", "S", "L", "R"])

    return paths


# --------------------------------------------------------------------------- #
# CCSCC 族辅助基元（五段路径）
# --------------------------------------------------------------------------- #


def LRSLR(x, y, phi):
    """左弧–右弧–直线–左弧–右弧（L-R-S-L-R）基元，CCSCC 族的核心子类型。

    中间两段弧固定为 ±π/2，因此5段路径由 t, u, v 参数化：
      [t, -π/2, u, -π/2, v] → 类型 L-R-S-L-R

    注意：原始 RS 论文 8.11 节中存在笔误（*** TYPO IN PAPER ***），
    此处实现已修正。

    返回：(True, t, u, v) 或 (False, 0, 0, 0)
    """
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(xi, eta)  # 极坐标，获取辅助距离 rho 和角度 theta

    if rho >= 2.0:  # 距离约束
        u = 4.0 - math.sqrt(rho * rho - 4.0)  # 直线段长度（≤0 表示后退）
        if u <= 0.0:
            # 通过 atan2 从几何约束中解出第一段弧长 t
            t = M(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = M(t - phi)  # 末段弧长

            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


# --------------------------------------------------------------------------- #
# CCSCC 族（五段路径：曲线-曲线-直线-曲线-曲线）
# --------------------------------------------------------------------------- #


def CCSCC(x, y, phi, paths):
    """枚举所有五段 曲线-曲线-直线-曲线-曲线（C-C-S-C-C）路径，共 4 种变体。

    中间两段弧固定为 ±π/2（各 90°），因此路径形如：
      [t, -π/2, u, -π/2, v]  类型 L-R-S-L-R
    通过坐标翻转覆盖 后退 / 左右对称 / 两者同时 共 4 种情况。
    """
    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path(
            paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["L", "R", "S", "L", "R"]
        )

    flag, t, u, v = LRSLR(-x, y, -phi)  # 后退等价
    if flag:
        paths = set_path(
            paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["L", "R", "S", "L", "R"]
        )

    flag, t, u, v = LRSLR(x, -y, -phi)  # 左右镜像 → R-L-S-R-L
    if flag:
        paths = set_path(
            paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["R", "L", "S", "R", "L"]
        )

    flag, t, u, v = LRSLR(-x, -y, phi)  # 后退 + 镜像
    if flag:
        paths = set_path(
            paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["R", "L", "S", "R", "L"]
        )

    return paths


# --------------------------------------------------------------------------- #
# 路径离散化：将参数化路径转换为实际坐标序列
# --------------------------------------------------------------------------- #


def generate_local_course(L, lengths, mode, maxc, step_size):
    """在局部归一化坐标系中，将 RS 路径参数转换为离散路径点序列。

    算法
    ----
    1. 预分配足够大的数组：最多 int(L/step_size) + len(lengths) + 3 个点。
    2. 逐段处理，每段沿弧长方向以 step_size 为步长调用 interpolate 插值。
    3. 段与段之间通过 ll（剩余弧长）保证连接处精确对齐，避免累积误差。
    4. 清除末尾未使用的零值槽位。

    参数
    ----
    L         : 路径总长度（归一化，各段 |lengths| 之和）
    lengths   : 各段有符号弧长列表（正→前进，负→后退）
    mode      : 各段类型列表（'S'/'L'/'R'）
    maxc      : 最大曲率，用于将归一化曲率转换为实际曲率
    step_size : 归一化坐标系下的采样步长（= STEP_SIZE × maxc）

    返回
    ----
    px, py, pyaw, directions : 路径点的 x/y 坐标、航向角、行驶方向列表
    """
    # 预分配点数组（含冗余，后续会截断）
    point_num = int(L / step_size) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]  # 各路径点 x 坐标（归一化）
    py = [0.0 for _ in range(point_num)]  # 各路径点 y 坐标（归一化）
    pyaw = [0.0 for _ in range(point_num)]  # 各路径点航向角
    directions = [0 for _ in range(point_num)]  # 各路径点行驶方向
    ind = 1  # 当前写入位置（0 号点为起点，从 1 开始填充）

    # 初始化起点方向（由第一段弧长符号决定）
    if lengths[0] > 0.0:
        directions[0] = 1  # 第一段前进
    else:
        directions[0] = -1  # 第一段后退

    # 初始化步进方向
    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    pd = d  # 当前段内的累积弧长（从上一段末尾校正值开始）
    ll = 0.0  # 上一段产生的剩余弧长（用于保证连接处精度）

    for m, l, i in zip(mode, lengths, range(len(mode))):
        # 根据当前段符号决定步进方向
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        # 保存当前段起点坐标（从上一段末尾继承）
        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1  # 回退一格，下面的循环会从 ind+1 开始写
        # 计算本段的起始插值参数 pd，需要补偿上一段剩余的弧长 ll
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            # 相邻两段行驶方向相同，不需要换向补偿
            pd = -d - ll
        else:
            # 换向或第一段：pd 从 step_size 减去上一段剩余量
            pd = d - ll

        # 在当前段内均匀插值，直到超过段长度
        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = interpolate(
                ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions
            )
            pd += d

        # 计算本段产生的剩余弧长（用于下一段起点对齐）
        ll = l - pd - d  # calc remain length

        # 精确插入段末尾节点（保证终点坐标准确）
        ind += 1
        px, py, pyaw, directions = interpolate(
            ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions
        )

    # 清除末尾未使用的零值槽位（预分配的冗余空间）
    while px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    """计算路径上第 ind 个点的坐标、航向角和行驶方向。

    参数
    ----
    ind        : 待写入的数组下标
    l          : 到本段起点的有符号归一化弧长（负值表示后退方向行驶）
    m          : 本段路径类型 ('S' / 'L' / 'R')
    maxc       : 最大曲率 [1/m]
    ox, oy, oyaw : 本段起点坐标（归一化）和航向角
    px, py, pyaw, directions : 全局路径点数组（原地写入）

    返回
    ----
    更新后的 px, py, pyaw, directions 数组
    """
    if m == "S":  # 直线段：按当前航向前进 l/maxc 米
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw  # 直线不改变航向
    else:  # 圆弧段：通过弦长+横移量计算终点
        # 局部坐标系（以本段起点为原点，起点航向为 x 轴）下的位移
        ldx = math.sin(l) / maxc  # 切向分量（= 弦长在 x 方向投影）
        if m == "L":  # 左转：横向偏移为正
            ldy = (1.0 - math.cos(l)) / maxc
        elif m == "R":  # 右转：横向偏移取反
            ldy = (1.0 - math.cos(l)) / (-maxc)

        # 将局部坐标旋转到全局坐标系（以起点起点航向为基准）
        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    # 更新航向角：左转累加 l，右转累减 l（弧度等于转角）
    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    # 行驶方向由弧长符号决定
    if l > 0.0:
        directions[ind] = 1  # 前进
    else:
        directions[ind] = -1  # 后退

    return px, py, pyaw, directions


# --------------------------------------------------------------------------- #
# 路径生成入口
# --------------------------------------------------------------------------- #


def generate_path(q0, q1, maxc):
    """将全局起终点坐标变换到局部归一化坐标系，并枚举所有 RS 路径族。

    变换说明
    --------
    以起点 q0 为原点，以起点航向为 x 轴建立局部坐标系，
    并用 maxc（最大曲率）对距离归一化，使算法结果与转弯半径无关。
    最终坐标 (x, y) 及航向差 dth 即为各路径基元函数的输入。

    参数
    ----
    q0 : [sx, sy, syaw] 起点状态
    q1 : [gx, gy, gyaw] 终点状态
    maxc : 最大曲率 [1/m]

    返回
    ----
    list[PATH] : 所有枚举到的合法 RS 路径（仅含参数，无离散点）
    """
    dx = q1[0] - q0[0]  # 全局 x 差
    dy = q1[1] - q0[1]  # 全局 y 差
    dth = q1[2] - q0[2]  # 航向差（注意：未归一化到 [-π, π]，由各基元内部处理）
    c = math.cos(q0[2])  # 起点航向余弦
    s = math.sin(q0[2])  # 起点航向正弦
    # 将全局坐标差旋转到局部坐标系并以 maxc 归一化
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc

    # 依次调用各路径族枚举函数，收集所有合法路径
    paths = []
    paths = SCS(x, y, dth, paths)  # 直线-圆弧-直线（3 段）
    paths = CSC(x, y, dth, paths)  # 圆弧-直线-圆弧（3 段，8 变体）
    paths = CCC(x, y, dth, paths)  # 圆弧-圆弧-圆弧（3 段，8 变体）
    paths = CCCC(x, y, dth, paths)  # 圆弧×4（4 段，8 变体）
    paths = CCSC(x, y, dth, paths)  # 圆弧-圆弧-直线-圆弧（4 段，16 变体）
    paths = CCSCC(x, y, dth, paths)  # 圆弧-圆弧-直线-圆弧-圆弧（5 段，4 变体）

    return paths


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def pi_2_pi(theta):
    """将角度规范化到区间 (-π, π]。

    使用循环迭代（适合角度差不太大时），保证 -π < theta ≤ π。
    对于大角度差建议改用 M() 函数（基于取模运算）。
    """
    while theta > PI:
        theta -= 2.0 * PI

    while theta < -PI:
        theta += 2.0 * PI

    return theta


def R(x, y):
    """计算点 (x, y) 的极坐标 (r, theta)。

    返回
    ----
    r     : 到原点的距离（欧式距离），r = hypot(x, y)
    theta : 极角 [rad]，theta = atan2(y, x) ∈ (-π, π]
    """
    r = math.hypot(x, y)
    theta = math.atan2(y, x)

    return r, theta


def M(theta):
    """将角度规范化到区间 (-π, π]（基于取模运算，效率高于循环迭代）。

    与 pi_2_pi 功能相同但实现不同：
    1. 先对 2π 取模，使结果落在 [0, 2π)
    2. 再调整到 (-π, π]

    参数：theta 为任意实数角度 [rad]
    返回：phi ∈ (-π, π]
    """
    phi = theta % (2.0 * PI)  # 先映射到 [0, 2π)（Python 取模结果恒为非负）

    if phi < -PI:
        phi += 2.0 * PI
    if phi > PI:
        phi -= 2.0 * PI

    return phi


def get_label(path):
    """生成路径的可读标签字符串，用于调试和可视化。

    格式：每段依次输出类型字母（'S'/'L'/'R'）和方向符号（'+' 前进 / '-' 后退）。
    示例：路径 ['L','S','R']，长度 [0.5, 1.0, -0.3] → 标签 "L+S+R-"

    参数：path 为 PATH 对象
    返回：字符串标签
    """
    label = ""

    for m, l in zip(path.ctypes, path.lengths):
        label = label + m
        if l > 0.0:
            label = label + "+"  # 前进方向
        else:
            label = label + "-"  # 后退方向

    return label


def calc_curvature(x, y, yaw, directions):
    """在离散路径点序列上数值估计各点的曲率和弧长微元。

    方法：中心差分法（对相邻三点计算一次导数和二次导数，再求曲率）。
    公式：κ = (y'' x' - x'' y') / (x'^2 + y'^2)
    首尾两点分别复制相邻值填充（避免越界）。
    后退段的曲率取反（方向约定一致）。

    参数
    ----
    x, y       : 路径点 x/y 坐标序列 [m]
    yaw        : 各点航向角序列 [rad]（当前未使用，保留接口兼容性）
    directions : 各点行驶方向序列（1=前进，-1=后退）

    返回
    ----
    c  : 各点曲率列表 [1/m]
    ds : 各点弧长微元列表 [m]
    """
    c, ds = [], []

    for i in range(1, len(x) - 1):
        # 前后相邻点的位移向量
        dxn = x[i] - x[i - 1]  # 向后差分
        dxp = x[i + 1] - x[i]  # 向前差分
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = math.hypot(dxn, dyn)  # 后向弧长微元
        dp = math.hypot(dxp, dyp)  # 前向弧长微元
        # 加权中心差分：一阶导数
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)  # 二阶导数（曲率分子分量）
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        # 曲率公式：κ = (y''x' - x''y') / (x'^2 + y'^2)
        curvature = (ddy * dx - ddx * dy) / (dx**2 + dy**2)
        d = (dn + dp) / 2.0  # 该点处弧长微元（前后平均）

        if np.isnan(curvature):  # 防止除零导致的 NaN
            curvature = 0.0

        if directions[i] <= 0.0:  # 后退段曲率取反以保持符号约定一致
            curvature = -curvature

        if len(c) == 0:  # 第一个内部点同时写入 ds 和 c（对齐首点）
            ds.append(d)
            c.append(curvature)

        ds.append(d)
        c.append(curvature)

    # 末尾复制最后一个值（保持 c 和 ds 与 x/y 等长）
    ds.append(ds[-1])
    c.append(c[-1])

    return c, ds


def check_path(sx, sy, syaw, gx, gy, gyaw, maxc):
    """验证 calc_all_paths 生成的所有路径的正确性（单元测试用途）。

    断言内容
    --------
    1. 至少生成了 1 条路径。
    2. 每条路径的起点坐标与 (sx, sy, syaw) 误差 ≤ 0.01。
    3. 每条路径的终点坐标与 (gx, gy, gyaw) 误差 ≤ 0.01。
    4. 相邻路径点间距与 STEP_SIZE 误差 ≤ 0.001（验证离散步长均匀性）。

    参数：与 calc_all_paths 相同。
    """
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc)

    assert len(paths) >= 1  # 必须找到至少一条路径

    for path in paths:
        # 验证路径起点与终点坐标精度
        assert abs(path.x[0] - sx) <= 0.01
        assert abs(path.y[0] - sy) <= 0.01
        assert abs(path.yaw[0] - syaw) <= 0.01
        assert abs(path.x[-1] - gx) <= 0.01
        assert abs(path.y[-1] - gy) <= 0.01
        assert abs(path.yaw[-1] - gyaw) <= 0.01

        # 验证路径点间距均匀性（每步应恰好等于 STEP_SIZE）
        d = [
            math.hypot(dx, dy)
            for dx, dy in zip(
                np.diff(path.x[0 : len(path.x) - 1]),
                np.diff(path.y[0 : len(path.y) - 1]),
            )
        ]

        for i in range(len(d)):
            assert abs(d[i] - STEP_SIZE) <= 0.001
