"""代价函数唯一所在地：J = lambda*D + W；lambda_distance 与 difficulty 只许在本模块内相乘，搜索实际最小化 C = (1-w)*J + w*P*T。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import geometry
import timing
from config import Config


# --- 代价的两项 ---
def motion_cost(cfg: Config, distance: float) -> float:
    """行驶 distance 米所耗能量，即 lambda*D 项。"""
    return cfg.lambda_distance * distance


def manipulation_work(difficulty: float, distance: float) -> float:
    """克服障碍物摩擦力移动 distance 米所耗能量，即 W 项。"""
    return difficulty * distance


def turn_cost(cfg: Config, dtheta: float) -> float:
    """机器人原地转 dtheta 弧度所耗能量，即并入 J 的转弯项。

    把机器人半径处的一点沿弧线拖出这么长一段位移，等效为走同样长度直线所需的功——
    和 se2_path_length 用 mean_rotation_radius 把障碍物自转折算成距离是同一套物理
    类比（那边算的是障碍物自己转，这边算的是机器人原地转）。恒定计入 J，与
    time_importance 无关：转弯本来就要克服真实的滑动/转向阻力，不是只有赶时间才
    该在乎的事。
    """
    if dtheta <= 0.0:
        return 0.0
    return motion_cost(cfg, cfg.robot_radius * dtheta)


# --- 路径度量 ---
def se2_path_length(obs, poses, cfg: Config) -> float:
    """SE(2) 路线长度；原地旋转也做功，按平均旋转半径折算成等效平移计入。"""
    if poses is None or len(poses) < 2:
        return 0.0
    rot_weight = (geometry.mean_rotation_radius(obs.l, obs.d)
                  if cfg.se2_rot_weight is None else float(cfg.se2_rot_weight))
    total = 0.0
    for a, b in zip(poses, poses[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        total += rot_weight * abs(geometry.wrap_dtheta(a[2], b[2]))
    return total


# --- 能量与时间的换算 ---
def time_price(cfg: Config) -> float:
    """一秒任务时间折合的焦耳数，取巡航功率 lambda*v_max [W]，非新调参数。

    好处是普通行驶项恰好抵消：(1-w)*lambda*d + w*(lambda*v)*(d/v) = lambda*d，
    故 time_importance 只改变搬运障碍物的代价，不影响普通行驶。
    """
    return cfg.lambda_distance * cfg.v_max


def blend(cfg: Config, joules: float, seconds: float) -> float:
    """搜索实际最小化的 C = (1-w)*J + w*P*T。"""
    w = cfg.time_importance
    if w <= 0.0:
        return joules      # 精确返回，w=0 时逐位复现纯能量模型
    return (1.0 - w) * joules + w * time_price(cfg) * seconds


# --- 搜索策略 ---
@dataclass(frozen=True)
class StrategyWeights:
    """规划策略对搜索期 W 项的缩放；行驶距离从不打折，执行期始终按真实物理代价结算。"""
    work_mult: float


def strategy_weights(cfg: Config) -> StrategyWeights:
    if cfg.strategy == "shortest":
        # 完全忽略 W，让搜索走几何最短路线
        return StrategyWeights(work_mult=0.0)
    return StrategyWeights(work_mult=1.0)


# --- 搜索期的计费 ---
def search_motion_cost(cfg: Config, distance: float) -> float:
    """搜索视角的行驶代价；与 motion_cost 恒等（见 time_price 的抵消），显式写出 blend 以免该隐式抵消被悄悄破坏。"""
    return blend(cfg, motion_cost(cfg, distance), timing.drive_seconds(cfg, distance))


def search_turn_cost(cfg: Config, dtheta: float) -> float:
    """拐角处停车原地转 dtheta 的搜索代价：能量项 turn_cost 恒定计入(与 search_motion_cost
    对行驶距离的处理是同一套逻辑，w=0 时 blend() 精确返回这一项、不多不少)；时间项只在
    w>0 时通过 blend 混入。以前这里只有时间项、w=0 时恒为零，搜索会完全无视拐角——
    不只是"用一堆加减速换更短的路"，更严重的是转弯本身产生的真实能耗(见 turn_cost)
    从未进过 J，分支限界会剪掉本该更省的方案，见 README「时间权重」一节的实测案例。
    """
    if dtheta <= 0.0:
        return 0.0
    return blend(cfg, turn_cost(cfg, dtheta), timing.turn_seconds(cfg, dtheta))


def removal_cost(cfg: Config, work: float, contact_travel: float) -> float:
    """搜索对搬走一个障碍物的计费。

    work 为障碍物摩擦功的估计，contact_travel 为机器人贴身接送随行所行驶的距离。
    """
    joules = (work * strategy_weights(cfg).work_mult
              + motion_cost(cfg, contact_travel))
    return blend(cfg, joules, timing.removal_seconds(cfg, contact_travel))

