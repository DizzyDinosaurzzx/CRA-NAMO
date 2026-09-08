"""计算能量、时间、搬移和风险代价。"""

from __future__ import annotations

import math

import geometry
import kinematics
import risk as risk_model
from config import Config

def motion_cost(cfg: Config, distance: float) -> float:
    """行驶 distance 米消耗的能量，即 lambda*D 项。"""
    return cfg.lambda_distance * distance


def manipulation_work(difficulty: float, distance: float) -> float:
    """克服障碍物在 distance 米内的摩擦所需能量，即 W 项。"""
    return difficulty * distance


def time_cost(cfg: Config, seconds: float) -> float:
    """将 seconds 秒折算为能量，即计价后的 T 项。"""
    return cfg.time_value * seconds


def risk_cost(cfg: Config, level) -> float:
    """返回目标函数单位下的风险附加代价。"""
    return (cfg.risk_weight * cfg.lambda_distance
            * risk_model.detour_equivalent_m(level))


def combine(cfg: Config, joules: float, seconds: float) -> float:
    """按配置将能量、时间和风险合成为目标函数。"""
    w = cfg.time_importance
    return (1.0 - w) * joules + w * time_cost(cfg, seconds)


def drive_time(cfg: Config, distance: float) -> float:
    """返回一条直线路线图边的起停时间。"""
    return cfg.free_profile().translate_time(distance)


def manipulation_time(cfg: Config, cplan, n_poses: int,
                      move_dist: float) -> float:
    """返回一次搬移的接近、伴随和离开时间。"""
    free = cfg.free_profile()
    loaded = cfg.loaded_profile()
    path = cplan.robot_path
    off = cplan.move_offset
    last = off + max(n_poses - 1, 0)
    if last >= len(path):
        return kinematics.path_time(free, path)
    escort = kinematics.path_time(loaded, path[off:last + 1])
    if escort <= 0.0 and move_dist > 0.0:
        escort = loaded.translate_time(move_dist)
    return (kinematics.path_time(free, path[:off + 1])
            + escort
            + kinematics.path_time(free, path[last:]))


def se2_path_length(obs, poses, cfg: Config) -> float:
    """返回 SE(2) 路径长度，并将旋转折算为等效距离。"""
    if poses is None or len(poses) < 2:
        return 0.0
    rot_weight = (geometry.mean_rotation_radius(obs.l, obs.d)
                  if cfg.se2_rot_weight is None else float(cfg.se2_rot_weight))
    total = 0.0
    for a, b in zip(poses, poses[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        total += rot_weight * abs(geometry.wrap_dtheta(a[2], b[2]))
    return total


def edge_cost(cfg: Config, length: float) -> float:
    """返回搜索通过一条畅通路线图边的代价。"""
    return combine(cfg, motion_cost(cfg, length), drive_time(cfg, length))


def removal_cost(cfg: Config, work: float, contact_travel: float,
                 seconds: float, risk_level=None) -> float:
    """返回从一条边上清除一个障碍物的目标代价。"""
    joules = work + motion_cost(cfg, contact_travel)
    return combine(cfg, joules, seconds) + risk_cost(cfg, risk_level)


def heuristic(cfg: Config, distance: float) -> float:
    """返回剩余路线距离的可采纳下界。"""
    longest = 2.0 * cfg.conn_radius
    per_metre = cfg.free_profile().translate_time(longest) / longest
    return combine(cfg, motion_cost(cfg, distance), distance * per_metre)
