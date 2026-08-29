"""Compute energy, time, manipulation, and risk costs."""

from __future__ import annotations

import math

import geometry
import kinematics
import risk as risk_model
from config import Config

def motion_cost(cfg: Config, distance: float) -> float:
    """Joules spent driving `distance` metres. The lambda*D term."""
    return cfg.lambda_distance * distance


def manipulation_work(difficulty: float, distance: float) -> float:
    """Joules spent overcoming an obstacle's friction over `distance`. The W term."""
    return difficulty * distance


def time_cost(cfg: Config, seconds: float) -> float:
    """Joules-equivalent of `seconds` spent. The T term, priced."""
    return cfg.time_value * seconds


def risk_cost(cfg: Config, level) -> float:
    """Return the risk surcharge in the objective's cost units."""
    return (cfg.risk_weight * cfg.lambda_distance
            * risk_model.detour_equivalent_m(level))


def combine(cfg: Config, joules: float, seconds: float) -> float:
    """Combine energy, time and risk into the configured objective."""
    w = cfg.time_importance
    return (1.0 - w) * joules + w * time_cost(cfg, seconds)


def drive_time(cfg: Config, distance: float) -> float:
    """Return rest-to-rest time for one straight roadmap edge."""
    return cfg.free_profile().translate_time(distance)


def manipulation_time(cfg: Config, cplan, n_poses: int,
                      move_dist: float) -> float:
    """Return approach, escort and exit time for one manipulation."""
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
    """Return SE(2) route length with rotation converted to equivalent distance."""
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
    """What the search charges for driving one clear roadmap edge."""
    return combine(cfg, motion_cost(cfg, length), drive_time(cfg, length))


def removal_cost(cfg: Config, work: float, contact_travel: float,
                 seconds: float, risk_level=None) -> float:
    """Return the objective cost of clearing one obstacle from an edge."""
    joules = work + motion_cost(cfg, contact_travel)
    return combine(cfg, joules, seconds) + risk_cost(cfg, risk_level)


def heuristic(cfg: Config, distance: float) -> float:
    """Return an admissible lower bound for the remaining route distance."""
    longest = 2.0 * cfg.conn_radius
    per_metre = cfg.free_profile().translate_time(longest) / longest
    return combine(cfg, motion_cost(cfg, distance), distance * per_metre)
