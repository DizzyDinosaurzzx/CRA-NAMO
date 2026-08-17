"""The cost function, in one place.

    J = lambda * D + W          [J]

    lambda * D  lambda_distance [N] times every metre the robot drives — both
                ordinary roadmap travel and the excursion it makes while holding
                an obstacle.
    W           sum over manipulations of (friction force [N] x distance moved).

Both terms come out in joules: `difficulty` is a real friction force
f = mu*rho*V*g, and `lambda_distance` is the robot's equivalent driving
resistance, so the two have the same dimension and J is dimensionally sound.

A metre spent moving an obstacle costs `lambda + difficulty`, because the robot
has to overcome its own driving resistance *and* the obstacle's friction over
that same metre. That falls out of adding the two terms — it is not a separate
case anywhere in the code.

**Nothing outside this module should multiply by `lambda_distance` or by
`difficulty`.** If it does, the accounting has leaked back out again, which is
how it came to be spread over four files in the first place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import geometry
from config import Config

# An `easiest`-strategy surcharge per obstacle, expressed in metres of detour so
# it tracks the physical cost scale rather than being an arbitrary constant.
_EASIEST_DETOUR_EQUIV_M = 50.0


# --- the two terms ---
def motion_cost(cfg: Config, distance: float) -> float:
    """Joules spent driving `distance` metres. The lambda*D term."""
    return cfg.lambda_distance * distance


def manipulation_work(difficulty: float, distance: float) -> float:
    """Joules spent overcoming an obstacle's friction over `distance`. The W term."""
    return difficulty * distance


# --- path measurement ---
def se2_path_length(obs, poses, cfg: Config) -> float:
    """Length of an SE(2) route, with rotation folded in as equivalent translation.

    Turning in place still costs work, so an angle is converted to a distance via
    the body's mean rotation radius. This is the `distance` that goes into
    `manipulation_work`.
    """
    if poses is None or len(poses) < 2:
        return 0.0
    rot_weight = (geometry.mean_rotation_radius(obs.l, obs.d)
                  if cfg.se2_rot_weight is None else float(cfg.se2_rot_weight))
    total = 0.0
    for a, b in zip(poses, poses[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        total += rot_weight * abs(geometry.wrap_dtheta(a[2], b[2]))
    return total


# --- search strategies ---
@dataclass(frozen=True)
class StrategyWeights:
    """How a planning strategy distorts the obstacle-work term during search.

    Only W is affected. The robot's own travel is never discounted — it is real
    distance under every strategy — and execution always settles at true physical
    cost regardless of which strategy chose the route.
    """
    work_mult: float
    work_bias: float


def strategy_weights(cfg: Config) -> StrategyWeights:
    if cfg.strategy == "shortest":
        # ignore W entirely, so the search takes the geometrically shortest route
        return StrategyWeights(work_mult=0.0, work_bias=0.0)
    if cfg.strategy == "easiest":
        # heavily penalise touching anything, so detours win where they exist
        return StrategyWeights(
            work_mult=1.0,
            work_bias=_EASIEST_DETOUR_EQUIV_M * cfg.lambda_distance)
    return StrategyWeights(work_mult=1.0, work_bias=0.0)


def removal_cost(cfg: Config, work: float, contact_travel: float) -> float:
    """What the search charges for clearing one obstacle off an edge.

    `work` is the estimated obstacle friction work; `contact_travel` is how far
    the robot itself drives to fetch, escort and leave it.
    """
    w = strategy_weights(cfg)
    return work * w.work_mult + w.work_bias + motion_cost(cfg, contact_travel)

