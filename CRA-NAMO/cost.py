"""The cost function, in one place.

    C = (1 - w) * J  +  w * (time_value * T)  +  R        [J]

    J = lambda * D + W          [J]
    T = move time + plan time   [s]
    R = sum of risk surcharges  [J]

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

T is the simulated clock: every second the robot spends driving, turning, or
standing still while the planner thinks. `w = cfg.time_importance` slides between
the two — 0 is the pure-energy objective the project started with and is the
default, 1 costs nothing but time. `time_value` [J/s] is what makes that sum
legal: it prices a second in joules, so both halves of C are energies.

The search and the executor charge time from different sides of the same model.
The search knows a route's geometry but not which way the robot will be facing
when it gets there, so it costs roadmap edges as pure translation; the executor
knows the heading exactly and pays the turns as well. Planning time is worse
still — it is real algorithm time, so it cannot be known before it is spent, and
only the executor's T contains it. Both effects push the same way, executed time
>= planned time, which is the same relationship the estimated and true
difficulties already have.

R is what it costs to have moved dangerous things: one surcharge per obstacle,
the first time it is moved, priced by the level `risk` assigns it. It sits
*outside* the (1-w)/w split rather than inside J, and that placement is the whole
design. Folded into J it would be scaled by (1-w), so a robot told to care only
about time (w=1) would price a wheelchair with someone in it at zero. Risk is not
the kind of thing a speed preference is allowed to discount.

**Nothing outside this module should multiply by `lambda_distance`, by
`difficulty`, by `time_value`, or by `risk_weight`.** If it does, the accounting has leaked back
out again, which is how it came to be spread over four files in the first place.
"""

from __future__ import annotations

import math

import geometry
import kinematics
import risk as risk_model
from config import Config

# --- the two terms ---
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
    """One obstacle's risk surcharge. The R term.

    Stated in the dataset as the detour worth taking to avoid it, so multiplying
    by lambda turns it into joules on the same scale as everything else — the
    ladder keeps its meaning if the robot's driving resistance is retuned.
    `None` means never assessed, or already paid for, and costs nothing.
    """
    return (cfg.risk_weight * cfg.lambda_distance
            * risk_model.detour_equivalent_m(level))


def combine(cfg: Config, joules: float, seconds: float) -> float:
    """The objective itself: C = (1 - w) * J + w * (time_value * T).

    Linear in both arguments, which is what lets the search accumulate C edge by
    edge and still get the same answer as combining the totals at the end.
    """
    w = cfg.time_importance
    return (1.0 - w) * joules + w * time_cost(cfg, seconds)


# --- how long things take ---
def drive_time(cfg: Config, distance: float) -> float:
    """Seconds to drive one roadmap edge, rest to rest and in a straight line.

    No turn: which way the robot arrives facing depends on the edge it came in
    on, which the search does not carry in its state. The executor pays that
    turn for real.
    """
    return cfg.free_profile().translate_time(distance)


def manipulation_time(cfg: Config, cplan, n_poses: int,
                      move_dist: float) -> float:
    """Seconds for one manipulation, from setting off to being back on the roadmap.

    Three legs at two speeds: walk out to the grip point free, escort the
    obstacle loaded, walk back out free. With `contact_required` off the robot
    never moves during the middle leg, so there is no escort path to measure and
    the obstacle's own travel stands in for it — otherwise moving something would
    take no time at all.
    """
    free = cfg.free_profile()
    loaded = cfg.loaded_profile()
    path = cplan.robot_path
    off = cplan.move_offset
    last = off + max(n_poses - 1, 0)
    if last >= len(path):                     # degenerate plan, nothing to escort
        return kinematics.path_time(free, path)
    escort = kinematics.path_time(loaded, path[off:last + 1])
    if escort <= 0.0 and move_dist > 0.0:
        escort = loaded.translate_time(move_dist)
    return (kinematics.path_time(free, path[:off + 1])
            + escort
            + kinematics.path_time(free, path[last:]))


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
def work_multiplier(cfg: Config) -> float:
    """How much of the obstacle-work term the search believes, per strategy.

    Only W is scaled. The robot's own travel is never discounted — it is real
    distance under every strategy — and execution always settles at true physical
    cost regardless of which strategy chose the route.
    """
    # "shortest" ignores W entirely, taking the geometrically shortest route
    return 0.0 if cfg.strategy == "shortest" else 1.0


def edge_cost(cfg: Config, length: float) -> float:
    """What the search charges for driving one clear roadmap edge."""
    return combine(cfg, motion_cost(cfg, length), drive_time(cfg, length))


def removal_cost(cfg: Config, work: float, contact_travel: float,
                 seconds: float, risk_level=None) -> float:
    """What the search charges for clearing one obstacle off an edge.

    `work` is the estimated obstacle friction work, `contact_travel` how far the
    robot itself drives to fetch, escort and leave it, and `seconds` how long all
    of that takes. `risk_level` is charged whole, outside the energy/time split —
    pass None for an obstacle that has already been moved once and so has already
    paid its surcharge.
    """
    joules = work * work_multiplier(cfg) + motion_cost(cfg, contact_travel)
    return combine(cfg, joules, seconds) + risk_cost(cfg, risk_level)


def heuristic(cfg: Config, distance: float) -> float:
    """Lower bound on the cost of covering `distance` still to go.

    Both halves have to be bounds for the search to stay admissible. Distance is
    easy — the straight line is the shortest route there is. Time takes more
    care, because the robot stops at every roadmap node and pays an acceleration
    ramp on each edge, so a metre is cheaper on a long edge than on a short one.
    The fastest any metre can be covered is therefore on the longest edge the
    roadmap can hold, which is a terminal's, at twice the connection radius.

    Bounding by `v_max` alone would also be admissible, but it assumes an
    acceleration the robot does not have: on these maps that bound is roughly a
    third of the real per-metre time, and a heuristic that loose makes the search
    expand its way to the answer rather than aim at it. Since planning time is
    itself part of T, a lazy bound makes the very thing being minimised worse.
    """
    longest = 2.0 * cfg.conn_radius          # `Roadmap.add_terminal` allows this much
    per_metre = cfg.free_profile().translate_time(longest) / longest
    return combine(cfg, motion_cost(cfg, distance), distance * per_metre)

