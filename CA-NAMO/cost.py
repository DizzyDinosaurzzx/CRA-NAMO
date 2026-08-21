"""The cost function, in one place.

What the robot is billed
------------------------

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

What the search optimises
-------------------------

Not J. J is what the run *costs*; the route is chosen against

    C = (1 - w) * J + w * P * T      w = cfg.time_importance in [0, 1]

so the robot can be told to care about finishing quickly as well as cheaply.
`P = lambda * v_max` [W] is the exchange rate — see `time_price`. At w = 0 this
collapses to J exactly and the search is bit-identical to the energy-only model;
at w = 1 it is a pure minimum-time search.

J and T stay separately measured and separately reported throughout. The blend
exists only inside the search; execution always settles at true physical cost
and true simulated duration, whichever route the blend picked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import geometry
import timing
from config import Config


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


# --- energy against time ---
def time_price(cfg: Config) -> float:
    """Joules that one second of mission time is worth.

    Not a new hand-tuned constant: it is the power the robot burns while
    cruising, lambda [N] x v_max [m/s]. Pricing a second at what a second of
    driving already costs makes the two terms exchangeable without inventing a
    rate, and it has a tidy consequence — for ordinary driving the blend cancels
    exactly:

        (1-w) * lambda*d  +  w * (lambda*v) * (d/v)  =  lambda*d

    So `time_importance` leaves plain travel alone and moves only the price of
    *handling obstacles*, which is the trade-off it exists to control. Energy
    cares how heavy a thing is; the clock does not, and cares instead that the
    robot crawls at v_max_contact while it has hold of it.
    """
    return cfg.lambda_distance * cfg.v_max


def blend(cfg: Config, joules: float, seconds: float) -> float:
    """C = (1-w)*J + w*P*T — the number the search actually minimises."""
    w = cfg.time_importance
    if w <= 0.0:
        return joules      # exact, so w=0 reproduces the energy-only model bit for bit
    return (1.0 - w) * joules + w * time_price(cfg) * seconds


# --- search strategies ---
@dataclass(frozen=True)
class StrategyWeights:
    """How a planning strategy distorts the obstacle-work term during search.

    Only W is affected. The robot's own travel is never discounted — it is real
    distance under every strategy — and execution always settles at true physical
    cost regardless of which strategy chose the route.
    """
    work_mult: float


def strategy_weights(cfg: Config) -> StrategyWeights:
    if cfg.strategy == "shortest":
        # ignore W entirely, so the search takes the geometrically shortest route
        return StrategyWeights(work_mult=0.0)
    return StrategyWeights(work_mult=1.0)


# --- what the search charges ---
def search_motion_cost(cfg: Config, distance: float) -> float:
    """Driving `distance` metres, as the *search* prices it.

    Equal to `motion_cost` for every w (see `time_price`), but written as the
    blend so the cancellation is visible here rather than being an unwritten
    assumption that breaks silently if the exchange rate ever changes.
    """
    return blend(cfg, motion_cost(cfg, distance), timing.drive_seconds(cfg, distance))


def search_turn_cost(cfg: Config, dtheta: float) -> float:
    """Stopping and pivoting through `dtheta` at a corner, as the search prices it.

    Turning burns no *distance*, so it has no place in J at all — this term is
    pure time and is therefore exactly zero at w = 0, which is why adding it
    leaves the energy-only model untouched. Without it the search is blind to
    corners and, once time starts to matter, will happily buy a shorter route
    with a dozen extra stop-turn-go cycles that cost more seconds than they save.
    """
    if cfg.time_importance <= 0.0 or dtheta <= 0.0:
        return 0.0
    return blend(cfg, 0.0, timing.turn_seconds(cfg, dtheta))


def removal_cost(cfg: Config, work: float, contact_travel: float,
                 move_dist: float) -> float:
    """What the search charges for clearing one obstacle off an edge.

    `work` is the estimated obstacle friction work, `contact_travel` how far the
    robot itself drives to fetch, escort and leave it, and `move_dist` the
    obstacle's own route — the last only sets the clock under `--no-contact`.
    """
    joules = (work * strategy_weights(cfg).work_mult
              + motion_cost(cfg, contact_travel))
    return blend(cfg, joules, timing.removal_seconds(cfg, contact_travel, move_dist))

