"""How long the mission actually takes, in one place.

    T = T_motion + T_decision

    T_motion    simulated wall-clock of the robot physically driving the route
                it ended up taking — accumulated here from the same travel
                events that `cost.py` bills, so distance and duration can never
                disagree about what the robot did.
    T_decision  real measured wall-clock spent planning (A* + LLM calls). It is
                timed in `executor.py`, not modelled here.

Motion model
------------
The robot is a differential-drive disc: it **turns in place, then drives
straight**. So a polyline route costs

    sum over straight runs   trapezoid(run length, v_max, a_max)
  + sum over corners         trapezoid(|turn|,     w_max, alpha_max)

Consecutive collinear segments are merged into one straight run before the
profile is applied — otherwise the roadmap's 0.3 m grid spacing would make the
robot brake to a stop thirty times per straight corridor and the number would be
meaningless.  `trapezoid` is the standard accelerate–cruise–decelerate profile,
degrading to a triangular one when the run is too short to ever reach v_max.

Escorting an obstacle uses a second, slower set of limits (`*_contact`): the
robot is pressed against a load it has to keep from slipping, so it drives and
turns more gently than it does in free space.

**Nothing outside this module should divide a distance by a speed.**  Same rule
as `cost.py` and for the same reason — the moment two places own the conversion
they drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from config import Config

XY = Tuple[float, float]

# Below this a "segment" is numerical noise from the contact planner (the grip
# point barely slides while the obstacle rotates), not a drive the robot makes.
_MIN_SEGMENT_M = 1e-9
# Heading change under this is the same straight line continuing, so the run is
# not broken and no turn is billed.
_MIN_TURN_RAD = 1e-6


def trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """Seconds to cover `distance` from rest to rest under |v| <= v_max, |a| <= a_max.

    Works for angles too — feed radians, rad/s and rad/s^2.
    """
    if distance <= 0.0:
        return 0.0
    # distance consumed by accelerating to v_max and back down again
    ramp = v_max * v_max / a_max
    if distance >= ramp:
        return distance / v_max + v_max / a_max      # accelerate, cruise, decelerate
    return 2.0 * math.sqrt(distance / a_max)         # triangular: never reaches v_max


def _wrap_pi(angle: float) -> float:
    """Shortest signed turn, modulo 2*pi.

    Not `geometry.wrap_dtheta`, which is modulo *pi* because a rectangle's
    footprint repeats every half turn. A heading does not: driving north and
    driving south are not the same manoeuvre.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_of(a: XY, b: XY) -> Optional[float]:
    """Direction of travel from a to b, or None when they coincide."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if math.hypot(dx, dy) <= _MIN_SEGMENT_M:
        return None
    return math.atan2(dy, dx)


# --- planner-side estimates ---
# The executor above measures what the robot *did*, turn by turn.  The search
# needs a price before the fact, for an edge it may never take, and it does not
# know which way the robot will be facing when it arrives — heading depends on
# the route, so pricing turns would mean searching over (node, heading) instead
# of node.  So these estimate at cruise speed with no ramps and no pivots.
#
# That makes them a *lower bound* on the executor's number, which is what keeps
# the A* heuristic admissible.  It is the same bargain the codebase already
# makes with obstacle difficulty: plan against an estimate, settle against the
# truth.  The reported mission time always comes from the executor.
def drive_seconds(cfg: Config, distance: float) -> float:
    """Estimated seconds to drive `distance` metres in free space."""
    return distance / cfg.v_max


def turn_seconds(cfg: Config, dtheta: float) -> float:
    """Seconds to pivot through `dtheta` radians in free space.

    Unlike the two estimates around it this one is exact — the executor bills a
    corner with the same profile. It is separate from `drive_seconds` because
    the search only knows a corner is coming when it looks at two consecutive
    edges, which happens in the expansion loop rather than in the edge cost.
    """
    return trapezoid_time(abs(dtheta), cfg.w_max, cfg.alpha_max)


def turn_between(h_in: Optional[float], h_out: Optional[float]) -> float:
    """Size of the pivot from heading `h_in` to `h_out`, or 0 if either is unknown."""
    if h_in is None or h_out is None:
        return 0.0
    return abs(_wrap_pi(h_out - h_in))


def removal_seconds(cfg: Config, contact_travel: float, move_dist: float) -> float:
    """Estimated seconds to clear one obstacle out of an edge.

    `contact_travel` is how far the robot walks to fetch, escort and leave it;
    `move_dist` is the obstacle's own SE(2) route, which only sets the clock in
    the `--no-contact` model where the robot does not walk with it.
    """
    handling = (contact_travel if cfg.contact_required else move_dist)
    return handling / cfg.v_max_contact + 2.0 * cfg.grip_time


@dataclass(frozen=True)
class MotionProfile:
    """Velocity and acceleration limits, in free space and while in contact."""
    v_max: float
    a_max: float
    w_max: float
    alpha_max: float
    v_max_contact: float
    a_max_contact: float
    w_max_contact: float
    alpha_max_contact: float
    grip_time: float

    @classmethod
    def from_config(cls, cfg: Config) -> "MotionProfile":
        return cls(v_max=cfg.v_max, a_max=cfg.a_max,
                   w_max=cfg.w_max, alpha_max=cfg.alpha_max,
                   v_max_contact=cfg.v_max_contact, a_max_contact=cfg.a_max_contact,
                   w_max_contact=cfg.w_max_contact,
                   alpha_max_contact=cfg.alpha_max_contact,
                   grip_time=cfg.grip_time)

    def limits(self, in_contact: bool) -> Tuple[float, float, float, float]:
        if in_contact:
            return (self.v_max_contact, self.a_max_contact,
                    self.w_max_contact, self.alpha_max_contact)
        return self.v_max, self.a_max, self.w_max, self.alpha_max


class MotionTimer:
    """Accumulates simulated seconds as the executor drives the robot.

    Call `travel` once per segment the robot actually covers, in order.  Collinear
    same-mode segments merge into one straight run, which is only converted to
    time when the run ends — so `total` and `contact_total` are exact only after
    `flush`.  `run()` flushes for you.
    """

    def __init__(self, profile: MotionProfile, heading: float = 0.0):
        self.profile = profile
        self.heading = heading
        self.total = 0.0            # all simulated robot motion [s]
        self.contact_total = 0.0    # the part of it spent holding an obstacle [s]
        # open straight run, not yet converted to time
        self._run_dist = 0.0
        self._run_contact = False

    @property
    def elapsed(self) -> float:
        """Seconds so far, counting the still-open straight run.

        Read-only on purpose: calling `flush` to get an up-to-date figure would
        close the run, and the next collinear segment would then start a fresh
        one — turning one straight line into two and inflating the total. Use
        this for progress readouts, `flush` only when the motion really ends.
        """
        if self._run_dist <= 0.0:
            return self.total
        v_max, a_max, _w, _alpha = self.profile.limits(self._run_contact)
        return self.total + trapezoid_time(self._run_dist, v_max, a_max)

    # --- accumulation ---
    def travel(self, distance: float, heading: Optional[float] = None,
               in_contact: bool = False) -> None:
        """Drive `distance` metres facing `heading`.

        `heading=None` means "do not bill a turn" — the robot is reversing back
        over ground it just covered, which a differential drive does without
        turning around.  The straight run is still broken, because it does have
        to stop before reversing.
        """
        if distance <= _MIN_SEGMENT_M:
            return
        if heading is None:
            self.flush()
        elif (abs(_wrap_pi(heading - self.heading)) > _MIN_TURN_RAD
                or in_contact != self._run_contact):
            self.flush()
            self.turn_to(heading, in_contact)
        self._run_dist += distance
        self._run_contact = in_contact

    def turn_to(self, heading: float, in_contact: bool = False) -> None:
        """Pivot in place to face `heading`."""
        self.flush()
        dtheta = abs(_wrap_pi(heading - self.heading))
        _v, _a, w_max, alpha_max = self.profile.limits(in_contact)
        self._add(trapezoid_time(dtheta, w_max, alpha_max), in_contact)
        self.heading = heading

    def hold(self, seconds: float, in_contact: bool = True) -> None:
        """Time the robot spends stationary but occupied — gripping, releasing,
        or standing by while an obstacle it is not escorting moves."""
        if seconds <= 0.0:
            return
        self.flush()
        self._add(seconds, in_contact)

    def transport(self, distance: float) -> None:
        """Time for an obstacle to cover `distance` of its own path while the
        robot stands by instead of escorting it (the `--no-contact` model).

        Charged at the contact cruise speed with no ramps, because this is one
        sub-step of a continuous motion rather than a move from rest to rest.
        """
        self.hold(distance / self.profile.v_max_contact)

    def grip(self) -> None:
        """Latching onto an obstacle and letting go of it again, once each."""
        self.hold(2.0 * self.profile.grip_time)

    def flush(self) -> None:
        """Close the open straight run and convert it to time."""
        if self._run_dist <= 0.0:
            self._run_dist = 0.0
            return
        v_max, a_max, _w, _alpha = self.profile.limits(self._run_contact)
        self._add(trapezoid_time(self._run_dist, v_max, a_max), self._run_contact)
        self._run_dist = 0.0

    def _add(self, seconds: float, in_contact: bool) -> None:
        self.total += seconds
        if in_contact:
            self.contact_total += seconds
