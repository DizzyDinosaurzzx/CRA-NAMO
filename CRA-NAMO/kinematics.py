"""How long a motion takes.

Dependency-free like `geometry`, and the same division of labour: this module
holds the *physics* of moving — velocity limits, acceleration limits, the time a
trapezoidal profile needs — and nothing about robots, roadmaps or costs.

The robot is a disc, so its footprint has no heading, but a real differential
drive still has to point itself before it can go anywhere. The model here is
turn-in-place-then-drive: at every waypoint the robot stops, rotates to face the
next one, and drives that segment from rest to rest. It matches how the
simulation already executes a route — node by node, stopping at each to perceive
and replan — and it is what makes an angular acceleration limit observable at
all. A robot that never stopped would need a smoothed trajectory, which the
roadmap does not produce.

Everything is rest-to-rest, so times add: the time of a polyline is the sum of
its turns and its segments, with no cross-terms to keep track of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

XY = Tuple[float, float]


def _trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """Rest-to-rest time over `distance` under a velocity and acceleration cap.

    Either the profile reaches `v_max` and cruises (trapezoid), or it runs out of
    distance first and turns round at its peak (triangle). The two agree exactly
    at the crossover, so time is continuous in distance — a search that compares
    routes across it will not see a step.
    """
    distance = abs(float(distance))
    if distance <= 0.0:
        return 0.0
    if distance * a_max >= v_max * v_max:      # long enough to reach the cap
        return distance / v_max + v_max / a_max
    return 2.0 * math.sqrt(distance / a_max)   # triangular profile


@dataclass(frozen=True)
class MotionProfile:
    """What the robot can do, unloaded or loaded.

    Two of these exist per run: one for driving free, a slower one for while an
    obstacle is being held. Which applies is decided by the caller — nothing here
    knows what the robot is doing.
    """
    v_max: float          # [m/s]
    a_max: float          # [m/s^2]
    w_max: float          # [rad/s]
    alpha_max: float      # [rad/s^2]

    def __post_init__(self):
        for name in ("v_max", "a_max", "w_max", "alpha_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def translate_time(self, distance: float) -> float:
        """Seconds to drive `distance` metres in a straight line, rest to rest."""
        return _trapezoid_time(distance, self.v_max, self.a_max)

    def rotate_time(self, angle: float) -> float:
        """Seconds to turn `angle` radians in place, rest to rest."""
        return _trapezoid_time(angle, self.w_max, self.alpha_max)


def turn_between(heading: float, target: float) -> float:
    """Smallest rotation from `heading` to `target`, as a magnitude.

    Full circle, not modulo pi: unlike a rectangle's orientation, a heading of
    theta and theta+pi are opposite ways to face, and the robot drives forwards.
    """
    return abs((target - heading + math.pi) % (2.0 * math.pi) - math.pi)


def segment_time(profile: MotionProfile, a: XY, b: XY,
                 heading: float) -> Tuple[float, float]:
    """Time to get from `a` to `b` starting out facing `heading`.

    Returns (seconds, heading on arrival). A zero-length segment costs nothing
    and leaves the heading alone — there is no direction to turn towards.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return 0.0, heading
    target = math.atan2(dy, dx)
    seconds = (profile.rotate_time(turn_between(heading, target))
               + profile.translate_time(distance))
    return seconds, target


def path_time(profile: MotionProfile, points: Sequence[XY],
              heading: float = None) -> float:
    """Time to walk a polyline, turning at every vertex.

    `heading` is where the robot faces to begin with. Left out, the first turn is
    free — which is what a planner wants when it is costing a path it has not
    decided how to approach yet, and would otherwise have to guess at.
    """
    if points is None or len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        if heading is None:
            dx, dy = b[0] - a[0], b[1] - a[1]
            if math.hypot(dx, dy) > 1e-12:
                heading = math.atan2(dy, dx)
            total += profile.translate_time(math.hypot(dx, dy))
            continue
        seconds, heading = segment_time(profile, a, b, heading)
        total += seconds
    return total
