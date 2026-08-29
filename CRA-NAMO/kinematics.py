"""Compute rest-to-rest translation and rotation times."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

XY = Tuple[float, float]


def _trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """Return rest-to-rest time for a trapezoidal or triangular profile."""
    distance = abs(float(distance))
    if distance <= 0.0:
        return 0.0
    if distance * a_max >= v_max * v_max:
        return distance / v_max + v_max / a_max
    return 2.0 * math.sqrt(distance / a_max)   # triangular profile


@dataclass(frozen=True)
class MotionProfile:
    """Velocity and acceleration limits for one robot motion state."""
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
    """Return the smallest full-circle heading change."""
    return abs((target - heading + math.pi) % (2.0 * math.pi) - math.pi)


def segment_legs(profile: MotionProfile, a: XY, b: XY,
                 heading: float) -> Tuple[float, float, float]:
    """Return turn time, drive time and arrival heading for one segment."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return 0.0, 0.0, heading
    target = math.atan2(dy, dx)
    return (profile.rotate_time(turn_between(heading, target)),
            profile.translate_time(distance), target)


def segment_time(profile: MotionProfile, a: XY, b: XY,
                 heading: float) -> Tuple[float, float]:
    """Return travel time and arrival heading for one segment."""
    turn, drive, target = segment_legs(profile, a, b, heading)
    return turn + drive, target


def path_time(profile: MotionProfile, points: Sequence[XY],
              heading: float = None) -> float:
    """Return time to traverse a polyline with turns at its vertices."""
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
