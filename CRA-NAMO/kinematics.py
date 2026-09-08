"""计算起停平移和旋转时间。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

XY = Tuple[float, float]


def _trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """返回梯形或三角形速度曲线的起停时间。"""
    distance = abs(float(distance))
    if distance <= 0.0:
        return 0.0
    if distance * a_max >= v_max * v_max:
        return distance / v_max + v_max / a_max
    return 2.0 * math.sqrt(distance / a_max)   # 三角形速度曲线


@dataclass(frozen=True)
class MotionProfile:
    """一种机器人运动状态的速度和加速度限制。"""
    v_max: float          # [米/秒]
    a_max: float          # [米/秒^2]
    w_max: float          # [弧度/秒]
    alpha_max: float      # [弧度/秒^2]

    def __post_init__(self):
        for name in ("v_max", "a_max", "w_max", "alpha_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def translate_time(self, distance: float) -> float:
        """直线起停行驶 distance 米所需的秒数。"""
        return _trapezoid_time(distance, self.v_max, self.a_max)

    def rotate_time(self, angle: float) -> float:
        """原地起停旋转 angle 弧度所需的秒数。"""
        return _trapezoid_time(angle, self.w_max, self.alpha_max)


def turn_between(heading: float, target: float) -> float:
    """返回最小的整圆航向变化量。"""
    return abs((target - heading + math.pi) % (2.0 * math.pi) - math.pi)


def segment_legs(profile: MotionProfile, a: XY, b: XY,
                 heading: float) -> Tuple[float, float, float]:
    """返回一段路径的转向时间、行驶时间和到达航向。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return 0.0, 0.0, heading
    target = math.atan2(dy, dx)
    return (profile.rotate_time(turn_between(heading, target)),
            profile.translate_time(distance), target)


def segment_time(profile: MotionProfile, a: XY, b: XY,
                 heading: float) -> Tuple[float, float]:
    """返回一段路径的行驶时间和到达航向。"""
    turn, drive, target = segment_legs(profile, a, b, heading)
    return turn + drive, target


def path_time(profile: MotionProfile, points: Sequence[XY],
              heading: float = None) -> float:
    """返回带顶点转向的折线通过时间。"""
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
