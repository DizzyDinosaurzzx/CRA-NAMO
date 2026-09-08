"""参考场景共用的物理系数和几何检查。"""

from __future__ import annotations

from typing import Iterable, Sequence

from shapely.geometry import Point

G = 9.81                        # 重力加速度 [米/秒^2]

MU_CASTORS = 0.03               # 坚硬清洁地面上的自由脚轮
MU_CASTORS_FOULED = 0.22        # 被砂砾、碎屑或刹车卡住的脚轮
MU_RUBBER_WHEELS = 0.08         # 小车轮越过碎石或门槛
MU_BRAKED_WHEELS = 0.60         # 脚轮刹车锁定后被拖动
MU_FELT_PADS = 0.25             # 家具脚垫在硬地面上
MU_WOOD = 0.35                  # 裸木或塑料在硬地面上
MU_UPHOLSTERY = 0.50            # 沙发或床垫沿底面拖动
MU_STEEL = 0.45                 # 钢板底座在混凝土或瓷砖上
MU_CONCRETE = 0.60              # 混凝土或砌体在混凝土上

_OVERLAP_EPS = 1e-7


def push_force(mass_kg: float, mu: float) -> float:
    """返回真实物体的地面滑动阻力 mu * m * g [牛]。"""
    if mass_kg <= 0.0:
        raise ValueError(f"mass must be positive, got {mass_kg!r} kg")
    if mu <= 0.0:
        raise ValueError(f"friction coefficient must be positive, got {mu!r}")
    return round(mu * mass_kg * G, 3)


def bulk_density(mass_kg: float, l: float, d: float, h: float) -> float:
    """返回质量除以包围盒体积 [千克/米^3]，即估计器需要推断的量。"""
    volume = l * d * h
    if volume <= 0.0:
        raise ValueError("bounding box must have positive volume")
    return mass_kg / volume


def tip_over_width(cfg) -> float:
    """返回避免倾倒所需的障碍物最小宽度。"""
    if cfg.robot_push_height <= 0.0 or cfg.push_friction_mu <= 0.0:
        return 0.0
    return 2.0 * cfg.push_friction_mu * cfg.robot_push_height


def check_layout(name: str, *, workspace, static: Iterable, movable: Iterable,
                 start: Sequence[float], goal: Sequence[float], cfg,
                 require_pushable_width: bool = True) -> None:
    """若预设坐标违反地图几何约束则抛出异常。"""
    walls = list(static)
    obstacles = list(movable)

    seen: dict = {}
    for obs in obstacles:
        if obs.oid in seen:
            raise ValueError(f"{name}: duplicate obstacle id {obs.oid!r}")
        seen[obs.oid] = obs

    narrowest = tip_over_width(cfg) if require_pushable_width else 0.0

    for obs in obstacles:
        if not (obs.difficulty > 0.0 and obs.difficulty < float("inf")):
            raise ValueError(
                f"{name}: obstacle {obs.oid!r} has difficulty "
                f"{obs.difficulty!r}, which is not a positive force")
        if min(obs.l, obs.d) < narrowest - 1e-9:
            raise ValueError(
                f"{name}: obstacle {obs.oid!r} is {min(obs.l, obs.d):.2f} m "
                f"across, under the {narrowest:.2f} m the robot needs to push "
                "it without tipping it over; widen it or make it static")
        if not workspace.covers(obs.polygon):
            raise ValueError(
                f"{name}: obstacle {obs.oid!r} is not inside the workspace")
        for wall in walls:
            overlap = obs.polygon.intersection(wall.polygon).area
            if overlap > _OVERLAP_EPS:
                raise ValueError(
                    f"{name}: obstacle {obs.oid!r} overlaps wall "
                    f"{wall.name!r} by {overlap:.4f} m^2")

    for index, first in enumerate(obstacles):
        for second in obstacles[index + 1:]:
            overlap = first.polygon.intersection(second.polygon).area
            if overlap > _OVERLAP_EPS:
                raise ValueError(
                    f"{name}: obstacles {first.oid!r} and {second.oid!r} "
                    f"overlap by {overlap:.4f} m^2")

    blocked = [wall.polygon for wall in walls]
    blocked.extend(obs.polygon for obs in obstacles)
    for label, point in (("start", start), ("goal", goal)):
        footprint = Point(point).buffer(cfg.robot_radius)
        if not workspace.covers(footprint):
            raise ValueError(f"{name}: {label} {tuple(point)} is off the map")
        if any(footprint.intersects(poly) for poly in blocked):
            raise ValueError(
                f"{name}: {label} {tuple(point)} sits inside an obstacle")
