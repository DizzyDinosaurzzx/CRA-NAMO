"""覆盖代价、风险、隐藏状态和在线重规划的场景。"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from shapely.geometry import box

from config import Config
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle

XY = Tuple[float, float]

_SHELL_T = 0.5
_WALL_T = 0.6

    # 门洞阻塞物周围的总净空。
_DOOR_DAYLIGHT = 0.6

    # 门元组为（沿墙比例，开口宽度）。
_WEST_WALL = ((12.0, 0.3), (15.0, 23.7))
_CABINET_DOOR = (0.12, 2.6)
_WHEELCHAIR_DOOR = (0.45, 1.8)
_CART_DOOR = (0.92, 2.6)

    # 隐藏阻塞物的间隙必须窄于机器人。
_TRAP_GAP = 0.35

_EAST_WALL = ((25.0, 0.3), (22.0, 23.7))
_SEALED_DOOR = (0.70, 2.4)
_CRATE_DOOR = (0.15, 2.6)

_START = (3.5, 6.0)
_GOAL = (32.5, 20.5)

_SHELF_ENDS = ((25.2, 17.5), (34.9, 8.0))


def _lerp(a: XY, b: XY, t: float) -> XY:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _wall_with_doors(a: XY, b: XY, doors: Sequence[Tuple[float, float]],
                     name: str) -> Tuple[List[StaticObstacle], List[dict]]:
    """根据相对位置和宽度构建带开口的墙。"""
    span = math.hypot(b[0] - a[0], b[1] - a[1])
    theta = math.atan2(b[1] - a[1], b[0] - a[0])
    cuts = sorted((at, 0.5 * width / span) for at, width in doors)

    pieces: List[StaticObstacle] = []
    edge = 0.0
    for i, (at, half) in enumerate(cuts):
        if at - half - edge > 1e-9:
            pieces.append(StaticObstacle.segment(
                _lerp(a, b, edge), _lerp(a, b, at - half), _WALL_T, f"{name}_{i}"))
        edge = at + half
    if 1.0 - edge > 1e-9:
        pieces.append(StaticObstacle.segment(
            _lerp(a, b, edge), b, _WALL_T, f"{name}_{len(cuts)}"))

    gaps = [{"centre": _lerp(a, b, at), "theta": theta, "width": 2.0 * half * span}
            for at, half in cuts]
    return pieces, gaps


def _plug(gap: dict, d: float, h: float, material: str, oid: int,
          difficulty: float | None = None,
          contact_reveals: str = "") -> MovableObstacle:
    """创建与门洞对齐并填满门洞的障碍物。"""
    l = gap["width"] - _DOOR_DAYLIGHT
    if difficulty is None:
        difficulty = round(friction_force(material_mu_rho(material), l * d * h), 3)
    return MovableObstacle(
        x=gap["centre"][0], y=gap["centre"][1],
        l=l, d=d, h=h, theta=gap["theta"],
        material=material, difficulty=difficulty,
        contact_reveals=contact_reveals, oid=oid,
    )


def _behind(gap: dict, d: float, h: float, material: str, oid: int,
            overhang: float = 0.3) -> MovableObstacle:
    """在门洞远侧创建横跨门洞的隐藏阻挡物。"""
    theta = gap["theta"]
    normal = (math.sin(theta), -math.cos(theta))
    offset = _WALL_T / 2.0 + _TRAP_GAP + d / 2.0
    cx, cy = gap["centre"]
    return MovableObstacle(
        x=cx + normal[0] * offset, y=cy + normal[1] * offset,
        l=gap["width"] + 2.0 * overhang, d=d, h=h, theta=theta,
        material=material,
        difficulty=round(friction_force(
            material_mu_rho(material), (gap["width"] + 2.0 * overhang) * d * h), 3),
        oid=oid,
    )


def _shelf(ends: Tuple[XY, XY], d: float, h: float, material: str,
           oid: int) -> MovableObstacle:
    """创建沿两点连线放置、横向宽度为 d 的物体。"""
    (ax, ay), (bx, by) = ends
    l = math.hypot(bx - ax, by - ay)
    return MovableObstacle(
        x=0.5 * (ax + bx), y=0.5 * (ay + by),
        l=l, d=d, h=h, theta=math.atan2(by - ay, bx - ax),
        material=material,
        difficulty=round(friction_force(material_mu_rho(material), l * d * h), 3),
        oid=oid,
    )


def _reject_overlaps(walls: Sequence[StaticObstacle],
                     movable: Sequence[MovableObstacle]) -> None:
    """拒绝可移动障碍物与墙体重叠的场景几何。"""
    for obs in movable:
        body = obs.polygon
        for wall in walls:
            overlap = body.intersection(wall.polygon).area
            if overlap > 1e-9:
                raise ValueError(
                    f"obstacle {obs.oid} ({obs.material}) overlaps wall "
                    f"{wall.name!r} by {overlap:.4f} m^2")


def create():
    """创建策略演示场景。"""
    workspace = box(0, 0, 36, 24)
    t = _SHELL_T

    west_walls, west_doors = _wall_with_doors(
        *_WEST_WALL, (_CABINET_DOOR, _WHEELCHAIR_DOOR, _CART_DOOR), "west_divider")
    east_walls, east_doors = _wall_with_doors(
        *_EAST_WALL, (_SEALED_DOOR, _CRATE_DOOR), "east_divider")
    cabinet_door, wheelchair_door, cart_door = west_doors
    crate_door, sealed_door = east_doors

    walls = [
        StaticObstacle.segment((0.0, t / 2), (36.0, t / 2), t, "outer_bottom"),
        StaticObstacle.segment((0.0, 24.0 - t / 2), (36.0, 24.0 - t / 2), t, "outer_top"),
        StaticObstacle.segment((t / 2, 0.0), (t / 2, 24.0), t, "outer_left"),
        StaticObstacle.segment((36.0 - t / 2, 0.0), (36.0 - t / 2, 24.0), t, "outer_right"),
        *west_walls,
        *east_walls,
    ]

    movable = [
    # 物理代价低，但因人员风险而避开。
        _plug(wheelchair_door, d=0.85, h=1.3, material="occupied_wheelchair",
              difficulty=28.0, oid=1),
    # 廉价小车遮挡着代价高的阻塞物。
        _plug(cart_door, d=0.9, h=1.0, material="empty_cart", oid=2),
        _behind(cart_door, d=0.9, h=1.0, material="concrete_block", oid=3),
        _plug(cabinet_door, d=0.9, h=1.8, material="filing_cabinet", oid=4),
    # 接触后揭示危险内容物。
        _plug(sealed_door, d=0.7, h=1.0, material="sealed_crate",
              contact_reveals="crate_of_gas_cylinders", difficulty=1200.0, oid=5),
        _plug(crate_door, d=0.9, h=1.0, material="wooden_crate", oid=6),
    # 沉重货架使绕行更便宜。
        _shelf(_SHELF_ENDS, d=1.2, h=2.0, material="steel_shelf", oid=7),
    ]

    _reject_overlaps(walls, movable)

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "cfg": Config(),
    }
