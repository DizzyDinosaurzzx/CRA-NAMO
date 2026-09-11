"""Shared geometry for topology templates."""

from __future__ import annotations

import math

from shapely.geometry import LineString, Point, box

from obstacle import StaticObstacle
from scenario_generation.models import RandomScenarioRequest, TopologyResult


def shell(width: float, height: float, thickness: float) -> list[StaticObstacle]:
    return [
        StaticObstacle(box(0.0, 0.0, width, thickness), "shell_south"),
        StaticObstacle(box(0.0, height - thickness, width, height), "shell_north"),
        StaticObstacle(box(0.0, 0.0, thickness, height), "shell_west"),
        StaticObstacle(box(width - thickness, 0.0, width, height), "shell_east"),
    ]


def partition_with_openings(
    *, x: float, height: float, thickness: float,
    openings: list[tuple[float, float]], tilt: float, name: str,
) -> tuple[list[StaticObstacle], list[tuple[float, float, float]]]:
    """Build one slightly tilted partition and return its opening poses."""
    margin = thickness + 0.08
    lo, hi = margin, height - margin
    cuts = sorted((max(lo, y - size / 2), min(hi, y + size / 2), y)
                  for y, size in openings)
    spans: list[tuple[float, float]] = []
    edge = lo
    for cut_lo, cut_hi, _ in cuts:
        if cut_lo > edge:
            spans.append((edge, cut_lo))
        edge = max(edge, cut_hi)
    if edge < hi:
        spans.append((edge, hi))

    # The line direction is vertical plus a small tilt.
    vx, vy = math.sin(tilt), math.cos(tilt)
    walls = []
    for i, (a, b) in enumerate(spans):
        p = (x + vx * (a - height / 2), height / 2 + vy * (a - height / 2))
        q = (x + vx * (b - height / 2), height / 2 + vy * (b - height / 2))
        walls.append(StaticObstacle.segment(p, q, thickness, f"{name}_{i}"))

    poses = []
    for _, _, y in cuts:
        poses.append((x + vx * (y - height / 2),
                      height / 2 + vy * (y - height / 2),
                      math.atan2(vy, vx)))
    return walls, poses


def staged_topology(request: RandomScenarioRequest, rng, *, corridor: bool,
                    wall_angle_sigma: float | None = None) -> TopologyResult:
    width, height = request.width, request.height
    wall_t = 0.38
    workspace = box(0.0, 0.0, width, height)
    walls = shell(width, height, wall_t)

    direct_y = height * (0.38 if corridor else 0.31)
    bypass_y = height * (0.69 if corridor else 0.73)
    emergency_y = height * 0.89
    door_h = max(1.45, min(2.0, height * 0.13))
    gate_count = max(3, min(4, request.max_decision_points))
    gate_xs = [width * (i + 1) / (gate_count + 1) for i in range(gate_count)]
    gates = []
    for i, base_x in enumerate(gate_xs):
        x = base_x + rng.uniform(-0.22, 0.22)
        sigma = ((0.025 if corridor else 0.045)
                 if wall_angle_sigma is None else wall_angle_sigma)
        tilt = rng.gauss(0.0, sigma)
        pieces, poses = partition_with_openings(
            x=x, height=height, thickness=wall_t,
            openings=[(direct_y, door_h), (bypass_y, door_h),
                      (emergency_y, door_h)],
            tilt=tilt, name=f"gate_{i}")
        walls.extend(pieces)
        gates.append({
            "index": i,
            "direct": poses[0],
            "bypass": poses[1],
            "emergency": poses[2],
            "door_height": door_h,
            "wall_thickness": wall_t,
            "tilt": tilt,
        })

    start = (1.4, direct_y)
    goal = (width - 1.4, direct_y)
    direct_route = LineString([start, goal]).buffer(0.75)
    bypass_route = LineString([
        start, (width * 0.25, bypass_y),
        (width * 0.75, bypass_y), goal,
    ]).buffer(0.7)
    emergency_route = LineString([
        start, (width * 0.18, emergency_y),
        (width * 0.82, emergency_y), goal,
    ]).buffer(0.7)
    anchors = {
        "gates": gates,
        "direct_y": direct_y,
        "bypass_y": bypass_y,
        "emergency_y": emergency_y,
        "background_zones": [
            box(0.8, height * 0.48, width - 0.8, height * 0.60),
            box(0.8, height * 0.82, width - 0.8, height - 0.8),
            box(0.8, 0.8, width - 0.8, height * 0.16),
        ],
    }
    return TopologyResult(
        workspace=workspace, walls=walls, start=start, goal=goal,
        anchors=anchors, reserved_regions=[direct_route, bypass_route,
                                           emergency_route,
                                           Point(start).buffer(0.8),
                                           Point(goal).buffer(0.8)])
