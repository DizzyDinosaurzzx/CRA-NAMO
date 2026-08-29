"""Physical coefficients and geometry checks shared by reference scenarios."""

from __future__ import annotations

from typing import Iterable, Sequence

from shapely.geometry import Point

G = 9.81                        # gravitational acceleration [m/s^2]

MU_CASTORS = 0.03               # free castors on a hard, clean floor
MU_CASTORS_FOULED = 0.22        # castors jammed by grit, debris or a brake
MU_RUBBER_WHEELS = 0.08         # trolley wheels over rubble or a threshold
MU_BRAKED_WHEELS = 0.60         # castors with the brake set, dragged anyway
MU_FELT_PADS = 0.25             # furniture pads on hard floor
MU_WOOD = 0.35                  # bare wood or plastic on hard floor
MU_UPHOLSTERY = 0.50            # sofa or mattress dragged on its base
MU_STEEL = 0.45                 # sheet-steel base on concrete or tile
MU_CONCRETE = 0.60              # concrete or masonry on concrete

_OVERLAP_EPS = 1e-7


def push_force(mass_kg: float, mu: float) -> float:
    """Ground-truth sliding resistance mu * m * g [N] of one real object."""
    if mass_kg <= 0.0:
        raise ValueError(f"mass must be positive, got {mass_kg!r} kg")
    if mu <= 0.0:
        raise ValueError(f"friction coefficient must be positive, got {mu!r}")
    return round(mu * mass_kg * G, 3)


def bulk_density(mass_kg: float, l: float, d: float, h: float) -> float:
    """Mass over bounding-box volume [kg/m^3] -- what the estimator must guess."""
    volume = l * d * h
    if volume <= 0.0:
        raise ValueError("bounding box must have positive volume")
    return mass_kg / volume


def tip_over_width(cfg) -> float:
    """Return the minimum obstacle width that avoids tipping."""
    if cfg.robot_push_height <= 0.0 or cfg.push_friction_mu <= 0.0:
        return 0.0
    return 2.0 * cfg.push_friction_mu * cfg.robot_push_height


def check_layout(name: str, *, workspace, static: Iterable, movable: Iterable,
                 start: Sequence[float], goal: Sequence[float], cfg,
                 require_pushable_width: bool = True) -> None:
    """Raise if authored coordinates violate map geometry constraints."""
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
