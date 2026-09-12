"""Hospital route-planning scenario.

The courier starts in the north-west service corridor and finishes in the
south-east service corridor.  Patient-room beds remain part of the map; the
former temporary bed, carts, occupied bed, and mobile X-ray are intentionally
omitted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from shapely.geometry import Point, box

from config import Config
from dynamics import Event, MoveTo, at_time
from obstacle import MovableObstacle, StaticObstacle


# A much smaller footprint than the original reference-derived map.
_WIDTH = 42.0
_HEIGHT = 28.0
_WALL_T = 0.35

# Clearance dimensions for hospital beds.
_DOOR_W = 2.9                    # leaves grid/swept-motion margin around a 2.1 m bed
_CORRIDOR_WIDTH = 3.0            # wider than the 2.3 m bed footprint
_MAIN_CORRIDOR = (12.5, 15.5)
_SOUTH_BYPASS = (0.35, 0.35 + _CORRIDOR_WIDTH)
_NORTH_BYPASS = (27.65 - _CORRIDOR_WIDTH, 27.65)

_BED_LENGTH = 2.3
_BED_WIDTH = 2.1                # 0.8 m narrower than each 2.9 m door
_BED_HEIGHT = 0.9
_BED_DIFFICULTY = 35.0          # wheeled, empty hospital bed

_EMERGENCY_END_X = 19.5
_INPATIENT_START_X = 22.5

_EVENT_BED_ID = "hospitalBed106"
_EVENT_BED_HOME = (18.1, 21.4, math.pi / 2.0)
_EVENT_BED_CORRIDOR = (16.2, 26.15, math.pi / 2.0)
_MIDDLE_EVENT_BED_112 = "hospitalBed112"
_MIDDLE_EVENT_BED_201 = "hospitalBed201"
_BED_112_MIDDLE_CORRIDOR = (16.2, 14.0, math.pi / 2.0)
_BED_201_MIDDLE_CORRIDOR = (25.7, 14.0, math.pi / 2.0)
_LOWER_EVENT_BED_210 = "hospitalBed210"
_LOWER_EVENT_BED_211 = "hospitalBed211"
_EVENT_BED_210_HOME = (34.1, 6.6, math.pi / 2.0)
_BED_210_LOWER_CORRIDOR = (32.2, 1.85, math.pi / 2.0)
_BED_211_LOWER_CORRIDOR = (38.7, 1.85, math.pi / 2.0)

_START = (2.0, 25.5)            # upper-left red circle
_GOAL = (40.0, 2.5)             # lower-right red circle


def _wall(p: tuple[float, float], q: tuple[float, float],
          name: str) -> StaticObstacle:
    return StaticObstacle.segment(p, q, _WALL_T, name)


def _cut_wall(axis: str, fixed: float, start: float, end: float,
              gaps: Sequence[tuple[float, float]], name: str):
    """Build one wall while leaving the requested door openings."""
    pieces = []
    cursor = start
    for index, (gap_start, gap_end) in enumerate(sorted(gaps)):
        gap_start = max(start, gap_start)
        gap_end = min(end, gap_end)
        if gap_start > cursor:
            p = (cursor, fixed) if axis == "h" else (fixed, cursor)
            q = (gap_start, fixed) if axis == "h" else (fixed, gap_start)
            pieces.append(_wall(p, q, f"{name}_{index}"))
        cursor = max(cursor, gap_end)
    if cursor < end:
        p = (cursor, fixed) if axis == "h" else (fixed, cursor)
        q = (end, fixed) if axis == "h" else (fixed, end)
        pieces.append(_wall(p, q, f"{name}_{len(pieces)}"))
    return pieces


def _hwall(y: float, x0: float, x1: float,
           gaps: Sequence[tuple[float, float]], name: str):
    return _cut_wall("h", y, x0, x1, gaps, name)


def _door(center: float) -> tuple[float, float]:
    return center - _DOOR_W / 2.0, center + _DOOR_W / 2.0


def _partitions(xs: Iterable[float], y0: float, y1: float, prefix: str):
    return [
        _wall((x, y0), (x, y1), f"{prefix}_{index}")
        for index, x in enumerate(xs)
    ]


def _outer_shell():
    t = _WALL_T
    return [
        StaticObstacle(box(0.0, 0.0, _WIDTH, t), "outer_south"),
        StaticObstacle(box(0.0, _HEIGHT - t, _WIDTH, _HEIGHT), "outer_north"),
        StaticObstacle(box(0.0, 0.0, t, _HEIGHT), "outer_west"),
        StaticObstacle(box(_WIDTH - t, 0.0, _WIDTH, _HEIGHT), "outer_east"),
    ]


def _department_rooms(x0: float, x1: float, partitions: Sequence[float],
                      door_centers: Sequence[float], prefix: str):
    """Create upper/lower rooms with access to both route alternatives."""
    main_south, main_north = _MAIN_CORRIDOR
    south_bypass_top = _SOUTH_BYPASS[1]
    north_bypass_bottom = _NORTH_BYPASS[0]
    doors = [_door(center) for center in door_centers]

    walls = []

    # Upper rooms connect the main corridor to the northern bypass.
    walls.extend(_hwall(main_north, x0, x1, doors,
                        f"{prefix}_upper_main_doors"))
    walls.extend(_hwall(north_bypass_bottom, x0, x1, doors,
                        f"{prefix}_upper_bypass_doors"))
    walls.extend(_partitions(partitions, main_north, north_bypass_bottom,
                             f"{prefix}_upper_partition"))

    # Lower rooms mirror the arrangement and connect to the southern bypass.
    walls.extend(_hwall(main_south, x0, x1, doors,
                        f"{prefix}_lower_main_doors"))
    walls.extend(_hwall(south_bypass_top, x0, x1, doors,
                        f"{prefix}_lower_bypass_doors"))
    walls.extend(_partitions(partitions, south_bypass_top, main_south,
                             f"{prefix}_lower_partition"))

    # The 3 m opening between departments remains free for temporary objects.
    walls.extend([
        _wall((x0, main_north), (x0, north_bypass_bottom),
              f"{prefix}_upper_west"),
        _wall((x1, main_north), (x1, north_bypass_bottom),
              f"{prefix}_upper_east"),
        _wall((x0, south_bypass_top), (x0, main_south),
              f"{prefix}_lower_west"),
        _wall((x1, south_bypass_top), (x1, main_south),
              f"{prefix}_lower_east"),
    ])
    return walls


def _emergency_department():
    """Triage, resuscitation, treatment, and observation rooms."""
    return _department_rooms(
        x0=0.35,
        x1=_EMERGENCY_END_X,
        partitions=(6.5, 13.0),
        door_centers=(3.3, 9.7, 16.2),
        prefix="emergency",
    )


def _inpatient_ward():
    """Six inpatient rooms around the same three-route circulation system."""
    return _department_rooms(
        x0=_INPATIENT_START_X,
        x1=_WIDTH - 0.35,
        partitions=(29.0, 35.5),
        door_centers=(25.7, 32.2, 38.7),
        prefix="inpatient",
    )


def _department_beds(edges: Sequence[float], oid_start: int):
    """Place two beds in every upper and lower room, matching the blue marks."""
    beds = []
    row_heights = (
        ("upper", 19.1, 21.4),
        ("lower", 6.6, 8.9),
    )

    oid = oid_start
    for row_name, near_y, far_y in row_heights:
        for room_index, (left, right) in enumerate(zip(edges, edges[1:])):
            left_y, right_y = (
                (near_y, far_y) if room_index % 2 == 0
                else (far_y, near_y)
            )
            for side, x, y in (
                ("left", left + 1.4, left_y),
                ("right", right - 1.4, right_y),
            ):
                beds.append(MovableObstacle(
                    x=x,
                    y=y,
                    l=_BED_LENGTH,
                    d=_BED_WIDTH,
                    h=_BED_HEIGHT,
                    theta=math.pi / 2.0,
                    material="empty_hospital_bed",
                    difficulty=_BED_DIFFICULTY,
                    oid=f"hospitalBed{oid}",
                ))
                oid += 1
    return beds


def _hospital_beds():
    return [
        *_department_beds((0.35, 6.5, 13.0, _EMERGENCY_END_X), 101),
        *_department_beds((_INPATIENT_START_X, 29.0, 35.5,
                           _WIDTH - 0.35), 201),
    ]


def _validate_geometry(workspace, walls, movable, cfg: Config):
    """Fail early if an authored event object starts in invalid geometry."""
    inner = workspace.buffer(-_WALL_T)
    for obs in movable:
        if not inner.covers(obs.polygon):
            raise ValueError(f"hospital obstacle {obs.oid} lies outside the map")
        for wall in walls:
            overlap = obs.polygon.intersection(wall.polygon).area
            if overlap > 1e-9:
                raise ValueError(
                    f"hospital obstacle {obs.oid} overlaps {wall.name!r} "
                    f"by {overlap:.4f} m^2")

    for index, first in enumerate(movable):
        for second in movable[index + 1:]:
            overlap = first.polygon.intersection(second.polygon).area
            if overlap > 1e-9:
                raise ValueError(
                    f"hospital obstacles {first.oid} and {second.oid} overlap "
                    f"by {overlap:.4f} m^2")

    blocked = [wall.polygon for wall in walls]
    blocked.extend(obs.polygon for obs in movable)
    for name, point in (("start", _START), ("goal", _GOAL)):
        footprint = Point(point).buffer(cfg.robot_radius)
        if any(footprint.intersects(poly) for poly in blocked):
            raise ValueError(f"hospital {name} lies in an obstacle")


def create():
    """Build the compact hospital map."""
    workspace = box(0.0, 0.0, _WIDTH, _HEIGHT)
    walls = [
        *_outer_shell(),
        *_emergency_department(),
        *_inpatient_ward(),
    ]
    movable = _hospital_beds()

    # The bed takes about 6 s to reach the upper corridor.  It arrives near
    # t=31 s, remains across the corridor for roughly 10 s, then returns home.
    # A static run reaches this section around t=43--65 s; the robot perceives
    # the blockage while approaching, but is still far enough away for the bed
    # to use the doorway safely on its return trip.
    events = [
        Event(
            name="bed106 is pushed into the upper corridor",
            trigger=at_time(25.0),
            effect=MoveTo(
                oid=_EVENT_BED_ID,
                goal=_EVENT_BED_CORRIDOR,
                speed=1.0,
            ),
        ),
        Event(
            name="bed112 is pushed into the middle corridor and left there",
            trigger=at_time(28.0),
            effect=MoveTo(
                oid=_MIDDLE_EVENT_BED_112,
                goal=_BED_112_MIDDLE_CORRIDOR,
                speed=1.0,
            ),
        ),
        Event(
            name="bed106 is pushed back after blocking for 10 seconds",
            trigger=at_time(41.0),
            effect=MoveTo(
                oid=_EVENT_BED_ID,
                goal=_EVENT_BED_HOME,
                speed=1.0,
            ),
        ),
        Event(
            name="bed201 is pushed into the middle corridor and left there",
            trigger=at_time(58.0),
            effect=MoveTo(
                oid=_MIDDLE_EVENT_BED_201,
                goal=_BED_201_MIDDLE_CORRIDOR,
                speed=1.0,
            ),
        ),
        Event(
            name="bed210 is pushed into the lower corridor",
            trigger=at_time(170.0),
            effect=MoveTo(
                oid=_LOWER_EVENT_BED_210,
                goal=_BED_210_LOWER_CORRIDOR,
                speed=1.0,
            ),
        ),
        Event(
            name="bed210 is pushed back after blocking for 10 seconds",
            trigger=at_time(186.0),
            effect=MoveTo(
                oid=_LOWER_EVENT_BED_210,
                goal=_EVENT_BED_210_HOME,
                speed=1.0,
            ),
        ),
        Event(
            name="bed211 is pushed into the lower corridor and left there",
            trigger=at_time(190.0),
            effect=MoveTo(
                oid=_LOWER_EVENT_BED_211,
                goal=_BED_211_LOWER_CORRIDOR,
                speed=1.0,
            ),
        ),
    ]

    cfg = Config(
        # A 0.8 m courier cannot squeeze through the 0.35 m gaps beside
        # corridor beds; this footprint also applies to execution and contact.
        robot_radius=0.3,
        grid_step=0.3,
        conn_radius=0.65,
        R_perc=7.5,
        R_manip=4.5,
        time_importance=0.32,
        time_value=100.0,
        dynamic_speed=0.4,
        dynamic_step=0.25,
        dynamic_wait_step=1.5,
        dynamic_max_wait=90.0,
        use_llm_ordering=False,
        save_frames=True,
        show_global_obstacles=True,
    )

    _validate_geometry(workspace, walls, movable, cfg)

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "dynamics": events,
        "cfg": cfg,
        "decision_points": [],
    }
