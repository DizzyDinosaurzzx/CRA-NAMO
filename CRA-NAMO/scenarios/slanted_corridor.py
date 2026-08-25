"""Three rooms separated by slanted walls, each doorway plugged by an obstacle.

Nothing but the outer shell is axis-aligned: the map exists to exercise walls
that carry a heading the way obstacles do. Both partitions are laid out as a
line with a doorway cut out of it, and the obstacle filling each doorway lies
along the same heading as the wall it sits in — with less than one robot
diameter of daylight either side, so the run has to move it rather than squeeze
past. The cheap cart comes first; only once through it does the robot see the
heavy crate in the second doorway.
"""

from __future__ import annotations
import math
from typing import List, Tuple

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle

_WALL_T = 0.6                    # partition thickness
_SHELL_T = 0.5                   # outer shell thickness

# Diagonal partition between the middle wedge and the right room. Both endpoints
# overrun into the shell, so the joins seal instead of leaving a slit.
_BARRIER_A = (9.8, 0.2)
_BARRIER_B = (20.2, 15.8)
_BARRIER_DOOR = (0.5, 3.6)       # (fraction along the wall, gap width [m])

# Partition between the near room and the wedge, tilted the other way. It hangs
# off the top wall and runs down into the barrier, so the wedge has exactly one
# way in and one way out.
_BAFFLE_TOP = (6.0, 15.8)
_BAFFLE_FOOT_T = 0.12            # where it meets the barrier, as a fraction of it
_BAFFLE_OVERRUN = 0.5            # push past that join so the corner is sealed
_BAFFLE_DOOR = (0.45, 2.6)

# Total daylight left around the obstacle plugging a doorway: 0.3 m either
# side, against a robot 0.4 m across. Widen it past 0.4 and the robot slips
# through without moving anything, and the map stops testing what it is for.
_DOOR_DAYLIGHT = 0.6


def _lerp(a: Tuple[float, float], b: Tuple[float, float], t: float):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _wall_with_door(a, b, door, name: str) -> Tuple[List[StaticObstacle], dict]:
    """Split the wall a->b into two pieces, leaving a doorway of the given width.

    Returns the pieces plus the pose of the doorway itself — centre, heading and
    width — which is what an obstacle needs to sit in it flush.
    """
    at, width = door
    span = math.hypot(b[0] - a[0], b[1] - a[1])
    half = 0.5 * width / span
    return (
        [StaticObstacle.segment(a, _lerp(a, b, at - half), _WALL_T, f"{name}_lower"),
         StaticObstacle.segment(_lerp(a, b, at + half), b, _WALL_T, f"{name}_upper")],
        {"centre": _lerp(a, b, at),
         "theta": math.atan2(b[1] - a[1], b[0] - a[0]),
         "width": width},
    )


def create():
    workspace = box(0, 0, 30, 16)
    t = _SHELL_T

    barrier_walls, barrier_door = _wall_with_door(
        _BARRIER_A, _BARRIER_B, _BARRIER_DOOR, "barrier")

    # the baffle runs from the top wall down to (and a little past) the barrier
    foot = _lerp(_BARRIER_A, _BARRIER_B, _BAFFLE_FOOT_T)
    reach = math.hypot(foot[0] - _BAFFLE_TOP[0], foot[1] - _BAFFLE_TOP[1])
    baffle_foot = _lerp(_BAFFLE_TOP, foot, 1.0 + _BAFFLE_OVERRUN / reach)
    baffle_walls, baffle_door = _wall_with_door(
        _BAFFLE_TOP, baffle_foot, _BAFFLE_DOOR, "baffle")

    walls = [
        # outer shell, stated endpoint-to-endpoint like the slanted walls
        StaticObstacle.segment((0.0, t / 2), (30.0, t / 2), t, "outer_bottom"),
        StaticObstacle.segment((0.0, 16.0 - t / 2), (30.0, 16.0 - t / 2), t, "outer_top"),
        StaticObstacle.segment((t / 2, 0.0), (t / 2, 16.0), t, "outer_left"),
        StaticObstacle.segment((30.0 - t / 2, 0.0), (30.0 - t / 2, 16.0), t, "outer_right"),
        *baffle_walls,
        *barrier_walls,
    ]

    movable = [
        # cart plugging the baffle doorway: light, and the first thing seen
        MovableObstacle(
            x=baffle_door["centre"][0], y=baffle_door["centre"][1],
            l=baffle_door["width"] - _DOOR_DAYLIGHT, d=0.9, h=0.9,
            theta=baffle_door["theta"],
            material="empty_cart",
            difficulty=35.708,
            oid=1,
        ),
        # crate plugging the barrier doorway, hidden behind the baffle until then
        MovableObstacle(
            x=barrier_door["centre"][0], y=barrier_door["centre"][1],
            l=barrier_door["width"] - _DOOR_DAYLIGHT, d=0.9, h=0.8,
            theta=barrier_door["theta"],
            material="wooden_crate",
            difficulty=412.6,
            oid=2,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (3.0, 8.0),
        "goal": (27.0, 8.0),
        "cfg": Config(),
    }
