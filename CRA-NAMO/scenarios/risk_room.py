"""Two ways through one wall, and the cheap one is the one you must not take.

A demonstration map for the risk model. The direct doorway is plugged by an
occupied wheelchair — trivial to push, since it rolls, and unthinkable to push,
since someone is sitting in it. The long way round is plugged by a sealed crate
that looks like ordinary freight. Energy alone says take the short door; risk
says take the long one, and the gap between the two surcharges is far wider than
the detour, so the decision is not close.

The crate is the second half of the demonstration: it reads as `low` from a
distance, and only physical contact reveals what is inside it. That re-rating
arrives after the robot has committed to the long route and is standing next to
the crate, which is exactly when a re-assessment is worth having.
"""

from __future__ import annotations

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle

_WALL_T = 0.5
# Each doorway is 2.0 m; each obstacle is 1.4 m across it, leaving 0.3 m either
# side against a robot 0.4 m wide. Widen either and the robot slips through
# without touching anything, and the map stops testing what it is for.
_DIRECT_DOOR = (9.0, 11.0)      # y-range of the doorway on the straight line
_LONG_DOOR = (17.0, 19.0)       # y-range of the doorway that costs a detour


def create():
    workspace = box(0, 0, 30, 20)
    t = _WALL_T

    walls = [
        StaticObstacle.segment((0.0, t / 2), (30.0, t / 2), t, "outer_bottom"),
        StaticObstacle.segment((0.0, 20.0 - t / 2), (30.0, 20.0 - t / 2), t, "outer_top"),
        StaticObstacle.segment((t / 2, 0.0), (t / 2, 20.0), t, "outer_left"),
        StaticObstacle.segment((30.0 - t / 2, 0.0), (30.0 - t / 2, 20.0), t, "outer_right"),
        # the dividing wall, in three pieces: two doorways cut out of it
        StaticObstacle.segment((15.0, 0.0), (15.0, _DIRECT_DOOR[0]), 0.6, "divider_lower"),
        StaticObstacle.segment((15.0, _DIRECT_DOOR[1]), (15.0, _LONG_DOOR[0]), 0.6, "divider_middle"),
        StaticObstacle.segment((15.0, _LONG_DOOR[1]), (15.0, 20.0), 0.6, "divider_upper"),
    ]

    movable = [
        # straight ahead, and off limits: it rolls, so `difficulty` is tiny, and
        # only the risk term stands between the robot and pushing a person aside
        MovableObstacle(
            x=15.0, y=sum(_DIRECT_DOOR) / 2,
            l=1.1, d=1.4, h=1.3, theta=0.0,
            material="occupied_wheelchair",
            difficulty=28.0,
            oid=1,
        ),
        # the long way round: freight, until the robot lays a hand on it
        MovableObstacle(
            x=15.0, y=sum(_LONG_DOOR) / 2,
            l=1.1, d=1.4, h=1.0, theta=0.0,
            material="sealed_crate",
            contact_reveals="crate_of_gas_cylinders",
            difficulty=520.0,
            oid=2,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (4.0, 10.0),
        "goal": (26.0, 10.0),
        "cfg": Config(),
    }
