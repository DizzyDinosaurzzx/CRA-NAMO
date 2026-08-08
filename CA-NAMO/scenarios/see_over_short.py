"""See-over-a-low-obstacle demo.

Minimal scenario for the height rule in `perception.py — _point_visible`:
a blocker no taller than half the target (`w.h <= target.h / 2`) leaves the
target's upper half exposed, so it does not occlude it.

Layout — everything sits on the x = 10 line, straight ahead of the robot:

        y=14  * goal
        y=10.5  [====]      tall shelf   l=3.0  h=2.0   <- oid 2
        y=7   [==========]  low pallets  l=6.0  h=0.5   <- oid 1
        y=2   R start

The low stack is wider than the tall one and sits between the robot and it, so
oid 2 lies entirely inside oid 1's geometric shadow: every line of sight from
the start to any corner of oid 2 crosses oid 1's footprint. The only reason
oid 2 is perceived at all is that oid 1 is short enough to see over.

Expected result: both obstacles are revealed on the very first perception
sweep, before the robot has moved (Replan cycle 0 reports 2 obstacles).

To flip the test, raise the front obstacle above half the rear one's height —
`h=1.5` on oid 1 makes 1.5 > 2.0/2, so oid 2 stays hidden until the robot
walks around the stack.
"""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    workspace = box(0, 0, 20, 16)

    # No walls: the only occluder in play is the low obstacle itself, so nothing
    # about the result can be blamed on static line-of-sight blocking.
    walls: list[StaticObstacle] = []

    movable = [
        # front, low and wide — casts a full geometric shadow over oid 2
        MovableObstacle(
            x=10.0,
            y=7.0,
            l=6.0,
            d=1.0,
            h=0.5,
            theta=0.0,
            material="pallet",
            difficulty=2048.328,
            oid=1,
        ),
        # behind, tall and narrow — visible over the top of oid 1
        MovableObstacle(
            x=10.0,
            y=10.5,
            l=3.0,
            d=1.0,
            h=2.0,
            theta=0.0,
            material="steel_shelf",
            difficulty=8829.0,
            oid=2,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (10.0, 2.0),
        "goal": (10.0, 14.0),
        "cfg": Config(),
    }
