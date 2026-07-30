from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    workspace = box(0, 0, 40, 8)
    t = 0.5

    walls = [
        StaticObstacle(box(0.0, 7.5, 40.0, 8.0), "wall_top"),
        StaticObstacle(box(0.0, 0.0, 40.0, 0.5), "wall_bottom"),
        StaticObstacle(box(18.5, 6.5, 19.5, 7.0), "wall1"),
        StaticObstacle(box(20.5, 1.0, 21.5, 1.5), "wall2"),
    ]

    movable = [
        MovableObstacle(
            x=20.0, y=4.0,
            l=0.7, d=6, theta=0.0,
            material="wooden_crate",
            difficulty=1.0,
            oid=1,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (4.0, 4.0),
        "goal": (36.0, 4.0),
        "cfg": Config(),
    }
