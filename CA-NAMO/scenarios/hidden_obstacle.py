from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    workspace = box(0, 0, 28, 40)
    walls = [
        StaticObstacle(box(0.0, 19.5, 3.0, 20.5), "wall_left"),
        StaticObstacle(box(7.0, 19.5, 20.75, 20.5), "wall_mid"),
        StaticObstacle(box(25.25, 19.5, 28.0, 20.5), "wall_right"),
    ]

    movable = [
        MovableObstacle(
            x=5.0,
            y=20.0,
            l=3.4,
            d=1.4,
            h=1.6,
            theta=0.0,
            material="industrial_machines",
            difficulty=26149.536,
            oid=1,
        ),
        MovableObstacle(
            x=23.0,
            y=20.5,
            l=3.8,
            d=1.8,
            h=1,
            theta=0.0,
            material="styrofoam_box",
            difficulty=352.277,
            oid=2,
        ),
        MovableObstacle(
            x=23.0,
            y=18.8,
            l=3.8,
            d=1.4,
            h=1.6,
            theta=0.0,
            material="industrial_machine",
            difficulty=29225.952,
            oid=3,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (5.0, 35.0),
        "goal": (5.0, 5.0),
        "cfg": Config(),
    }
