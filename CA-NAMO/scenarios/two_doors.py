from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    workspace = box(0, 0, 30, 20)
    walls = [
        StaticObstacle(box(14.5, 0.0, 15.5, 3.0), "wall_lo"),
        StaticObstacle(box(14.5, 6.0, 15.5, 15.0), "wall_mid"),
        StaticObstacle(box(14.5, 18.0, 15.5, 20.0), "wall_hi"),
    ]

    movable = [
        MovableObstacle(
            x=15.0,
            y=4.5,
            l=1.4,
            d=2.6,
            h=1,
            theta=0.0,
            material="empty_cart",
            difficulty=0.291,
            oid=1,
        ),
        MovableObstacle(
            x=16.7,
            y=4.5,
            l=1.2,
            d=2.4,
            h=0.1,
            theta=0.0,
            material="foam_mat",
            difficulty=0.014,
            oid=2,
        ),
        MovableObstacle(
            x=9.0,
            y=12.0,
            l=2.0,
            d=2.0,
            h=0.9,
            theta=0.0,
            material="plastic_chair",
            difficulty=0.36,
            oid=3,
        ),
        MovableObstacle(
            x=22.0,
            y=10.0,
            l=2.0,
            d=1.5,
            h=0.8,
            theta=0.0,
            material="cardboard_box",
            difficulty=0.168,
            oid=4,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (5.0, 4.5),
        "goal": (25.0, 4.5),
        "cfg": Config(),
    }
