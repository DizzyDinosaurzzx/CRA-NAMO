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
        # A：堵住近处门口，很容易推开。
        MovableObstacle(
            x=15.0,
            y=4.5,
            l=1.4,
            d=2.6,
            theta=0.0,
            material="empty_cart",      # 0.08 x 3.64 = 0.29
            difficulty=0.3,
            oid=1,
        ),
        # B：紧邻墙体东侧，被 A 遮挡，几乎没有阻力。
        MovableObstacle(
            x=16.7,
            y=4.5,
            l=1.2,
            d=2.4,
            theta=0.0,
            material="foam_mat",        # 0.05 x 2.88 = 0.14
            difficulty=0.15,
            oid=2,
        ),
        # 其他位置的干扰物，用于测试感知。
        MovableObstacle(
            x=9.0,
            y=12.0,
            l=2.0,
            d=2.0,
            theta=0.0,
            material="plastic_chair",   # 0.10 x 4.00 = 0.40
            difficulty=0.4,
            oid=3,
        ),
        MovableObstacle(
            x=22.0,
            y=10.0,
            l=2.0,
            d=1.5,
            theta=0.0,
            material="cardboard_box",   # 0.07 x 3.00 = 0.21
            difficulty=0.2,
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
