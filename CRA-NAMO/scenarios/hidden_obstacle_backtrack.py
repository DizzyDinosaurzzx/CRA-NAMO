"""双门回溯场景：诱饵后藏超重障碍，逼出折返搬另一门障碍的回溯。"""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    """构建双门回溯场景。"""
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
            h=2,
            theta=0.0,
            material="steel_shelf",
            difficulty=14008.68,
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
        # 诱饵后方的陷阱：须重到让"折返约 20 m 搬开近门钢架"优于硬推——
        # oid 3 只需挪约 0.7 m 即可让开门洞，普通重物（29 kN / 33 kN）会输掉该比较而不触发回溯。
        MovableObstacle(
            x=23.0,
            y=18.8,
            l=3.8,
            d=1.4,
            h=1.6,
            theta=0.0,
            material="concrete_block",
            difficulty=120243.917,
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
