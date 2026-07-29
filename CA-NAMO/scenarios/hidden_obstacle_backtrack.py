from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    workspace = box(0, 0, 28, 40)
    walls = [
        StaticObstacle(box(0.0, 19.5, 3.0, 20.5), "wall_left"),
        # 近门：x = 3.0 ~ 7.0（宽 4.0）  远门：x = 20.75 ~ 25.25（宽 4.5）
        StaticObstacle(box(7.0, 19.5, 20.75, 20.5), "wall_mid"),
        StaticObstacle(box(25.25, 19.5, 28.0, 20.5), "wall_right"),
    ]

    movable = [
        # A：堵住近门。中等偏重但推得动——这正是折返方案能赢的原因。
        MovableObstacle(
            x=5.0,
            y=20.0,
            l=3.4,
            d=1.4,
            theta=0.0,
            material="steel_shelf",          # 4.20 x 4.76 = 19.99
            difficulty=20.0,
            oid=1,
        ),
        # B：堵住远门，位于 C 的上方。便宜到几乎免费，是把机器人"骗"去远门的诱饵。
        # B 的底部与 C 的顶部相接，但二者不重叠。
        MovableObstacle(
            x=23.0,
            y=20.5,
            l=3.8,
            d=1.8,
            theta=0.0,
            material="styrofoam_box",        # 0.004 x 6.84 = 0.027
            difficulty=0.03,
            oid=2,
        ),
        # C：从上方房间观察时被 B 完全遮挡，难度极高。
        # C 的顶部与墙体底面相接，移开 B 后无法从细缝绕过 C。
        MovableObstacle(
            x=23.0,
            y=18.8,
            l=3.8,
            d=1.4,
            theta=0.0,
            material="industrial_machine",   # 37.5 x 5.32 = 199.5
            difficulty=200.0,
            oid=3,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (5.0, 35.0),
        "goal": (5.0, 5.0),
        "cfg": Config(lambda_distance=1.0),
    }
