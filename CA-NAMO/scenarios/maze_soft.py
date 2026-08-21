"""软障碍迷宫场景：迷宫路径被低难度泡沫垫类障碍阻挡，考验推挪。"""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    """构建软障碍迷宫场景。"""
    workspace = box(0, 0, 30, 30)

    t = 0.45
    walls = [
        StaticObstacle(box(0.0, 0.0, 10.0, t), "outer_bottom_left"),
        StaticObstacle(box(15.0, 0.0, 30.0, t), "outer_bottom_right"),
        StaticObstacle(box(29.55, 0.0, 30.0, 30.0), "outer_right"),
        StaticObstacle(box(5.0, 29.55, 30.0, 30.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 25.0), "outer_left"),

        StaticObstacle(box(5.0, 25.5, 5.0 + t, 29.6), "lu_vertical"),
        StaticObstacle(box(5.0, 25.5, 14.8, 25.5 + t), "lu_horizontal"),

        StaticObstacle(box(0.0, 20.8, 10.0, 20.8 + t), "left_upper_horizontal"),
        StaticObstacle(box(5.0, 16.5, 5.0 + t, 20.8), "left_upper_vertical"),

        StaticObstacle(box(14.5, 20.8, 14.5 + t, 29.6), "upper_mid_vertical"),

        StaticObstacle(box(24.5, 25.5, 24.5 + t, 29.6), "ru_vertical_top"),
        StaticObstacle(box(19.8, 20.8, 30.0, 20.8 + t), "ru_horizontal_bottom"),
        StaticObstacle(box(19.8, 20.8, 19.8 + t, 25.6), "ru_vertical_left"),
        StaticObstacle(box(24.7, 16.7, 24.7 + t, 20.8), "ru_vertical_down"),

        StaticObstacle(box(2.0, 12.0, 10.0, 12.0+t), "mid_left_horizontal"),

        StaticObstacle(box(9.8, 7.8, 10.25, 16.7), "center_left_vertical"),
        StaticObstacle(box(9.8, 16.25, 19.8, 16.7), "center_top_horizontal"),
        StaticObstacle(box(9.8, 7.8, 24.8, 8.25), "center_bottom_horizontal"),
        StaticObstacle(box(14.5, 12.1, 14.95, 16.7), "center_inner_vertical"),

        StaticObstacle(box(20.0, 12.2, 30.0, 12.65), "mid_right_horizontal"),

        StaticObstacle(box(5.0, 0.0, 5.45, 5.2), "ll_vertical"),
    ]

    movable = [
        MovableObstacle(
            x=1.5,
            y=13.5,
            l=2,
            d=2,
            h=0.1,
            theta=0.0,
            material="foam_mat",
            difficulty=58.86,
            oid=1,
        ),

        MovableObstacle(
            x=19,
            y=14.4,
            l=3,
            d=3.4,
            h=0.1,
            theta=0.0,
            material="foam_mat",
            difficulty=150.093,
            oid=2,
        ),

        MovableObstacle(
            x=12.2,
            y=21.8,
            l=4,
            d=1.8,
            h=1.8,
            theta=0.0,
            material="empty_shelf",
            difficulty=2002.417,
            oid=3,
        ),
    ]

    start = (12.5, 1.5)
    goal = (2.5, 28.0)
    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": Config(),
    }
