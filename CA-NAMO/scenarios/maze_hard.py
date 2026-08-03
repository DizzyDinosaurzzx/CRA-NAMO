from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    """30x30 maze with 3 high-difficulty obstacles (steel_shelf 15 / steel_safe 20 / concrete_block 100)."""
    workspace = box(0, 0, 30, 30)
    t = 0.45
    walls = [
        StaticObstacle(box(0.0, 0.0, 10.0, t), "outer_bottom_left"),
        StaticObstacle(box(15.0, 0.0, 30.0, t), "outer_bottom_right"),
        StaticObstacle(box(29.55, 0.0, 30.0, 30.0), "outer_right"),
        StaticObstacle(box(5.0, 29.55, 30.0, 30.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 25.0), "outer_left"),

        StaticObstacle(box(5.5, 25.5, 14.8, 25.5 + t), "lu_horizontal"),

        StaticObstacle(box(0.0, 20.8, 10.0, 20.8 + t), "left_upper_horizontal"),
        StaticObstacle(box(5.0, 16.5, 5.0 + t, 20.8), "left_upper_vertical"),

        StaticObstacle(box(14.5, 20.8, 14.5 + t, 28), "upper_mid_vertical"),
        StaticObstacle(box(11.5, 28, 11.5 + t, 29.6), "upper_mid_vertical"),
        StaticObstacle(box(9.5, 25.6, 9.5 + t, 28), "upper_mid_vertical"),
        StaticObstacle(box(7.5, 28, 7.5 + t, 29.6), "upper_mid_vertical"),
        StaticObstacle(box(5.5, 25.6, 5.5 + t, 28), "upper_mid_vertical"),

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
            h=0.8,
            theta=0.0,
            material="box",
            difficulty=0.224,
            oid=1,
        ),

        MovableObstacle(
            x=11,
            y=21.8,
            l=2,
            d=1.8,
            h=1.5,
            theta=0.0,
            material="steel_safe",
            difficulty=29.7,
            oid=3,
        ),

                MovableObstacle(
                    x=13.5,
                    y=23,
                    l=2,
                    d=1.8,
                    h=2,
                    theta=0.0,
                    material="steel_shelf",
                    difficulty=30.24,
                    oid=4,
                ),
    ]

    start = (12.5, 1.5)
    goal= (2.5, 28.0)

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": Config(),
    }
