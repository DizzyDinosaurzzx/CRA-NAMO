"""近门有 A、远门有 B，且 C 被 B 遮挡的双门地图。"""

from __future__ import annotations

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    """创建 two_doors_hidden_c 地图。"""
    workspace = box(0, 0, 28, 40)

    # y 位于 [19.5, 20.5] 的水平墙。
    # 近门 x[3, 7] 与起点和终点对齐；远门 x[21, 25]。
    walls = [
        StaticObstacle(box(0.0, 19.5, 3.0, 20.5), "wall_left"),
        StaticObstacle(box(7.0, 19.5, 21.0, 20.5), "wall_mid"),
        StaticObstacle(box(25.0, 19.5, 28.0, 20.5), "wall_right"),
    ]

    movable = [
        # A：堵住近门，难度排名第二。
        MovableObstacle(
            x=5.0,
            y=20.0,
            l=3.4,
            d=1.4,
            theta=0.0,
            material="loaded_pallet",
            difficulty=20.0,
            work=20.0,
            oid=1,
        ),
        # B：堵住远门、位于 C 的上方，难度排名第三。
        # B 的底部与 C 的顶部相接，但二者不重叠。
        MovableObstacle(
            x=23.0,
            y=20.5,
            l=3.8,
            d=1.8,
            theta=0.0,
            material="wooden_crate",
            difficulty=2.0,
            work=2.0,
            oid=2,
        ),
        # C：从上方房间观察时被 B 完全遮挡，难度排名第一。
        # C 的顶部与墙体底面相接，移开 B 后无法从细缝绕过 C。
        MovableObstacle(
            x=23.0,
            y=18.8,
            l=3.8,
            d=1.4,
            theta=0.0,
            material="steel_safe",
            difficulty=200.0,
            work=200.0,
            oid=3,
        ),
    ]

    return {
        "name": "two_doors_hidden_c",
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (5.0, 35.0),
        "goal": (5.0, 5.0),
        "cfg": Config(),
    }
