"""原始双门演示地图。"""

from __future__ import annotations

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    """创建 two_doors 地图，每次调用都返回一份新的场景数据。"""
    workspace = box(0, 0, 30, 20)

    # x 位于 [14.5, 15.5] 的竖直墙，包含两个缺口：
    # 近处 y[3, 6]，远处 y[15, 18]。
    walls = [
        StaticObstacle(box(14.5, 0.0, 15.5, 3.0), "wall_lo"),
        StaticObstacle(box(14.5, 6.0, 15.5, 15.0), "wall_mid"),
        StaticObstacle(box(14.5, 18.0, 15.5, 20.0), "wall_hi"),
    ]

    movable = [
        # A：堵住近处门口，容易移动。
        MovableObstacle(
            x=15.0,
            y=4.5,
            l=1.4,
            d=2.6,
            theta=0.0,
            material="empty_cart",
            difficulty=0.3,
            work=0.3,
            oid=1,
        ),
        # B：紧邻墙体东侧，被 A 遮挡，移动难度中等。
        MovableObstacle(
            x=16.7,
            y=4.5,
            l=1.2,
            d=2.4,
            theta=0.0,
            material="wooden_table",
            difficulty=1.2,
            work=1.2,
            oid=2,
        ),
        # 其他位置的干扰物，用于测试感知。
        MovableObstacle(
            x=9.0,
            y=12.0,
            l=2.0,
            d=2.0,
            theta=0.0,
            material="chair",
            difficulty=0.4,
            work=0.4,
            oid=3,
        ),
        MovableObstacle(
            x=22.0,
            y=10.0,
            l=2.0,
            d=1.5,
            theta=0.0,
            material="cardboard_box",
            difficulty=0.2,
            work=0.2,
            oid=4,
        ),
    ]

    return {
        "name": "two_doors",
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (5.0, 4.5),
        "goal_region": box(24.0, 3.5, 26.0, 5.5),
        "cfg": Config(),
    }
