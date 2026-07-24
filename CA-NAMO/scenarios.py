"""
演示场景。

`two_doors` is the canonical illustration of the cost trade-off:

    +------------------------------------------+ 20
    |            room L        ||   room R      |
    |                          ||               |
    |                        (gap B, far)       |   <- long detour, always open
    |                          ||               |
    |   S ......(gap A, near).[A][B].... G      |   <- short route, blocked by
    |                          ||               |      movable obstacles A, B
    +------------------------------------------+ 0
       x=0                x=15                x=30

一面不可移动的墙将空间分成两个房间，并设有两个门口。靠近的门口（与起点 S 和目标 G
对齐）被容易移动的障碍物 A 阻挡，其后方还隐藏着障碍物 B。远处门口畅通，但必须
绕行很长距离。规划器必须权衡“移开 A（可能还包括 B）”与“绕路行驶”。

机器人只有接近到 R_perc 范围内才会发现 A 和 B；B 最初被 A 遮挡，移动 A 后会揭示 B，
并触发重规划。
"""
from __future__ import annotations

from shapely.geometry import box, Polygon

from obstacle import MovableObstacle, StaticObstacle
from config import Config


def two_doors():
    workspace = box(0, 0, 30, 20)

    # x 位于 [14.5, 15.5] 的竖直墙，包含两个缺口：近处 y[3,6]，远处 y[15,18]
    walls = [
        StaticObstacle(box(14.5, 0.0, 15.5, 3.0), "wall_lo"),
        StaticObstacle(box(14.5, 6.0, 15.5, 15.0), "wall_mid"),
        StaticObstacle(box(14.5, 18.0, 15.5, 20.0), "wall_hi"),
    ]

    # 可移动障碍物（真实世界，包含真实难度）
    movable = [
        # A：堵住近处门口，容易移动
        MovableObstacle(x=15.0, y=4.5, l=1.4, d=2.6, theta=0.0,
                        material="empty_cart", difficulty=0.3, oid=1),
        # B：紧邻墙体东侧，被 A 遮挡，移动难度中等
        MovableObstacle(x=16.7, y=4.5, l=1.2, d=2.4, theta=0.0,
                        material="wooden_table", difficulty=1.2, oid=2),
        # 其他位置的干扰物（无关杂物，用于测试感知）
        MovableObstacle(x=9.0, y=12.0, l=2.0, d=2.0, theta=0.0,
                        material="chair", difficulty=0.4, oid=3),
        MovableObstacle(x=22.0, y=10.0, l=2.0, d=1.5, theta=0.0,
                        material="cardboard_box", difficulty=0.2, oid=4),
    ]

    start = (5.0, 4.5)
    goal_region = box(24.0, 3.5, 26.0, 5.5)

    cfg = Config(
        lambda_d=1.0, lambda_w=1.0,
        R_perc=8.0, R_push=5.0,
        grid_step=1.5, conn_radius=2.4, robot_radius=0.35,
        step_execute_edges=1, max_replans=120,
    )
    return dict(name="two_doors", workspace=workspace, static=walls,
                movable=movable, start=start, goal_region=goal_region, cfg=cfg)


SCENARIOS = {"two_doors": two_doors}


def load(name: str = "two_doors"):
    return SCENARIOS[name]()
