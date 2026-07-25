"""
演示场景
"""

from __future__ import annotations

from shapely.geometry import box, Polygon

from obstacle import MovableObstacle, StaticObstacle
from config import Config


def two_doors():

    """
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

    # 所有可调参数统一在 config.py 中控制，这里不再覆盖
    cfg = Config()
    return dict(name="two_doors", workspace=workspace, static=walls,
                movable=movable, start=start, goal_region=goal_region, cfg=cfg)


def two_doors_hidden_c():

    """双门场景：近门有 A，远门有 B，且 C 被 B 遮挡。

        +-----------------------------+ 40
        |  S                          |
        |  .                          |
        |  A==========wall======B     |  <- 近门在 A，远门在 B
        |  .                   C      |     C 被 B 遮挡
        |  G                          |
        +-----------------------------+ 0
           x=0                     x=28

    障碍物编号与名称的对应关系为 1=A、2=B、3=C。难度数值是仿真世界中的
    ground truth；在进入感知半径之前，规划器并不知道障碍物的存在与难度。
    近门由第二难移动的 A 阻挡；远门由第三难移动的 B 阻挡，B 后面隐藏着第一难
    移动的 C。真实难度满足 C >> A >> B。机器人接近远门时先发现 B，移动 B 后
    才发现 C，从而触发在线重规划。
    """
    workspace = box(0, 0, 28, 40)

    # y 位于 [19.5, 20.5] 的水平墙。
    # 近门 x[3,7] 与 S/G 对齐；远门 x[21,25] 迫使机器人额外绕行约 36 m。
    # S 位于上方房间，因此 B 在视线上挡住其下方的 C；B 的首选移除方向则朝上，
    # 不会穿过 C。
    walls = [
        StaticObstacle(box(0.0, 19.5, 3.0, 20.5), "wall_left"),
        StaticObstacle(box(7.0, 19.5, 21.0, 20.5), "wall_mid"),
        StaticObstacle(box(25.0, 19.5, 28.0, 20.5), "wall_right"),
    ]

    movable = [
        # A：堵住近门，难度排名第二。
        MovableObstacle(x=5.0, y=20.0, l=3.4, d=1.4, theta=0.0,
                        material="loaded_pallet", difficulty=20.0, oid=1),
        # B：堵住远门、位于 C 的上方，难度排名第三。
        # 上移到 y=20.5（底部 19.6），正好压在 C 的顶部之上而不与之重叠。
        MovableObstacle(x=23.0, y=20.5, l=3.8, d=1.8, theta=0.0,
                        material="wooden_crate", difficulty=2.0, oid=2),
        # C：与 B、远门同轴，从上方房间观察时被 B 完全遮挡，难度排名第一。
        # 抬高到 y=18.8（顶部 19.5）与墙体底面严丝合缝，封死原来 C 顶与墙底之间
        # 那条 0.8 宽的竖直缝——移开 B 后机器人无法再从缝中横向绕过 C。
        MovableObstacle(x=23.0, y=18.8, l=3.8, d=1.4, theta=0.0,
                        material="steel_safe", difficulty=200.0, oid=3),
    ]

    start = (5.0, 35.0)
    goal_region = box(4.0, 4.0, 6.0, 6.0)

    # 所有可调参数统一在 config.py 中控制，这里不再覆盖
    cfg = Config()
    return dict(name="two_doors_hidden_c", workspace=workspace, static=walls,
                movable=movable, start=start, goal_region=goal_region, cfg=cfg)


SCENARIOS = {
    "two_doors": two_doors,
    "two_doors_hidden_c": two_doors_hidden_c,
}


def load(name: str = "two_doors"):
    return SCENARIOS[name]()
