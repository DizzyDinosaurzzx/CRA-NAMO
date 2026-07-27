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
    goal = (25.0, 4.5)

    # 所有可调参数统一在 config.py 中控制，这里不再覆盖
    cfg = Config()
    return dict(name="two_doors", workspace=workspace, static=walls,
                movable=movable, start=start, goal=goal, cfg=cfg)


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
    goal = (5.0, 5.0)

    # 所有可调参数统一在 config.py 中控制，这里不再覆盖
    cfg = Config()
    return dict(name="two_doors_hidden_c", workspace=workspace, static=walls,
                movable=movable, start=start, goal=goal, cfg=cfg)


def maze_three_movable():
    """根据迷宫图片近似复刻的场景，含 3 个岔路口可移动障碍物。

    坐标系：左下角 (0,0)，右上角 (30,30)，墙厚统一 t=0.45。起点在底部中央开口，
    目标在左上区域。机器人从底部入口进入后依次经过三个岔路口障碍物：
        A(纸箱, 易)   -> 底部入口后第一个岔路口
        B(椅子, 中)   -> 迷宫中央左右分流处
        C(木箱, 较难) -> 靠近左上目标区的岔路口
    难度为仿真世界的 ground truth；机器人在进入感知半径前并不知情。
    """
    workspace = box(0, 0, 30, 30)
    t = 0.45

    walls = [
        # ===== 外框 =====
        StaticObstacle(box(0.0, 0.0, 10.0, t), "outer_bottom_left"),
        StaticObstacle(box(15.0, 0.0, 30.0, t), "outer_bottom_right"),
        StaticObstacle(box(29.55, 0.0, 30.0, 30.0), "outer_right"),
        StaticObstacle(box(5.0, 29.55, 30.0, 30.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 25.0), "outer_left"),

        # ===== 左上区域 =====
        StaticObstacle(box(5.0, 25.5, 5.0 + t, 29.6), "lu_vertical"),
        StaticObstacle(box(5.0, 25.5, 14.8, 25.5 + t), "lu_horizontal"),
        StaticObstacle(box(0.0, 20.8, 10.0, 20.8 + t), "left_upper_horizontal"),
        StaticObstacle(box(5.0, 16.5, 5.0 + t, 20.8), "left_upper_vertical"),
        StaticObstacle(box(14.5, 20.8, 14.5 + t, 29.6), "upper_mid_vertical"),

        # ===== 右上区域 =====
        StaticObstacle(box(24.5, 25.5, 24.5 + t, 29.6), "ru_vertical_top"),
        StaticObstacle(box(19.8, 20.8, 30.0, 20.8 + t), "ru_horizontal_bottom"),
        StaticObstacle(box(19.8, 20.8, 19.8 + t, 25.6), "ru_vertical_left"),
        StaticObstacle(box(24.7, 16.7, 24.7 + t, 20.8), "ru_vertical_down"),

        # ===== 中部（回字型结构） =====
        StaticObstacle(box(9.8, 7.8, 10.25, 16.7), "center_left_vertical"),
        StaticObstacle(box(9.8, 16.25, 19.8, 16.7), "center_top_horizontal"),
        StaticObstacle(box(9.8, 7.8, 24.8, 8.25), "center_bottom_horizontal"),
        StaticObstacle(box(14.5, 12.1, 14.95, 16.7), "center_inner_vertical"),
        StaticObstacle(box(20.0, 12.2, 30.0, 12.65), "mid_right_horizontal"),

        # ===== 左下区域 =====
        StaticObstacle(box(5.0, 0.0, 5.45, 5.2), "ll_vertical"),
        StaticObstacle(box(0.0, 9.5, 9.8, 9.95), "ll_top_horizontal"),
    ]

    movable = [
        # A：底部入口进入后第一个岔路口。
        MovableObstacle(x=16.0, y=10.0, l=4.0, d=3.0, theta=0.0,
                        material="cardboard_box", difficulty=0.4, oid=1),
        # B：迷宫中央左右分流处。
        MovableObstacle(x=23.0, y=15.0, l=3.0, d=4.0, theta=0.0,
                        material="chair", difficulty=0.8, oid=2),
        # C：靠近左上目标区的岔路口。
        MovableObstacle(x=12.0, y=21.8, l=4.0, d=1.8, theta=0.0,
                        material="wooden_crate", difficulty=1.5, oid=3),
    ]

    start = (12.5, 1.5)     # 底部入口（图片红色箭头）
    goal = (2.5, 28.0)      # 左上目标点（图片红旗）

    cfg = Config(
        grid_step=0.55,
        conn_radius=1.2,
        max_expansions=800000,
    )
    return dict(name="maze_three_movable", workspace=workspace, static=walls,
                movable=movable, start=start, goal=goal, cfg=cfg)

def maze_three_movable2():
    """创建 maze_three_movable2 场景。"""
    workspace = box(0, 0, 30, 30)

    # 墙厚统一设为 0.45。
    # 坐标系：左下角为 (0, 0)，右上角为 (30, 30)。
    # 起点位于底部中央开口，目标位于左上区域。
    t = 0.45

    walls = [
        # ===== 外框 =====
        StaticObstacle(box(0.0, 0.0, 10.0, t), "outer_bottom_left"),
        StaticObstacle(box(15.0, 0.0, 30.0, t), "outer_bottom_right"),
        StaticObstacle(box(29.55, 0.0, 30.0, 30.0), "outer_right"),
        StaticObstacle(box(5.0, 29.55, 30.0, 30.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 25.0), "outer_left"),

        # ===== 左上区域 =====
        # 顶部左侧的倒 L 形结构
        StaticObstacle(box(5.0, 25.5, 5.0 + t, 29.6), "lu_vertical"),
        StaticObstacle(box(5.0, 25.5, 14.8, 25.5 + t), "lu_horizontal"),

        # 左侧中上横墙
        StaticObstacle(box(0.0, 20.8, 10.0, 20.8 + t), "left_upper_horizontal"),
        StaticObstacle(box(5.0, 16.5, 5.0 + t, 20.8), "left_upper_vertical"),

        # 中上竖墙
        StaticObstacle(box(14.5, 20.8, 14.5 + t, 29.6), "upper_mid_vertical"),

        # ===== 右上区域 =====
        # 右上房间的 U / Γ 形结构
        StaticObstacle(box(24.5, 25.5, 24.5 + t, 29.6), "ru_vertical_top"),
        StaticObstacle(box(19.8, 20.8, 30.0, 20.8 + t), "ru_horizontal_bottom"),
        StaticObstacle(box(19.8, 20.8, 19.8 + t, 25.6), "ru_vertical_left"),
        StaticObstacle(box(24.7, 16.7, 24.7 + t, 20.8), "ru_vertical_down"),

        # ===== 中部 =====
        # 左中横墙，与中部矩形区域相接
        StaticObstacle(box(2.0, 12.0, 10.0, 12.0+t), "mid_left_horizontal"),

        # 中部大矩形 / 回字型结构
        StaticObstacle(box(9.8, 7.8, 10.25, 16.7), "center_left_vertical"),
        StaticObstacle(box(9.8, 16.25, 19.8, 16.7), "center_top_horizontal"),
        StaticObstacle(box(9.8, 7.8, 24.8, 8.25), "center_bottom_horizontal"),
        StaticObstacle(box(14.5, 12.1, 14.95, 16.7), "center_inner_vertical"),

        # 右中横墙
        StaticObstacle(box(20.0, 12.2, 30.0, 12.65), "mid_right_horizontal"),

        # ===== 左下区域 =====
        # 左下小房间
        StaticObstacle(box(5.0, 0.0, 5.45, 5.2), "ll_vertical"),
        #StaticObstacle(box(0.0, 9.5, 9.8, 9.95), "ll_top_horizontal"),
    ]

    movable = [
        # A：底部入口进入后第一个岔路口。
        MovableObstacle(
            x=1.5,
            y=13.5,
            l=2,
            d=2,
            theta=0.0,
            material="cardboard_box",
            difficulty=100,
            oid=1,
        ),

        # B：迷宫中央左右分流处。
        MovableObstacle(
            x=19,
            y=14.4,
            l=3,
            d=3.4,
            theta=0.0,
            material="chair",
            difficulty=0.5,
            oid=2,
        ),

        # C：靠近左上目标区的岔路口。
        MovableObstacle(
            x=12,
            y=21.8,
            l=4,
            d=1.8,
            theta=0.0,
            material="wooden_crate",
            difficulty=1.5,
            oid=3,
        ),
    ]

    # 图片底部红色箭头对应入口。
    start = (12.5, 1.5)

    # 图片左上红旗对应目标点。
    goal = (2.5, 28.0)

    cfg = Config(
        grid_step=0.55,
        conn_radius=1.2,
        max_expansions=800000,
    )
    return dict(name="maze_three_movable2", workspace=workspace, static=walls,
                movable=movable, start=start, goal=goal, cfg=cfg)


SCENARIOS = {
    "two_doors": two_doors,
    "two_doors_hidden_c": two_doors_hidden_c,
    "maze_three_movable": maze_three_movable,
    "maze_three_movable2": maze_three_movable2,
}


def load(name: str = "two_doors"):
    return SCENARIOS[name]()
