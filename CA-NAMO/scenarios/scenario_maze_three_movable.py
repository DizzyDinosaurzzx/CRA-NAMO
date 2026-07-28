from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    """创建 maze_three_movable 场景。"""
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
            material="concrete_block",  # 25.0 x 4.00 = 100.0
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
            material="foam_mat",        # 0.05 x 10.20 = 0.51
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
            material="empty_shelf",     # 0.21 x 7.20 = 1.51
            difficulty=1.5,
            oid=3,
        ),
    ]

    start = (12.5, 1.5)
    goal = (2.5, 28.0)
    return {
        "name": "maze_three_movable",
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": Config(),
    }
