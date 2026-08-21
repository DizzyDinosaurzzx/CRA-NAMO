"""越障可视场景：低矮宽障碍挡路但遮不住后方高处目标，考验遮挡推理。"""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    """构建越障可视场景。"""
    workspace = box(0, 0, 20, 16)

    t = 0.4
    # 三个房间上下排布、由两道门相连；门都足够宽，不会挡住起点到任一障碍的视线，
    # 低矮障碍始终是机器人与 oid 2 之间唯一的遮挡。
    walls = [
        StaticObstacle(box(0.0, 0.0, 20.0, t), "outer_bottom"),
        StaticObstacle(box(0.0, 16.0 - t, 20.0, 16.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 16.0), "outer_left"),
        StaticObstacle(box(20.0 - t, 0.0, 20.0, 16.0), "outer_right"),

        # 入口隔墙——门洞 x 在 [7.5, 12.5]，正对起点
        StaticObstacle(box(t, 4.0, 7.5, 4.0 + t), "entrance_left"),
        StaticObstacle(box(12.5, 4.0, 20.0 - t, 4.0 + t), "entrance_right"),

        # 与 oid 1 同层的两侧封壁，使其成为唯一闸口：两侧仅剩 0.4 m 缝隙窄于机器人，
        # 通行必须把它推开。
        StaticObstacle(box(t, 6.5, 6.6, 7.5), "gate_flank_left"),
        StaticObstacle(box(13.4, 6.5, 20.0 - t, 7.5), "gate_flank_right"),

        # 凹室短墙，足够远，视线不会触及
        StaticObstacle(box(t, 9.0, 3.5, 9.0 + t), "alcove_left"),
        StaticObstacle(box(16.5, 9.0, 20.0 - t, 9.0 + t), "alcove_right"),

        # 出口隔墙——门洞 x 在 [6.0, 11.0]，故意错开，逼机器人绕过 oid 2 而非直奔目标
        StaticObstacle(box(t, 12.5, 6.0, 12.5 + t), "exit_left"),
        StaticObstacle(box(11.0, 12.5, 20.0 - t, 12.5 + t), "exit_right"),
    ]

    movable = [
        # 前方低矮宽大——在几何上完全遮挡 oid 2
        MovableObstacle(
            x=10.0,
            y=7.0,
            l=6.0,
            d=1.0,
            h=0.5,
            theta=0.0,
            material="pallet",
            difficulty=2048.328,
            oid=1,
        ),
        # 后方高瘦——可越过 oid 1 顶部被看到
        MovableObstacle(
            x=10.0,
            y=10.5,
            l=3.0,
            d=1.0,
            h=2.0,
            theta=0.0,
            material="steel_shelf",
            difficulty=8829.0,
            oid=2,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (10.0, 2.0),
        "goal": (10.0, 14.0),
        "cfg": Config(),
    }
