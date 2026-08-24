"""corridor 场景的变体：唯一一个可动障碍物挂了 drift，机器人没有绕路可选，必须
去搬一个"正在自己动"的东西——专门用来测试贴身搬运和自主漂移撞在一起会怎样。

几何取自 corridor.py（已验证过 λ=350 默认会选择搬走而非绕行，见 README「接触模型
带来了什么」）：0.7×6 的长条 y 方向几乎贴满 8m 高的走廊(只留 0.5m 余量给外墙)，
沿 x 方向随便停在哪都会把走廊整个截断，不需要额外的缺口墙来加码——corridor.py
原版那两道缺口墙(wall1/wall2)刻意没抄过来：它们卡在 x=18.5-21.5，正好在过道中间，
第一版把巡逻区间设在这一带时，长条扫过去时 y 方向整段撞进缺口墙里(第一次跑直接
success=False，教训记在 drift-clock-integration 备忘里)。去掉这两道墙之后，
巡逻区间可以放心覆盖一大段空旷区域。

唯一的改动是给这根长条挂了 drift.scripted：沿走廊方向来回滑动，机器人赶到之前它
一直在动，赶到之后能不能"抓住一个还在动的东西"、抓住期间它是否老实停在机器人手里、
松手后是否恢复自己的时刻表——是这个场景要验证的三件事。
"""

from __future__ import annotations

from shapely.geometry import box

import drift
from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    workspace = box(0, 0, 40, 8)

    walls = [
        StaticObstacle(box(0.0, 7.5, 40.0, 8.0), "wall_top"),
        StaticObstacle(box(0.0, 0.0, 40.0, 0.5), "wall_bottom"),
    ]

    blocker = MovableObstacle(
        x=20.0, y=4.0, l=0.7, d=6.0, h=1.0, theta=0.0,
        material="wooden_crate", difficulty=1112.454, oid="MOVING_BLOCKER",
        drift=drift.scripted([
            (0.0, (20.0, 4.0, 0.0)),
            (10.0, (26.0, 4.0, 0.0)),
            (20.0, (14.0, 4.0, 0.0)),
            (30.0, (20.0, 4.0, 0.0)),
        ], loop=True),
    )

    return {
        "workspace": workspace,
        "static": walls,
        "movable": [blocker],
        "start": (4.0, 4.0),
        "goal": (36.0, 4.0),
        "cfg": Config(deepseek_api_key=""),   # wooden_crate 已锚定，强制离线只为保险
    }
