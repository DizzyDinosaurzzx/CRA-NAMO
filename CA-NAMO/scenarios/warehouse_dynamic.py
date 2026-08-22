"""在 warehouse 地图基础上给三个障碍物挂自主运动策略，验证"地图会动"的机制。

三种脚本化时刻表模式，覆盖 drift.scripted 的三类典型用法：
- patrol：沿开放纵向通道往返巡逻——位置变化，朝向不变，loop=True。
- sway：原地摆动——朝向变化，位置不变，loop=True。
- relocate：一次性搬迁——不循环，挪完就停，loop=False。

三个障碍物都摆在 warehouse.py 原有可动物体从未占用过的两条纵向空隙（货架列 3/4
之间、货架列 5/6 之间）里，不与既有布局冲突。这是机制验证场景，摆放不代表真实
仓储设计，见 drift-clock-integration 备忘。
"""

from __future__ import annotations

import drift
from obstacle import MovableObstacle
from scenarios.warehouse import create as _base_create


def create():
    base = _base_create()
    movable = list(base["movable"])

    gap34_x = 27.75   # 货架列 3(21-26) 与列 4(29.5-34) 之间的纵向空隙
    gap56_x = 44.25   # 货架列 5(38-42.5) 与列 6(46-50) 之间的纵向空隙

    patrol = MovableObstacle(
        x=gap34_x, y=3.0, l=1.6, d=0.8, h=1.2, theta=0.0,
        material="forklift", difficulty=4000.0, oid="DRIFT_PATROL",
        drift=drift.scripted([
            (0.0, (gap34_x, 3.0, 0.0)),
            (60.0, (gap34_x, 37.0, 0.0)),
            (120.0, (gap34_x, 3.0, 0.0)),
        ], loop=True),
    )

    sway = MovableObstacle(
        x=gap56_x, y=5.0, l=1.2, d=1.2, h=1.5, theta=0.0,
        material="rotating_arm", difficulty=2000.0, oid="DRIFT_SWAY",
        drift=drift.scripted([
            (0.0, (gap56_x, 5.0, 0.0)),
            (10.0, (gap56_x, 5.0, 1.2)),
            (20.0, (gap56_x, 5.0, 0.0)),
        ], loop=True),
    )

    relocate = MovableObstacle(
        x=gap56_x, y=20.0, l=1.5, d=1.5, h=1.0, theta=0.0,
        material="pallet", difficulty=1500.0, oid="DRIFT_RELOCATE",
        drift=drift.scripted([
            (0.0, (gap56_x, 20.0, 0.0)),
            (30.0, (gap56_x, 10.0, 0.0)),
        ], loop=False),
    )

    movable.extend([patrol, sway, relocate])
    base["movable"] = movable
    return base
