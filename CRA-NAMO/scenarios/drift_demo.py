"""独立于 warehouse 的最小场景，专门用于快速验证 drift.py 三种触发方式。

20m x 8m 单走廊，中间一堵墙留一道 2m 宽的门；起点终点隔门相对，机器人必须穿过门。

- PATROL(scripted)：沿起点-终点直线来回巡逻，周期性穿过门，逼机器人真的要和它打交道。
- EVENT_MOVER(event)：起初蹲在直线末段，机器人靠近到一定距离才让开，见 drift.robot_within。
- WANDERER(reactive)：在左下角一小块与机器人路线完全不重叠的区域随机游走，纯观察用，
  不参与任何交互——留一个"安全"的对照，免得三个都在互相干扰时分不清是谁的问题。

材质全用 llm_difficulty.PROMPT_ANCHORS 里已收录的名字(cart/wooden_crate/pallet)，
且 cfg 强制 deepseek_api_key="" 退到 heuristic 模式，保证这个场景每次跑都是纯本地、
确定性、秒级完成，不会像 warehouse_dynamic 那样因为材质没锚定而打真实网络请求，
见 drift-clock-integration 备忘里的踩坑记录。默认还开了 wait_on_block_s，用来练
Step 6 那条"被 drift 障碍物挡路时先等等看"的执行层等待。
"""

from __future__ import annotations

from shapely.geometry import box

import drift
from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    workspace = box(0.0, 0.0, 20.0, 8.0)
    wall_t = 0.5

    static = [
        StaticObstacle(box(0.0, 0.0, 20.0, wall_t), "outer_bottom"),
        StaticObstacle(box(0.0, 8.0 - wall_t, 20.0, 8.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, wall_t, 8.0), "outer_left"),
        StaticObstacle(box(20.0 - wall_t, 0.0, 20.0, 8.0), "outer_right"),
        # 中间隔墙，留 y in [3.0, 5.0] 一道 2m 门
        StaticObstacle(box(9.75, wall_t, 10.25, 3.0), "divider_bottom"),
        StaticObstacle(box(9.75, 5.0, 10.25, 8.0 - wall_t), "divider_top"),
    ]

    patrol = MovableObstacle(
        x=6.0, y=4.0, l=1.0, d=0.8, h=1.0, theta=0.0,
        material="cart", difficulty=600.0, oid="PATROL",
        drift=drift.scripted([
            (0.0, (6.0, 4.0, 0.0)),
            (5.0, (14.0, 4.0, 0.0)),
            (10.0, (6.0, 4.0, 0.0)),
        ], loop=True),
    )

    event_mover = MovableObstacle(
        x=16.0, y=4.0, l=1.0, d=1.0, h=0.6, theta=0.0,
        material="wooden_crate", difficulty=300.0, oid="EVENT_MOVER",
        drift=drift.event(
            drift.robot_within((16.0, 4.0), 3.0),
            [(0.0, (16.0, 4.0, 0.0)), (3.0, (16.0, 6.5, 0.0))],
            loop=False,
        ),
    )

    wanderer = MovableObstacle(
        x=4.0, y=1.5, l=1.2, d=0.8, h=0.3, theta=0.0,
        material="pallet", difficulty=200.0, oid="WANDERER",
        drift=drift.reactive(rng_seed=7, speed=0.4,
                             bounds=(1.5, 7.0, 0.7, 2.5), avoid_radius=1.0),
    )

    start = (2.0, 4.0)
    goal = (18.0, 4.0)

    return {
        "workspace": workspace,
        "static": static,
        "movable": [patrol, event_mover, wanderer],
        "start": start,
        "goal": goal,
        "cfg": Config(
            grid_step=0.3,
            conn_radius=0.6,
            se2_cell=0.3,
            deepseek_api_key="",     # 强制 heuristic，保证本场景纯本地、秒级完成
            wait_on_block_s=3.0,     # 被 drift 障碍物挡路时先等 3s 再重规划
            wait_substep_s=0.5,
        ),
    }
