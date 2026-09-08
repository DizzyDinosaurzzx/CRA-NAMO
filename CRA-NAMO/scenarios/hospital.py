
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from shapely.geometry import box

import cost
from config import Config
from dynamics import Event, MoveTo, at_time
from obstacle import MovableObstacle, StaticObstacle
from scenarios import _realism
from scenarios._realism import MU_BRAKED_WHEELS, MU_CASTORS, push_force


# 紧凑医院平面。
_WIDTH = 42.0
_HEIGHT = 28.0
_WALL_T = 0.35

# 临时设备所需净空尺寸。
_DOOR_W = 2.4
_MAIN_CORRIDOR = (10.5, 17.5)   # 宽 7 米
_SOUTH_BYPASS = (0.35, 5.0)     # 宽 4.65 米
_NORTH_BYPASS = (23.0, 27.65)   # 宽 4.65 米

# 病床含床头长 2.20 米、护栏宽 1.00 米，含床垫重 160 千克。刹车松开时可滚动，
# 因此机器人克服的是脚轮滚动阻力，而不是滑动摩擦。
_BED_LENGTH = 2.20
_BED_WIDTH = 1.00
_BED_HEIGHT = 1.05
_BED_MASS = 160.0
_BED_DIFFICULTY = push_force(_BED_MASS, MU_CASTORS)

# 两面双开口墙形成明确的路线决策。
_CENTRAL_GATE_X = 20.8
_ICU_GATE_X = 33.2
# 两个服务门是 1.5 米单扇开口；ICU 上方开口为 2.4 米病区入口，只有它能让
# 2.20 米病床横放而不使两端撞墙。
_CENTRAL_GATE_GAPS = ((11.4, 13.4), (14.9, 16.9))
_ICU_GATE_GAPS = ((11.4, 13.4), (14.5, 16.9))

_TRIAGE_BED_ID = "tempBedA"
_XRAY_ID = "mobileXrayB"
_LINEN_CART_ID = "linenCartB"
_OCCUPIED_BED_ID = "occupiedBedC"
_CLEANING_CART_ID = "cleaningCartC"

_EMERGENCY_END_X = 19.5
_INPATIENT_START_X = 22.5

_START = (2.0, 25.5)            # 左上红色圆点
_GOAL = (40.0, 2.5)             # 右下红色圆点


# 阻塞物停在门洞旁而不是嵌入门洞。SE(2) 规划器会将目标姿态吸附到网格，
# 正好堵住开口的物体会因舍入损失一半净空，移动器可能在到达前放弃；X 光机和
# 清洁车曾因此失败。将其停在距墙略有间隙的走廊中，既保留路线也能关门：
# 剩余空隙小于 0.12 米，而机器人需要 0.20 米。
# 需要关闭的开口从 t=0 起全部关闭，事件只负责清除其中一个。这是规划约束：
# plan_path 对起点允许有限碰撞豁免，但严格检查目标格，因此阻塞物可被移出开口，
# 却不能再被移入开口。之前的关门事件正是因此无声失败。


def _wall(p: tuple[float, float], q: tuple[float, float],
          name: str) -> StaticObstacle:
    return StaticObstacle.segment(p, q, _WALL_T, name)


def _cut_wall(axis: str, fixed: float, start: float, end: float,
              gaps: Sequence[tuple[float, float]], name: str):
    """构建墙体，同时保留指定门洞。"""
    pieces = []
    cursor = start
    for index, (gap_start, gap_end) in enumerate(sorted(gaps)):
        gap_start = max(start, gap_start)
        gap_end = min(end, gap_end)
        if gap_start > cursor:
            p = (cursor, fixed) if axis == "h" else (fixed, cursor)
            q = (gap_start, fixed) if axis == "h" else (fixed, gap_start)
            pieces.append(_wall(p, q, f"{name}_{index}"))
        cursor = max(cursor, gap_end)
    if cursor < end:
        p = (cursor, fixed) if axis == "h" else (fixed, cursor)
        q = (end, fixed) if axis == "h" else (fixed, end)
        pieces.append(_wall(p, q, f"{name}_{len(pieces)}"))
    return pieces


def _hwall(y: float, x0: float, x1: float,
           gaps: Sequence[tuple[float, float]], name: str):
    return _cut_wall("h", y, x0, x1, gaps, name)


def _door(center: float) -> tuple[float, float]:
    return center - _DOOR_W / 2.0, center + _DOOR_W / 2.0


def _partitions(xs: Iterable[float], y0: float, y1: float, prefix: str):
    return [
        _wall((x, y0), (x, y1), f"{prefix}_{index}")
        for index, x in enumerate(xs)
    ]


def _outer_shell():
    t = _WALL_T
    return [
        StaticObstacle(box(0.0, 0.0, _WIDTH, t), "outer_south"),
        StaticObstacle(box(0.0, _HEIGHT - t, _WIDTH, _HEIGHT), "outer_north"),
        StaticObstacle(box(0.0, 0.0, t, _HEIGHT), "outer_west"),
        StaticObstacle(box(_WIDTH - t, 0.0, _WIDTH, _HEIGHT), "outer_east"),
    ]


def _department_rooms(x0: float, x1: float, partitions: Sequence[float],
                      door_centers: Sequence[float], prefix: str):
    """创建可通往两条替代路线的上下房间。"""
    main_south, main_north = _MAIN_CORRIDOR
    south_bypass_top = _SOUTH_BYPASS[1]
    north_bypass_bottom = _NORTH_BYPASS[0]
    doors = [_door(center) for center in door_centers]

    walls = []

    # 上方房间连接主走廊和北侧绕行通道。
    walls.extend(_hwall(main_north, x0, x1, doors,
                        f"{prefix}_upper_main_doors"))
    walls.extend(_hwall(north_bypass_bottom, x0, x1, doors,
                        f"{prefix}_upper_bypass_doors"))
    walls.extend(_partitions(partitions, main_north, north_bypass_bottom,
                             f"{prefix}_upper_partition"))

    # 下方房间连接主走廊和南侧绕行通道。
    walls.extend(_hwall(main_south, x0, x1, doors,
                        f"{prefix}_lower_main_doors"))
    walls.extend(_hwall(south_bypass_top, x0, x1, doors,
                        f"{prefix}_lower_bypass_doors"))
    walls.extend(_partitions(partitions, south_bypass_top, main_south,
                             f"{prefix}_lower_partition"))

    # 保留部门之间的开口供临时物体使用。
    walls.extend([
        _wall((x0, main_north), (x0, north_bypass_bottom),
              f"{prefix}_upper_west"),
        _wall((x1, main_north), (x1, north_bypass_bottom),
              f"{prefix}_upper_east"),
        _wall((x0, south_bypass_top), (x0, main_south),
              f"{prefix}_lower_west"),
        _wall((x1, south_bypass_top), (x1, main_south),
              f"{prefix}_lower_east"),
    ])
    return walls


def _emergency_department():
    """分诊、复苏、治疗和观察房间。"""
    return _department_rooms(
        x0=0.35,
        x1=_EMERGENCY_END_X,
        partitions=(6.5, 13.0),
        door_centers=(3.3, 9.7, 16.2),
        prefix="emergency",
    )


def _inpatient_ward():
    """围绕同一三路线循环系统布置六个住院病房。"""
    return _department_rooms(
        x0=_INPATIENT_START_X,
        x1=_WIDTH - 0.35,
        partitions=(29.0, 35.5),
        door_centers=(25.7, 32.2, 38.7),
        prefix="inpatient",
    )


def _decision_walls():
    """增加两个双开口门组，不改变三层路线。"""
    main_south, main_north = _MAIN_CORRIDOR
    return [
        *_cut_wall(
            "v", _CENTRAL_GATE_X, main_south, main_north,
            _CENTRAL_GATE_GAPS, "central_decision_gate",
        ),
        *_cut_wall(
            "v", _ICU_GATE_X, main_south, main_north,
            _ICU_GATE_GAPS, "icu_decision_gate",
        ),
    ]


def _department_beds(edges: Sequence[float], oid_start: int):
    """按蓝色标记在每个上下房间放置两张病床。"""
    beds = []
    row_heights = (
        ("upper", 19.1, 21.4),
        ("lower", 6.6, 8.9),
    )

    oid = oid_start
    for row_name, near_y, far_y in row_heights:
        for room_index, (left, right) in enumerate(zip(edges, edges[1:])):
            left_y, right_y = (
                (near_y, far_y) if room_index % 2 == 0
                else (far_y, near_y)
            )
            for side, x, y in (
                ("left", left + 1.4, left_y),
                ("right", right - 1.4, right_y),
            ):
                beds.append(MovableObstacle(
                    x=x,
                    y=y,
                    l=_BED_LENGTH,
                    d=_BED_WIDTH,
                    h=_BED_HEIGHT,
                    theta=math.pi / 2.0,
                    material="empty_hospital_bed",
                    difficulty=_BED_DIFFICULTY,
                    oid=f"hospitalBed{oid}",
                ))
                oid += 1
    return beds


def _hospital_beds():
    return [
        *_department_beds((0.35, 6.5, 13.0, _EMERGENCY_END_X), 101),
        *_department_beds((_INPATIENT_START_X, 29.0, 35.5,
                           _WIDTH - 0.35), 201),
    ]


def _temporary_obstacles():
    """位置使等待、绕行和搬移可比较的物体。"""
    return [
        # E1：虚构的病床，刹车松开，横在第一个上方治疗室最近出口。
        MovableObstacle(
            x=3.3,
            y=_MAIN_CORRIDOR[1],
            l=_BED_LENGTH,
            d=_BED_WIDTH,
            h=_BED_HEIGHT,
            theta=0.0,
            material="empty_hospital_bed",
            difficulty=_BED_DIFFICULTY,
            oid=_TRIAGE_BED_ID,
        ),
        # E2：450 千克移动 X 光机停在中央下门洞。它有脚轮但刹车锁定，机器人
        # 无法释放，只能在乙烯基地面上拖动 2.6 千牛橡胶，或选择长路；接触后
        # 才能确定真实情况。
        MovableObstacle(
            x=_CENTRAL_GATE_X,
            y=12.40,
            l=1.60,
            d=0.62,
            h=1.95,
            theta=math.pi / 2.0,
            material="mobile_xray_unit",
            difficulty=push_force(450.0, MU_BRAKED_WHEELS),
            contact_reveals="parked_xray_unit_with_brakes_set",
            oid=_XRAY_ID,
        ),
        # 载物布草车用自由脚轮堵住中央上门洞：阻力 28 牛，比行驶一米便宜；
        # 另一选择是等待布草车自行离开。
        MovableObstacle(
            x=_CENTRAL_GATE_X,
            y=15.90,
            l=1.60,
            d=0.70,
            h=1.30,
            theta=math.pi / 2.0,
            material="linen_cart",
            difficulty=push_force(95.0, MU_CASTORS),
            oid=_LINEN_CART_ID,
        ),
        # ICU 病区入口被载人的病床堵住，刹车锁定：阻力 1.4 千牛且风险高。
        # 它物理上可移动，但设计意图是不要搬动。
        MovableObstacle(
            x=_ICU_GATE_X,
            y=15.70,
            l=_BED_LENGTH,
            d=_BED_WIDTH,
            h=_BED_HEIGHT,
            theta=math.pi / 2.0,
            material="occupied_bed",
            difficulty=push_force(_BED_MASS + 80.0, MU_BRAKED_WHEELS),
            contact_reveals="patient_in_bed",
            oid=_OCCUPIED_BED_ID,
        ),
        # E4/E5：清洁车在清洁班次前堵住 ICU 服务门；离开后，病区入口及其病床
        # 成为穿过该墙的唯一其他路线。
        MovableObstacle(
            x=_ICU_GATE_X,
            y=12.40,
            l=1.60,
            d=0.62,
            h=1.05,
            theta=math.pi / 2.0,
            material="cleaning_cart",
            difficulty=push_force(70.0, MU_CASTORS),
            oid=_CLEANING_CART_ID,
        ),
    ]


def _option_costs(cfg: Config, obstacle: MovableObstacle, *,
                  move_distance: float, contact_distance: float,
                  detour_distance: float, wait_seconds: float) -> dict:
    """估计局部搬移、绕行和等待代价，用于场景报告。"""
    move_work = cost.manipulation_work(obstacle.difficulty, move_distance)
    move_motion = cost.motion_cost(
        cfg, contact_distance + move_distance)
    move_time = (
        cfg.free_profile().translate_time(contact_distance)
        + cfg.loaded_profile().translate_time(move_distance)
    )

    detour_energy = cost.motion_cost(cfg, detour_distance)
    detour_time = cfg.free_profile().translate_time(detour_distance)

    def option(energy_j: float, seconds: float, *, work_j: float = 0.0):
        return {
            "energy_j": round(energy_j, 2),
            "work_j": round(work_j, 2),
            "time_s": round(seconds, 2),
            "objective_delta": round(
                cost.combine(cfg, energy_j, seconds), 2),
        }

    return {
        "move": option(move_motion + move_work, move_time, work_j=move_work),
        "detour": option(detour_energy, detour_time),
        "wait": option(0.0, wait_seconds),
        "assumption": (
            "local estimate only; online planning adds exact turning, contact, "
            "and risk costs"
        ),
    }


def create():
    """构建紧凑医院地图及其确定性移动事件。"""
    workspace = box(0.0, 0.0, _WIDTH, _HEIGHT)
    walls = [
        *_outer_shell(),
        *_emergency_department(),
        *_inpatient_ward(),
        *_decision_walls(),
    ]
    temporary = _temporary_obstacles()
    movable = [*_hospital_beds(), *temporary]
    by_id = {obs.oid: obs for obs in temporary}

    events = [
        Event(
            name="E1 triage transfer clears the doorway",
            trigger=at_time(58.0),
            effect=MoveTo(
                oid=_TRIAGE_BED_ID,
                goal=(6.0, 14.6, math.pi / 2.0),
                speed=0.52,
            ),
        ),
        Event(
            name="E2 mobile X-ray is wheeled out of the central lower door",
            trigger=at_time(88.0),
            effect=MoveTo(
                oid=_XRAY_ID,
                goal=(17.4, 12.40, math.pi / 2.0),
                speed=0.34,
            ),
        ),
        Event(
            name="E3 linen round clears the central upper gate",
            trigger=at_time(128.0),
            effect=MoveTo(
                oid=_LINEN_CART_ID,
                goal=(24.0, 15.90, math.pi / 2.0),
                speed=0.48,
            ),
        ),
        Event(
            name="E4 cleaning round clears the ICU service door",
            trigger=at_time(148.0),
            effect=MoveTo(
                oid=_CLEANING_CART_ID,
                goal=(36.0, 12.40, math.pi / 2.0),
                speed=0.40,
            ),
        ),
        Event(
            name="E5 cleaning trolley carries on down the ward",
            trigger=at_time(188.0),
            effect=MoveTo(
                oid=_CLEANING_CART_ID,
                goal=(38.5, 12.40, math.pi / 2.0),
                speed=0.44,
            ),
        ),
    ]

    cfg = Config()

    decision_points = [
        {
            "name": "triage_exit",
            "location": (3.3, _MAIN_CORRIDOR[1]),
            "temporary_obstacles": [_TRIAGE_BED_ID],
            "events": ["E1"],
            "tradeoff": _option_costs(
                cfg, by_id[_TRIAGE_BED_ID],
                move_distance=1.25,
                contact_distance=1.8,
                detour_distance=6.4,
                wait_seconds=18.0,
            ),
        },
        {
            "name": "central_double_gate",
            "location": (_CENTRAL_GATE_X, 14.2),
            "temporary_obstacles": [_XRAY_ID, _LINEN_CART_ID],
            "events": ["E2", "E3"],
            "tradeoffs": {
                "move_heavy_xray": _option_costs(
                    cfg, by_id[_XRAY_ID],
                    move_distance=1.15,
                    contact_distance=2.2,
                    detour_distance=9.5,
                    wait_seconds=30.0,
                ),
                "move_light_linen_cart": _option_costs(
                    cfg, by_id[_LINEN_CART_ID],
                    move_distance=1.15,
                    contact_distance=2.0,
                    detour_distance=9.5,
                    wait_seconds=30.0,
                ),
            },
        },
        {
            "name": "icu_double_gate",
            "location": (_ICU_GATE_X, 14.1),
            "temporary_obstacles": [_OCCUPIED_BED_ID, _CLEANING_CART_ID],
            "events": ["E4", "E5"],
            "tradeoffs": {
                "move_occupied_bed": _option_costs(
                    cfg, by_id[_OCCUPIED_BED_ID],
                    move_distance=1.2,
                    contact_distance=2.0,
                    detour_distance=6.8,
                    wait_seconds=24.0,
                ),
                "move_cleaning_cart": _option_costs(
                    cfg, by_id[_CLEANING_CART_ID],
                    move_distance=1.1,
                    contact_distance=1.8,
                    detour_distance=6.8,
                    wait_seconds=24.0,
                ),
            },
        },
    ]

    _realism.check_layout(
        "hospital", workspace=workspace, static=walls,
        movable=movable, start=_START, goal=_GOAL, cfg=cfg,
    )

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "dynamics": events,
        "cfg": cfg,
        "decision_points": decision_points,
    }
