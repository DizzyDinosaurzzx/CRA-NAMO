"""复杂迷宫带门场景：更多门板与斜置障碍，考验门板选择与推挪规划。"""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from llm_difficulty import friction_force, material_height, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle

DOOR_OID_BASE = 900

DEFAULT_DOOR_MATERIAL = "wooden_crate"
DOOR_CLEARANCE = 0.2

def _door_blocker(label: str, x: float, y: float, l: float, d: float,
                  material: str = DEFAULT_DOOR_MATERIAL,
                  difficulty: float | None = None) -> MovableObstacle:
    """生成封堵门洞的门板障碍，oid 从 DOOR_OID_BASE 起按编号分配。"""
    # 长边留出 DOOR_CLEARANCE 间隙，避免门板完全封死门洞
    if l > d:
        l -= DOOR_CLEARANCE
    else:
        d -= DOOR_CLEARANCE
    h = material_height(material)
    if difficulty is None:
        difficulty = round(friction_force(material_mu_rho(material), l * d * h), 3)
    return MovableObstacle(
        x=x,
        y=y,
        l=l,
        d=d,
        h=h,
        theta=0.0,
        material=material,
        difficulty=difficulty,
        oid=DOOR_OID_BASE + int(label[1:]),
    )

def create():
    """构建复杂迷宫带门场景。"""
    workspace = box(0, 0, 30, 30)
    t = 0.45

    walls = [
        StaticObstacle(box(0.0, 0.0, 30.0, t), "outer_bottom"),
        StaticObstacle(box(29.55, 0.0, 30.0, 30.0), "outer_right"),
        StaticObstacle(box(5.5, 29.55, 30.0, 30.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 25.5), "outer_left"),

        StaticObstacle(box(0.0, 25.5, 2.0, 25.5 + t), "upper_left_horizontal_top_far_left"),
        StaticObstacle(box(4.0, 25.5, 12.5, 25.5 + t), "upper_left_horizontal_top_middle"),
        StaticObstacle(box(14.0, 25.5, 16.5, 25.5 + t), "upper_left_horizontal_top_right_1"),
        StaticObstacle(box(18.5, 25.5, 20, 25.5 + t), "upper_left_horizontal_top_right_2"),

        StaticObstacle(box(0.0, 20.8, 3.0, 20.8 + t), "upper_left_horizontal_bottom_left"),
        StaticObstacle(box(4.5, 20.8, 7.0, 20.8 + t), "upper_left_horizontal_bottom_middle_1"),
        StaticObstacle(box(9.0, 20.8, 11.0, 20.8 + t), "upper_left_horizontal_bottom_middle_2"),
        StaticObstacle(box(13.0, 20.8, 17.0, 20.8 + t), "upper_left_horizontal_bottom_middle_3"),
        StaticObstacle(box(18.5, 20.8, 20, 20.8 + t), "upper_left_horizontal_bottom_right"),

        StaticObstacle(box(14.8, 20.8, 14.8 + t, 22.5), "upper_middle_vertical_1_lower"),
        StaticObstacle(box(14.8, 24.5, 14.8 + t, 28), "upper_middle_vertical_1_upper"),
        StaticObstacle(box(11.5, 28, 11.5 + t, 29.6), "upper_middle_vertical_2"),
        StaticObstacle(box(9.5, 25.6, 9.5 + t, 28), "upper_middle_vertical_3"),
        StaticObstacle(box(7.5, 28, 7.5 + t, 29.6), "upper_middle_vertical_4"),
        StaticObstacle(box(5.5, 25.6, 5.5 + t, 27.0), "upper_middle_vertical_5_lower"),
        StaticObstacle(box(5.5, 28.5, 5.5 + t, 29.6), "upper_middle_vertical_5_upper"),

        StaticObstacle(box(20, 20.8, 23.0, 20.8 + t), "upper_right_horizontal_bottom_left"),
        StaticObstacle(box(24.5, 20.8, 27.0, 20.8 + t), "upper_right_horizontal_bottom_middle"),
        StaticObstacle(box(28.5, 20.8, 30.0, 20.8 + t), "upper_right_horizontal_bottom_right"),

        StaticObstacle(box(20, 20.8, 20 + t, 22.5), "upper_right_vertical_left_lower"),
        StaticObstacle(box(20, 24.0, 20 + t, 26.5), "upper_right_vertical_left_middle"),
        StaticObstacle(box(20, 28.5, 20 + t, 30.0), "upper_right_vertical_left_upper"),

        StaticObstacle(box(0.0, 12.0, 1.5, 12.0+t), "middle_left_horizontal_top_far_left"),
        StaticObstacle(box(3.5, 12.0, 7.0, 12.0+t), "middle_left_horizontal_top_middle"),
        StaticObstacle(box(8.5, 12.0, 10.0, 12.0+t), "middle_left_horizontal_top_right"),
        StaticObstacle(box(0.0, 5.0, 1.5, 5.0+t), "middle_left_horizontal_bottom_far_left"),
        StaticObstacle(box(3.5, 5.0, 7.0, 5.0+t), "middle_left_horizontal_bottom_middle"),
        StaticObstacle(box(8.5, 5.0, 10.0, 5.0+t), "middle_left_horizontal_bottom_right"),
        StaticObstacle(box(9.8, 0, 10.25, 2.0), "middle_center_vertical_left_lower"),
        StaticObstacle(box(9.8, 3.5, 10.25, 8.0), "middle_center_vertical_left_middle"),
        StaticObstacle(box(9.8, 9.5, 10.25, 16.7), "middle_center_vertical_left_upper"),
        StaticObstacle(box(9.8, 16.25, 14.5, 16.7), "middle_center_horizontal_top_left"),
        StaticObstacle(box(16.5, 16.25, 19.8, 16.7), "middle_center_horizontal_top_right"),
        StaticObstacle(box(25.0, 12, 27.0, 12.45), "middle_right_horizontal_top_left"),
        StaticObstacle(box(28.5, 12, 30.0, 12.45), "middle_right_horizontal_top_right"),
        StaticObstacle(box(20.0, 8, 22.0, 8.45), "middle_right_horizontal_bottom_far_left"),
        StaticObstacle(box(24.0, 8, 27.0, 8.45), "middle_right_horizontal_bottom_middle"),
        StaticObstacle(box(28.5, 8, 30.0, 8.45), "middle_right_horizontal_bottom_right"),
        StaticObstacle(box(20.0, 0, 20+t, 3.0), "middle_right_vertical_left_lower"),
        StaticObstacle(box(20.0, 4.5, 20+t, 8.45), "middle_right_vertical_left_upper"),

        StaticObstacle(box(5.0, 0.0, 5.45, 2.0), "lower_left_vertical_lower"),
        StaticObstacle(box(5.0, 3.5, 5.45, 5.0), "lower_left_vertical_upper"),

        StaticObstacle(box(25.0, 0.0, 25+t, 2.0), "right_section_vertical_lower"),
        StaticObstacle(box(25.0, 3.5, 25+t, 9.5), "right_section_vertical_lower_middle"),
        StaticObstacle(box(25.0, 11.0, 25+t, 15.0), "right_section_vertical_upper_middle"),
        StaticObstacle(box(25.0, 17.0, 25+t, 20.8), "right_section_vertical_upper"),

    ]

    door_blockers = [
        _door_blocker("d1", 5.5 + t / 2, 27.75, t, 1.5),
        _door_blocker("d2", 20 + t / 2, 27.5, t, 2.0),
        _door_blocker("d3", 3.0, 25.5 + t / 2, 2.0, t),
        _door_blocker("d4", 13.25, 25.5 + t / 2, 1.5, t),
        _door_blocker("d5", 20 + t / 2, 23.25, t, 1.5),

        _door_blocker("d6", 3.75, 20.8 + t / 2, 1.5, t),
        _door_blocker("d7", 8.0, 20.8 + t / 2, 2.0, t),
        _door_blocker("d8", 12.0, 20.8 + t / 2, 2.0, t),
        _door_blocker("d9", 17.75, 20.8 + t / 2, 1.5, t),
        _door_blocker("d10", 23.75, 20.8 + t / 2, 1.5, t),
        _door_blocker("d11", 27.75, 20.8 + t / 2, 1.5, t),

        _door_blocker("d12", 25 + t / 2, 16.0, t, 2.0),
        _door_blocker("d13", 7.75, 12.0 + t / 2, 1.5, t),
        _door_blocker("d14", 27.75, 12.0 + t / 2, 1.5, t),
        _door_blocker("d15", 25 + t / 2, 10.25, t, 1.5),
        _door_blocker("d16", 10.025, 8.75, 0.45, 1.5),
        _door_blocker("d17", 27.75, 8.0 + t / 2, 1.5, t),
        _door_blocker("d18", 7.75, 5.0 + t / 2, 1.5, t),

        _door_blocker("d19", 20 + t / 2, 3.75, t, 1.5),
        _door_blocker("d20", 5.0 + t / 2, 2.75, t, 1.5),
        _door_blocker("d21", 10.025, 2.75, 0.45, 1.5),
        _door_blocker(
            "d22", 25 + t / 2, 2.75, t, 1.5,
            material="concrete_block",
        ),

        _door_blocker("d23", 2.5, 12.0 + t / 2, 2.0, t),
        _door_blocker("d24", 15.5, 16.25 + t / 2, 2.0, t),
        _door_blocker("d25", 23.0, 8.0 + t / 2, 2.0, t),
        _door_blocker("d26", 2.5, 5.0 + t / 2, 2.0, t),

        _door_blocker("d27", 17.5, 25.5 + t / 2, 2.0, t),
        _door_blocker("d28", 14.8 + t / 2, 23.5, t, 2.0),
    ]

    start = (28, 2)
    goal = (2,27)

    manual_obstacles = [
        MovableObstacle(
            x=1.5,
            y=13.5,
            l=2.0,
            d=2.0,
            h=1,
            theta=0.0,
            material="concrete_block",
            difficulty=56505.6,
            oid=1,
        ),
        MovableObstacle(
            x=12.8, y=11, l=4.6, d=6.6, h=1.6, theta=0.0,
            material="industrial_machine", difficulty=166785.696, oid=2,
        ),
        MovableObstacle(
            x=17.7, y=11.5, l=2.0, d=3.8, h=1, theta=0.0,
            material="wooden_crate", difficulty=2013.012, oid=3,
        ),
        MovableObstacle(
            x=12.0, y=6.65, l=1.6, d=1.2, h=1, theta=0.0,
            material="wooden_crate", difficulty=508.55, oid=4,
        ),
        MovableObstacle(
            x=12.0, y=3.25, l=1.5, d=2.0, h=1, theta=0.0,
            material="wooden_crate", difficulty=794.61, oid=5,
        ),
        MovableObstacle(
            x=13.8, y=5.2, l=0.8, d=1.0, h=1, theta=0.0,
            material="wooden_crate", difficulty=211.896, oid=6,
        ),
        MovableObstacle(
            x=18.9, y=4.7, l=1.2, d=1.3, h=1, theta=0.0,
            material="wooden_crate", difficulty=413.197, oid=7,
        ),
        MovableObstacle(
            x=18.9, y=3.4, l=1.2, d=1.3, h=1, theta=0.0,
            material="wooden_crate", difficulty=413.197, oid=8,
        ),
        MovableObstacle(
            x=27.75, y=6.7, l=1.4, d=1.2, h=0.8, theta=0.0,
            material="cardboard_box", difficulty=184.585, oid=10,
        ),
        MovableObstacle(
            x=13.05, y=27.7, l=1.2, d=1.2, h=1, theta=0.0,
            material="wooden_crate", difficulty=381.413, oid=11,
        ),
        MovableObstacle(
            x=23.75, y=27.75, l=4.5, d=3.5, h=1, theta=0.0,
            material="wooden_crate", difficulty=4171.703, oid=12,
        ),
        MovableObstacle(
            x=10.25, y=24.3, l=4.5, d=2.1, h=1, theta=0.0,
            material="wooden_crate", difficulty=2503.022, oid=13,
        ),
        MovableObstacle(
            x=23.0, y=23.05, l=1.1, d=3.0, h=1, theta=0.0,
            material="wooden_crate", difficulty=874.071, oid=14,
        ),

        MovableObstacle(
            x=14.0, y=18.7, l=2.5, d=2.3, h=1, theta=0.0,
            material="wooden_crate", difficulty=1523.003, oid=16,
        ),
        MovableObstacle(
            x=28.0, y=16.75, l=1.8, d=2.8, h=1, theta=0.0,
            material="wooden_crate", difficulty=1334.945, oid=17,
        ),
        MovableObstacle(
            x=3.8, y=13.5, l=1.8, d=1.6, h=1, theta=0.0,
            material="wooden_crate", difficulty=762.826, oid=18,
        ),
        MovableObstacle(
            x=6.6, y=7.4, l=5.0, d=1.4, h=1, theta=0.53,
            material="wooden_crate", difficulty=1854.09, oid=19,
        ),
        MovableObstacle(
            x=2.6, y=3.1, l=1.6, d=1.5, h=1, theta=0.0,
            material="wooden_crate", difficulty=635.688, oid=20,
        ),
        MovableObstacle(
            x=6.9, y=1.5, l=2.2, d=1.8, h=1, theta=0.0,
            material="wooden_crate", difficulty=1048.885, oid=21,
        ),
        MovableObstacle(
            x=17.1, y=27.5, l=3.0, d=0.75, h=1, theta=0.92,
            material="wooden_crate", difficulty=595.957, oid=22,
        ),
        MovableObstacle(
            x=6.4, y=23.7, l=4.4, d=0.75, h=1, theta=-0.65,
            material="wooden_crate", difficulty=874.071, oid=23,
        ),
        MovableObstacle(
            x=17.2, y=22.8, l=2.7, d=0.85, h=1, theta=-0.82,
            material="wooden_crate", difficulty=607.877, oid=24,
        ),
        MovableObstacle(
            x=7.5, y=14.6, l=3.8, d=0.8, h=1, theta=-0.91,
            material="wooden_crate", difficulty=805.205, oid=25,
        ),
        MovableObstacle(
            x=22.5, y=12.3, l=8.25, d=0.7, h=1, theta=1.07,
            material="wooden_crate", difficulty=1529.624, oid=26,
        ),
        MovableObstacle(
            x=3.8, y=9.5, l=4.3, d=0.75, h=1, theta=-1.05,
            material="wooden_crate", difficulty=854.206, oid=27,
        ),
        MovableObstacle(
            x=23.0, y=4.0, l=3.8, d=0.9, h=1, theta=-0.7,
            material="wooden_crate", difficulty=905.855, oid=28,
        ),
        MovableObstacle(
            x=15, y=2.0, l=4.6, d=0.75, h=1, theta=-0.45,
            material="wooden_crate", difficulty=913.801, oid=29,
        ),
        MovableObstacle(
            x=28.1, y=23.2, l=0.5, d=0.5, h=1, theta=0.00,
            material="wooden_crate", difficulty=66.218, oid=30,
        ),
        MovableObstacle(
            x=26.2, y=23.2, l=0.5, d=0.5, h=1, theta=0.17,
            material="wooden_crate", difficulty=66.218, oid=31,
        ),
        MovableObstacle(
            x=27, y=23.2, l=0.5, d=0.5, h=1, theta=0.35,
            material="wooden_crate", difficulty=66.218, oid=32,
        ),
        MovableObstacle(
            x=26.5, y=22, l=0.5, d=0.5, h=1, theta=0.52,
            material="wooden_crate", difficulty=66.218, oid=33,
        ),
        MovableObstacle(
            x=27.3, y=22.2, l=0.5, d=0.5, h=1, theta=0.70,
            material="wooden_crate", difficulty=66.218, oid=34,
        ),
        MovableObstacle(
            x=27.9, y=22, l=0.5, d=0.5, h=1, theta=0.87,
            material="wooden_crate", difficulty=66.218, oid=35,
        ),
        MovableObstacle(
            x=27, y=24, l=0.5, d=0.5, h=1, theta=1.05,
            material="wooden_crate", difficulty=66.218, oid=36,
        ),
        MovableObstacle(
            x=28, y=24, l=0.5, d=0.5, h=1, theta=1.22,
            material="wooden_crate", difficulty=66.218, oid=37,
        ),

    ]

    movable = [
        *manual_obstacles,
        *door_blockers,
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": Config(),
    }
