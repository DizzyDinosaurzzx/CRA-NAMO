from __future__ import annotations
from shapely.geometry import box
from config import Config
from llm_difficulty import material_density
from obstacle import MovableObstacle, StaticObstacle

# 门洞障碍物的 oid 号段：d1 -> 901，d28 -> 928。
# 必须是整数——原先用 "d1" 这类字符串，一旦和杂物的整数 oid 落进同一次排序
# （planner 汇报推动碰撞时会 sorted(...)），Python 3 会直接抛 TypeError。
DOOR_OID_BASE = 900

DEFAULT_DOOR_MATERIAL = "wooden_crate"
# 连续 SE(2) 推动要求障碍物初始位姿不能与墙体相切。门障碍物沿门洞方向
# 比门洞短 0.2 m（两端各留 0.1 m）；余量仍小于机器人直径，因此不会让
# 机器人从障碍物旁边直接穿过。
DOOR_CLEARANCE = 0.2


def _door_blocker(label: str, x: float, y: float, l: float, d: float,
                  material: str = DEFAULT_DOOR_MATERIAL,
                  difficulty: float | None = None) -> MovableObstacle:
    """创建可手工配置材质和难度、且与门框留有余量的门洞障碍物。"""
    if l > d:
        l -= DOOR_CLEARANCE
    else:
        d -= DOOR_CLEARANCE
    if difficulty is None:
        difficulty = round(material_density(material) * l * d, 3)
    return MovableObstacle(
        x=x,
        y=y,
        l=l,
        d=d,
        theta=0.0,
        material=material,
        difficulty=difficulty,
        oid=DOOR_OID_BASE + int(label[1:]),
    )


def create():
    workspace = box(0, 0, 30, 30)
    t = 0.45

    walls = [
        # ===== 外框 =====
        StaticObstacle(box(0.0, 0.0, 30.0, t), "outer_bottom"),
        StaticObstacle(box(29.55, 0.0, 30.0, 30.0), "outer_right"),
        StaticObstacle(box(5.5, 29.55, 30.0, 30.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 25.5), "outer_left"),

        # ===== 左上区域 =====
        # 顶部横墙在 x=2.0~4.0、x=12.5~14.0、x=16.5~18.5 留门。
        StaticObstacle(box(0.0, 25.5, 2.0, 25.5 + t), "upper_left_horizontal_top_far_left"),
        StaticObstacle(box(4.0, 25.5, 12.5, 25.5 + t), "upper_left_horizontal_top_middle"),
        StaticObstacle(box(14.0, 25.5, 16.5, 25.5 + t), "upper_left_horizontal_top_right_1"),
        StaticObstacle(box(18.5, 25.5, 20, 25.5 + t), "upper_left_horizontal_top_right_2"),

        # 底部横墙在红框位置增加 x=7.0~9.0、x=11.0~13.0 两扇门。
        StaticObstacle(box(0.0, 20.8, 3.0, 20.8 + t), "upper_left_horizontal_bottom_left"),
        StaticObstacle(box(4.5, 20.8, 7.0, 20.8 + t), "upper_left_horizontal_bottom_middle_1"),
        StaticObstacle(box(9.0, 20.8, 11.0, 20.8 + t), "upper_left_horizontal_bottom_middle_2"),
        StaticObstacle(box(13.0, 20.8, 17.0, 20.8 + t), "upper_left_horizontal_bottom_middle_3"),
        StaticObstacle(box(18.5, 20.8, 20, 20.8 + t), "upper_left_horizontal_bottom_right"),
        #StaticObstacle(box(5.0, 16.5, 5.0 + t, 20.8), "upper_left_vertical_connector"),

        # 中上竖墙在 y=22.5~24.5 留门。
        StaticObstacle(box(14.8, 20.8, 14.8 + t, 22.5), "upper_middle_vertical_1_lower"),
        StaticObstacle(box(14.8, 24.5, 14.8 + t, 28), "upper_middle_vertical_1_upper"),
        StaticObstacle(box(11.5, 28, 11.5 + t, 29.6), "upper_middle_vertical_2"),
        StaticObstacle(box(9.5, 25.6, 9.5 + t, 28), "upper_middle_vertical_3"),
        StaticObstacle(box(7.5, 28, 7.5 + t, 29.6), "upper_middle_vertical_4"),
        # 最左侧竖墙在 y=27.0~28.5 留门，连接目标房间。
        StaticObstacle(box(5.5, 25.6, 5.5 + t, 27.0), "upper_middle_vertical_5_lower"),
        StaticObstacle(box(5.5, 28.5, 5.5 + t, 29.6), "upper_middle_vertical_5_upper"),
        

        # ===== 右上区域 =====
        # 右上房间的 U / Γ 形结构
        # 底墙在 x=23.0~24.5、x=27.0~28.5 留门。
        StaticObstacle(box(20, 20.8, 23.0, 20.8 + t), "upper_right_horizontal_bottom_left"),
        StaticObstacle(box(24.5, 20.8, 27.0, 20.8 + t), "upper_right_horizontal_bottom_middle"),
        StaticObstacle(box(28.5, 20.8, 30.0, 20.8 + t), "upper_right_horizontal_bottom_right"),

        # 左墙按红框在 y=22.5~24.0、y=26.5~28.5 留门。
        StaticObstacle(box(20, 20.8, 20 + t, 22.5), "upper_right_vertical_left_lower"),
        StaticObstacle(box(20, 24.0, 20 + t, 26.5), "upper_right_vertical_left_middle"),
        StaticObstacle(box(20, 28.5, 20 + t, 30.0), "upper_right_vertical_left_upper"),

        # ===== 中部 =====
        # 左中横墙在 x=1.5~3.5、x=7.0~8.5 留门。
        StaticObstacle(box(0.0, 12.0, 1.5, 12.0+t), "middle_left_horizontal_top_far_left"),
        StaticObstacle(box(3.5, 12.0, 7.0, 12.0+t), "middle_left_horizontal_top_middle"),
        StaticObstacle(box(8.5, 12.0, 10.0, 12.0+t), "middle_left_horizontal_top_right"),
        # 左下横墙在 x=1.5~3.5、x=7.0~8.5 留门。
        StaticObstacle(box(0.0, 5.0, 1.5, 5.0+t), "middle_left_horizontal_bottom_far_left"),
        StaticObstacle(box(3.5, 5.0, 7.0, 5.0+t), "middle_left_horizontal_bottom_middle"),
        StaticObstacle(box(8.5, 5.0, 10.0, 5.0+t), "middle_left_horizontal_bottom_right"),

        # 中部大矩形左墙按红框在 y=2.0~3.5、y=8.0~9.5 留门。
        StaticObstacle(box(9.8, 0, 10.25, 2.0), "middle_center_vertical_left_lower"),
        StaticObstacle(box(9.8, 3.5, 10.25, 8.0), "middle_center_vertical_left_middle"),
        StaticObstacle(box(9.8, 9.5, 10.25, 16.7), "middle_center_vertical_left_upper"),
        # 中央上横墙在 x=14.5~16.5 留门。
        StaticObstacle(box(9.8, 16.25, 14.5, 16.7), "middle_center_horizontal_top_left"),
        StaticObstacle(box(16.5, 16.25, 19.8, 16.7), "middle_center_horizontal_top_right"),

        # 右侧上下房间的横墙均在 x=27.0~28.5 留门。
        StaticObstacle(box(25.0, 12, 27.0, 12.45), "middle_right_horizontal_top_left"),
        StaticObstacle(box(28.5, 12, 30.0, 12.45), "middle_right_horizontal_top_right"),
        # 右下横墙在 x=22.0~24.0 留门。
        StaticObstacle(box(20.0, 8, 22.0, 8.45), "middle_right_horizontal_bottom_far_left"),
        StaticObstacle(box(24.0, 8, 27.0, 8.45), "middle_right_horizontal_bottom_middle"),
        StaticObstacle(box(28.5, 8, 30.0, 8.45), "middle_right_horizontal_bottom_right"),

        # 左竖墙在 y=3.0~4.5 留门，连接中央空间。
        StaticObstacle(box(20.0, 0, 20+t, 3.0), "middle_right_vertical_left_lower"),
        StaticObstacle(box(20.0, 4.5, 20+t, 8.45), "middle_right_vertical_left_upper"),

        # ===== 左下区域 =====
        # 左下小房间的右墙在 y=2.0~3.5 留门。
        StaticObstacle(box(5.0, 0.0, 5.45, 2.0), "lower_left_vertical_lower"),
        StaticObstacle(box(5.0, 3.5, 5.45, 5.0), "lower_left_vertical_upper"),

        # ===== 右侧分隔墙 =====
        # 右侧分隔墙在 y=2.0~3.5、y=9.5~11.0、y=15.0~17.0 留门。
        StaticObstacle(box(25.0, 0.0, 25+t, 2.0), "right_section_vertical_lower"),
        StaticObstacle(box(25.0, 3.5, 25+t, 9.5), "right_section_vertical_lower_middle"),
        StaticObstacle(box(25.0, 11.0, 25+t, 15.0), "right_section_vertical_upper_middle"),
        StaticObstacle(box(25.0, 17.0, 25+t, 20.8), "right_section_vertical_upper"),

    ]

    # ===== 门洞障碍物 =====
    # 编号顺序：从上到下、同一高度从左到右。
    # 传入门洞的完整尺寸；_door_blocker 会沿门洞方向扣除 DOOR_CLEARANCE。
    # 横墙门洞：l 是门宽，d 是墙厚 t；竖墙门洞反之。
    door_blockers = [
        # 顶部区域
        _door_blocker("d1", 5.5 + t / 2, 27.75, t, 1.5),
        _door_blocker("d2", 20 + t / 2, 27.5, t, 2.0),
        _door_blocker("d3", 3.0, 25.5 + t / 2, 2.0, t),
        _door_blocker("d4", 13.25, 25.5 + t / 2, 1.5, t),
        _door_blocker("d5", 20 + t / 2, 23.25, t, 1.5),

        # y=20.8 横墙上的六个门洞
        _door_blocker("d6", 3.75, 20.8 + t / 2, 1.5, t),
        _door_blocker("d7", 8.0, 20.8 + t / 2, 2.0, t),
        _door_blocker("d8", 12.0, 20.8 + t / 2, 2.0, t),
        _door_blocker("d9", 17.75, 20.8 + t / 2, 1.5, t),
        _door_blocker("d10", 23.75, 20.8 + t / 2, 1.5, t),
        _door_blocker("d11", 27.75, 20.8 + t / 2, 1.5, t),

        # 中部和右侧区域
        _door_blocker("d12", 25 + t / 2, 16.0, t, 2.0),
        _door_blocker("d13", 7.75, 12.0 + t / 2, 1.5, t),
        _door_blocker("d14", 27.75, 12.0 + t / 2, 1.5, t),
        _door_blocker("d15", 25 + t / 2, 10.25, t, 1.5),
        _door_blocker("d16", 10.025, 8.75, 0.45, 1.5),
        _door_blocker("d17", 27.75, 8.0 + t / 2, 1.5, t),
        _door_blocker("d18", 7.75, 5.0 + t / 2, 1.5, t),

        # 下部区域
        _door_blocker("d19", 20 + t / 2, 3.75, t, 1.5),
        _door_blocker("d20", 5.0 + t / 2, 2.75, t, 1.5),
        _door_blocker("d21", 10.025, 2.75, 0.45, 1.5),
        _door_blocker(
            "d22", 25 + t / 2, 2.75, t, 1.5,
            material="concrete_block",
        ),

        # 新增的四个门洞（对应图中的四个红框）
        _door_blocker("d23", 2.5, 12.0 + t / 2, 2.0, t),
        _door_blocker("d24", 15.5, 16.25 + t / 2, 2.0, t),
        _door_blocker("d25", 23.0, 8.0 + t / 2, 2.0, t),
        _door_blocker("d26", 2.5, 5.0 + t / 2, 2.0, t),

        # 新增的两扇门洞
        _door_blocker("d27", 17.5, 25.5 + t / 2, 2.0, t),
        _door_blocker("d28", 14.8 + t / 2, 23.5, t, 2.0),
    ]

    start = (28, 2)
    goal = (4.5, 11)

    # ===== 手工障碍物（以后只需要编辑这个列表）=====
    # oid 必须唯一，并建议使用 1～899，避免与门的 901～928 冲突。
    # x/y 是中心坐标，l/d 是长宽，theta 是弧度。
    manual_obstacles = [
        MovableObstacle(
            x=1.5,
            y=13.5,
            l=2.0,
            d=2.0,
            theta=0.0,
            material="concrete_block",
            difficulty=100.0,
            oid=1,
        ),
        # 图中红框位置：每个红框对应一个固定障碍物。
        MovableObstacle(
            x=12.8, y=11, l=4.6, d=6.6, theta=0.0,
            material="industrial_machine", difficulty=37.5, oid=2,
        ),
        MovableObstacle(
            x=17.7, y=11.5, l=2.0, d=3.8, theta=0.0,
            material="wooden_crate", difficulty=5.7, oid=3,
        ),
        MovableObstacle(
            x=12.0, y=6.65, l=1.6, d=1.2, theta=0.0,
            material="wooden_crate", difficulty=1.44, oid=4,
        ),
        MovableObstacle(
            x=12.0, y=3.25, l=1.5, d=2.0, theta=0.0,
            material="wooden_crate", difficulty=2.25, oid=5,
        ),
        MovableObstacle(
            x=13.8, y=5.2, l=0.8, d=1.0, theta=0.0,
            material="wooden_crate", difficulty=0.6, oid=6,
        ),
        MovableObstacle(
            x=18.9, y=4.7, l=1.2, d=1.3, theta=0.0,
            material="wooden_crate", difficulty=1.17, oid=7,
        ),
        MovableObstacle(
            x=18.9, y=3.4, l=1.2, d=1.3, theta=0.0,
            material="wooden_crate", difficulty=1.17, oid=8,
        ),
        MovableObstacle(
            x=27.75, y=6.7, l=1.4, d=1.2, theta=0.0,
            material="cardboard_box", difficulty=0.118, oid=10,
        ),
        MovableObstacle(
            x=13.05, y=27.7, l=1.2, d=1.2, theta=0.0,
            material="wooden_crate", difficulty=1.08, oid=11,
        ),
        MovableObstacle(
            x=23.75, y=27.75, l=4.5, d=3.5, theta=0.0,
            material="wooden_crate", difficulty=11.812, oid=12,
        ),
        MovableObstacle(
            x=10.25, y=24.3, l=4.5, d=2.1, theta=0.0,
            material="wooden_crate", difficulty=7.088, oid=13,
        ),
        MovableObstacle(
            x=23.0, y=23.05, l=1.1, d=3.0, theta=0.0,
            material="wooden_crate", difficulty=2.475, oid=14,
        ),
    
        MovableObstacle(
            x=14.0, y=18.7, l=2.5, d=2.3, theta=0.0,
            material="wooden_crate", difficulty=4.312, oid=16,
        ),
        MovableObstacle(
            x=28.0, y=16.75, l=1.8, d=2.8, theta=0.0,
            material="wooden_crate", difficulty=3.78, oid=17,
        ),
        MovableObstacle(
            x=3.8, y=13.5, l=1.8, d=1.6, theta=0.0,
            material="wooden_crate", difficulty=2.16, oid=18,
        ),
        MovableObstacle(
            x=6.6, y=7.4, l=5.0, d=1.4, theta=0.53,
            material="wooden_crate", difficulty=5.25, oid=19,
        ),
        MovableObstacle(
            x=2.6, y=3.1, l=1.6, d=1.5, theta=0.0,
            material="wooden_crate", difficulty=1.8, oid=20,
        ),
        MovableObstacle(
            x=6.9, y=1.5, l=2.2, d=1.8, theta=0.0,
            material="wooden_crate", difficulty=2.97, oid=21,
        ),
        # 用户标注的 8 个斜向障碍物（从上到下排列）。
        MovableObstacle(
            x=17.1, y=27.5, l=3.0, d=0.75, theta=0.92,
            material="wooden_crate", difficulty=1.688, oid=22,
        ),
        MovableObstacle(
            x=6.4, y=23.7, l=4.4, d=0.75, theta=-0.65,
            material="wooden_crate", difficulty=2.475, oid=23,
        ),
        MovableObstacle(
            x=17.2, y=22.8, l=2.7, d=0.85, theta=-0.82,
            material="wooden_crate", difficulty=1.721, oid=24,
        ),
        MovableObstacle(
            x=7.5, y=14.6, l=3.8, d=0.8, theta=-0.91,
            material="wooden_crate", difficulty=2.28, oid=25,
        ),
        MovableObstacle(
            x=22.5, y=12.3, l=8.3, d=0.7, theta=1.07,
            material="wooden_crate", difficulty=4.62, oid=26,
        ),
        MovableObstacle(
            x=3.8, y=9.5, l=4.3, d=0.75, theta=-1.05,
            material="wooden_crate", difficulty=2.419, oid=27,
        ),
        MovableObstacle(
            x=23.0, y=4.0, l=3.8, d=0.9, theta=-0.7,
            material="wooden_crate", difficulty=2.22, oid=28,
        ),
        MovableObstacle(
            x=15, y=2.0, l=4.6, d=0.75, theta=-0.45,
            material="wooden_crate", difficulty=2.025, oid=29,
        ),

        # 右上房间红框区域：8 个 0.5 m × 0.5 m、朝向各异的小障碍物。
        # 正方形每旋转 90° 几何形状重复，因此角度取 0°～70°。
        MovableObstacle(
            x=28.1, y=23.2, l=0.5, d=0.5, theta=0.00,
            material="wooden_crate", difficulty=0.188, oid=30,
        ),
        MovableObstacle(
            x=26.2, y=23.2, l=0.5, d=0.5, theta=0.17,
            material="wooden_crate", difficulty=0.188, oid=31,
        ),
        MovableObstacle(
            x=27, y=23.2, l=0.5, d=0.5, theta=0.35,
            material="wooden_crate", difficulty=0.188, oid=32,
        ),
        MovableObstacle(
            x=26.5, y=22, l=0.5, d=0.5, theta=0.52,
            material="wooden_crate", difficulty=0.188, oid=33,
        ),
        MovableObstacle(
            x=27.3, y=22.2, l=0.5, d=0.5, theta=0.70,
            material="wooden_crate", difficulty=0.188, oid=34,
        ),
        MovableObstacle(
            x=27.9, y=22, l=0.5, d=0.5, theta=0.87,
            material="wooden_crate", difficulty=0.188, oid=35,
        ),
        MovableObstacle(
            x=27, y=24, l=0.5, d=0.5, theta=1.05,
            material="wooden_crate", difficulty=0.188, oid=36,
        ),
        MovableObstacle(
            x=28, y=24, l=0.5, d=0.5, theta=1.22,
            material="wooden_crate", difficulty=0.188, oid=37,
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
