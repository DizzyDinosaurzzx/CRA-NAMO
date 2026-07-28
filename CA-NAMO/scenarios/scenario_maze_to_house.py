from __future__ import annotations
import random
from shapely.geometry import Point, box
from shapely.ops import unary_union
from config import Config
from llm_difficulty import material_density
from obstacle import MovableObstacle, StaticObstacle

# 门洞障碍物的 oid 号段：d1 -> 901，d22 -> 922。
# 必须是整数——原先用 "d1" 这类字符串，一旦和杂物的整数 oid 落进同一次排序
# （planner 汇报推动碰撞时会 sorted(...)），Python 3 会直接抛 TypeError。
DOOR_OID_BASE = 900

# 随机杂物 / 门洞障碍物各自的材质调色板。
CLUTTER_MATERIALS = (
    "styrofoam_box", "foam_mat", "cardboard_box", "empty_cart",
    "plastic_chair", "trash_bin", "stool", "chair", "empty_shelf",
    "cart", "wooden_table", "wooden_crate", "shelf", "sofa", "cabinet",
)
DOOR_MATERIALS = (
    "cardboard_box", "chair", "cart", "wooden_table", "wooden_crate",
    "cabinet", "pallet", "loaded_pallet", "filing_cabinet",
)

# 真实 difficulty = 材质密度 x 底面积 x 随机扰动。
# 扰动是刻意留的：机器人只能从材质标签和尺寸估出 密度 x 面积，这个系数就是
# 它估不准的部分，必须靠触碰才能得知真值。
DIFFICULTY_JITTER = (0.6, 1.6)
RNG_SEED = 20260728


def _sample_difficulty(rng: random.Random, material: str, area: float) -> float:
    """按材质与尺寸采样真实难度，带随机扰动。"""
    base = material_density(material) * area
    return max(0.01, round(base * rng.uniform(*DIFFICULTY_JITTER), 3))


def _door_blocker(rng: random.Random, label: str, x: float, y: float,
                  l: float, d: float) -> MovableObstacle:
    """创建与门洞同宽、与墙体同厚的可移动障碍物。"""
    material = rng.choice(DOOR_MATERIALS)
    return MovableObstacle(
        x=x,
        y=y,
        l=l,
        d=d,
        theta=0.0,
        material=material,
        difficulty=_sample_difficulty(rng, material, l * d),
        oid=DOOR_OID_BASE + int(label[1:]),
    )

def _random_room_obstacles(workspace, walls, door_blockers,
                           existing, start, goal):
    """在各空间中稳定随机生成不同尺寸的障碍物，并形成少量近接障碍物组。"""
    barriers = unary_union(
        [wall.polygon for wall in walls]
        + [door.polygon for door in door_blockers]
    )
    free_space = workspace.difference(barriers)
    spaces = (
        list(free_space.geoms)
        if free_space.geom_type == "MultiPolygon"
        else [free_space]
    )
    # 从上到下、同一高度从左到右处理，使整数编号的位置顺序稳定。
    spaces.sort(key=lambda area: (-area.centroid.y, area.centroid.x))

    rng = random.Random(RNG_SEED)
    occupied = [obs.polygon for obs in existing]
    protected = unary_union([
        Point(start).buffer(1.1),
        Point(goal).buffer(1.1),
    ])
    door_keepout = unary_union(
        [door.polygon for door in door_blockers]
    ).buffer(0.5)
    # 为一条可完成但仍需决策的路线保留定向推动空间：
    # d17 -> d15 -> d7 -> d3。其他门洞周围仍可随机堆放障碍物。
    decision_push_zones = unary_union([
        box(27.0, 5.5, 28.5, 8.45),   # d17 向下推
        box(22.5, 9.5, 25.45, 11.0),  # d15 向左推
        box(7.0, 18.3, 9.0, 21.25),   # d7 向下推
        box(2.0, 23.0, 4.0, 25.95),   # d3 向下推
    ])
    door_keepout = door_keepout.union(decision_push_zones)

    generated = []
    next_oid = 2

    def make_obstacle(oid, x, y, l, d):
        l, d = round(l, 2), round(d, 2)
        material = rng.choice(CLUTTER_MATERIALS)
        return MovableObstacle(
            x=round(x, 3),
            y=round(y, 3),
            l=l,
            d=d,
            theta=0.0,
            material=material,
            difficulty=_sample_difficulty(rng, material, l * d),
            oid=oid,
        )

    def valid_against_map(obstacle, space):
        padded = obstacle.polygon.buffer(0.08)
        return (
            space.contains(padded)
            and not padded.intersects(protected)
            and not padded.intersects(door_keepout)
        )

    def clear_of_existing(obstacle, gap=0.18):
        padded = obstacle.polygon.buffer(gap)
        return not any(padded.intersects(poly) for poly in occupied)

    for room_index, space in enumerate(spaces, 1):
        minx, miny, maxx, maxy = space.bounds
        room_width = maxx - minx
        room_height = maxy - miny
        target_count = max(4, min(10, int(space.area / 18) + 3))
        room_obstacles = []

        # 面积较大的空间先放一组仅相隔 0.04 的障碍物。
        # 视觉上形成连续阻挡，但几何上仍是两个可分别推动的物体。
        if space.area >= 30:
            for _attempt in range(4000):
                horizontal = rng.choice((True, False))
                l1 = rng.uniform(1.3, min(2.8, max(1.3, room_width * 0.4)))
                d1 = rng.uniform(1.1, min(2.4, max(1.1, room_height * 0.35)))
                l2 = rng.uniform(0.7, min(1.6, max(0.7, room_width * 0.3)))
                d2 = rng.uniform(0.7, min(1.6, max(0.7, room_height * 0.3)))
                cluster_gap = 0.12

                if horizontal:
                    group_width = l1 + cluster_gap + l2
                    group_height = max(d1, d2)
                    if group_width + 0.2 >= room_width:
                        continue
                    left = rng.uniform(minx + 0.1, maxx - group_width - 0.1)
                    cy = rng.uniform(
                        miny + group_height / 2 + 0.1,
                        maxy - group_height / 2 - 0.1,
                    )
                    first = make_obstacle(
                        next_oid,
                        left + l1 / 2, cy, l1, d1,
                    )
                    second = make_obstacle(
                        next_oid + 1,
                        left + l1 + cluster_gap + l2 / 2,
                        cy, l2, d2,
                    )
                else:
                    group_width = max(l1, l2)
                    group_height = d1 + cluster_gap + d2
                    if group_height + 0.2 >= room_height:
                        continue
                    cx = rng.uniform(
                        minx + group_width / 2 + 0.1,
                        maxx - group_width / 2 - 0.1,
                    )
                    bottom = rng.uniform(
                        miny + 0.1, maxy - group_height - 0.1
                    )
                    first = make_obstacle(
                        next_oid,
                        cx, bottom + d1 / 2, l1, d1,
                    )
                    second = make_obstacle(
                        next_oid + 1,
                        cx, bottom + d1 + cluster_gap + d2 / 2,
                        l2, d2,
                    )

                if not valid_against_map(first, space):
                    continue
                if not valid_against_map(second, space):
                    continue
                if not clear_of_existing(first):
                    continue
                if not clear_of_existing(second):
                    continue
                if first.polygon.intersects(second.polygon):
                    continue

                generated.extend([first, second])
                room_obstacles.extend([first, second])
                occupied.extend([first.polygon, second.polygon])
                next_oid += 2
                break

        while len(room_obstacles) < target_count:
            for _attempt in range(5000):
                # 每个空间至少尝试一个大件；其余物体保持明显的尺寸差异。
                make_large = (
                    space.area >= 30
                    and (
                        not room_obstacles
                        or (len(room_obstacles) % 5 == 0 and space.area >= 50)
                    )
                )
                if make_large:
                    l = rng.uniform(
                        1.4, min(2.8, max(1.4, room_width * 0.45))
                    )
                    d = rng.uniform(
                        1.2, min(2.4, max(1.2, room_height * 0.45))
                    )
                else:
                    l = rng.uniform(0.45, 1.5)
                    d = rng.uniform(0.45, 1.5)

                x_min = minx + l / 2 + 0.1
                x_max = maxx - l / 2 - 0.1
                y_min = miny + d / 2 + 0.1
                y_max = maxy - d / 2 - 0.1
                if x_min >= x_max or y_min >= y_max:
                    continue

                obstacle = make_obstacle(
                    next_oid,
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    l,
                    d,
                )
                if not valid_against_map(obstacle, space):
                    continue
                if not clear_of_existing(obstacle):
                    continue

                generated.append(obstacle)
                room_obstacles.append(obstacle)
                occupied.append(obstacle.polygon)
                next_oid += 1
                break
            else:
                raise RuntimeError(
                    f"无法在第 {room_index} 个空间中放置足够的随机障碍物"
                )

    return generated


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
        # 顶部横墙在 x=2.0~4.0、x=12.5~14.0 留门。
        StaticObstacle(box(0.0, 25.5, 2.0, 25.5 + t), "upper_left_horizontal_top_far_left"),
        StaticObstacle(box(4.0, 25.5, 12.5, 25.5 + t), "upper_left_horizontal_top_middle"),
        StaticObstacle(box(14.0, 25.5, 20, 25.5 + t), "upper_left_horizontal_top_right"),

        # 底部横墙在红框位置增加 x=7.0~9.0、x=11.0~13.0 两扇门。
        StaticObstacle(box(0.0, 20.8, 3.0, 20.8 + t), "upper_left_horizontal_bottom_left"),
        StaticObstacle(box(4.5, 20.8, 7.0, 20.8 + t), "upper_left_horizontal_bottom_middle_1"),
        StaticObstacle(box(9.0, 20.8, 11.0, 20.8 + t), "upper_left_horizontal_bottom_middle_2"),
        StaticObstacle(box(13.0, 20.8, 17.0, 20.8 + t), "upper_left_horizontal_bottom_middle_3"),
        StaticObstacle(box(18.5, 20.8, 20, 20.8 + t), "upper_left_horizontal_bottom_right"),
        #StaticObstacle(box(5.0, 16.5, 5.0 + t, 20.8), "upper_left_vertical_connector"),

        # 中上竖墙（按当前列表从右向左编号）
        StaticObstacle(box(14.8, 20.8, 14.8 + t, 28), "upper_middle_vertical_1"),
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
        # 左中横墙分别在 x=7.0~8.5 留门。
        StaticObstacle(box(0.0, 12.0, 7.0, 12.0+t), "middle_left_horizontal_top_left"),
        StaticObstacle(box(8.5, 12.0, 10.0, 12.0+t), "middle_left_horizontal_top_right"),
        StaticObstacle(box(0.0, 5.0, 7.0, 5.0+t), "middle_left_horizontal_bottom_left"),
        StaticObstacle(box(8.5, 5.0, 10.0, 5.0+t), "middle_left_horizontal_bottom_right"),

        # 中部大矩形左墙按红框在 y=2.0~3.5、y=8.0~9.5 留门。
        StaticObstacle(box(9.8, 0, 10.25, 2.0), "middle_center_vertical_left_lower"),
        StaticObstacle(box(9.8, 3.5, 10.25, 8.0), "middle_center_vertical_left_middle"),
        StaticObstacle(box(9.8, 9.5, 10.25, 16.7), "middle_center_vertical_left_upper"),
        StaticObstacle(box(9.8, 16.25, 19.8, 16.7), "middle_center_horizontal_top"),

        # 右侧上下房间的横墙均在 x=27.0~28.5 留门。
        StaticObstacle(box(25.0, 12, 27.0, 12.45), "middle_right_horizontal_top_left"),
        StaticObstacle(box(28.5, 12, 30.0, 12.45), "middle_right_horizontal_top_right"),
        StaticObstacle(box(20.0, 8, 27.0, 8.45), "middle_right_horizontal_bottom_left"),
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
    # 横墙门洞：l 等于门宽，d 等于墙厚 t。
    # 竖墙门洞：l 等于墙厚 t，d 等于门高。
    door_rng = random.Random(RNG_SEED + 1)
    door_blockers = [
        # 顶部区域
        _door_blocker(door_rng, "d1", 5.5 + t / 2, 27.75, t, 1.5),
        _door_blocker(door_rng, "d2", 20 + t / 2, 27.5, t, 2.0),
        _door_blocker(door_rng, "d3", 3.0, 25.5 + t / 2, 2.0, t),
        _door_blocker(door_rng, "d4", 13.25, 25.5 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d5", 20 + t / 2, 23.25, t, 1.5),

        # y=20.8 横墙上的六个门洞
        _door_blocker(door_rng, "d6", 3.75, 20.8 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d7", 8.0, 20.8 + t / 2, 2.0, t),
        _door_blocker(door_rng, "d8", 12.0, 20.8 + t / 2, 2.0, t),
        _door_blocker(door_rng, "d9", 17.75, 20.8 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d10", 23.75, 20.8 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d11", 27.75, 20.8 + t / 2, 1.5, t),

        # 中部和右侧区域
        _door_blocker(door_rng, "d12", 25 + t / 2, 16.0, t, 2.0),
        _door_blocker(door_rng, "d13", 7.75, 12.0 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d14", 27.75, 12.0 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d15", 25 + t / 2, 10.25, t, 1.5),
        _door_blocker(door_rng, "d16", 10.025, 8.75, 0.45, 1.5),
        _door_blocker(door_rng, "d17", 27.75, 8.0 + t / 2, 1.5, t),
        _door_blocker(door_rng, "d18", 7.75, 5.0 + t / 2, 1.5, t),

        # 下部区域
        _door_blocker(door_rng, "d19", 20 + t / 2, 3.75, t, 1.5),
        _door_blocker(door_rng, "d20", 5.0 + t / 2, 2.75, t, 1.5),
        _door_blocker(door_rng, "d21", 10.025, 2.75, 0.45, 1.5),
        _door_blocker(door_rng, "d22", 25 + t / 2, 2.75, t, 1.5),
    ]

    start = (28,2)
    goal = (2.5, 28.0)

    first_obstacle = MovableObstacle(
        x=1.5,
        y=13.5,
        l=2,
        d=2,
        theta=0.0,
        material="concrete_block",  # 25.0 x 4.00 = 100.0
        difficulty=100,
        oid=1,
    )
    random_obstacles = _random_room_obstacles(
        workspace,
        walls,
        door_blockers,
        [first_obstacle],
        start,
        goal,
    )
    movable = [
        first_obstacle,
        *random_obstacles,
        *door_blockers,
    ]
    return {
        "name": "maze_to_house",
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": Config(),
    }
