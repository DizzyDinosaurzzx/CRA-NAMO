from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from dynamics import Event, MoveTo, at_time
from obstacle import MovableObstacle, StaticObstacle
from scenarios import _realism
from scenarios._realism import (
    MU_BRAKED_WHEELS, MU_CASTORS, MU_STEEL, MU_WOOD, push_force,
)


def _load(oid: str, x: float, y: float, l: float, d: float, h: float,
          theta: float, material: str, *, mass: float,
          mu: float) -> MovableObstacle:
    """按真实尺寸、质量和地面摩擦系数放置一件仓库货物。"""
    return MovableObstacle(
        x=x, y=y, l=l, d=d, h=h, theta=theta, material=material,
        difficulty=push_force(mass, mu), oid=oid,
    )


def create():
    """Create the 60 m by 40 m warehouse map."""

    # Coordinates use metres with origin at the lower-left corner.
    workspace = box(0.0, 0.0, 60.0, 40.0)
    wall_t = 0.75

    # Build fixed walls and shelves.
    walls = [
        # Outer boundary walls.
        StaticObstacle.rect(
            x=30.0,
            y=wall_t / 2.0,
            l=60.0,
            d=wall_t,
            theta=math.radians(0.0),
            name="outer_bottom",
        ),
        StaticObstacle.rect(
            x=30.0,
            y=40.0 - wall_t / 2.0,
            l=60.0,
            d=wall_t,
            theta=math.radians(0.0),
            name="outer_top",
        ),
        StaticObstacle.rect(
            x=wall_t / 2.0,
            y=20.0,
            l=40.0,
            d=wall_t,
            theta=math.radians(90.0),
            name="outer_left",
        ),
        StaticObstacle.rect(
            x=60.0 - wall_t / 2.0,
            y=20.0,
            l=40.0,
            d=wall_t,
            theta=math.radians(90.0),
            name="outer_right",
        ),
    ]

    # Seven shelf columns leave four horizontal aisles.
    shelf_columns = (
        (8.5, 10.0),    # Column 1.
        (15, 16.50),    # Column 2.
        (21.0, 22.5),   # Column 3.
        (32, 33.50),    # Column 4.
        (38.5, 40),      # Column 5.
        (47.5, 49.0),   # Column 6.
        (54.5, 56.0),   # Column 7; upper section is shipping space.
    )

    # Top shelves occupy columns 1 through 6.
    for column, (x0, x1) in enumerate(shelf_columns[:6], start=1):
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(32.0 + 40) / 2.0,
                l=40 - 32.0,
                d=x1 - x0,
                theta=math.radians(90.0),
                name=f"shelf_{column}_top",
            )
        )

    # Left shelves are split around aisle B.
    for column, (x0, x1) in enumerate(shelf_columns[:4], start=1):
        walls.extend(
            (
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(24.5 + 29.0) / 2.0,
                    l=29.0 - 24.5,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_upper",
                ),
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(16.0 + 21.5) / 2.0,
                    l=21.5 - 16.0,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_middle",
                ),
            )
        )

    # Right shelves span the middle aisles.
    for column, (x0, x1) in enumerate(shelf_columns[4:], start=5):
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(16.0 + 29.0) / 2.0,
                l=29.0 - 16.0,
                d=x1 - x0,
                theta=math.radians(90.0),
                name=f"shelf_{column}_middle_tall",
            )
        )

    # All columns have lower and bottom shelf segments.
    for column, (x0, x1) in enumerate(shelf_columns, start=1):
        walls.extend(
            (
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(9.0 + 13.0) / 2.0,
                    l=13.0 - 9.0,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_lower",
                ),
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(2.5 + 6.5) / 2.0,
                    l=6.5 - 2.5,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_bottom",
                ),
            )
        )

    # Shelf bridges close selected aisle segments.
    shelf_bridges = (
        # Aisle A.
        (1, 29.0, 32.0, "A"),
        (3, 29.0, 32.0, "A"),
        (6, 29.0, 32.0, "A"),

        # Aisle B.
        (2, 21.5, 24.5, "B"),
        (4, 21.5, 24.5, "B"),

        # Aisle C.
        (1, 13.0, 16.0, "C"),
        (2, 13.0, 16.0, "C"),
        (3, 13.0, 16.0, "C"),

        # Aisle D.
        (1, 6.5, 9.0, "D"),
        (3, 6.5, 9.0, "D"),
        (4, 6.5, 9.0, "D"),
        (5, 6.5, 9.0, "D"),
        (7, 6.5, 9.0, "D"),

        # Extend selected bridges to the lower boundary.
        (2, wall_t, 2.5, "floor"),
        (3, wall_t, 2.5, "floor"),
        (5, wall_t, 2.5, "floor"),
        (6, wall_t, 2.5, "floor"),
        (7, wall_t, 2.5, "floor"),
    )
    for column, y0, y1, region in shelf_bridges:
        x0, x1 = shelf_columns[column - 1]
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                l=y1 - y0,
                d=x1 - x0,
                theta=math.radians(90.0),
                name=f"shelf_{column}_bridge_{region}",
            )
        )

    # Small fixed shelves in the shipping area.
    # 发货区右侧的六组小型固定货架，按 (x0, x1, y0, y1) 给出占地范围。
    shipping_shelves = (
        (58.00, 59.25, 28.5, 29.0),
        (56.00, 58.00, 27.0, 27.5),
        (58.00, 59.25, 25.5, 26.0),
        (56.00, 58.00, 24.0, 24.5),
        (58.00, 59.25, 22.0, 22.5),
        (56.00, 58.00, 20.0, 20.5),
    )
    for index, (x0, x1, y0, y1) in enumerate(shipping_shelves, start=1):
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                l=x1 - x0,
                d=y1 - y0,
                theta=math.radians(0.0),
                name=f"shelf_shipping_{index}",
            )
        )

    # Build movable obstacles by map region.
    # ======================================================================
    # 可移动障碍物
    # 尺寸取自真实仓储器具：欧标托盘 1.2 x 0.8 m，笼车 0.8 x 0.72 m，
    # 手动液压车 1.6 x 0.55 m，叉车 2.4 x 1.15 m。阻力由质量和地面摩擦
    # 系数决定：带刹车或直接拖动的货物很重，装在轮子上的很轻。
    # ======================================================================
    movable = [
        # ------------------------------------------------------------------
        # 左侧区域（x < 30）
        # ------------------------------------------------------------------
        # 通道 D 里遗留的一段辊道输送机。
        _load("conveyor101", 18.7, 10.5, 3.6, 0.6, 0.85, 0.5,
              "roller_conveyor", mass=180.0, mu=MU_STEEL),
        # 通道 B 附近缠好膜的整托纸箱。
        _load("wrappedPallet101", 18.0, 25.0, 2.4, 0.85, 1.05, 1.2,
              "wrapped_pallet", mass=380.0, mu=MU_WOOD),
        # 电动搬运车拖着一板货，轮子能滚，所以并不难推。
        _load("palletTruck101", 19.2, 21.0, 2.6, 0.85, 1.2, 0.5,
              "electric_pallet_truck", mass=1_100.0, mu=0.02),
        # 通道 B 中间的 2 x 2 堆垛：1.6 t，几乎总是绕开更划算。
        _load("blockStack101", 12.0, 23.0, 2.4, 2.4, 1.6, 0.0,
              "block_stacked_pallets", mass=1_600.0, mu=MU_WOOD),

        # ------------------------------------------------------------------
        # 中央区域（约 x=30--45）
        # ------------------------------------------------------------------
        # 三节笼车串在一起，脚轮自由。
        _load("rollCage101", 36.0, 23.0, 2.4, 0.75, 1.75, 0.23,
              "roll_cage", mass=620.0, mu=MU_CASTORS),
        # 两块空托盘并排放在通道 C 上。
        _load("pallet101", 32.5, 14.5, 2.4, 1.2, 0.15, 0.0,
              "pallet", mass=55.0, mu=MU_WOOD),
        # 一板满载货物：620 kg 直接在地面上拖。
        _load("loadedPallet101", 39.0, 14.3, 1.2, 0.8, 1.35, 0.8,
              "loaded_pallet", mass=620.0, mu=MU_WOOD),

        # ------------------------------------------------------------------
        # 右侧区域（x > 45）
        # ------------------------------------------------------------------
        _load("cartonBlock201", 52.0, 16.0, 2.4, 1.6, 1.1, 0.0,
              "cardboard_box", mass=480.0, mu=MU_WOOD),
        # 十块空托盘叠成一摞。
        _load("palletStack201", 52.0, 21.0, 1.2, 1.0, 1.45, 0.0,
              "pallet", mass=250.0, mu=MU_WOOD),
        # 吨袋箱（gaylord）连托盘。
        _load("gaylord201", 52.0, 25.0, 1.2, 1.0, 1.15, 0.3,
              "bulk_box", mass=400.0, mu=MU_WOOD),

        # ------------------------------------------------------------------
        # 通道内的零散器具
        # ------------------------------------------------------------------
        # 通道 D 里上下相邻的两个小木箱。
        _load("crate101", 16.0, 7.35, 0.5, 0.5, 0.45, 0.0,
              "wooden_crate", mass=28.0, mu=MU_STEEL),
        _load("crate102", 16.0, 8.05, 0.5, 0.5, 0.45, 0.0,
              "wooden_crate", mass=28.0, mu=MU_STEEL),

        # 通道 A 里停着一台上了驻车制动的叉车：2.5 t，14.7 kN，
        # 相当于要绕行 40 m 才划算，实际上就是一堵能看见的墙。
        _load("forklift101", 31.5, 30.5, 2.4, 1.15, 2.1, 0.7,
              "parked_forklift", mass=2_500.0, mu=MU_BRAKED_WHEELS),

        # 仓库底部两台空的手动液压搬运车。
        _load("handTruck101", 31.0, 1.8, 1.6, 0.55, 1.2, 0.7,
              "hand_pallet_truck", mass=120.0, mu=0.02),
        _load("handTruck102", 33.5, 1.8, 1.6, 0.55, 1.2, 0.5,
              "hand_pallet_truck", mass=120.0, mu=0.02),

        # 通道 C 与 D 里的三台笼车。
        _load("rollCage201", 47.5, 14.6, 0.8, 0.72, 1.75, 0.3,
              "roll_cage", mass=260.0, mu=MU_CASTORS),
        _load("rollCage202", 48.9, 14.6, 0.8, 0.72, 1.75, 0.0,
              "roll_cage", mass=260.0, mu=MU_CASTORS),
        _load("rollCage203", 48.0, 8.0, 0.8, 0.72, 1.75, 0.4,
              "roll_cage", mass=260.0, mu=MU_CASTORS),

        # ------------------------------------------------------------------
        # 由 AGV 载着自主行驶的送货托盘。停下时驻车制动，
        # 所以推它比等它让路贵得多。
        # ------------------------------------------------------------------
        _load("deliveryBox101", 13.5, 28.0, 1.3, 1.1, 1.4, 0.0,
              "agv_pallet", mass=420.0, mu=MU_BRAKED_WHEELS),
    ]

    _AGV_GRID = (
        (24.5, 27.0, "deliveryBox102"), (24.5, 29.0, "deliveryBox103"),
        (24.5, 31.0, "deliveryBox104"), (24.5, 33.0, "deliveryBox105"),
        (28.0, 27.0, "deliveryBox106"), (26.0, 27.0, "deliveryBox107"),
        (28.0, 29.0, "deliveryBox108"), (26.0, 29.0, "deliveryBox109"),
        (28.0, 31.0, "deliveryBox110"), (26.0, 31.0, "deliveryBox111"),
        (28.0, 33.0, "deliveryBox112"), (26.0, 33.0, "deliveryBox113"),
    )
    movable.extend(
        _load(oid, x, y, 1.2, 0.9, 1.35, 0.0, "agv_pallet",
              mass=380.0, mu=MU_BRAKED_WHEELS)
        for x, y, oid in _AGV_GRID
    )
    movable.append(
        _load("deliveryBox114", 57.0, 3.0, 1.2, 0.9, 1.35, 0.0,
              "agv_pallet", mass=380.0, mu=MU_BRAKED_WHEELS))

    events = [
        Event(
            name="post-0 enters the upper delivery aisle",
            trigger=at_time(20.0),
            effect=MoveTo(
                oid="deliveryBox101",
                goal=(18, 35, 0.0),
                speed=0.1,
            ),
        ),
        Event(
            name="post-1 enters the upper delivery aisle",
            trigger=at_time(40.0),
            effect=MoveTo(
                oid="deliveryBox102",
                goal=(24.5, 5, 0.0),
                speed=0.2,
            ),
        ),

        Event(
            name="post-2 enters the upper delivery aisle",
            trigger=at_time(50.0),
            effect=MoveTo(
                oid="deliveryBox106",
                goal=(28, 5, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3 enters the upper delivery aisle",
            trigger=at_time(60.0),
            effect=MoveTo(
                oid="deliveryBox107",
                goal=(26, 5, 0.0),
                speed=0.3,
            ),
        ),

        Event(
            name="post-1-2 enters the upper delivery aisle",
            trigger=at_time(45.0),
            effect=MoveTo(
                oid="deliveryBox103",
                goal=(24.5, 7, 0.0),
                speed=0.3,
            ),
        ),

        Event(
            name="post-2-2 enters the upper delivery aisle",
            trigger=at_time(55.0),
            effect=MoveTo(
                oid="deliveryBox108",
                goal=(28, 7, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3-2 enters the upper delivery aisle",
            trigger=at_time(65.0),
            effect=MoveTo(
                oid="deliveryBox109",
                goal=(26, 7, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-1-3 enters the upper delivery aisle",
            trigger=at_time(50.0),
            effect=MoveTo(
                oid="deliveryBox104",
                goal=(24.5, 9, 0.0),
                speed=0.3,
            ),
        ),


        Event(
            name="post-2-3 enters the upper delivery aisle",
            trigger=at_time(60.0),
            effect=MoveTo(
                oid="deliveryBox110",
                goal=(28, 9, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3-3 enters the upper delivery aisle",
            trigger=at_time(70.0),
            effect=MoveTo(
                oid="deliveryBox111",
                goal=(26, 9, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-1-4 enters the upper delivery aisle",
            trigger=at_time(55.0),
            effect=MoveTo(
                oid="deliveryBox105",
                goal=(24.5, 11, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-2-4 enters the upper delivery aisle",
            trigger=at_time(65.0),
            effect=MoveTo(
                oid="deliveryBox112",
                goal=(28, 11, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3-4 enters the upper delivery aisle",
            trigger=at_time(75.0),
            effect=MoveTo(
                oid="deliveryBox113",
                goal=(26, 11, 0.0),
                speed=0.3,
            ),
        ),

        
        Event(
            name="post-4 enters the upper delivery aisle",
            trigger=at_time(110.0),
            effect=MoveTo(
                oid="deliveryBox114",
                goal=(57, 15, 0.5),
                speed=0.3,
            ),
        ),
    ]

   


    # Start in receiving and finish in shipping.
    start = (5.0, 1.25)
    goal = (56.0, 33.0)

    cfg = Config(
        grid_step=0.5,
        conn_radius=0.75,
        se2_cell=0.5,
    )
    _realism.check_layout(
        "warehouse", workspace=workspace, static=walls, movable=movable,
        start=start, goal=goal, cfg=cfg,
    )

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "dynamics": events,
        "cfg": cfg,
    }
