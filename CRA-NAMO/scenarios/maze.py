"""Self-storage facility: one maze, and stock that only looks alike."""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle
from scenarios import _realism
from scenarios._realism import MU_STEEL, MU_WOOD, push_force

DOOR_OID_BASE = 900

_WALL_T = 0.45
# Trim the long side so blockers fit between wall stubs.
DOOR_CLEARANCE = 0.2

# Stock kinds: material label, height, packed density [kg/m^3], floor grip.
# `carton` and `books` are deliberately the same label at the same size. A
# metre cube of bedding weighs 45 kg and a metre cube of books weighs 420, and
# nothing the robot can see from the aisle tells the two apart -- only the
# force it measures once it is already pushing.
_STORED_KINDS = {
    "carton":     ("storage_carton",  1.10,  45.0, MU_WOOD),
    "books":      ("storage_carton",  1.10, 420.0, MU_WOOD),
    "crate":      ("wooden_crate",    1.00, 160.0, MU_STEEL),
    "files":      ("filing_cabinet",  1.32, 124.0, MU_STEEL),
    "shelving":   ("steel_shelf",     1.95, 111.0, MU_STEEL),
    "drums":      ("steel_drum",      0.90, 350.0, MU_STEEL),
    "tyres":      ("tyre_stack",      1.10, 130.0, 0.55),
    "whitegoods": ("white_goods",     0.90, 150.0, MU_STEEL),
    "safe":       ("steel_safe",      1.45, 731.0, MU_STEEL),
}

# What contact reveals about a unit that looked like any other.
_STORED_REVEALS = {
    "books": "cartons_of_books",
    "drums": "sealed_steel_drums",
}

# Doorway rows: label, centre, opening dimensions, stock kind.
_DOORWAYS = (
    ("d1",  5.5 + _WALL_T / 2, 27.75, _WALL_T, 1.5, "carton"),
    ("d2",  20 + _WALL_T / 2,  27.5,  _WALL_T, 2.0, "books"),
    ("d3",  3.0,  25.5 + _WALL_T / 2, 2.0, _WALL_T, "crate"),
    ("d4",  13.25, 25.5 + _WALL_T / 2, 1.5, _WALL_T, "carton"),
    ("d5",  20 + _WALL_T / 2, 23.25, _WALL_T, 1.5, "files"),

    ("d6",  3.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "carton"),
    ("d7",  8.0,  20.8 + _WALL_T / 2, 2.0, _WALL_T, "books"),
    ("d8",  12.0, 20.8 + _WALL_T / 2, 2.0, _WALL_T, "crate"),
    ("d9",  17.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "carton"),
    ("d10", 23.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "tyres"),
    ("d11", 27.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "carton"),

    ("d12", 25 + _WALL_T / 2, 16.0, _WALL_T, 2.0, "shelving"),
    ("d13", 7.75, 12.0 + _WALL_T / 2, 1.5, _WALL_T, "files"),
    ("d14", 27.75, 12.0 + _WALL_T / 2, 1.5, _WALL_T, "carton"),
    ("d15", 25 + _WALL_T / 2, 10.25, _WALL_T, 1.5, "crate"),
    ("d16", 10.025, 8.75, 0.45, 1.5, "books"),
    ("d17", 27.75, 8.0 + _WALL_T / 2, 1.5, _WALL_T, "carton"),
    ("d18", 7.75, 5.0 + _WALL_T / 2, 1.5, _WALL_T, "drums"),

    ("d19", 20 + _WALL_T / 2, 3.75, _WALL_T, 1.5, "carton"),
    ("d20", 5.0 + _WALL_T / 2, 2.75, _WALL_T, 1.5, "whitegoods"),
    ("d21", 10.025, 2.75, 0.45, 1.5, "carton"),
    # The one door not worth clearing: a 700 kg floor safe left in the frame.
    ("d22", 25 + _WALL_T / 2, 2.75, _WALL_T, 1.5, "safe"),

    ("d23", 2.5,  12.0 + _WALL_T / 2, 2.0, _WALL_T, "crate"),
    ("d24", 15.5, 16.25 + _WALL_T / 2, 2.0, _WALL_T, "books"),
    ("d25", 23.0, 8.0 + _WALL_T / 2, 2.0, _WALL_T, "shelving"),
    ("d26", 2.5,  5.0 + _WALL_T / 2, 2.0, _WALL_T, "carton"),

    ("d27", 17.5, 25.5 + _WALL_T / 2, 2.0, _WALL_T, "crate"),
    ("d28", 14.8 + _WALL_T / 2, 23.5, _WALL_T, 2.0, "carton"),
)


def _door_blocker(label: str, x: float, y: float, l: float, d: float,
                  kind: str) -> MovableObstacle:
    """Stand one unit of stored goods in a doorway, trimmed to fit the frame."""
    material, h, density, mu = _STORED_KINDS[kind]
    if l > d:
        l -= DOOR_CLEARANCE
    else:
        d -= DOOR_CLEARANCE
    return MovableObstacle(
        x=x,
        y=y,
        l=l,
        d=d,
        h=h,
        theta=0.0,
        material=material,
        difficulty=push_force(density * l * d * h, mu),
        contact_reveals=_STORED_REVEALS.get(kind, ""),
        oid=DOOR_OID_BASE + int(label[1:]),
    )


def _stored(oid: int, x: float, y: float, l: float, d: float, h: float,
            theta: float, material: str, *, density: float,
            mu: float) -> MovableObstacle:
    """Place stored goods whose mass follows from their bulk and their volume."""
    return MovableObstacle(
        x=x, y=y, l=l, d=d, h=h, theta=theta, material=material,
        difficulty=push_force(density * l * d * h, mu), oid=oid,
    )


def create():
    """Create the complex maze scenario."""
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

    door_blockers = [_door_blocker(*spec) for spec in _DOORWAYS]

    start = (28, 2)
    goal = (2,27)

    # Stored goods, unit by unit. Footprints are the ones the map was built
    # around; what fills them is bulk density and floor grip, so a 2 x 2 m
    # concrete block is 9.6 t and a 2 x 2 m stack of bedding cartons is 200 kg.
    stored = (
        # oid,  x,     y,    l,    d,    h,   theta, material, density, mu
        (1,   1.5,  13.5,  2.0,  2.0, 1.00,  0.00, "concrete_block",  2400.0, 0.60),
        (2,  12.8,  11.0,  4.6,  6.6, 1.60,  0.00, "palletised_stock", 250.0, 0.40),
        (3,  17.7,  11.5,  2.0,  3.8, 1.20,  0.00, "palletised_stock", 250.0, 0.40),
        (4,  12.0,   6.65, 1.6,  1.2, 1.00,  0.00, "wooden_crate",     160.0, 0.45),
        (5,  12.0,   3.25, 1.5,  2.0, 1.00,  0.00, "wooden_crate",     160.0, 0.45),
        (6,  13.8,   5.2,  0.8,  1.0, 1.00,  0.00, "storage_carton",    45.0, 0.35),
        # 7 and 8 are the same cartons at the same size 1.3 m apart. One holds
        # bedding, the other holds books, and only pushing tells them apart.
        (7,  18.9,   4.7,  1.2,  1.3, 1.10,  0.00, "storage_carton",    45.0, 0.35),
        (8,  18.9,   3.4,  1.2,  1.3, 1.10,  0.00, "storage_carton",   420.0, 0.35),
        (10, 27.75,  6.7,  1.4,  1.2, 0.90,  0.00, "white_goods",      150.0, 0.45),
        (11, 13.05, 27.7,  1.2,  1.2, 1.10,  0.00, "storage_carton",    45.0, 0.35),
        (12, 23.75, 27.75, 4.5,  3.5, 1.00,  0.00, "palletised_stock", 250.0, 0.40),
        (13, 10.25, 24.3,  4.5,  2.1, 1.00,  0.00, "palletised_stock", 250.0, 0.40),
        (14, 23.0,  23.05, 1.1,  3.0, 1.95,  0.00, "steel_shelf",      111.0, 0.45),
        (16, 14.0,  18.7,  2.5,  2.3, 1.00,  0.00, "palletised_stock", 250.0, 0.40),
        (17, 28.0,  16.75, 1.8,  2.8, 1.00,  0.00, "wooden_crate",     160.0, 0.45),
        (18,  3.8,  13.5,  1.8,  1.6, 1.10,  0.00, "storage_carton",    45.0, 0.35),
        (19,  6.6,   7.4,  5.0,  1.4, 1.00,  0.53, "racking_rails",     87.0, 0.45),
        (20,  2.6,   3.1,  1.6,  1.5, 0.90,  0.00, "steel_drum",       350.0, 0.45),
        (21,  6.9,   1.5,  2.2,  1.8, 1.00,  0.00, "wooden_crate",     160.0, 0.45),
        (22, 17.1,  27.5,  3.0,  0.75, 1.00, 0.92, "racking_rails",     87.0, 0.45),
        (23,  6.4,  23.7,  4.4,  0.75, 1.00, -0.65, "racking_rails",    87.0, 0.45),
        (24, 17.2,  22.8,  2.7,  0.85, 1.00, -0.82, "racking_rails",    87.0, 0.45),
        (25,  7.5,  14.6,  3.8,  0.8, 1.95, -0.91, "steel_shelf",      111.0, 0.45),
        (26, 22.5,  12.3,  8.25, 0.7, 1.00,  1.07, "racking_rails",     87.0, 0.45),
        (27,  3.8,   9.5,  4.3,  0.75, 1.00, -1.05, "racking_rails",    87.0, 0.45),
        (28, 23.0,   4.0,  3.8,  0.9, 1.00, -0.70, "racking_rails",     87.0, 0.45),
        (29, 15.0,   2.0,  4.6,  0.75, 1.00, -0.45, "racking_rails",    87.0, 0.45),
        # The eight cartons in the far corner: two of them hold books.
        (30, 28.1,  23.2,  0.5,  0.5, 0.50,  0.00, "storage_carton",    45.0, 0.35),
        (31, 26.2,  23.2,  0.5,  0.5, 0.50,  0.17, "storage_carton",    45.0, 0.35),
        (32, 27.0,  23.2,  0.5,  0.5, 0.50,  0.35, "storage_carton",    45.0, 0.35),
        (33, 26.5,  22.0,  0.5,  0.5, 0.50,  0.52, "storage_carton",   420.0, 0.35),
        (34, 27.3,  22.2,  0.5,  0.5, 0.50,  0.70, "storage_carton",    45.0, 0.35),
        (35, 27.9,  22.0,  0.5,  0.5, 0.50,  0.87, "storage_carton",    45.0, 0.35),
        (36, 27.0,  24.0,  0.5,  0.5, 0.50,  1.05, "storage_carton",   420.0, 0.35),
        (37, 28.0,  24.0,  0.5,  0.5, 0.50,  1.22, "storage_carton",    45.0, 0.35),
    )
    manual_obstacles = [
        _stored(oid, x, y, l, d, h, theta, material, density=density, mu=mu)
        for oid, x, y, l, d, h, theta, material, density, mu in stored
    ]

    movable = [
        *manual_obstacles,
        *door_blockers,
    ]

    cfg = Config()
    _realism.check_layout(
        "maze", workspace=workspace, static=walls, movable=movable,
        start=start, goal=goal, cfg=cfg,
    )

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": cfg,
    }
