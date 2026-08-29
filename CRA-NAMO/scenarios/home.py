"""House scenario with heterogeneous furniture blocking interior doorways."""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle
from scenarios import _realism
from scenarios._realism import (
    MU_FELT_PADS, MU_STEEL, MU_UPHOLSTERY, MU_WOOD, push_force,
)

DOOR_OID_BASE = 900

_WALL_T = 0.45
# Trim the long side so blockers fit between wall stubs.
DOOR_CLEARANCE = 0.2

# Doorway furniture: material, height, packed density [kg/m^3], floor grip.
# Density rather than mass, so the same piece is heavier in a 2.0 m opening
# than in a 1.5 m one. Depth is the doorway's, so these are the slim cases:
# a flat-pack wardrobe carcass, a table on its side, a 0.45 m bookcase.
_DOOR_KINDS = {
    "drawers":   ("chest_of_drawers",     0.82,   121.0, MU_FELT_PADS),
    "wardrobe":  ("flat_packed_wardrobe", 2.02,    56.0, MU_WOOD),
    "cartons":   ("cardboard_box",        1.20,    43.0, MU_WOOD),
    "bookcase":  ("loaded_bookcase",      1.85,    97.0, MU_WOOD),
    "mattress":  ("rolled_mattress",      0.62,    94.0, MU_UPHOLSTERY),
    "sideboard": ("sideboard",            0.86,   147.0, MU_FELT_PADS),
    "table":     ("dining_table",         0.76,    75.0, MU_WOOD),
    "files":     ("filing_cabinet",       1.32,   124.0, MU_STEEL),
    "shelving":  ("steel_shelf",          1.95,   111.0, MU_STEEL),
    "safe":      ("steel_safe",           1.45,   731.0, MU_STEEL),
}

# Doorway rows contain label, centre, opening dimensions, and furniture kind.
_DOORWAYS = (
    ("d1",  5.5 + _WALL_T / 2, 27.75, _WALL_T, 1.5, "mattress"),
    ("d2",  20 + _WALL_T / 2,  27.5,  _WALL_T, 2.0, "wardrobe"),
    ("d3",  3.0,  25.5 + _WALL_T / 2, 2.0, _WALL_T, "drawers"),
    ("d4",  13.25, 25.5 + _WALL_T / 2, 1.5, _WALL_T, "bookcase"),
    ("d5",  20 + _WALL_T / 2, 23.25, _WALL_T, 1.5, "cartons"),

    ("d6",  3.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "drawers"),
    ("d7",  8.0,  20.8 + _WALL_T / 2, 2.0, _WALL_T, "sideboard"),
    ("d8",  12.0, 20.8 + _WALL_T / 2, 2.0, _WALL_T, "table"),
    ("d9",  17.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "mattress"),
    ("d10", 23.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "cartons"),
    ("d11", 27.75, 20.8 + _WALL_T / 2, 1.5, _WALL_T, "bookcase"),

    ("d12", 25 + _WALL_T / 2, 16.0, _WALL_T, 2.0, "wardrobe"),
    ("d13", 7.75, 12.0 + _WALL_T / 2, 1.5, _WALL_T, "files"),
    ("d14", 27.75, 12.0 + _WALL_T / 2, 1.5, _WALL_T, "cartons"),
    ("d15", 25 + _WALL_T / 2, 10.25, _WALL_T, 1.5, "drawers"),
    ("d16", 10.025, 8.75, 0.45, 1.5, "bookcase"),
    ("d17", 27.75, 8.0 + _WALL_T / 2, 1.5, _WALL_T, "cartons"),
    ("d18", 7.75, 5.0 + _WALL_T / 2, 1.5, _WALL_T, "sideboard"),

    ("d19", 20 + _WALL_T / 2, 3.75, _WALL_T, 1.5, "drawers"),
    ("d20", 5.0 + _WALL_T / 2, 2.75, _WALL_T, 1.5, "mattress"),
    ("d21", 10.025, 2.75, 0.45, 1.5, "cartons"),
    # The floor safe is intentionally impractical to move.
    ("d22", 25 + _WALL_T / 2, 2.75, _WALL_T, 1.5, "safe"),

    ("d23", 2.5,  12.0 + _WALL_T / 2, 2.0, _WALL_T, "bookcase"),
    ("d24", 15.5, 16.25 + _WALL_T / 2, 2.0, _WALL_T, "table"),
    ("d25", 23.0, 8.0 + _WALL_T / 2, 2.0, _WALL_T, "shelving"),
    ("d26", 2.5,  5.0 + _WALL_T / 2, 2.0, _WALL_T, "sideboard"),

    ("d27", 17.5, 25.5 + _WALL_T / 2, 2.0, _WALL_T, "wardrobe"),
    ("d28", 14.8 + _WALL_T / 2, 23.5, _WALL_T, 2.0, "cartons"),
)


def _door_blocker(label: str, x: float, y: float, l: float, d: float,
                  kind: str) -> MovableObstacle:
    """Stand one piece of furniture in a doorway, trimmed to fit the frame."""
    material, h, density, mu = _DOOR_KINDS[kind]
    if l > d:
        l -= DOOR_CLEARANCE
    else:
        d -= DOOR_CLEARANCE
    mass = density * l * d * h
    return MovableObstacle(
        x=x,
        y=y,
        l=l,
        d=d,
        h=h,
        theta=0.0,
        material=material,
        difficulty=push_force(mass, mu),
        oid=f"door{DOOR_OID_BASE + int(label[1:])}",
    )


def _furniture(oid: str, x: float, y: float, l: float, d: float, h: float,
               theta: float, material: str, *, mass: float,
               mu: float) -> MovableObstacle:
    """Place one piece of furniture from its real size, mass and floor grip."""
    return MovableObstacle(
        x=x, y=y, l=l, d=d, h=h, theta=theta, material=material,
        difficulty=push_force(mass, mu), oid=oid,
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

    # Furniture is grouped by room; mass and friction determine difficulty.
    manual_obstacles = [
        # Home office: loaded steel shelving.
        _furniture("homeOffice1", 1.5, 13.5, 2.0, 1.9, 2.0, 0.0,
                   "steel_shelf", mass=640.0, mu=MU_STEEL),
        _furniture("homeOffice2", 3.8, 13.5, 1.8, 0.9, 0.75, 0.0,
                   "desk", mass=65.0, mu=MU_WOOD),
        # Angled loaded shelving unit.
        _furniture("homeOffice3", 7.5, 14.6, 2.0, 0.42, 1.95, -0.91,
                   "loaded_bookcase", mass=130.0, mu=MU_WOOD),

        # Living room: sectional sofa.
        _furniture("livingRoom1", 12.8, 11.0, 3.3, 2.6, 0.85, 0.0,
                   "sofa", mass=155.0, mu=MU_UPHOLSTERY),
        _furniture("livingRoom2", 17.7, 11.5, 2.4, 0.55, 2.0, 0.0,
                   "media_wall_unit", mass=120.0, mu=MU_WOOD),

        # Kitchen: island on locked castors.
        _furniture("kitchen1", 12.0, 6.65, 1.6, 0.9, 0.92, 0.0,
                   "kitchen_island", mass=110.0, mu=MU_FELT_PADS),
        _furniture("kitchen2", 12.0, 3.25, 0.92, 0.75, 1.78, 0.0,
                   "fridge_freezer", mass=130.0, mu=MU_WOOD),
        _furniture("kitchen3", 13.8, 5.2, 0.6, 0.6, 0.85, 0.0,
                   "dishwasher", mass=48.0, mu=MU_WOOD),
        _furniture("kitchen4", 18.9, 4.7, 0.9, 0.65, 0.9, 0.0,
                   "range_cooker", mass=85.0, mu=MU_STEEL),
        _furniture("kitchen5", 18.9, 3.4, 1.1, 0.65, 0.85, 0.0,
                   "chest_freezer", mass=95.0, mu=MU_WOOD),
        # Unbolted worktop run.
        _furniture("kitchen6", 15.0, 2.0, 2.6, 0.65, 0.9, -0.45,
                   "counter_run", mass=90.0, mu=MU_WOOD),

        # Dining room.
        _furniture("diningRoom1", 6.6, 7.4, 2.4, 1.1, 0.76, 0.53,
                   "dining_table", mass=75.0, mu=MU_FELT_PADS),
        _furniture("diningRoom2", 3.8, 9.5, 1.8, 0.5, 0.9, -1.05,
                   "sideboard", mass=70.0, mu=MU_FELT_PADS),

        # Utility rooms.
        _furniture("powderRoom1", 2.6, 3.1, 1.0, 0.55, 0.85, 0.0,
                   "vanity_unit", mass=48.0, mu=MU_FELT_PADS),
        # Plumbed washer and dryer.
        _furniture("laundryRoom1", 6.9, 1.5, 1.24, 0.65, 0.9, 0.0,
                   "washing_machine", mass=145.0, mu=MU_STEEL),

        # Entry and storage.
        _furniture("entryway1", 27.75, 6.7, 1.2, 0.8, 1.1, 0.0,
                   "cardboard_box", mass=42.0, mu=MU_WOOD),
        _furniture("storageRoom1", 23.0, 4.0, 1.8, 0.5, 1.95, -0.7,
                   "steel_shelf", mass=180.0, mu=MU_STEEL),
        # Rolled hall carpet.
        _furniture("hallway1", 22.5, 12.3, 3.2, 0.45, 0.45, 1.07,
                   "rolled_carpet", mass=45.0, mu=MU_UPHOLSTERY),

        # Family room and bathrooms.
        _furniture("familyRoom1", 14.0, 18.7, 2.2, 0.95, 0.85, 0.0,
                   "sofa", mass=85.0, mu=MU_UPHOLSTERY),
        # Empty cast-iron bathtub.
        _furniture("bathroom1", 28.0, 16.75, 1.75, 0.8, 0.6, 0.0,
                   "cast_iron_bathtub", mass=130.0, mu=MU_STEEL),
        _furniture("ensuiteBathroom1", 17.2, 22.8, 1.2, 0.6, 0.85, -0.82,
                   "vanity_unit", mass=58.0, mu=MU_FELT_PADS),

        # Bedrooms.
        _furniture("masterBedroom1", 10.25, 24.3, 2.15, 2.0, 0.6, 0.0,
                   "king_bed", mass=145.0, mu=MU_WOOD),
        # Empty bookcase tipped onto its back.
        _furniture("masterBedroom2", 6.4, 23.7, 2.4, 1.0, 0.6, -0.65,
                   "flat_packed_wardrobe", mass=95.0, mu=MU_WOOD),
        _furniture("guestBedroom1", 13.05, 27.7, 1.2, 0.55, 1.1, 0.0,
                   "chest_of_drawers", mass=62.0, mu=MU_FELT_PADS),
        _furniture("guestBedroom2", 17.1, 27.5, 2.0, 0.95, 0.55, 0.92,
                   "single_bed", mass=68.0, mu=MU_WOOD),
        _furniture("kidsBedroom1", 23.75, 27.75, 2.05, 1.45, 1.7, 0.0,
                   "bunk_bed", mass=110.0, mu=MU_WOOD),
        _furniture("kidsBedroom2", 23.0, 23.05, 1.1, 0.6, 2.05, 0.0,
                   "flat_packed_wardrobe", mass=88.0, mu=MU_WOOD),
    ]

    # Children's room contains a dense cluster of light cartons.
    toy_cartons = (
        (28.1, 23.2, 0.00), (26.2, 23.2, 0.17), (27.0, 23.2, 0.35),
        (26.5, 22.0, 0.52), (27.3, 22.2, 0.70), (27.9, 22.0, 0.87),
        (27.0, 24.0, 1.05), (28.0, 24.0, 1.22),
    )
    manual_obstacles.extend(
        _furniture(f"kidsBedroom{3 + index}", x, y, 0.5, 0.5, 0.45, theta,
                   "cardboard_box", mass=9.0, mu=MU_WOOD)
        for index, (x, y, theta) in enumerate(toy_cartons)
    )

    movable = [
        *manual_obstacles,
        *door_blockers,
    ]

    cfg = Config()
    _realism.check_layout(
        "home", workspace=workspace, static=walls, movable=movable,
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
