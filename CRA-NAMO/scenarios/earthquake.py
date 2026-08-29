"""Earthquake-rescue map with three coupled obstacle decisions."""

from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle
from scenarios import _realism
from scenarios._realism import (
    MU_CASTORS, MU_CASTORS_FOULED, MU_CONCRETE, MU_RUBBER_WHEELS,
    MU_STEEL, MU_UPHOLSTERY, MU_WOOD, push_force,
)


_WIDTH = 24.0
_HEIGHT = 20.0
_WALL_T = 0.32

_START = (1.35, 11.25)
_GOAL = (18.0, 16.0)
# The pocket the survivor is in. It has to contain the goal; it did not.
_SURVIVOR_REGION = box(17.3, 15.2, 18.7, 16.8)


def _wall(p, q, name: str) -> StaticObstacle:
    return StaticObstacle.segment(p, q, _WALL_T, name)


def _fixed(x: float, y: float, l: float, d: float,
           angle_deg: float, name: str) -> StaticObstacle:
    return StaticObstacle.rect(
        x=x,
        y=y,
        l=l,
        d=d,
        theta=math.radians(angle_deg),
        name=name,
    )


def _movable(x: float, y: float, l: float, d: float, h: float,
             angle_deg: float, material: str, oid: int,
             *, mass: float, mu: float, difficulty: float | None = None,
             contact_reveals: str = "") -> MovableObstacle:
    """Place an obstacle from mass and floor friction, with an optional override."""
    if difficulty is None:
        difficulty = push_force(mass, mu)
    return MovableObstacle(
        x=x,
        y=y,
        l=l,
        d=d,
        h=h,
        theta=math.radians(angle_deg),
        material=material,
        difficulty=round(difficulty, 3),
        contact_reveals=contact_reveals,
        oid=oid,
    )


def _couple(primary: MovableObstacle, partners, description: str) -> None:
    """Attach the explicit obstacle-interaction graph to a risky blocker."""
    primary.interacts_with = tuple(obs.oid for obs in partners)
    primary.interaction_risk = description


def _walls():
    """Dark structural strokes, including the earthquake-damaged fragments."""
    segments = (
        # Broken upper boundary.
        ((0.00, 20.00), (3.05, 20.00), "north_01"),
        ((4.00, 20.00), (7.00, 20.00), "north_02"),
        ((8.00, 20.00), (12.20, 20.00), "north_03"),
        ((13.20, 20.00), (21.00, 20.00), "north_04"),
        ((22.00, 20.00), (24.00, 20.00), "north_05"),

        # Room 1.
        ((0.00, 20.00), (0.00, 15.50), "room1_west"),
        ((0.00, 15.50), (1.65, 15.50), "room1_south_01"),
        ((2.70, 15.50), (5.00, 15.50), "room1_south_02"),
        ((1.85, 16.15), (2.70, 15.50), "room1_fallen_door"),
        ((5.00, 15.50), (5.00, 20.00), "room1_east"),

        # Room 2.
        ((9.35, 20.00), (9.35, 17.25), "room2_east"),
        ((5.00, 16.75), (8.20, 16.75), "room2_south"),

        # Room 3 and its broken lower wall.
        ((13.85, 20.00), (13.85, 16.35), "room3_east"),
        ((9.35, 16.35), (11.35, 15.55), "room3_south_rubble_01"),
        ((11.35, 15.55), (11.50, 16.25), "room3_south_rubble_02"),
        ((12.65, 16.30), (15.55, 16.30), "room3_south_rubble_03"),

        # Rooms 4 and 5.
        ((20.00, 20.00), (20.00, 16.25), "room4_east"),
        ((20.00, 16.25), (20.35, 15.55), "room5_west_rubble"),
        ((20.35, 15.55), (24.00, 15.55), "room5_south"),
        ((22.20, 19.25), (23.25, 15.55), "room5_fallen_wall"),

        # Right and lower outer boundary.
        ((24.00, 20.00), (24.00, 0.00), "outer_east"),
        ((12.20, 0.00), (24.00, 0.00), "outer_south"),
        ((12.20, 0.00), (12.20, 1.55), "south_step"),
        ((7.60, 1.55), (12.20, 1.55), "southwest_02"),
        ((1.00, 1.55), (6.50, 1.55), "southwest_01"),

        # West boundary and entrance.
        ((1.00, 15.50), (1.00, 14.20), "west_01"),
        ((1.00, 12.80), (1.00, 11.65), "west_02"),
        ((1.00, 10.30), (1.00, 1.55), "west_03"),

        # Room 8 and Bathroom 2.
        ((20.40, 15.55), (20.40, 14.05), "room8_west_01"),
        ((19.80, 12.75), (20.50, 11.20), "room8_west_02"),
        ((20.50, 11.20), (24.00, 11.20), "room8_south"),
        ((20.50, 11.20), (20.50, 6.55), "bathroom2_west"),
        ((20.50, 6.55), (21.05, 6.55), "bathroom2_south_01"),
        ((21.85, 6.55), (24.00, 6.55), "bathroom2_south_02"),
        ((23.55, 11.20), (23.55, 8.95), "bathroom2_divider_01"),
        ((23.55, 8.10), (23.55, 6.55), "bathroom2_divider_02"),

        # Three damaged dividers form the decision bands.
        ((6.50, 10.00), (6.459, 10.55), "decision_a_01"),
        ((6.385, 11.55), (6.333, 12.25), "decision_a_02"),
        ((6.252, 13.35), (6.156, 14.65), "decision_a_03"),
        ((6.074, 15.75), (6.00, 16.75), "decision_a_04"),

        ((10.65, 10.00), (10.537, 10.55), "decision_b_01"),
        ((10.333, 11.55), (10.179, 12.30), "decision_b_02"),
        ((9.974, 13.40), (9.698, 14.65), "decision_b_03"),
        ((9.473, 15.75), (9.35, 16.35), "decision_b_04"),

        ((15.30, 10.00), (15.322, 10.55), "decision_c_01"),
        ((15.362, 11.55), (15.415, 12.90), "decision_c_02"),
        ((15.459, 14.00), (15.509, 15.25), "decision_c_03"),
        ((15.547, 16.20), (15.55, 16.30), "decision_c_04"),

        # Secondary rubble preserves scene irregularity.
        ((2.60, 14.90), (3.55, 14.15), "hall1_secondary_rubble_01"),
        ((11.55, 11.10), (12.55, 10.35), "hall1_secondary_rubble_02"),
        ((17.30, 14.70), (20.00, 16.25), "hall1_secondary_rubble_03"),
        ((15.30, 10.00), (18.60, 6.05), "hall2_rubble_01"),
        ((21.90, 0.15), (23.70, 4.65), "hall2_rubble_02"),

        # Bathroom 1 and Room 9.
        ((1.00, 10.00), (4.10, 10.00), "bathroom1_north"),
        ((4.10, 10.00), (4.45, 8.90), "bathroom1_northeast"),
        ((4.00, 6.55), (4.35, 7.55), "bathroom1_southeast"),
        ((1.00, 6.55), (4.00, 6.55), "bathroom1_south"),
        ((4.75, 5.45), (5.55, 4.10), "room9_northeast"),
        ((5.20, 4.35), (5.20, 1.55), "room9_east"),

        # Rooms 6 and 7.
        ((6.50, 10.00), (15.30, 10.00), "rooms6_7_north"),
        ((6.50, 6.20), (6.50, 10.00), "room6_west"),
        ((10.65, 6.20), (10.65, 10.00), "rooms6_7_divider"),
        ((15.30, 6.20), (15.30, 10.00), "room7_east"),
        ((6.50, 6.20), (8.90, 6.20), "rooms6_7_south_01"),
        ((9.80, 6.20), (12.30, 6.20), "rooms6_7_south_02"),
        ((13.30, 6.20), (15.30, 6.20), "rooms6_7_south_03"),

        # Damaged wall between Room 9 and Hall 2.
        ((8.95, 1.55), (10.25, 4.25), "hall2_west_rubble"),
    )
    return [_wall(p, q, name) for p, q, name in segments]


def _fixed_obstacles():
    """Pale-grey fixed obstacles and the two fixed room fixtures."""
    specs = (
        (0.82, 18.97, 0.65, 0.85, 90.0, "fixed_room1_01"),
        (1.07, 17.29, 0.67, 1.00, 90.0, "fixed_room1_02"),
        (6.14, 19.07, 0.67, 0.60, 0.0, "fixed_room2_01"),
        (8.38, 18.35, 0.87, 0.60, 0.0, "fixed_room2_02"),
        (10.11, 17.38, 1.20, 0.70, 90.0, "fixed_room3_01"),
        (15.30, 18.96, 0.69, 0.60, 0.0, "fixed_room4_01"),
        (17.36, 18.96, 0.45, 0.60, 90.0, "fixed_room4_02"),
        (17.73, 17.41, 1.40, 0.74, 0.0, "fixed_room4_03"),
        (20.91, 16.46, 0.93, 0.57, 90.0, "fixed_room5_01"),
        (3.10, 13.55, 0.75, 0.45, 74.0, "fixed_hall1_01"),
        (8.20, 14.70, 0.70, 0.55, -12.0, "fixed_hall1_02"),
        (12.75, 11.90, 1.10, 0.45, 18.0, "fixed_hall1_03"),
        (13.10, 15.30, 0.82, 0.42, 72.0, "fixed_hall1_04"),
        (20.82, 12.56, 0.88, 0.62, 90.0, "fixed_room8_01"),
        (21.06, 10.04, 0.93, 0.50, 90.0, "fixed_bathroom2_01"),
        (20.98, 8.43, 0.48, 0.33, 90.0, "fixed_bathroom2_02"),
        (7.47, 9.09, 0.67, 0.60, 0.0, "fixed_room6_01"),
        (14.44, 9.13, 0.67, 0.50, 0.0, "fixed_room7_01"),
        (14.18, 7.24, 1.20, 0.65, 90.0, "fixed_room7_02"),
        (1.69, 9.05, 0.52, 0.33, 90.0, "fixed_bathroom1_01"),
        (1.80, 7.12, 0.80, 0.65, 90.0, "fixed_bathroom1_02"),
        (1.74, 5.65, 0.45, 0.35, 90.0, "fixed_room9_01"),
        (1.77, 2.18, 0.70, 0.55, 0.0, "fixed_room9_02"),
        (9.05, 3.81, 1.50, 0.65, 65.0, "fixed_hall2_01"),

        # Bed and washbasin are fixed fixtures.
        (23.14, 13.86, 2.10, 0.95, 90.0, "fixed_room8_bed"),
        (22.55, 7.06, 1.15, 0.72, 0.0, "fixed_bathroom2_washbasin"),
    )
    return [_fixed(*spec) for spec in specs]


def _movable_obstacles():
    """Create thirteen obstacles arranged as three coupled decision groups."""

    # Decision A couples a bracing cart, beam, gas cylinders, and wheelchair.
    brace_cart = _movable(
        6.293, 12.80, 0.95, 0.58, 0.98, 94.2, "empty_cart", 1,
        mass=34.0 + 160.0, mu=MU_CASTORS_FOULED,
        contact_reveals=(
            "cart_bracing_cracked_load_bearing_beam_above_gas_cylinders"),
    )
    # Reinforced-concrete beam resting on broken concrete.
    cracked_beam = _movable(
        8.05, 13.45, 2.10, 0.38, 0.42, 18.0, "collapsed_beam", 2,
        mass=804.0, mu=MU_CONCRETE,
    )
    # Steel cage containing three compressed-gas cylinders.
    gas_cylinder_a = _movable(
        7.42, 12.12, 0.80, 0.60, 1.55, 78.0, "gas_cylinder", 3,
        mass=220.0, mu=MU_STEEL,
    )
    # Lightweight relief cartons.
    safe_boxes_a = _movable(
        6.422, 11.05, 0.80, 0.52, 0.62, 94.2, "cardboard_box", 4,
        mass=14.0, mu=MU_WOOD,
    )
    # Occupied wheelchair on rubble.
    occupied_wheelchair = _movable(
        6.115, 15.20, 1.05, 0.68, 0.95, 94.2, "occupied_wheelchair", 5,
        mass=88.0, mu=MU_RUBBER_WHEELS,
    )
    _couple(
        brace_cart, (cracked_beam, gas_cylinder_a),
        "moving the cart unloads a cracked beam onto a cage of gas cylinders",
    )

    # Decision B couples a cart to a damaged gas cylinder.
    free_cart_resistance = push_force(30.0, MU_CASTORS)
    dragged_cylinder_resistance = push_force(75.0, MU_WOOD)
    tethered_cart = _movable(
        10.077, 12.85, 0.95, 0.56, 0.98, 101.6, "empty_cart", 6,
        mass=30.0, mu=MU_CASTORS,
        difficulty=free_cart_resistance + dragged_cylinder_resistance,
        contact_reveals="cart_tethered_to_damaged_gas_cylinder",
    )
    # Full gas cylinder on its foot ring.
    gas_cylinder_b = _movable(
        11.25, 13.05, 0.34, 0.34, 1.52, 74.0,
        "damaged_gas_cylinder", 7, mass=75.0, mu=MU_WOOD,
    )
    safe_cart_b = _movable(
        9.586, 15.20, 0.92, 0.55, 0.96, 101.6, "empty_cart", 8,
        mass=28.0, mu=MU_CASTORS,
    )
    # Concrete lintel fragment.
    blocked_beam_b = _movable(
        10.435, 11.05, 0.90, 0.40, 0.38, 101.6,
        "collapsed_beam", 9, mass=328.0, mu=MU_CONCRETE,
    )
    _couple(
        tethered_cart, (gas_cylinder_b,),
        "the cart is mechanically tethered to a damaged gas cylinder",
    )

    # Decision C couples an electrical cabinet to a flooded water system.
    electrical_cabinet = _movable(
        15.528, 15.700, 0.80, 0.45, 1.80, 87.7,
        "filing_cabinet", 10, mass=105.0, mu=MU_STEEL,
        contact_reveals=(
            "live_electrical_cabinet_in_floodwater_beside_damaged_water_pipe"),
    )
    # Partly filled sectional water tank.
    water_tank = _movable(
        16.55, 14.78, 1.00, 0.60, 0.80, 12.0,
        "water_tank", 11, mass=275.0, mu=MU_STEEL,
    )
    # Damaged cast-iron water riser.
    damaged_pipe = _movable(
        16.45, 16.95, 1.60, 0.34, 0.34, -22.0,
        "damaged_water_pipe", 12, mass=120.0, mu=MU_STEEL,
    )
    safe_boxes_c = _movable(
        15.437, 13.45, 0.85, 0.50, 0.65, 87.7,
        "cardboard_box", 13, mass=16.0, mu=MU_WOOD,
    )
    _couple(
        electrical_cabinet, (water_tank, damaged_pipe),
        "moving the cabinet can energise water released by the damaged pair",
    )

    return [
        brace_cart, cracked_beam, gas_cylinder_a, safe_boxes_a,
        occupied_wheelchair, tethered_cart, gas_cylinder_b, safe_cart_b,
        blocked_beam_b, electrical_cabinet, water_tank, damaged_pipe,
        safe_boxes_c,
    ]


def _scene_obstacles():
    """The rest of the building's contents, shaken off their feet.

    These carry no authored decision. They are what the robot has to see,
    price and mostly drive around on its way between the three that do, and
    they are why the hall is a room with things in it rather than a corridor
    with three gates. Rows are oid, centre, size, heading, label, mass and the
    friction of whatever it is standing on.
    """
    specs = (
        # Hall 1, west of the first divider: what the robot meets first.
        (14, 3.30, 12.30, 1.30, 0.95, 0.55,   0.0, "debris_pile",
         950.0, MU_CONCRETE),
        (15, 4.90, 13.60, 1.60, 0.80, 0.75,  15.0, "office_desk",
         55.0, MU_WOOD),
        (16, 4.90, 11.60, 0.62, 0.55, 1.05, -25.0, "evacuation_chair",
         22.0, MU_RUBBER_WHEELS),

        # Hall 1, between the first and second dividers.
        (17, 8.20, 10.95, 1.60, 1.00, 0.35,   8.0, "ceiling_panel_stack",
         95.0, MU_WOOD),
        (18, 7.10, 16.10, 0.42, 0.42, 1.30,   0.0, "water_cooler",
         55.0, MU_WOOD),

        # Hall 1, between the second and third dividers.
        (19, 12.20, 14.30, 1.80, 0.50, 0.90, -12.0, "toppled_locker_bank",
         130.0, MU_STEEL),
        (20, 14.10, 12.90, 1.20, 0.80, 1.00,  20.0, "supply_pallet",
         320.0, MU_WOOD),

        # Room 1, off the entrance hall.
        (21, 3.60, 18.40, 1.32, 0.62, 0.47, 100.0, "filing_cabinet",
         65.0, MU_STEEL),
        (22, 2.30, 17.10, 0.60, 0.60, 1.05,  40.0, "office_chair",
         14.0, MU_RUBBER_WHEELS),

        # Rooms 2 and 3.
        (23, 7.30, 18.60, 0.80, 0.60, 1.90,   0.0, "server_rack",
         180.0, MU_STEEL),
        (24, 12.30, 18.70, 1.60, 0.55, 1.85,  0.0, "steel_shelf",
         150.0, MU_STEEL),

        # Rooms 4 and 5, the rooms either side of the survivor.
        (25, 15.30, 18.10, 2.00, 0.70, 0.45,  5.0, "folding_cot",
         18.0, MU_WOOD),
        (26, 19.10, 18.60, 1.00, 0.80, 1.85,  0.0, "vending_machine",
         260.0, MU_STEEL),
        (27, 21.20, 18.30, 0.90, 0.70, 0.80, 12.0, "wooden_crate",
         110.0, MU_STEEL),

        # Rooms 6 and 7, south of the hall.
        (28, 8.90, 7.60, 1.10, 0.90, 1.00,  -8.0, "wooden_crate",
         180.0, MU_STEEL),
        (29, 12.90, 8.20, 0.62, 0.62, 0.92,  20.0, "chemical_drum",
         200.0, MU_STEEL),

        # Hall 2, the southern hall.
        (30, 14.50, 3.20, 1.40, 1.40, 1.40,   0.0, "cable_drum",
         480.0, MU_WOOD),

        # Room 8 and Bathroom 2, on the eastern side.
        (31, 21.80, 14.10, 1.50, 0.85, 0.30,  8.0, "mattress",
         30.0, MU_UPHOLSTERY),
        (32, 22.10, 9.30, 0.60, 0.60, 1.45,   0.0, "water_heater",
         45.0, MU_STEEL),

        # Bathroom 1 and Room 9, on the western side.
        (33, 2.90, 8.30, 1.00, 0.80, 0.45, -15.0, "debris_pile",
         520.0, MU_CONCRETE),
        (34, 3.60, 3.40, 0.80, 0.60, 0.70,  30.0, "wooden_crate",
         75.0, MU_STEEL),
    )
    return [
        _movable(x, y, l, d, h, angle, material, oid, mass=mass, mu=mu)
        for oid, x, y, l, d, h, angle, material, mass, mu in specs
    ]


def create():
    """Build the reference earthquake-rescue scenario."""
    workspace = box(0.0, 0.0, _WIDTH, _HEIGHT)
    static = [*_walls(), *_fixed_obstacles()]
    movable = [*_movable_obstacles(), *_scene_obstacles()]
    cfg = Config(
        grid_step=0.30,
        conn_radius=0.70,
        se2_cell=0.15,
        R_perc=8.0,
        R_manip=4.0,
    )
    _realism.check_layout(
        "earthquake", workspace=workspace, static=static, movable=movable,
        start=_START, goal=_GOAL, cfg=cfg,
    )
    return {
        "workspace": workspace,
        "static": static,
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "cfg": cfg,
        "entrance": _START,
        "survivor_region": _SURVIVOR_REGION,
        "decision_points": [
            {
                "name": "structural_support_chain",
                "risky": 1,
                "partners": [2, 3],
                "safer_alternative": 4,
                "avoid": 5,
            },
            {
                "name": "pressurised_cylinder_chain",
                "risky": 6,
                "partners": [7],
                "safer_alternative": 8,
                "avoid": 9,
            },
            {
                "name": "water_electrical_chain",
                "risky": 10,
                "partners": [11, 12],
                "safer_alternative": 13,
                "avoid": "lower_detour",
            },
        ],
    }
