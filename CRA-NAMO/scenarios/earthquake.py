"""Earthquake-rescue floor plan with three coupled-obstacle decisions.

The 24 m x 20 m footprint, entrance, survivor goal, planner configuration and
the thirteen movable-obstacle budget are retained from the supplied map.  The
damaged upper hall is reorganised into three non-parallel breached dividers.
Each divider offers several passages and therefore a genuine route decision:

1. a cart that appears harmless is actually bracing a cracked beam above gas
   cylinders;
2. a second cart is tethered to a damaged pressurised cylinder; and
3. a light filing cabinet is coupled to a damaged water/electrical assembly.

The coupled hazards are placed as visible obstacle groups.  ``contact_reveals``
lets the existing online risk estimator revise the apparently low-risk blocker
when the robot interacts with it, while ``decision_points`` and
``interacts_with`` retain the authored causal relationships for inspection.
"""

from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle


_WIDTH = 24.0
_HEIGHT = 20.0
_WALL_T = 0.32

_START = (1.35, 11.25)
_SURVIVOR_REGION = box(18.80, 15, 19, 17)
_GOAL = (18,16)


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
             *, difficulty: float | None = None,
             contact_reveals: str = "") -> MovableObstacle:
    if difficulty is None:
        difficulty = friction_force(material_mu_rho(material), l * d * h)
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
        # Broken upper boundary (the five clear spans match the reference).
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

        # Room 3 and the broken wall along its lower side.
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

        # West boundary below Room 1; the lower opening is the entrance.
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

        # Three damaged, non-parallel dividers across the rescue hall.  Each
        # has three breaches; movable obstacles below turn those breaches into
        # successive route-versus-risk decisions.
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

        # Secondary collapsed members preserve the irregular earthquake scene
        # without obscuring the three authored decision bands.
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

        # The bed and plumbed washbasin are fixtures rather than two of the
        # thirteen movable obstacles stated in the reference note.
        (23.14, 13.86, 2.10, 0.75, 90.0, "fixed_room8_bed"),
        (22.55, 7.06, 1.15, 0.72, 0.0, "fixed_bathroom2_washbasin"),
    )
    return [_fixed(*spec) for spec in specs]


def _movable_obstacles():
    """Thirteen obstacles arranged as three coupled decision groups."""

    # Decision A — the central breach is geometrically attractive.  The cart
    # initially reads as low risk, but contact reveals that it is a structural
    # brace above a pressurised cylinder.  The low breach is the safe fallback;
    # the high breach is blocked by an occupied stretcher.
    brace_cart = _movable(
        6.293, 12.80, 0.82, 0.54, 1.00, 94.2, "empty_cart", 1,
        difficulty=24.0,
        contact_reveals=(
            "cart_bracing_cracked_load_bearing_beam_above_gas_cylinders"),
    )
    cracked_beam = _movable(
        7.72, 13.42, 1.65, 0.38, 1.65, 18.0, "collapsed_beam", 2,
        difficulty=2800.0,
    )
    gas_cylinder_a = _movable(
        7.42, 12.12, 0.58, 0.46, 1.35, 78.0, "gas_cylinder", 3,
        difficulty=180.0,
    )
    safe_boxes_a = _movable(
        6.422, 11.05, 0.72, 0.48, 0.72, 94.2, "cardboard_box", 4,
        difficulty=18.0,
    )
    occupied_stretcher = _movable(
        6.115, 15.20, 0.82, 0.62, 1.05, 94.2,
        "occupied_stretcher", 5, difficulty=30.0,
    )
    _couple(
        brace_cart, (cracked_beam, gas_cylinder_a),
        "moving the cart unloads a cracked beam onto a gas cylinder",
    )

    # Decision B — a second apparently empty cart occupies the middle breach.
    # Its tether to a damaged cylinder is only identified on interaction.  The
    # upper cart is a safe alternative; the lower breach contains a collapsed
    # beam that should never be disturbed.
    tethered_cart = _movable(
        10.077, 12.85, 0.82, 0.54, 0.95, 101.6, "empty_cart", 6,
        difficulty=22.0,
        contact_reveals="cart_tethered_to_damaged_gas_cylinder",
    )
    gas_cylinder_b = _movable(
        11.25, 13.05, 0.62, 0.48, 1.40, 74.0,
        "damaged_gas_cylinder", 7, difficulty=220.0,
    )
    safe_cart_b = _movable(
        9.586, 15.20, 0.82, 0.52, 0.90, 101.6, "empty_cart", 8,
        difficulty=20.0,
    )
    blocked_beam_b = _movable(
        10.435, 11.05, 0.78, 0.58, 1.70, 101.6,
        "collapsed_beam", 9, difficulty=4100.0,
    )
    _couple(
        tethered_cart, (gas_cylinder_b,),
        "the cart is mechanically tethered to a damaged gas cylinder",
    )

    # Decision C — the direct upper breach looks cheap because the filing
    # cabinet slides easily.  Interaction exposes its electrical coupling to a
    # displaced water tank and damaged pipe.  Cardboard boxes in the middle
    # breach provide the lower-risk route to the survivor.
    electrical_cabinet = _movable(
        15.528, 15.725, 0.70, 0.56, 1.35, 87.7,
        "filing_cabinet", 10, difficulty=28.0,
        contact_reveals=(
            "live_electrical_cabinet_in_floodwater_beside_damaged_water_pipe"),
    )
    water_tank = _movable(
        16.65, 14.80, 1.05, 0.62, 1.25, 12.0,
        "water_tank", 11, difficulty=250.0,
    )
    damaged_pipe = _movable(
        16.75, 17.05, 1.45, 0.32, 0.55, -22.0,
        "damaged_water_pipe", 12, difficulty=300.0,
    )
    safe_boxes_c = _movable(
        15.437, 13.45, 0.82, 0.48, 0.70, 87.7,
        "cardboard_box", 13, difficulty=20.0,
    )
    _couple(
        electrical_cabinet, (water_tank, damaged_pipe),
        "moving the cabinet can energise water released by the damaged pair",
    )

    return [
        brace_cart, cracked_beam, gas_cylinder_a, safe_boxes_a,
        occupied_stretcher, tethered_cart, gas_cylinder_b, safe_cart_b,
        blocked_beam_b, electrical_cabinet, water_tank, damaged_pipe,
        safe_boxes_c,
    ]


def create():
    """Build the reference earthquake-rescue scenario."""
    movable = _movable_obstacles()
    return {
        "workspace": box(0.0, 0.0, _WIDTH, _HEIGHT),
        "static": [*_walls(), *_fixed_obstacles()],
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "cfg": Config(
            grid_step=0.30,
            conn_radius=0.70,
            se2_cell=0.15,
            R_perc=8.0,
            R_manip=4.0,
        ),
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
