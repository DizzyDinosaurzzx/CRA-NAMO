"""Static ten-gate corridor for measuring route impact of estimator gaps."""

from __future__ import annotations

from shapely.geometry import box

import risk as risk_model
from config import Config
from obstacle import MovableObstacle, StaticObstacle

# Geometry constants.
WALL_T = 0.4                 # wall and shell thickness [m]
CORRIDOR_H = 18.0            # workspace height [m]
GATE_SPACING = 5.0           # wall-to-wall pitch [m]
FIRST_GATE_X = 5.0           # x of the first wall [m]
N_GATES = 10

DOOR_H = 1.8                 # opening size along y [m]
DOOR_A_Y = 7.6               # door A occupies [7.6, 9.4]
DOOR_B_Y = 9.7               # door B occupies [9.7, 11.5]
TRAVEL_Y = 9.55              # the corridor axis, midway between A and B
BLOCKER_CLEARANCE = 0.2      # blocker is this much smaller than its doorway

WORKSPACE_W = FIRST_GATE_X * 2 + GATE_SPACING * (N_GATES - 1)   # 55.0 m

A_OID_BASE = 100             # door A of gate i is oid 100 + 2i, door B is +1

# Gate rows: bypass position, materials, and true difficulties.
GATES = (
    (4.0,  "wooden_crate",    1600.0, "wooden_crate",    2600.0),  # near break-even
    (13.6, "wooden_crate",     400.0, "wooden_crate",    3000.0),  # cheap control
    (3.0,  "wooden_crate",    2200.0, "wooden_crate",    1800.0),  # near break-even
    (14.6, "wooden_crate",    3400.0, "wooden_crate",    2900.0),  # near break-even
    (4.4,  "glassware_crate",  800.0, "wooden_crate",    2400.0),  # risk-sensitive
    (15.6, "wooden_crate",    1200.0, "glassware_crate",  900.0),  # risk-sensitive
    (2.2,  "wooden_crate",    2600.0, "wooden_crate",    2000.0),  # near break-even
    (13.0, "wooden_crate",    5000.0, "wooden_crate",    4500.0),  # expensive control
    (1.2,  "glassware_crate", 1000.0, "wooden_crate",    2600.0),  # risk-sensitive
    (15.8, "wooden_crate",    2000.0, "glassware_crate", 1500.0),  # risk-sensitive
)

# Per-gate bypass distance in metres.
DETOUR_M = tuple(round(2.0 * abs(y0 + DOOR_H / 2.0 - TRAVEL_Y), 2)
                 for y0, *_ in GATES)


def gate_x(i: int) -> float:
    """x of the near face of gate i's wall."""
    return FIRST_GATE_X + GATE_SPACING * i


def _blocker(oid: int, x: float, door_y: float, material: str,
             difficulty: float) -> MovableObstacle:
    """Create a doorway plug that leaves less than robot-width clearance."""
    return MovableObstacle(
        x=x + WALL_T / 2.0,
        y=door_y + DOOR_H / 2.0,
        l=WALL_T,
        d=DOOR_H - BLOCKER_CLEARANCE,
        h=1.0,
        theta=0.0,
        material=material,
        difficulty=float(difficulty),
        oid=oid,
    )


def gate_table() -> list:
    """Return ground-truth gate records for the gap study."""
    rows = []
    for i, (bypass_y0, mat_a, diff_a, mat_b, diff_b) in enumerate(GATES):
        for side, material, difficulty in (("A", mat_a, diff_a),
                                           ("B", mat_b, diff_b)):
            rows.append({
                "gate": i,
                "side": side,
                "oid": A_OID_BASE + 2 * i + (0 if side == "A" else 1),
                "material": material,
                "difficulty": float(difficulty),
                "risk": risk_model.keyword_level(material),
                "detour_m": DETOUR_M[i],
                "bypass_y": bypass_y0,
            })
    return rows


def _wall_segments(bypass_y0: float):
    """The wall left over once the three openings are cut out of it."""
    openings = sorted([(DOOR_A_Y, DOOR_A_Y + DOOR_H),
                       (DOOR_B_Y, DOOR_B_Y + DOOR_H),
                       (bypass_y0, bypass_y0 + DOOR_H)])
    edges = [0.0]
    for lo, hi in openings:
        edges += [lo, hi]
    edges.append(CORRIDOR_H)
    return [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]


def create():
    """Build the ten-gate corridor."""
    workspace = box(0.0, 0.0, WORKSPACE_W, CORRIDOR_H)

    walls = [
        StaticObstacle(box(0.0, 0.0, WORKSPACE_W, WALL_T), "shell_bottom"),
        StaticObstacle(box(0.0, CORRIDOR_H - WALL_T, WORKSPACE_W, CORRIDOR_H),
                       "shell_top"),
        StaticObstacle(box(0.0, 0.0, WALL_T, CORRIDOR_H), "shell_left"),
        StaticObstacle(box(WORKSPACE_W - WALL_T, 0.0, WORKSPACE_W, CORRIDOR_H),
                       "shell_right"),
    ]

    movable = []
    for i, (bypass_y0, mat_a, diff_a, mat_b, diff_b) in enumerate(GATES):
        x = gate_x(i)
        for k, (y0, y1) in enumerate(_wall_segments(bypass_y0)):
            if y1 - y0 > 1e-9:
                walls.append(StaticObstacle(box(x, y0, x + WALL_T, y1),
                                            f"gate{i}_seg{k}"))
        movable.append(_blocker(A_OID_BASE + 2 * i, x, DOOR_A_Y, mat_a, diff_a))
        movable.append(_blocker(A_OID_BASE + 2 * i + 1, x, DOOR_B_Y, mat_b, diff_b))

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (2.2, TRAVEL_Y),
        "goal": (WORKSPACE_W - 2.2, TRAVEL_Y),
        "cfg": Config(),
        # Ground truth used by the estimator-gap study.
        "gates": gate_table(),
    }
