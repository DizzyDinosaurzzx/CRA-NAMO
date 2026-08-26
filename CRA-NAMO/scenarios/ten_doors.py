"""Ten gates, three ways through each one: move A, move B, or walk around.

A benchmark map for one question — what does a wrong `difficulty` or `risk`
estimate cost a route? — built so the answer is readable rather than tangled.
The robot crosses a straight corridor from end to end along the axis y =
`TRAVEL_Y`. Ten walls cross its path, and every wall has exactly three openings:

    door A      just below the axis, plugged by one movable obstacle
    door B      just above the axis, plugged by a second movable obstacle
    the bypass  far off the axis, open, and reached only by leaving the axis
                and coming back to it

So each wall is one independent three-way decision and the run is ten of them in
a row. Nothing else competes for the planner's attention: the cells between the
walls are empty, and A and B sit the same small distance either side of the
axis, so walking to one costs exactly what walking to the other costs and the
choice between them is decided by push cost and risk alone.

Why the bypasses alternate sides
--------------------------------
A bypass has to be a *local* detour. If every bypass were high up the wall the
robot would climb once, run along the top through all ten, and never pay for the
detour again — the first version of this map did exactly that, and no gate was
ever worth a push. So the bypasses alternate: gate 0 low, gate 1 high, gate 2
low, and so on. Using two in a row means crossing the full height of the
corridor twice, which is never cheaper than coming back to the axis. What each
one costs is then simply the round trip off the axis and back, which is the
number `DETOUR_M` records per gate.

Why the three options are close together
----------------------------------------
A decision only reveals an estimator error if the estimator can change it, so
each gate is placed near one of the two break-evens the planner computes:

    push instead of detour when   difficulty * push_distance < lambda * detour
    push at all only when         the risk surcharge is worth paying
                                  (`risk.RISK_DETOUR_EQUIV_M`: low 0 m,
                                  medium 20 m, and up)

With the default lambda = 350 N a gate whose bypass costs 10 m is a 3500 J
budget, a ~2 m push of a 1600 N obstacle spends ~3200 J of it, and a single
`medium` risk verdict adds 7000 J and settles the question by itself. The gate
table spans that band deliberately: some gates sit a few percent from their
break-even and flip under a small error, others are far from it and must not
flip whatever the estimator says. Those are the controls — an experiment where
every gate is knife-edge cannot tell a real effect from noise.

Two gates carry a genuinely `medium` risk obstacle (a crate of glassware) in the
cheap door, so under-calling its risk is the error that makes the planner push
exactly the thing it should have walked around.

The map is static: no dynamics, no hidden labels, no contact reveals. Every
difference between two runs of this scenario comes from what the estimator was
told to believe.
"""

from __future__ import annotations

from shapely.geometry import box

import risk as risk_model
from config import Config
from obstacle import MovableObstacle, StaticObstacle

# -- geometry ---------------------------------------------------------------
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

# -- the gates --------------------------------------------------------------
# (bypass_y0, A material, A true difficulty [N], B material, B true difficulty)
#
# `bypass_y0` alternates low / high; the comment on each row is the intent, and
# the oracle run is what settles what the map actually does.
#
# `wooden_crate` reads as low risk in `risk.RISK_LABELS`, `glassware_crate` as
# medium. Difficulty is stated outright rather than derived from the material,
# because what is being tuned here is the distance from the break-even, not the
# plausibility of a crate.
GATES = (
    (4.0,  "wooden_crate",    1600.0, "wooden_crate",    2600.0),  # 0 near:    A just beats the bypass
    (13.6, "wooden_crate",     400.0, "wooden_crate",    3000.0),  # 1 control: A far too cheap to skip
    (3.0,  "wooden_crate",    2200.0, "wooden_crate",    1800.0),  # 2 near:    B beats both A and the bypass
    (14.6, "wooden_crate",    3400.0, "wooden_crate",    2900.0),  # 3 near:    the bypass wins, but only just
    (4.4,  "glassware_crate",  800.0, "wooden_crate",    2400.0),  # 4 risk:    A is cheap and must not be touched
    (15.6, "wooden_crate",    1200.0, "glassware_crate",  900.0),  # 5 risk:    A is right, over-calling it loses
    (2.2,  "wooden_crate",    2600.0, "wooden_crate",    2000.0),  # 6 near:    B by a small margin
    (13.0, "wooden_crate",    5000.0, "wooden_crate",    4500.0),  # 7 control: both doors hopeless
    (1.2,  "glassware_crate", 1000.0, "wooden_crate",    2600.0),  # 8 risk:    B against a long bypass
    (15.8, "wooden_crate",    2000.0, "glassware_crate", 1500.0),  # 9 control: A comfortably right
)

# What walking around gate i costs, in metres: off the axis to the middle of the
# bypass and back. This is the number the planner weighs each push against.
DETOUR_M = tuple(round(2.0 * abs(y0 + DOOR_H / 2.0 - TRAVEL_Y), 2)
                 for y0, *_ in GATES)


def gate_x(i: int) -> float:
    """x of the near face of gate i's wall."""
    return FIRST_GATE_X + GATE_SPACING * i


def _blocker(oid: int, x: float, door_y: float, material: str,
             difficulty: float) -> MovableObstacle:
    """One doorway plug: as thick as the wall, and 0.2 m short of its opening.

    0.2 m of daylight is less than the robot's 0.4 m diameter, so the door is
    shut until the obstacle is moved — but the gap is real, which keeps the
    doorway from reading as a solid wall.
    """
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
    """One record per obstacle, for a study that needs to know what it is.

    Everything here is ground truth: the true difficulty the world will charge,
    the true risk level its label carries, and the detour its gate offers.
    """
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
        # Ground truth for the estimator-error study; ignored by the planner.
        "gates": gate_table(),
    }
