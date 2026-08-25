"""Detour or push, and what it takes to make the robot turn back.

Nothing but the outer shell is axis-aligned: two slanted partitions tilted
opposite ways, one cut by three doorways and the other by two, and every
obstacle plugging a doorway lying along the heading of the wall it sits in.

The route is a sequence of decisions, each of a different kind:

  1. **Detour or push, decided by risk.** The straight line to the goal runs
     through the middle doorway of the first wall, which is plugged by an
     occupied wheelchair. It rolls, so it is the cheapest thing on the map to
     shift and the energy term alone would shove it aside without hesitating.
     The risk surcharge for moving a person is what sends the robot round.
  2. **Push, decided by energy — and wrong.** Of the two doorways left it takes
     the one plugged by an empty cart, which rolls too: cheaper to move than the
     filing cabinet in the third doorway by more than the extra walk to reach
     that one. Same question as (1), opposite answer.
  3. **Turn back, decided by what moving the cart uncovers.** The cart was
     hiding a concrete block lying flat against the far face of that same
     doorway, close enough to it that nothing can pass down the slot between the
     two. The block cannot be seen at all until the cart is out of the way, and
     by then the walk and the push are spent for nothing — metre for metre
     the block is an order of magnitude dearer to shift than the cabinet the
     robot passed over, so it turns round and walks the length of the wall to
     the third doorway after all. The cheapest obstacle was never the cheapest
     way through, and finding that out the hard way is what this doorway is for.
  4. **Turn back, decided by what contact reveals.** Through the cabinet is the
     middle chamber, with a doorway at each end. The near one holds a sealed
     crate that reads as ordinary freight, and the robot goes for it. Touching
     it resolves the label into a crate of gas cylinders — heavier than it
     looked and, more to the point, hazardous. That verdict arrives when the
     robot is already standing next to it, and the long doorway at the far end
     of the chamber is suddenly worth the walk back.
  5. **Detour, decided by sheer mass.** A run of steel racking lies across the
     last room at a third angle, sealed against the outer wall at one end. Going
     round the open end costs a couple of kilojoules; moving it would cost
     forty times that per metre, so the robot goes round without the question
     being close.

Two of those turn the robot round, and for different reasons: (3) is something
it could not see, (4) something it could not know until it touched. Distances
are set so none of the five is close.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from shapely.geometry import box

from config import Config
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle

XY = Tuple[float, float]

_SHELL_T = 0.5                   # outer shell thickness
_WALL_T = 0.6                    # partition thickness

# Total daylight left around an obstacle plugging a doorway, split evenly either
# side. The robot is 0.4 m across, so anything at or above that lets it squeeze
# past without touching and the doorway stops posing a question.
_DOOR_DAYLIGHT = 0.6

# The two partitions, stated endpoint-to-endpoint and tilted opposite ways, and
# where along each one the doorways are cut: (fraction of the way along, width).
#
# The positions are what make the map pose its questions, so they are set by the
# answers they have to produce, measured in metres of travel:
#
#   * through the wheelchair is ~7 m shorter than round to the cart, which is
#     more than the wheelchair costs to move — so energy alone takes it, and
#     only the risk surcharge sends the robot the long way;
#   * the cabinet doorway is the nearer of the two that are left, so the cart has
#     to beat it on the obstacle instead rather than on the walk — it does, and
#     that is what sends the robot to the doorway that turns out to be a dead
#     end, with the cabinet already in plain sight from the start;
#   * out through the sealed crate is ~19 m shorter than out through the wooden
#     one, so the robot commits to it before touching it, and the surcharge that
#     lands when it does is worth far more than walking back down the chamber.
_WEST_WALL = ((12.0, 0.3), (15.0, 23.7))
_CABINET_DOOR = (0.12, 2.6)         # bottom of the wall, and the answer in the end
_WHEELCHAIR_DOOR = (0.45, 1.8)      # on the direct line, and the cheap way through
_CART_DOOR = (0.92, 2.6)            # top of the wall, and a dead end behind the cart

# What the concrete block leaves between itself and the wall it lies against.
# Under a robot's width, so clearing the cart out of that doorway opens nothing:
# what looked like the cheap way through is not a way through at all.
_TRAP_GAP = 0.35

_EAST_WALL = ((25.0, 0.3), (22.0, 23.7))
_SEALED_DOOR = (0.70, 2.4)          # the near exit, and the one that turns out badly
_CRATE_DOOR = (0.15, 2.6)           # the far exit, down at the bottom of the chamber

_START = (3.5, 6.0)
_GOAL = (32.5, 20.5)

# The racking in the last room, stated as the two ends of its long axis. The east
# end stops 0.2 m short of the outer wall — no overlap, and no way past either —
# so the only way round is the far end, which is most of a room away from the
# line the robot would otherwise drive.
_SHELF_ENDS = ((25.2, 17.5), (34.9, 8.0))


def _lerp(a: XY, b: XY, t: float) -> XY:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _wall_with_doors(a: XY, b: XY, doors: Sequence[Tuple[float, float]],
                     name: str) -> Tuple[List[StaticObstacle], List[dict]]:
    """Lay the wall a->b, leaving a gap at each (fraction along, width).

    Returns the wall pieces plus, per gap, everything an obstacle needs to sit in
    it flush: the centre, the wall's heading, and how wide the gap is.
    """
    span = math.hypot(b[0] - a[0], b[1] - a[1])
    theta = math.atan2(b[1] - a[1], b[0] - a[0])
    cuts = sorted((at, 0.5 * width / span) for at, width in doors)

    pieces: List[StaticObstacle] = []
    edge = 0.0
    for i, (at, half) in enumerate(cuts):
        if at - half - edge > 1e-9:
            pieces.append(StaticObstacle.segment(
                _lerp(a, b, edge), _lerp(a, b, at - half), _WALL_T, f"{name}_{i}"))
        edge = at + half
    if 1.0 - edge > 1e-9:
        pieces.append(StaticObstacle.segment(
            _lerp(a, b, edge), b, _WALL_T, f"{name}_{len(cuts)}"))

    gaps = [{"centre": _lerp(a, b, at), "theta": theta, "width": 2.0 * half * span}
            for at, half in cuts]
    return pieces, gaps


def _plug(gap: dict, d: float, h: float, material: str, oid: int,
          difficulty: float | None = None,
          contact_reveals: str = "") -> MovableObstacle:
    """An obstacle filling *gap*, lying along the heading of the wall it plugs.

    `l` runs along the wall and is what the daylight is measured on; `d` runs
    across it. With `difficulty` left out the true value is the friction the
    material implies, so the ground truth and the label agree — pass one to make
    an obstacle that is not what it looks like.
    """
    l = gap["width"] - _DOOR_DAYLIGHT
    if difficulty is None:
        difficulty = round(friction_force(material_mu_rho(material), l * d * h), 3)
    return MovableObstacle(
        x=gap["centre"][0], y=gap["centre"][1],
        l=l, d=d, h=h, theta=gap["theta"],
        material=material, difficulty=difficulty,
        contact_reveals=contact_reveals, oid=oid,
    )


def _behind(gap: dict, d: float, h: float, material: str, oid: int,
            overhang: float = 0.3) -> MovableObstacle:
    """An obstacle lying across the far face of *gap*, parallel to the wall.

    Set back far enough to clear the wall and the doorway's own plug without
    touching either, and no further: the slot it leaves against the wall has to
    stay under a robot's width, or the doorway is not sealed and the trap is not
    a trap. `overhang` is how far past each side of the gap it reaches, so it
    cannot be walked round through the doorway either.
    """
    theta = gap["theta"]
    normal = (math.sin(theta), -math.cos(theta))        # away from the near room
    offset = _WALL_T / 2.0 + _TRAP_GAP + d / 2.0
    cx, cy = gap["centre"]
    return MovableObstacle(
        x=cx + normal[0] * offset, y=cy + normal[1] * offset,
        l=gap["width"] + 2.0 * overhang, d=d, h=h, theta=theta,
        material=material,
        difficulty=round(friction_force(
            material_mu_rho(material), (gap["width"] + 2.0 * overhang) * d * h), 3),
        oid=oid,
    )


def _shelf(ends: Tuple[XY, XY], d: float, h: float, material: str,
           oid: int) -> MovableObstacle:
    """A body lying along the line between two points, `d` wide across it."""
    (ax, ay), (bx, by) = ends
    l = math.hypot(bx - ax, by - ay)
    return MovableObstacle(
        x=0.5 * (ax + bx), y=0.5 * (ay + by),
        l=l, d=d, h=h, theta=math.atan2(by - ay, bx - ax),
        material=material,
        difficulty=round(friction_force(material_mu_rho(material), l * d * h), 3),
        oid=oid,
    )


def _reject_overlaps(walls: Sequence[StaticObstacle],
                     movable: Sequence[MovableObstacle]) -> None:
    """No obstacle may share floor with a wall.

    A doorway plug sits *in* the gap, not in the wall beside it, and an obstacle
    buried in a wall is one the robot can neither reach nor ever move — the run
    would just report the doorway as impassable and nothing would say why.
    """
    for obs in movable:
        body = obs.polygon
        for wall in walls:
            overlap = body.intersection(wall.polygon).area
            if overlap > 1e-9:
                raise ValueError(
                    f"obstacle {obs.oid} ({obs.material}) overlaps wall "
                    f"{wall.name!r} by {overlap:.4f} m^2")


def create():
    workspace = box(0, 0, 36, 24)
    t = _SHELL_T

    west_walls, west_doors = _wall_with_doors(
        *_WEST_WALL, (_CABINET_DOOR, _WHEELCHAIR_DOOR, _CART_DOOR), "west_divider")
    east_walls, east_doors = _wall_with_doors(
        *_EAST_WALL, (_SEALED_DOOR, _CRATE_DOOR), "east_divider")
    cabinet_door, wheelchair_door, cart_door = west_doors    # sorted along the wall
    crate_door, sealed_door = east_doors

    walls = [
        StaticObstacle.segment((0.0, t / 2), (36.0, t / 2), t, "outer_bottom"),
        StaticObstacle.segment((0.0, 24.0 - t / 2), (36.0, 24.0 - t / 2), t, "outer_top"),
        StaticObstacle.segment((t / 2, 0.0), (t / 2, 24.0), t, "outer_left"),
        StaticObstacle.segment((36.0 - t / 2, 0.0), (36.0 - t / 2, 24.0), t, "outer_right"),
        *west_walls,
        *east_walls,
    ]

    movable = [
        # Straight ahead, and off limits. It rolls, so the energy term prices it
        # at almost nothing; only the surcharge for moving a person stands
        # between the robot and shoving it out of the way.
        _plug(wheelchair_door, d=0.85, h=1.3, material="occupied_wheelchair",
              difficulty=28.0, oid=1),
        # The way round the wheelchair the robot takes, and the wrong one: light,
        # harmless, and cheaper to move than anything else it can reach — which
        # is exactly why it never looks behind the door first.
        _plug(cart_door, d=0.9, h=1.0, material="empty_cart", oid=2),
        # What the cart was hiding. Flat against the back of that doorway, so
        # moving the cart buys nothing at all, and no cheaper to shift than a
        # wall is. Standing where the cart stood, the robot cannot see it: the
        # cart is as tall as it is and fills the gap.
        _behind(cart_door, d=0.9, h=1.0, material="concrete_block", oid=3),
        # The doorway the robot passed over, and the one it comes back to. Real
        # weight rather than a rounding error, so the cart wins on first sight —
        # but a fortieth of what the block behind the cart would cost.
        _plug(cabinet_door, d=0.9, h=1.8, material="filing_cabinet", oid=4),
        # Freight, until the robot lays a hand on it. Heavier than the label
        # implies (a wooden crate this size would be ~430 N) and hazardous once
        # the contents are known, which is what turns the robot round.
        _plug(sealed_door, d=0.7, h=1.0, material="sealed_crate",
              contact_reveals="crate_of_gas_cylinders", difficulty=1200.0, oid=5),
        # The far end of the chamber: an honest crate, and the answer once the
        # near doorway has priced itself out.
        _plug(crate_door, d=0.9, h=1.0, material="wooden_crate", oid=6),
        # A run of racking lying across the last room at a third angle, all but
        # touching the outer wall at one end and leaving the room's whole width
        # to be walked around at the other. Immovable in practice — tens of kN
        # of friction against a detour of a few metres — so this one is the
        # trade-off made in the other direction, and made without hesitating.
        _shelf(_SHELF_ENDS, d=1.2, h=2.0, material="steel_shelf", oid=7),
    ]

    _reject_overlaps(walls, movable)

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "cfg": Config(),
    }
