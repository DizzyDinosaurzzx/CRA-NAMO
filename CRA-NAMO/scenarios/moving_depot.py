"""A depot that will not hold still: the map changes while the robot crosses it.

Two rooms joined by one door, and beyond the door a neck plugged by a crate.
None of what follows is in the map the robot is handed — it is in the world, and
the robot finds it out the way it finds out everything else, by looking.

    the trolley     Waits in the first room until the robot is halfway to the
                    door, then crosses through it. 1.6 m of trolley in a 2.0 m
                    door leaves nothing to squeeze past, and at 6 cm/s it is in
                    the way for the best part of a minute. The robot can wait for
                    the door to clear, follow it through, or take hold of it and
                    set it aside — and if it does that, the trolley carries on to
                    where it was going from wherever it was put down.

    the crate       1.5 m of crate in the 2.0 m neck beyond the door. Nothing
                    dynamic about it: it is there to be moved, and moving it is
                    what sets the pallet off.

    the pallet      Parked in the far room, where the robot sees it early and
                    prices it as an empty pallet. Partway through the run the
                    load settles: it becomes a bigger, far heavier concrete
                    block. The robot goes on believing the old figure until it
                    next looks at it, and only learns the new weight by taking
                    hold of it, because weight is not a thing you can see. Then,
                    once the crate has been moved, the pallet rolls into the last
                    aisle and the robot has to deal with it at its new weight —
                    which is most of what the run ends up costing.

One trigger of each kind the dynamics support: a place the robot reaches, a
moment on the clock, an obstacle the robot moves.
"""

from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from dynamics import Event, Mutate, MoveTo, after_moved, at_time, near_point
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle

_WIDTH = 20.0
_HEIGHT = 10.0
_SHELL = 0.5

_DIVIDER_X = (9.8, 10.2)         # the wall between the two rooms
_DOOR = (4.0, 6.0)               # the one gap in it
_NECK_X = (13.0, 13.4)           # the thin neck beyond it
_NECK = (4.0, 6.0)

_START = (2.5, 5.0)
_GOAL = (17.5, 5.0)


def _difficulty(material: str, l: float, d: float, h: float) -> float:
    return round(friction_force(material_mu_rho(material), l * d * h), 3)


def create():
    t = _SHELL
    walls = [
        StaticObstacle(box(0.0, 0.0, _WIDTH, t), "shell_south"),
        StaticObstacle(box(0.0, _HEIGHT - t, _WIDTH, _HEIGHT), "shell_north"),
        StaticObstacle(box(0.0, 0.0, t, _HEIGHT), "shell_west"),
        StaticObstacle(box(_WIDTH - t, 0.0, _WIDTH, _HEIGHT), "shell_east"),
        StaticObstacle(box(_DIVIDER_X[0], t, _DIVIDER_X[1], _DOOR[0]),
                       "divider_south"),
        StaticObstacle(box(_DIVIDER_X[0], _DOOR[1], _DIVIDER_X[1], _HEIGHT - t),
                       "divider_north"),
        StaticObstacle(box(_NECK_X[0], t, _NECK_X[1], _NECK[0]), "neck_south"),
        StaticObstacle(box(_NECK_X[0], _NECK[1], _NECK_X[1], _HEIGHT - t),
                       "neck_north"),
    ]

    movable = [
        # Waiting in the first room with its own errand to run.
        MovableObstacle(
            x=9.0, y=6.6, l=1.6, d=0.6, h=0.9, theta=math.pi / 2.0,
            material="service_trolley",
            difficulty=_difficulty("service_trolley", 1.6, 0.6, 0.9),
            oid=1,
        ),
        # Plugs the neck: 1.5 m of crate across a 2.0 m gap.
        MovableObstacle(
            x=13.2, y=5.0, l=1.5, d=1.1, h=1.0, theta=math.pi / 2.0,
            material="wooden_crate",
            difficulty=_difficulty("wooden_crate", 1.5, 1.1, 1.0),
            oid=2,
        ),
        # Parked out of the way in the far room, for now.
        MovableObstacle(
            x=16.5, y=8.4, l=1.2, d=1.0, h=0.9, theta=0.0,
            material="empty_pallet",
            difficulty=_difficulty("empty_pallet", 1.2, 1.0, 0.9),
            oid=3,
        ),
    ]

    events = [
        Event(
            name="trolley sets off",
            trigger=near_point((5.5, 5.0), 1.2),
            effect=MoveTo(oid=1, goal=(11.6, 7.8, math.pi / 2.0), speed=0.06),
        ),
        Event(
            name="pallet rolls into the aisle",
            trigger=after_moved(2),
            effect=MoveTo(oid=3, goal=(16.5, 5.0, 0.0), speed=0.12),
        ),
        Event(
            name="the load settles",
            trigger=at_time(90.0),
            effect=Mutate(oid=3, material="concrete_block", l=1.5, d=1.2,
                          difficulty=_difficulty("concrete_block", 1.5, 1.2, 0.9)),
        ),
    ]

    return {
        "workspace": box(0, 0, _WIDTH, _HEIGHT),
        "static": walls,
        "movable": movable,
        "start": _START,
        "goal": _GOAL,
        "dynamics": events,
        "cfg": Config(),
    }
