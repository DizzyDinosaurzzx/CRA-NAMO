"""Dynamic depot scenario with moving, triggered, and mutated obstacles."""

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

_DIVIDER_X = (9.8, 10.2)         # Wall separating the two rooms.
_DOOR = (4.0, 6.0)               # Door opening in the divider.
_NECK_X = (13.0, 13.4)           # Narrow neck beyond the divider.
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
        # Trolley waits in the first room.
        MovableObstacle(
            x=9.0, y=6.6, l=1.6, d=0.6, h=0.9, theta=math.pi / 2.0,
            material="service_trolley",
            difficulty=_difficulty("service_trolley", 1.6, 0.6, 0.9),
            oid=1,
        ),
        # Crate plugs the narrow neck.
        MovableObstacle(
            x=13.2, y=5.0, l=1.5, d=1.1, h=1.0, theta=math.pi / 2.0,
            material="wooden_crate",
            difficulty=_difficulty("wooden_crate", 1.5, 1.1, 1.0),
            oid=2,
        ),
        # Pallet starts parked in the far room.
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
