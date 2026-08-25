"""Corridor scenario requiring a wedged crate to rotate in SE(2).

The crate nearly spans the hall, while staggered wall stubs prevent a pure
translation. The scenario exercises rotational planning, leverage-constrained
contact, and grip movement around the obstacle perimeter.
"""

from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle

_WIDTH = 16.0
_HEIGHT = 8.0
_SHELL_T = 0.5
_DEPTH = _HEIGHT - 2 * _SHELL_T

_CRATE_X = 7.0
_CRATE_L = 6.7
_CRATE_D = 0.7

# Staggered stubs block pure translation in either direction.
_STUB_DEPTH = 0.6
_STUB_WEST = (5.2, 6.4)
_STUB_EAST = (7.6, 8.8)

_START_X = 2.0
_GOAL_X = 13.0


def create():
    """Create the corridor scenario."""
    t = _SHELL_T
    mid_y = _HEIGHT / 2.0
    north = _HEIGHT - t

    walls = [
        StaticObstacle(box(0.0, 0.0, _WIDTH, t), "shell_south"),
        StaticObstacle(box(0.0, _HEIGHT - t, _WIDTH, _HEIGHT), "shell_north"),
        StaticObstacle(box(0.0, 0.0, t, _HEIGHT), "shell_west"),
        StaticObstacle(box(_WIDTH - t, 0.0, _WIDTH, _HEIGHT), "shell_east"),
        StaticObstacle(box(_STUB_WEST[0], north - _STUB_DEPTH,
                           _STUB_WEST[1], north), "stub_north"),
        StaticObstacle(box(_STUB_EAST[0], t,
                           _STUB_EAST[1], t + _STUB_DEPTH), "stub_south"),
    ]

    movable = [
        MovableObstacle(
            x=_CRATE_X, y=mid_y,
            l=_CRATE_L, d=_CRATE_D, h=1.0, theta=math.pi / 2.0,
            material="wooden_crate",
            difficulty=round(friction_force(material_mu_rho("wooden_crate"),
                                            _CRATE_L * _CRATE_D * 1.0), 3),
            oid=1,
        ),
    ]

    return {
        "workspace": box(0, 0, _WIDTH, _HEIGHT),
        "static": walls,
        "movable": movable,
        "start": (_START_X, mid_y),
        "goal": (_GOAL_X, mid_y),
        "cfg": Config(),
    }
