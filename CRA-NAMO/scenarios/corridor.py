"""One crate, wedged, and the only way past it is to work it round.

A hall twice as wide as it is deep, walled all the way round, with the goal
straight ahead on the far side of the one thing in the way. There is no route
around and nothing else to decide: the whole map is the manoeuvre.

Across the hall stands a crate 6.7 m long, with 15 cm to spare at each end. It
cannot be squeezed past, and it cannot be shoved aside either — two stubs jut
from the walls, one from the ceiling just west of it and one from the floor just
east, staggered so that whichever way it is pushed one of them is in the road.
Sliding is what a planner reaches for first, because it is cheap; here it simply
is not available.

What is left is to turn it, and the hall has just enough room: the crate sweeps
6.74 m coming round and the hall is 7.00 m deep. Because the goal lies straight
beyond the crate rather than off to one side, turning alone will not do either —
every metre the robot gains is a metre the crate has to be taken as well. So the
two go together, and the run works out as nine manipulations that alternate
between them: turn 15 degrees to bring an end clear of a stub, push half a metre
into the room that makes, turn again, push again. By the end the crate has
travelled 2.6 m up the hall and come round 45 degrees, and the robot has re-set
its grip nine times to do it.

    hall depth          7.00 m
    crate standing      6.70 m   0.15 m clear at each end, so nothing gets past
    crate turning       6.74 m   fits the hall, but not between the stubs
    crate at 45 deg     5.23 m   leaves a gap, and the robot walks through

Fifteen degrees is the planner's angular step, and 45 is where it stops — the
first heading that opens a gap wide enough to drive through. Nothing about that
is chosen by hand; it falls out of pricing rotation against translation and
buying the cheaper one at every step. `se2_planner` finds the sequence, and
`contact` keeps the robot flush against the crate throughout, walking its grip
round the perimeter as the crate turns under it.
"""

from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle

_WIDTH = 16.0                    # map, long side [m]
_HEIGHT = 8.0                    # map, short side [m] — 2:1
_SHELL_T = 0.5                   # the wall that runs all the way round
_DEPTH = _HEIGHT - 2 * _SHELL_T  # 7.0 m of hall between the north and south walls

_CRATE_X = 7.0                   # where the crate stands
_CRATE_L = 6.7                   # across the hall — 0.15 m clear at each end
_CRATE_D = 0.7                   # along it — keeps the turning sweep inside 7 m

# The two stubs that wedge it. Staggered: one drops from the north wall west of
# the crate, one rises from the south wall east of it, and each reaches far
# enough across to foul the crate whichever way it is pushed. Both sit inside the
# sweep the crate makes when it turns, which is what stops it being turned on the
# spot without being edged along at the same time.
_STUB_DEPTH = 0.6
_STUB_WEST = (5.2, 6.4)          # x range of the one hanging from the north wall
_STUB_EAST = (7.6, 8.8)          # x range of the one standing on the south wall

_START_X = 2.0                   # this side of the crate
_GOAL_X = 13.0                   # and the far side of it


def create():
    t = _SHELL_T
    mid_y = _HEIGHT / 2.0                        # 4.0, centre of the hall
    north = _HEIGHT - t                          # 7.5, inner face of the north wall

    walls = [
        StaticObstacle(box(0.0, 0.0, _WIDTH, t), "shell_south"),
        StaticObstacle(box(0.0, _HEIGHT - t, _WIDTH, _HEIGHT), "shell_north"),
        StaticObstacle(box(0.0, 0.0, t, _HEIGHT), "shell_west"),
        StaticObstacle(box(_WIDTH - t, 0.0, _WIDTH, _HEIGHT), "shell_east"),
        # the stubs that leave turning as the only way out
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
