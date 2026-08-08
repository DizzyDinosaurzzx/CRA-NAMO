"""See-over-a-low-obstacle demo"""

from __future__ import annotations
from shapely.geometry import box
from config import Config
from obstacle import MovableObstacle, StaticObstacle

def create():
    workspace = box(0, 0, 20, 16)

    t = 0.4
    # Three chambers stacked bottom-to-top, joined by two doorways. Both doorways
    # are wide enough that every line of sight from the start to either obstacle
    # stays clear of them — the low obstacle remains the only thing standing
    # between the robot and oid 2.
    walls = [
        StaticObstacle(box(0.0, 0.0, 20.0, t), "outer_bottom"),
        StaticObstacle(box(0.0, 16.0 - t, 20.0, 16.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, t, 16.0), "outer_left"),
        StaticObstacle(box(20.0 - t, 0.0, 20.0, 16.0), "outer_right"),

        # entrance divider — doorway x in [7.5, 12.5], straight ahead of the start
        StaticObstacle(box(t, 4.0, 7.5, 4.0 + t), "entrance_left"),
        StaticObstacle(box(12.5, 4.0, 20.0 - t, 4.0 + t), "entrance_right"),

        # flanks level with oid 1, closing the hall off on both sides so oid 1 is
        # the only gate: the 0.4 m slivers left over beside it are narrower than
        # the robot, so reaching the goal means pushing it out of the way.
        StaticObstacle(box(t, 6.5, 6.6, 7.5), "gate_flank_left"),
        StaticObstacle(box(13.4, 6.5, 20.0 - t, 7.5), "gate_flank_right"),

        # alcove stubs, far enough out that no sight line reaches them
        StaticObstacle(box(t, 9.0, 3.5, 9.0 + t), "alcove_left"),
        StaticObstacle(box(16.5, 9.0, 20.0 - t, 9.0 + t), "alcove_right"),

        # exit divider — doorway x in [6.0, 11.0], offset so the robot has to
        # steer around oid 2 rather than walk straight at the goal
        StaticObstacle(box(t, 12.5, 6.0, 12.5 + t), "exit_left"),
        StaticObstacle(box(11.0, 12.5, 20.0 - t, 12.5 + t), "exit_right"),
    ]

    movable = [
        # front, low and wide — casts a full geometric shadow over oid 2
        MovableObstacle(
            x=10.0,
            y=7.0,
            l=6.0,
            d=1.0,
            h=0.5,
            theta=0.0,
            material="pallet",
            difficulty=2048.328,
            oid=1,
        ),
        # behind, tall and narrow — visible over the top of oid 1
        MovableObstacle(
            x=10.0,
            y=10.5,
            l=3.0,
            d=1.0,
            h=2.0,
            theta=0.0,
            material="steel_shelf",
            difficulty=8829.0,
            oid=2,
        ),
    ]

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (10.0, 2.0),
        "goal": (10.0, 14.0),
        "cfg": Config(),
    }
