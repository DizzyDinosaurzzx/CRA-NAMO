"""Obstacle data structures"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from shapely.geometry import Polygon

import geometry

def _rect_polygon(x: float, y: float, l: float, d: float, theta: float) -> Polygon:
    """Shapely view of the same rectangle `geometry.rect_corners` returns.

    One implementation of "rectangle at a pose", two representations of it —
    numpy corners for the grid planner, a shapely polygon for everything that
    does set operations. They used to be two separate implementations that had to
    be kept in agreement by hand.
    """
    return Polygon(geometry.rect_corners(x, y, l, d, theta))

@dataclass
class StaticObstacle:
    """A wall, or any other piece of geometry the robot may never move.

    Stored as a polygon rather than as a pose, because the rest of the system
    only ever asks it for its outline — and because `se2_planner` builds its
    C-space with `inside_convex`, which means whatever shape is put in here has
    to be *convex*. A rotated rectangle still is, so a wall can carry a heading
    exactly like a `MovableObstacle` does.

    Build one with `rect` (centre + heading, mirroring `MovableObstacle`) or with
    `segment` (two endpoints + thickness, which is how a tilted wall is usually
    easiest to state); `theta` is then filled in for whoever wants to read the
    wall's direction back. Passing a polygon directly is still fine — an L-shaped
    or otherwise non-convex wall must be handed over as several convex pieces.
    """
    polygon: Polygon
    name: str = "wall"
    theta: Optional[float] = None   # heading of the long axis; None if built from a raw polygon

    def __post_init__(self):
        # se2_planner treats every wall as convex (it Minkowski-sums the outline
        # and tests the result with `inside_convex`), so a concave one would not
        # fail — it would quietly plan obstacle routes straight through its
        # notch. Cheaper to refuse it here than to debug that later.
        hull_area = self.polygon.convex_hull.area
        if hull_area - self.polygon.area > 1e-9 * max(1.0, hull_area):
            raise ValueError(
                f"static obstacle {self.name!r} is not convex; "
                "split it into convex pieces")

    @classmethod
    def rect(cls, x: float, y: float, l: float, d: float, theta: float = 0.0,
             name: str = "wall") -> "StaticObstacle":
        """Wall centred at (x, y): length *l* along *theta*, thickness *d* across it.

        Same parameterisation as `MovableObstacle`, so a wall and an obstacle at
        the same (x, y, l, d, theta) have the same footprint.
        """
        return cls(_rect_polygon(x, y, l, d, theta), name, float(theta))

    @classmethod
    def segment(cls, p: Tuple[float, float], q: Tuple[float, float],
                thickness: float, name: str = "wall") -> "StaticObstacle":
        """Wall running from *p* to *q*, *thickness* wide, centred on that line.

        The heading follows from the endpoints, so a slanted wall can be laid out
        by naming the two corners of the room it runs between rather than by
        working out its centre and angle by hand. The rectangle ends flush with
        *p* and *q*: butt two of them together and they meet with a notch on the
        outside of the corner, so overlap them if the join has to be sealed.
        """
        if thickness <= 0:
            raise ValueError("wall thickness must be positive")
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        if length < 1e-12:
            raise ValueError("wall endpoints coincide; a wall needs a direction")
        return cls.rect((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0,
                        length, thickness, math.atan2(dy, dx), name)

@dataclass
class MovableObstacle:
    x: float
    y: float
    l: float
    d: float
    h: float = 1.0                 # height, used for volume and occlusion reasoning
    theta: float = 0.0
    material: str = "unknown"      # semantic label for LLM reasoning
    difficulty: float = 1.0        # true sliding resistance f = mu*rho*V*g [N]; W = difficulty * distance moved [J]
    oid: int = -1

    # runtime flags
    removed: bool = False          # has been physically moved out of the way

    def __post_init__(self):
        if self.difficulty < 0:
            raise ValueError("difficulty must be non-negative")

    @property
    def polygon(self) -> Polygon:
        return _rect_polygon(self.x, self.y, self.l, self.d, self.theta)

    @property
    def area(self) -> float:
        return self.l * self.d

    @property
    def volume(self) -> float:
        return self.l * self.d * self.h

    def center(self):
        return (self.x, self.y)

    def perceived_copy(self) -> "MovableObstacle":
        """Return a copy for Belief tracking, with difficulty set to NaN"""
        return MovableObstacle(
            x=self.x, y=self.y, l=self.l, d=self.d, h=self.h, theta=self.theta,
            material=self.material, difficulty=math.nan, oid=self.oid,
        )

    def polygon_at(self, x: float, y: float, theta: Optional[float] = None) -> Polygon:
        return _rect_polygon(x, y, self.l, self.d,
                             self.theta if theta is None else theta)

    # --- observation ---
    def observation(self) -> dict:
        """Information revealed upon perception"""
        return {
            "oid": self.oid,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "l": self.l, "d": self.d, "h": self.h, "theta": round(self.theta, 3),
            "area": round(self.area, 2),
            "volume": round(self.volume, 2),
            "material": self.material,
        }

    def __repr__(self):
        return (f"Obs#{self.oid}({self.material}, c=({self.x:.1f},{self.y:.1f}), "
                f"{self.l}x{self.d}x{self.h}, diff={self.difficulty:,})")

