"""Obstacle data structures"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
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
    polygon: Polygon
    name: str = "wall"

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
                f"{self.l}x{self.d}x{self.h}, diff={self.difficulty})")

