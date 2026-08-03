"""Obstacle data structures"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
from shapely.geometry import Polygon

def _rect_polygon(x: float, y: float, l: float, d: float, theta: float) -> Polygon:
    hl, hd = l / 2.0, d / 2.0
    verts = np.array([[hl, hd], [hl, -hd], [-hl, -hd], [-hl, hd]])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T + np.array([x, y])
    return Polygon(verts)

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
    difficulty: float = 1.0        # true work coefficient per unit push distance
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

    # ---- Information available to the robot once the obstacle is perceived ----
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
