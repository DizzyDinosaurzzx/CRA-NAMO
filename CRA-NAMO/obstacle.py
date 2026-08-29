"""Define static and movable obstacle data structures."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from shapely.geometry import Polygon

import geometry

def _rect_polygon(x: float, y: float, l: float, d: float, theta: float) -> Polygon:
    """Convert the shared rectangle representation to a Shapely polygon."""
    return Polygon(geometry.rect_corners(x, y, l, d, theta))


@dataclass
class StaticObstacle:
    """Convex geometry that the robot cannot move."""
    polygon: Polygon
    name: str = "wall"
    theta: Optional[float] = None   # Long-axis heading, or None for raw polygons.

    def __post_init__(self):
        # SE(2) collision checks require convex static polygons.
        hull_area = self.polygon.convex_hull.area
        if hull_area - self.polygon.area > 1e-9 * max(1.0, hull_area):
            raise ValueError(
                f"static obstacle {self.name!r} is not convex; "
                "split it into convex pieces")

    @classmethod
    def rect(cls, x: float, y: float, l: float, d: float, theta: float = 0.0,
             name: str = "wall") -> "StaticObstacle":
        """Create a wall from its center, dimensions, and heading."""
        return cls(_rect_polygon(x, y, l, d, theta), name, float(theta))

    @classmethod
    def segment(cls, p: Tuple[float, float], q: Tuple[float, float],
                thickness: float, name: str = "wall") -> "StaticObstacle":
        """Create a wall centered on a line segment."""
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
    h: float = 1.0                 # Height used for volume and occlusion.
    theta: float = 0.0
    material: str = "unknown"      # Semantic label used by estimators.
    difficulty: float = 1.0        # True sliding resistance in newtons.
    # Optional label revealed by contact.
    contact_reveals: str = ""
    # Coupled bodies whose risk must be considered when this one moves.
    interacts_with: Tuple[int, ...] = ()
    interaction_risk: str = ""
    oid: int = -1

    removed: bool = False          # Whether the obstacle has been moved.

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
        """Return a belief-state copy with unknown difficulty."""
        return MovableObstacle(
            x=self.x, y=self.y, l=self.l, d=self.d, h=self.h, theta=self.theta,
            material=self.material, difficulty=math.nan, oid=self.oid,
            interacts_with=tuple(self.interacts_with),
            interaction_risk=self.interaction_risk,
        )

    def polygon_at(self, x: float, y: float, theta: Optional[float] = None) -> Polygon:
        return _rect_polygon(x, y, self.l, self.d,
                             self.theta if theta is None else theta)

    def observation(self) -> dict:
        """Return information available through visual perception."""
        o = {
            "oid": self.oid,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "l": self.l, "d": self.d, "h": self.h, "theta": round(self.theta, 3),
            "area": round(self.area, 2),
            "volume": round(self.volume, 2),
            "material": self.material,
        }
        if self.interacts_with:
            o["interacts_with"] = tuple(self.interacts_with)
        if self.interaction_risk:
            o["interaction_risk"] = self.interaction_risk
        return o

    def contact_observation(self) -> dict:
        """Return visual information plus contact-only properties."""
        obs = self.observation()
        if self.contact_reveals:
            obs["contact_reveals"] = self.contact_reveals
        return obs

    def __repr__(self):
        return (f"Obs#{self.oid}({self.material}, c=({self.x:.1f},{self.y:.1f}), "
                f"{self.l}x{self.d}x{self.h}, diff={self.difficulty:,})")
