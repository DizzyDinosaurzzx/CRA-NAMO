"""定义静态和可移动障碍物数据结构。"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from shapely.geometry import Polygon

import geometry

def _rect_polygon(x: float, y: float, l: float, d: float, theta: float) -> Polygon:
    """将共用的矩形表示转换为 Shapely 多边形。"""
    return Polygon(geometry.rect_corners(x, y, l, d, theta))


@dataclass
class StaticObstacle:
    """机器人不可搬移的凸几何体。"""
    polygon: Polygon
    name: str = "wall"
    theta: Optional[float] = None   # 长轴航向；原始多边形使用 None。

    def __post_init__(self):
        # SE(2) 碰撞检查要求静态多边形为凸多边形。
        hull_area = self.polygon.convex_hull.area
        if hull_area - self.polygon.area > 1e-9 * max(1.0, hull_area):
            raise ValueError(
                f"static obstacle {self.name!r} is not convex; "
                "split it into convex pieces")

    @classmethod
    def rect(cls, x: float, y: float, l: float, d: float, theta: float = 0.0,
             name: str = "wall") -> "StaticObstacle":
        """根据中心、尺寸和航向创建墙体。"""
        return cls(_rect_polygon(x, y, l, d, theta), name, float(theta))

    @classmethod
    def segment(cls, p: Tuple[float, float], q: Tuple[float, float],
                thickness: float, name: str = "wall") -> "StaticObstacle":
        """创建中心位于线段上的墙体。"""
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
    h: float = 1.0                 # 用于计算体积和遮挡的高度。
    theta: float = 0.0
    material: str = "unknown"      # 估计器使用的语义标签。
    difficulty: float = 1.0        # 真实滑动阻力，单位为牛。
    # 接触时可选的揭示标签。
    contact_reveals: str = ""
    # 与其耦合、搬移时需要共同考虑风险的物体。
    interacts_with: Tuple[int, ...] = ()
    interaction_risk: str = ""
    oid: int = -1

    removed: bool = False          # 障碍物是否已经搬移。

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
        """返回难度未知的 belief 状态副本。"""
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
        """返回视觉感知可获得的信息。"""
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
        """返回视觉信息及仅接触后可获得的属性。"""
        obs = self.observation()
        if self.contact_reveals:
            obs["contact_reveals"] = self.contact_reveals
        return obs

    def __repr__(self):
        return (f"Obs#{self.oid}({self.material}, c=({self.x:.1f},{self.y:.1f}), "
                f"{self.l}x{self.d}x{self.h}, diff={self.difficulty:,})")
