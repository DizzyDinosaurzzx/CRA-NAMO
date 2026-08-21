"""障碍物数据结构定义。"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from shapely.geometry import Polygon

import geometry

def _rect_polygon(x: float, y: float, l: float, d: float, theta: float) -> Polygon:
    """由 geometry.rect_corners 生成矩形角点，再包装成 shapely 多边形；一处实现两种表示。"""
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
    h: float = 1.0                 # 高度，用于体积与遮挡推理
    theta: float = 0.0
    material: str = "unknown"      # 材质语义标签，供 LLM 推理
    difficulty: float = 1.0        # 真实滑动阻力 f = mu*rho*V*g [N]；做功 W = difficulty * 移动距离 [J]
    oid: int = -1

    # 运行时状态
    removed: bool = False          # 是否已被实际挪开

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
        """返回用于信念追踪的副本，difficulty 置为 NaN。"""
        return MovableObstacle(
            x=self.x, y=self.y, l=self.l, d=self.d, h=self.h, theta=self.theta,
            material=self.material, difficulty=math.nan, oid=self.oid,
        )

    def polygon_at(self, x: float, y: float, theta: Optional[float] = None) -> Polygon:
        return _rect_polygon(x, y, self.l, self.d,
                             self.theta if theta is None else theta)

    # --- 观测 ---
    def observation(self) -> dict:
        """感知时对外暴露的障碍物信息。"""
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

