"""碍物数据结构"""

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
    theta: float = 0.0
    material: str = "unknown"      # LLM 用于推理的语义标签
    difficulty: float = 1.0        # 真实的单位推动距离做功系数
    oid: int = -1                  

    # 运行时标志
    removed: bool = False          # 已经被实际移到不碍事的位置

    def __post_init__(self):
        if self.difficulty < 0:
            raise ValueError("difficulty 必须为非负数")

    @property
    def polygon(self) -> Polygon:
        return _rect_polygon(self.x, self.y, self.l, self.d, self.theta)

    @property
    def area(self) -> float:
        return self.l * self.d

    def center(self):
        return (self.x, self.y)

    def perceived_copy(self) -> "MovableObstacle":
        """复制一份供 Belief 持有,difficulty为NaN"""
        return MovableObstacle(
            x=self.x, y=self.y, l=self.l, d=self.d, theta=self.theta,
            material=self.material, difficulty=math.nan, oid=self.oid,
        )

    def polygon_at(self, x: float, y: float, theta: Optional[float] = None) -> Polygon:
        return _rect_polygon(x, y, self.l, self.d,
                             self.theta if theta is None else theta)

    # ---- 障碍物被感知后机器人允许看到的信息 ----
    def observation(self) -> dict:
        """感知时揭示的信息"""
        return {
            "oid": self.oid,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "l": self.l, "d": self.d, "theta": round(self.theta, 3),
            "area": round(self.area, 2),
            "material": self.material,
        }

    def __repr__(self):
        return (f"Obs#{self.oid}({self.material}, c=({self.x:.1f},{self.y:.1f}), "
                f"{self.l}x{self.d}, diff={self.difficulty})")
