"""
障碍物数据结构
共有两类障碍物：
1. StaticObstacle  - 不可移动的墙体/柱体
2. MovableObstacle - 位置、尺寸、材质和难度系数在障碍物进入感知圆之前对机器人全部未知。

推开障碍物的做功由难度系数乘以实际推动距离得到： W = difficulty * push_distance
`difficulty` 是仿真世界的 ground truth，机器人只能通过触碰获知真实值，在此之前只能依据 `material` 与几何尺寸由 LLM/启发式估计。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate


def _rect_polygon(x: float, y: float, l: float, d: float, theta: float) -> Polygon:
    """创建以 (x, y) 为中心、尺寸为 l x d 并旋转 theta 的矩形。"""
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
    difficulty: float = 1.0        # 真实的单位推动距离做功系数（ground truth）
    oid: int = -1                  # 由场景/世界分配的唯一 ID

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

    def moved_copy(self, x: float, y: float, theta: Optional[float] = None) -> "MovableObstacle":
        """复制一个移动到新位姿的障碍物（用于候选放置位置）。"""
        return MovableObstacle(
            x=x, y=y, l=self.l, d=self.d,
            theta=self.theta if theta is None else theta,
            material=self.material, difficulty=self.difficulty,
            oid=self.oid,
        )

    def polygon_at(self, x: float, y: float, theta: Optional[float] = None) -> Polygon:
        return _rect_polygon(x, y, self.l, self.d,
                             self.theta if theta is None else theta)

    # ---- 障碍物被感知后机器人允许看到的信息 ----
    def observation(self) -> dict:
        """感知时揭示的信息：几何信息与材质标签。

        `difficulty` 有意排除——它是 ground truth，机器人只能估计或触碰获知。
        """
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
