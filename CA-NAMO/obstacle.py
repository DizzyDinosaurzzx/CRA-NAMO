"""
障碍物数据结构。

问题中包含两类障碍物（见问题定义）：

* StaticObstacle  - 不可移动的墙体/柱体（集合 L）。预先已知；用于一次性构建路网。
* MovableObstacle - 集合 M。位置、尺寸、*材质*和真实的*难度*系数在障碍物进入
                    感知圆之前对机器人全部未知。

`difficulty` 是单位推动距离功的真实系数（可理解为质量 x 摩擦力）。几何计算器返回的
实际操作功为
    work = difficulty * push_distance
LLM 永远看不到该数值；它只能看到 `material`/`description`，并预测一个用于*排列*
搜索顺序的难度。
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
    """由 shapely Polygon 直接描述的不可移动障碍物。"""
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
    difficulty: float = 1.0        # 真实的单位距离操作功系数
    oid: int = -1                  # 由场景/世界分配的唯一 ID

    # 运行时标志
    removed: bool = False          # 已经被实际移到不碍事的位置

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
            material=self.material, difficulty=self.difficulty, oid=self.oid,
        )

    def polygon_at(self, x: float, y: float, theta: Optional[float] = None) -> Polygon:
        return _rect_polygon(x, y, self.l, self.d,
                             self.theta if theta is None else theta)

    # ---- 障碍物被感知后机器人允许看到的信息 ----
    def observation(self) -> dict:
        """感知时揭示的信息：几何信息和材质标签。

        有意排除真实的 `difficulty`；机器人必须根据可见特征使用 LLM 或启发式方法估计。
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
