"""
感知与信念状态。

机器人只有在可移动障碍物进入半径为R_perc的感知圆且未被更近的障碍物遮挡时，才知道该障碍物的存在。
因此，重新放置障碍物会揭示其后方隐藏的物体。

信念状态按路网边维护当前已知的阻挡障碍物集合（即“付费解锁”边）。
未知或未探索的边不携带阻挡信息，搜索会乐观地将其视为畅通。
更新是增量式的：只重新评估通道与新发现或刚移动的障碍物相交的边。

"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from shapely.geometry import LineString, Point

from obstacle import MovableObstacle
from roadmap import Roadmap, EdgeKey
from config import Config


class Belief:
    def __init__(self, roadmap: Roadmap, cfg: Config):
        self.roadmap = roadmap
        self.cfg = cfg
        self.perceived: Dict[int, MovableObstacle] = {}     # oid -> 障碍物（已感知到的真实难度）
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}    # 边 -> 阻挡该边的障碍物 oid 集合
        self.newly_revealed: List[int] = []

    # -------------------- 感知 ----------------------
    def perceive(self, world_obstacles: List[MovableObstacle],
                 robot_pos: Tuple[float, float]) -> List[int]:
        """Reveal every visible, not-yet-known obstacle around `robot_pos`."""
        self.newly_revealed = []
        rp = Point(robot_pos)
        for w in world_obstacles:
            if w.oid in self.perceived:
                continue
            if rp.distance(Point(w.center())) > self.cfg.R_perc:
                continue
            if self._occluded(robot_pos, w, world_obstacles):
                continue
            self.perceived[w.oid] = w
            self.newly_revealed.append(w.oid)
            self._update_edges_for(w)
        return self.newly_revealed

    def _occluded(self, robot_pos, target: MovableObstacle,
                  world_obstacles: List[MovableObstacle]) -> bool:
        sight = LineString([robot_pos, target.center()])
        d_target = math.hypot(robot_pos[0] - target.x, robot_pos[1] - target.y)
        for w in world_obstacles:
            if w.oid == target.oid:
                continue
            d_w = math.hypot(robot_pos[0] - w.x, robot_pos[1] - w.y)
            if d_w < d_target and w.polygon.intersects(sight):
                return True
        return False

    # -------------------- 更新 ----------------------
    def _update_edges_for(self, obs: MovableObstacle):
        """Incrementally (re)compute which edges `obs` blocks."""
        poly = obs.polygon
        minx, miny, maxx, maxy = poly.bounds
        pad = 1.0
        for key, corridor in self.roadmap.edge_corridor.items():
            cminx, cminy, cmaxx, cmaxy = corridor.bounds
            if cmaxx < minx - pad or cminx > maxx + pad:      # 低成本的包围盒快速排除
                continue
            if cmaxy < miny - pad or cminy > maxy + pad:
                continue
            blockers = self.edge_blockers.setdefault(key, set())
            if corridor.intersects(poly):
                blockers.add(obs.oid)
            else:
                blockers.discard(obs.oid)

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):
        """Apply an executed push: move the obstacle and refresh its blocked edges."""
        # 删除它原有的阻挡关系
        for blockers in self.edge_blockers.values():
            blockers.discard(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self._update_edges_for(obs)     # 无反效果 => 新的阻挡集是原阻挡集的子集

    # -------------------- 查询 ----------------------
    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]
