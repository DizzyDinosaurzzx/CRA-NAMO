"""感知与信念状态维护与更新"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from shapely.geometry import LineString, Point, Polygon
from obstacle import MovableObstacle
from roadmap import Roadmap, EdgeKey
from config import Config


class Belief:
    def __init__(self, roadmap: Roadmap, cfg: Config):
        self.roadmap = roadmap
        self.cfg = cfg
        self.perceived: Dict[int, MovableObstacle] = {}     # oid -> 已感知障碍物（含直接给定的 W）
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}    # 边 -> 阻挡该边的障碍物 oid 集合
        self.newly_revealed: List[int] = []
        # 通过"推动碰撞"发现的匿名占据区域：只知道"这里有东西"，不知其身份/几何/难度。
        # 仅用作放置/推动的硬约束；无法据此规划"移除"它（不知道怎么移、多难移）。
        self.contacts: List[Polygon] = []
        # 需求4: 已触摸获知真实难度的障碍物
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {}

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
            if not self._visible(robot_pos, w, world_obstacles):
                continue
            self.perceived[w.oid] = w
            self.newly_revealed.append(w.oid)
            self._update_edges_for(w)
            # 该物体现在被完整感知，之前对它的匿名"接触"记录可以清除（由真实 footprint 取代）
            self._clear_contacts_overlapping(w.polygon)
        return self.newly_revealed

    # -------------------- 可见性（多点采样 + 墙体遮挡） ----------------------
    def _half_edge_samples(self, obs: MovableObstacle):
        """矩形的八条半边，每条半边用 (端点, 中点, 端点) 三个采样点表示。
        半边 = 一条边从角点到边中点的那一半。
        只要任意一条半边完全可见，就认为机器人看清了该障碍物。
        """
        coords = list(obs.polygon.exterior.coords)[:-1]   # 4 个角（去掉闭合重复点）
        n = len(coords)
        halves = []
        for i in range(n):
            a = coords[i]
            b = coords[(i + 1) % n]
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            for p, q in ((a, mid), (mid, b)):
                c = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
                halves.append((p, c, q))
        return halves

    def _point_visible(self, robot_pos, p,
                       target: MovableObstacle,
                       world_obstacles: List[MovableObstacle]) -> bool:
        """从机器人到采样点 p 的视线是否畅通（不被墙体或其他障碍物遮挡）
        cfg.sight_width>0 时把视线视为具有该宽度的走廊（缓冲后判断），窄于该宽度的缝隙无法看穿；为 0 时退化为零宽度射线。
        """
        seg = LineString([robot_pos, p])
        width = self.cfg.sight_width
        sight = seg.buffer(width / 2.0, cap_style=2) if width > 0 else seg
        # 墙体遮挡：视线一旦离开静态自由空间（= 工作空间挖去墙体）即被墙挡住
        if not self.roadmap.static_free.contains(sight):
            return False
        # 其他可移动障碍物遮挡
        for w in world_obstacles:
            if w.oid == target.oid:
                continue
            if sight.intersects(w.polygon):
                return False
        return True

    def _visible(self, robot_pos, target: MovableObstacle,
                 world_obstacles: List[MovableObstacle]) -> bool:
        """需求1+2: 至少半条边完整可见即认为感知到该障碍物的全部信息。"""
        for p, c, q in self._half_edge_samples(target):
            if (self._point_visible(robot_pos, p, target, world_obstacles)
                    and self._point_visible(robot_pos, c, target, world_obstacles)
                    and self._point_visible(robot_pos, q, target, world_obstacles)):
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

    # -------------------- 碰撞接触（部分信息） ----------------------
    def register_contact(self, region: Polygon):
        """记录一块通过推动碰撞发现的匿名占据区域（不含身份/难度信息）。
        先扣掉已感知障碍物的 footprint：那部分不是"新信息"，而且此后不会再被
        `_clear_contacts_overlapping` 清理（该障碍物不会二次 newly_revealed）
        会退化成一块永久压在它身上的幽灵区域，把它自己锁成不可推动。
        """
        if region is None or region.is_empty:
            return
        for obs in self.perceived.values():
            region = region.difference(obs.polygon)
            if region.is_empty:
                return
        if region.area <= 1e-9:      # 只剩浮点碎片
            return
        self.contacts.append(region)

    def _clear_contacts_overlapping(self, poly: Polygon):
        """某障碍物被完整感知后，删除与其重叠的匿名接触记录（已被真实 footprint 取代）。"""
        if not self.contacts:
            return
        self.contacts = [c for c in self.contacts if not c.intersects(poly)]

    def force_reveal(self, obs: MovableObstacle) -> List[int]:
        """物理接触导致的强制揭示：无视感知半径与遮挡，直接把障碍物纳入信念。

        用于机器人推动某障碍物时撞上一个尚未感知的物体——碰撞本身就暴露了它。
        """
        if obs.oid in self.perceived:
            return []
        self.perceived[obs.oid] = obs
        self.newly_revealed.append(obs.oid)
        self._update_edges_for(obs)
        return [obs.oid]

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):
        """Apply an executed push: move the obstacle and refresh its blocked edges."""
        # 删除它原有的阻挡关系
        for blockers in self.edge_blockers.values():
            blockers.discard(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self._update_edges_for(obs)     # 无反效果 => 新的阻挡集是原阻挡集的子集

    # -------------------- 需求3: 机器人自身碰撞感知 ----------------------
    def check_robot_collision(
        self, from_pos, to_pos,
        world_obstacles: List[MovableObstacle],
        cfg: Config,
    ) -> List[int]:
        """检查机器人移动是否撞到未感知障碍物，撞到则 force_reveal。"""
        corridor = LineString([from_pos, to_pos]).buffer(cfg.robot_radius, cap_style=1)
        revealed: List[int] = []
        for w in world_obstacles:
            if w.oid in self.perceived:
                continue
            if corridor.intersects(w.polygon):
                self.force_reveal(w)
                revealed.append(w.oid)
        return revealed

    # -------------------- 需求4: 触摸感知难度 ----------------------
    def touch_check(
        self, robot_pos,
        world_obstacles: List[MovableObstacle],
        cfg: Config,
    ) -> List[int]:
        """触摸圆接触障碍物则揭示真实 difficulty。"""
        touch_radius = cfg.robot_radius + cfg.touch_margin
        rp = Point(robot_pos)
        revealed: List[int] = []
        for w in world_obstacles:
            if w.oid in self.touched:
                continue
            if rp.distance(Point(w.center())) > touch_radius + max(w.l, w.d):
                continue
            if rp.buffer(touch_radius).intersects(w.polygon):
                self.touched.add(w.oid)
                self.touched_difficulty[w.oid] = w.difficulty
                revealed.append(w.oid)
        return revealed

    def get_difficulty(self, oid: int, estimator) -> float:
        """搜索阶段获取难度: 触摸过用真实值，否则用 LLM/启发式估计。"""
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    # -------------------- 查询 ----------------------
    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]
