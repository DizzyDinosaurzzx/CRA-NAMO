"""感知与信念状态维护与更新"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from obstacle import MovableObstacle
from roadmap import Roadmap, EdgeKey
from config import Config


class Belief:
    def __init__(self, roadmap: Roadmap, cfg: Config):
        self.roadmap = roadmap
        self.cfg = cfg
        self.perceived: Dict[int, MovableObstacle] = {}     # 已感知到的障碍物的副本
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}    # 阻挡边的障碍物
        self.newly_revealed: List[int] = [] # 新感知到的障碍物
        self.contacts: List[Polygon] = [] # 通过推动碰撞知道的障碍物
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {} # 已获取障碍物的真的移动难度

    # -------------------- 感知 ----------------------
    def perceive(self, world_obstacles: List[MovableObstacle],
                 robot_pos: Tuple[float, float]) -> List[int]:
        """展示机器人周围所有可见障碍物；已知的同步状态"""
        self.newly_revealed = []
        rp = Point(robot_pos)
        for w in world_obstacles:
            known = self.perceived.get(w.oid)
            # 已知且位姿一致 —— 没有任何可同步的信息，直接跳过昂贵的可见性判定
            if known is not None and self._pose_matches(known, w):
                continue
            if rp.distance(Point(w.center())) > self.cfg.R_perc:
                continue
            if not self._visible(robot_pos, w, world_obstacles):
                continue
            if known is not None:
                self._sync_pose(known, w)
                continue
            obs = w.perceived_copy()
            self.perceived[w.oid] = obs
            self.newly_revealed.append(w.oid)
            self._update_edges_for(obs)
            # 该物体现在被完整感知，之前对它的匿名"接触"记录可以清除
            self._clear_contacts_overlapping(obs.polygon)
        return self.newly_revealed

    @staticmethod
    def _pose_matches(a: MovableObstacle, b: MovableObstacle) -> bool:
        return (abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9
                and abs(a.theta - b.theta) < 1e-9)

    def _sync_pose(self, known: MovableObstacle, world_obs: MovableObstacle):
        """把已感知副本的位姿对齐到世界真实位姿"""
        old_footprint = known.polygon
        self._forget_edges(known.oid)
        known.x, known.y, known.theta = world_obs.x, world_obs.y, world_obs.theta
        known.removed = world_obs.removed
        self._update_edges_for(known)
        self._clear_contacts_overlapping(old_footprint)

    # -------------------- 可见性（多点采样 + 墙体遮挡） ----------------------
    def _half_edge_samples(self, obs: MovableObstacle): # 障碍物8个点的视线
        coords = list(obs.polygon.exterior.coords)[:-1]  
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
        # 视线检测
        seg = LineString([robot_pos, p])
        width = self.cfg.sight_width
        sight = seg.buffer(width / 2.0, cap_style=2) if width > 0 else seg
        # 墙体遮挡：视线一旦离开静态自由空间（= 工作空间挖去墙体）即被墙挡住
        if not self.roadmap.static_free_prep.contains(sight):
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
        #返回是否真的能看见
        for p, c, q in self._half_edge_samples(target):
            if (self._point_visible(robot_pos, p, target, world_obstacles)
                    and self._point_visible(robot_pos, c, target, world_obstacles)
                    and self._point_visible(robot_pos, q, target, world_obstacles)):
                return True
        return False

    # -------------------- 更新 ----------------------
    def _update_edges_for(self, obs: MovableObstacle):
        # 更新这个障碍物挡住了哪几个路线
        poly = obs.polygon
        minx, miny, maxx, maxy = poly.bounds
        pad = 1.0
        for key, corridor in self.roadmap.edge_corridor.items():
            cminx, cminy, cmaxx, cmaxy = corridor.bounds
            if cmaxx < minx - pad or cminx > maxx + pad:      
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
        if region is None or region.is_empty:
            return
        for obs in self.perceived.values():
            region = region.difference(obs.polygon)
            if region.is_empty:
                return
        if region.area <= 1e-9:      # 只剩浮点碎片
            return
        merged = [region]
        rest = []
        for c in self.contacts:
            (merged if c.intersects(region) else rest).append(c)
        rest.append(unary_union(merged) if len(merged) > 1 else region)
        self.contacts = rest

    def _clear_contacts_overlapping(self, poly: Polygon):
        if not self.contacts:
            return
        self.contacts = [c for c in self.contacts if not c.intersects(poly)]

    def force_reveal(self, world_obs: MovableObstacle) -> List[int]: # 物理接触的障碍物性质显示
        if world_obs.oid in self.perceived:
            return []
        obs = world_obs.perceived_copy()
        self.perceived[obs.oid] = obs
        self.newly_revealed.append(obs.oid)
        self._update_edges_for(obs)
        self._clear_contacts_overlapping(obs.polygon)
        return [obs.oid]

    def _forget_edges(self, oid: int): # 清除障碍物的阻挡信息
        for blockers in self.edge_blockers.values():
            blockers.discard(oid)

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float): # 更新新的障碍物阻挡信息
        self._forget_edges(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self._update_edges_for(obs)

    # -------------------- 需求3: 机器人自身碰撞感知 ----------------------
    @staticmethod
    def _first_contact_t(from_pos, to_pos, poly: Polygon, radius: float,
                         coarse: int = 64, refine: int = 20) -> float:
        fx, fy = from_pos
        dx, dy = to_pos[0] - fx, to_pos[1] - fy

        def touching(t: float) -> bool:
            return poly.distance(Point(fx + dx * t, fy + dy * t)) <= radius

        if touching(0.0):
            return 0.0
        lo = 0.0
        for i in range(1, coarse + 1):
            t = i / coarse
            if touching(t):
                hi = t
                for _ in range(refine):
                    mid = 0.5 * (lo + hi)
                    if touching(mid):
                        hi = mid
                    else:
                        lo = mid
                return hi
            lo = t
        return 1.0

    def check_robot_collision(
        self, from_pos, to_pos,
        world_obstacles: List[MovableObstacle],
        cfg: Config,
    ) -> Tuple[List[int], float]: #检查机器人有没有撞上障碍物
        corridor = LineString([from_pos, to_pos]).buffer(cfg.robot_radius, cap_style=1)
        candidates = [w for w in world_obstacles
                      if w.oid not in self.perceived
                      and corridor.intersects(w.polygon)]
        if not candidates:
            return [], 1.0

        ts = [self._first_contact_t(from_pos, to_pos, w.polygon, cfg.robot_radius)
              for w in candidates]
        t_hit = min(ts)
        revealed: List[int] = []
        for w, t in zip(candidates, ts):
            if t <= t_hit + 1e-6:          # 同时撞上的算一起
                self.force_reveal(w)
                revealed.append(w.oid)
        return revealed, t_hit

    # -------------------- 需求4: 触摸感知真实难度 ----------------------
    def touch_check(
        self, robot_pos,
        world_obstacles: List[MovableObstacle],
        cfg: Config,
    ) -> List[int]:
        touch_radius = cfg.robot_radius
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

    def get_difficulty(self, oid: int, estimator) -> float:  #保存之前已知的正式的移动难度
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    # -------------------- 查询 ----------------------
    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]
