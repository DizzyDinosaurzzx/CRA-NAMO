"""感知与信念状态的维护和更新。"""

from __future__ import annotations
import math
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
        self.perceived: Dict[int, MovableObstacle] = {}     # 已感知障碍物的副本
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}    # 阻塞各边的障碍物
        self.newly_revealed: List[int] = []  # 新感知到的障碍物
        self.contacts: List[Polygon] = []  # 仅通过搬运碰撞得知的障碍物
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {}  # 触觉获得的障碍物真实难度
        self.touched_risk: Dict[int, str] = {}  # 触碰后评估过的次生风险档位，见 risk.py
        self.move_dir: Dict[int, Tuple[float, float]] = {}  # 各障碍物最近一次被移动的方向

    # --- 感知 ---
    def perceive(self, world_obstacles: List[MovableObstacle],
                 robot_pos: Tuple[float, float]) -> List[int]:
        """揭示机器人周围所有可见障碍物；已知障碍物同步位姿。"""
        self.newly_revealed = []
        rp = Point(robot_pos)
        for w in world_obstacles:
            known = self.perceived.get(w.oid)
            # 已知且位姿一致——无信息可同步，整体跳过昂贵的可见性检查
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
            # 障碍物已被完整感知，此前匿名的"接触"记录可清除
            self._clear_contacts_overlapping(obs.polygon)
        return self.newly_revealed

    @staticmethod
    def _pose_matches(a: MovableObstacle, b: MovableObstacle) -> bool:
        return (abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9
                and abs(a.theta - b.theta) < 1e-9)

    def _sync_pose(self, known: MovableObstacle, world_obs: MovableObstacle):
        """将感知副本位姿对齐到世界真实位姿。"""
        old_footprint = known.polygon
        self._forget_edges(known.oid)
        known.x, known.y, known.theta = world_obs.x, world_obs.y, world_obs.theta
        known.removed = world_obs.removed
        self._update_edges_for(known)
        self._clear_contacts_overlapping(old_footprint)

    # --- 可见性 ---
    def _half_edge_samples(self, obs: MovableObstacle):  # 取障碍物轮廓各半边上的视线采样点
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
        # 视线检查
        seg = LineString([robot_pos, p])
        width = self.cfg.sight_width
        sight = seg.buffer(width / 2.0, cap_style=2) if width > 0 else seg
        # 墙体遮挡：视线一旦离开静态自由空间（工作区减墙体）即被挡
        if not self.roadmap.static_free_prep.contains(sight):
            return False
        # 其他可移动障碍物遮挡：遮挡物不高于目标一半时目标上半部仍露出，不算遮挡
        for w in world_obstacles:
            if w.oid == target.oid or w.h <= target.h / 2.0 + 1e-9:
                continue
            if sight.intersects(w.polygon):
                return False
        return True

    def _visible(self, robot_pos, target: MovableObstacle,
                 world_obstacles: List[MovableObstacle]) -> bool:
        for p, c, q in self._half_edge_samples(target):
            if (self._point_visible(robot_pos, p, target, world_obstacles)
                    and self._point_visible(robot_pos, c, target, world_obstacles)
                    and self._point_visible(robot_pos, q, target, world_obstacles)):
                return True
        return False

    # --- 更新 ---
    def _update_edges_for(self, obs: MovableObstacle):
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

    # --- 碰撞接触 ---
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

    def force_reveal(self, world_obs: MovableObstacle) -> List[int]:  # 物理接触时揭示障碍物全部属性
        if world_obs.oid in self.perceived:
            return []
        obs = world_obs.perceived_copy()
        self.perceived[obs.oid] = obs
        self.newly_revealed.append(obs.oid)
        self._update_edges_for(obs)
        self._clear_contacts_overlapping(obs.polygon)
        return [obs.oid]

    def _forget_edges(self, oid: int):  # 清除该障碍物的边阻塞信息
        for blockers in self.edge_blockers.values():
            blockers.discard(oid)

    def record_move_direction(self, oid: int, from_xy, to_xy):
        """记录该障碍物最近被移动的方向（纯旋转保留旧方向）。"""
        dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
        n = math.hypot(dx, dy)
        if n > 1e-3:
            self.move_dir[oid] = (dx / n, dy / n)

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):  # 障碍物重新落位后更新阻塞信息
        self._forget_edges(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self._update_edges_for(obs)

    # --- 机器人自身碰撞 ---
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
    ) -> Tuple[List[int], float]:  # 检查机器人是否撞上障碍物
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
            if t <= t_hit + 1e-6:          # 同时命中的归为一组
                self.force_reveal(w)
                revealed.append(w.oid)
        return revealed, t_hit

    # --- 触觉感知 ---
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

    def reveal_by_interaction(self, oid: int,
                              world_obstacles: List[MovableObstacle]) -> bool:
        """任何物理交互（不只碰撞）都揭示真实难度：推动必然接触，搬过它就不可能还不知道它有多难推。"""
        if oid in self.touched:
            return False
        for w in world_obstacles:
            if w.oid == oid:
                self.touched.add(oid)
                self.touched_difficulty[oid] = w.difficulty
                return True
        return False

    def get_difficulty(self, oid: int, estimator) -> float:  # 返回正式难度，已知真实值则优先沿用
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    # --- 次生风险（见 risk.py） ---
    def assess_risk(self, oid: int, risk_estimator) -> str:
        """触碰后调用一次：评估并缓存该障碍物搬运的次生风险档位，同一障碍物只问
        一次 LLM。规划期从不主动调用——风险只在真正接触之后才有意义。"""
        if oid not in self.touched_risk:
            self.touched_risk[oid] = risk_estimator.assess(
                self.perceived[oid].observation())
        return self.touched_risk[oid]

    def get_risk_penalty(self, oid: int) -> float:
        """未触碰或功能关闭时视为零风险——规划期不能靠 LLM 瞎猜，只用已经问过的结果。"""
        if not self.cfg.risk_assessment_enabled:
            return 0.0
        tier = self.touched_risk.get(oid)
        return self.cfg.risk_tier_penalty.get(tier, 0.0) if tier else 0.0

    # --- 查询 ---
    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]

