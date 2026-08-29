"""Maintain the robot's partially observed world model."""

from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from obstacle import MovableObstacle
from roadmap import Roadmap, EdgeKey
from config import Config


class Belief:
    """Store only information the robot has perceived or discovered by contact."""

    def __init__(self, roadmap: Roadmap, cfg: Config, risk_estimator=None,
                 estimator=None):
        self.roadmap = roadmap
        self.cfg = cfg
        self.risk = risk_estimator
        self.estimator = estimator
        self.perceived: Dict[int, MovableObstacle] = {}
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}
        self.newly_revealed: List[int] = []
        self.updated: List[int] = []
        self.contacts: List[Polygon] = []
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {}
        self.disturbed: Set[int] = set()
        self.seen_moving: Set[int] = set()
        self.move_dir: Dict[int, Tuple[float, float]] = {}
        # Planner caches include this version to prevent stale route reuse.
        self.version = 0

    def _bump(self):
        self.version += 1

    def perceive(self, world_obstacles: List[MovableObstacle],
                 robot_pos: Tuple[float, float]) -> List[int]:
        """Reveal visible obstacles and synchronize known ones."""
        self.newly_revealed = []
        self.updated = []
        new_observations = []
        rp = Point(robot_pos)
        for w in world_obstacles:
            known = self.perceived.get(w.oid)
            if known is not None and self._matches(known, w):
                continue
            if rp.distance(Point(w.center())) > self.cfg.R_perc:
                continue
            if not self._visible(robot_pos, w, world_obstacles):
                continue
            if known is not None:
                self._sync(known, w)
                continue
            obs = w.perceived_copy()
            self.perceived[w.oid] = obs
            self.newly_revealed.append(w.oid)
            new_observations.append(obs.observation())
            self._update_edges_for(obs)
            self._clear_contacts_overlapping(obs.polygon)
        if self.risk is not None and new_observations:
            self.risk.assess_many(new_observations)
        self._forget_vacated(world_obstacles, robot_pos)
        if self.changed:
            self._bump()
        return self.newly_revealed

    def is_stale(self, world_obs: MovableObstacle) -> bool:
        """Return whether a perceived body no longer matches reality."""
        known = self.perceived.get(world_obs.oid)
        return known is not None and not self._matches(known, world_obs)

    @property
    def changed(self) -> bool:
        """Has anything the planner cached its work against moved or changed?"""
        return bool(self.newly_revealed or self.updated)

    def _assess_risk(self, obs: MovableObstacle):
        """Assess risk from the visually observed label."""
        if self.risk is not None:
            self.risk.assess(obs.observation())

    def _reassess_risk(self, world_obs: MovableObstacle):
        """Reassess risk using properties revealed by physical contact."""
        if self.risk is not None:
            self.risk.reassess(world_obs.contact_observation(),
                               world_obs.difficulty)

    @staticmethod
    def _matches(a: MovableObstacle, b: MovableObstacle) -> bool:
        """Is the remembered obstacle still what a look at the real one shows?"""
        return (abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9
                and abs(a.theta - b.theta) < 1e-9
                and a.l == b.l and a.d == b.d and a.h == b.h
                and a.material == b.material)

    def _sync(self, known: MovableObstacle, world_obs: MovableObstacle):
        """Update a remembered obstacle from what the robot can now see."""
        old_footprint = known.polygon
        old_x, old_y, old_theta = known.x, known.y, known.theta
        reshaped = (known.l != world_obs.l or known.d != world_obs.d
                    or known.h != world_obs.h
                    or known.material != world_obs.material)
        self._forget_edges(known.oid)
        known.x, known.y, known.theta = world_obs.x, world_obs.y, world_obs.theta
        known.l, known.d, known.h = world_obs.l, world_obs.d, world_obs.h
        known.material = world_obs.material
        known.removed = world_obs.removed
        self._update_edges_for(known)
        self._clear_contacts_overlapping(old_footprint)
        self.updated.append(known.oid)
        if (abs(known.x - old_x) > 1e-9 or abs(known.y - old_y) > 1e-9
                or abs(known.theta - old_theta) > 1e-9):
            # A moved obstacle invalidates the remembered pose unless relocated by the robot.
            self.seen_moving.add(known.oid)
        if reshaped:
            self._reconsider(known)

    def _reconsider(self, obs: MovableObstacle):
        """Clear estimates whose visible shape or label no longer matches."""
        self.touched.discard(obs.oid)
        self.touched_difficulty.pop(obs.oid, None)
        self.disturbed.discard(obs.oid)
        self._bump()
        if self.estimator is not None:
            self.estimator.forget(obs.oid)
        if self.risk is not None:
            self.risk.forget(obs.oid)
            self.risk.assess(obs.observation())

    def _forget_vacated(self, world_obstacles: List[MovableObstacle], robot_pos):
        """Remove visible memories whose stored footprint is gone."""
        real = {w.oid: w for w in world_obstacles}
        rp = Point(robot_pos)
        gone = [oid for oid, known in self.perceived.items()
                if oid in real and not self._matches(known, real[oid])
                and rp.distance(Point(known.center())) <= self.cfg.R_perc
                and self._visible(robot_pos, known, world_obstacles)]
        for oid in gone:
            self._forget_edges(oid)
            self.perceived.pop(oid)
            self.updated.append(oid)
        return gone

    def _half_edge_samples(self, obs: MovableObstacle):
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
        seg = LineString([robot_pos, p])
        width = self.cfg.sight_width
        sight = seg.buffer(width / 2.0, cap_style=2) if width > 0 else seg
        # Static free space handles wall occlusion.
        if not self.roadmap.static_free_prep.contains(sight):
            return False
        # A short blocker leaves the target's upper half visible.
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

    def _update_edges_for(self, obs: MovableObstacle):
        for key in self.roadmap.corridors_intersecting(obs.polygon):
            self.edge_blockers.setdefault(key, set()).add(obs.oid)

    def register_contact(self, region: Polygon):
        if region is None or region.is_empty:
            return
        for obs in self.perceived.values():
            region = region.difference(obs.polygon)
            if region.is_empty:
                return
        if region.area <= 1e-9:
            return
        merged = [region]
        rest = []
        for c in self.contacts:
            (merged if c.intersects(region) else rest).append(c)
        rest.append(unary_union(merged) if len(merged) > 1 else region)
        self.contacts = rest
        self._bump()

    def _clear_contacts_overlapping(self, poly: Polygon):
        if not self.contacts:
            return
        kept = [c for c in self.contacts if not c.intersects(poly)]
        if len(kept) != len(self.contacts):
            self._bump()
        self.contacts = kept

    def force_reveal(self, world_obs: MovableObstacle) -> List[int]:
        if world_obs.oid in self.perceived:
            return []
        obs = world_obs.perceived_copy()
        self.perceived[obs.oid] = obs
        self.newly_revealed.append(obs.oid)
        self._assess_risk(obs)
        self._update_edges_for(obs)
        self._clear_contacts_overlapping(obs.polygon)
        self._bump()
        return [obs.oid]

    def _forget_edges(self, oid: int):
        for blockers in self.edge_blockers.values():
            blockers.discard(oid)

    def record_move_direction(self, oid: int, from_xy, to_xy):
        """Store the latest non-zero movement direction."""
        dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
        n = math.hypot(dx, dy)
        if n > 1e-3:
            self.move_dir[oid] = (dx / n, dy / n)
            self._bump()

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):
        self._forget_edges(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self.disturbed.add(obs.oid)
        self._update_edges_for(obs)
        self._bump()

    def invalidate_contact(self, oid: int):
        """Forget contact measurements invalidated by hidden world changes."""
        self.touched.discard(oid)
        self.disturbed.discard(oid)
        self._bump()

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
    ) -> Tuple[List[int], float]:
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
            if t <= t_hit + 1e-6:
                self.force_reveal(w)
                revealed.append(w.oid)
        return revealed, t_hit

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
                self._reassess_risk(w)
                revealed.append(w.oid)
                self._bump()
        return revealed

    def reveal_by_interaction(self, oid: int,
                              world_obstacles: List[MovableObstacle]) -> bool:
        """Reveal true difficulty whenever the robot physically interacts."""
        if oid in self.touched:
            return False
        for w in world_obstacles:
            if w.oid == oid:
                self.touched.add(oid)
                self.touched_difficulty[oid] = w.difficulty
                self._reassess_risk(w)
                self._bump()
                return True
        return False

    def get_difficulty(self, oid: int, estimator) -> float:  # Return true contact difficulty when known.
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    def others_union(self, oid: int, relocated=None):
        """Return known exclusions for an obstacle, including planned relocations."""
        relocated = relocated or {}
        body = self.perceived[oid].polygon if oid in self.perceived else None
        polys = [(ob.polygon_at(*relocated[other]) if other in relocated
                  else ob.polygon)
                 for other, ob in self.perceived.items() if other != oid]
        for c in self.contacts:
            part = c if body is None else c.difference(body)
            if not part.is_empty and part.area > 1e-9:
                polys.append(part)
        return unary_union(polys) if polys else None

    def partners_of(self, oid: int) -> Tuple[int, ...]:
        """Return visible obstacles coupled to this one."""
        known = self.perceived.get(oid)
        if known is None:
            return ()
        return tuple(p for p in known.interacts_with if p in self.perceived)

    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]
