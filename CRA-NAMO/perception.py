"""Perception and belief state maintenance and update"""

from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from obstacle import MovableObstacle
from roadmap import Roadmap, EdgeKey
from config import Config


class Belief:
    """What the robot knows, and nothing else.

    Risk assessment lives here rather than in the executor because this is where
    the information arrives: an obstacle is assessed the moment it first enters
    `perceived`, and re-assessed the moment the robot touches it and learns what
    it actually weighs. Hooking those two events anywhere else would mean
    re-deriving "is this the first time?" at every call site.
    """

    def __init__(self, roadmap: Roadmap, cfg: Config, risk_estimator=None):
        self.roadmap = roadmap
        self.cfg = cfg
        self.risk = risk_estimator
        self.perceived: Dict[int, MovableObstacle] = {}     # copies of perceived obstacles
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}    # obstacles blocking edges
        self.newly_revealed: List[int] = []  # newly perceived obstacles
        self.contacts: List[Polygon] = []  # obstacles known only through manipulation collision
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {}  # obstacle true difficulties obtained via touch
        self.move_dir: Dict[int, Tuple[float, float]] = {}  # last direction moved per obstacle

    # --- Perception ---
    def perceive(self, world_obstacles: List[MovableObstacle],
                 robot_pos: Tuple[float, float]) -> List[int]:
        """Reveal all visible obstacles around the robot; sync state for known ones"""
        self.newly_revealed = []
        rp = Point(robot_pos)
        for w in world_obstacles:
            known = self.perceived.get(w.oid)
            # known and pose matches — no info to sync, skip expensive visibility check entirely
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
            self._assess_risk(obs)
            self._update_edges_for(obs)
            # obstacle is now fully perceived; any prior anonymous "contact" records for it can be cleared
            self._clear_contacts_overlapping(obs.polygon)
        return self.newly_revealed

    # --- risk ---
    def _assess_risk(self, obs: MovableObstacle):
        """First sight: judge the danger from the label, before going near it."""
        if self.risk is not None:
            self.risk.assess(obs.observation())

    def _reassess_risk(self, world_obs: MovableObstacle):
        """Physical contact: judge again, now knowing what it really weighs.

        Reads the *world* obstacle, because that is the only thing that has the
        knowledge contact confers — the label it resolves into, and its true
        difficulty. This is the one channel by which either crosses into belief,
        and it opens only when the robot is actually touching the obstacle.
        """
        if self.risk is not None:
            self.risk.reassess(world_obs.contact_observation(),
                               world_obs.difficulty)

    @staticmethod
    def _pose_matches(a: MovableObstacle, b: MovableObstacle) -> bool:
        return (abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9
                and abs(a.theta - b.theta) < 1e-9)

    def _sync_pose(self, known: MovableObstacle, world_obs: MovableObstacle):
        """Align perceived copy pose to world ground-truth pose"""
        old_footprint = known.polygon
        self._forget_edges(known.oid)
        known.x, known.y, known.theta = world_obs.x, world_obs.y, world_obs.theta
        known.removed = world_obs.removed
        self._update_edges_for(known)
        self._clear_contacts_overlapping(old_footprint)

    # --- visibility ---
    def _half_edge_samples(self, obs: MovableObstacle):  # line-of-sight samples from 8 obstacle points
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
        # line-of-sight check
        seg = LineString([robot_pos, p])
        width = self.cfg.sight_width
        sight = seg.buffer(width / 2.0, cap_style=2) if width > 0 else seg
        # wall occlusion: a ray is blocked once it leaves the static free space (= workspace minus walls)
        if not self.roadmap.static_free_prep.contains(sight):
            return False
        # occlusion by other movable obstacles; a blocker no taller than half the
        # target still leaves the target's upper half exposed, so it does not occlude
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

    # --- Update ---
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

    # --- collision contact ---
    def register_contact(self, region: Polygon):
        if region is None or region.is_empty:
            return
        for obs in self.perceived.values():
            region = region.difference(obs.polygon)
            if region.is_empty:
                return
        if region.area <= 1e-9:      # only floating-point fragments remain
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

    def force_reveal(self, world_obs: MovableObstacle) -> List[int]:  # reveal full obstacle properties on physical contact
        if world_obs.oid in self.perceived:
            return []
        obs = world_obs.perceived_copy()
        self.perceived[obs.oid] = obs
        self.newly_revealed.append(obs.oid)
        self._assess_risk(obs)
        self._update_edges_for(obs)
        self._clear_contacts_overlapping(obs.polygon)
        return [obs.oid]

    def _forget_edges(self, oid: int):  # forget edge-blocking info for this obstacle
        for blockers in self.edge_blockers.values():
            blockers.discard(oid)

    def record_move_direction(self, oid: int, from_xy, to_xy):
        """Remember which way this obstacle was last moved (pure rotations keep the old direction)"""
        dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
        n = math.hypot(dx, dy)
        if n > 1e-3:
            self.move_dir[oid] = (dx / n, dy / n)

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):  # update obstacle blocking info after relocation
        self._forget_edges(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self._update_edges_for(obs)

    # --- robot self-collision ---
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
    ) -> Tuple[List[int], float]:  # check whether the robot hits any obstacle
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
            if t <= t_hit + 1e-6:          # group simultaneous hits together
                self.force_reveal(w)
                revealed.append(w.oid)
        return revealed, t_hit

    # --- touch sensing ---
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
        return revealed

    def reveal_by_interaction(self, oid: int,
                              world_obstacles: List[MovableObstacle]) -> bool:
        """Any physical interaction reveals the true difficulty, not only collisions.

        Pushing requires contact by definition, so the robot cannot have moved an
        obstacle while still being ignorant of how hard it was to move.
        """
        if oid in self.touched:
            return False
        for w in world_obstacles:
            if w.oid == oid:
                self.touched.add(oid)
                self.touched_difficulty[oid] = w.difficulty
                self._reassess_risk(w)
                return True
        return False

    def get_difficulty(self, oid: int, estimator) -> float:  # return formal difficulty, preserving any previously known true value
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    # --- Query ---
    def others_union(self, oid: int):
        """Everything known to be in the way apart from obstacle *oid*.

        The perceived obstacles plus the anonymous contact regions, minus
        whatever part of a contact region is *oid* itself — a bump recorded
        against the obstacle being moved is not a second thing to steer around.
        Returns None when nothing else is known.
        """
        body = self.perceived[oid].polygon if oid in self.perceived else None
        polys = [ob.polygon for other, ob in self.perceived.items() if other != oid]
        for c in self.contacts:
            part = c if body is None else c.difference(body)
            if not part.is_empty and part.area > 1e-9:
                polys.append(part)
        return unary_union(polys) if polys else None

    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]

