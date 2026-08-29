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
        self.updated: List[int] = []     # known obstacles seen to have changed
        self.contacts: List[Polygon] = []
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {}
        self.disturbed: Set[int] = set()   # moved by the robot since last seen to change
        # Seen, with the robot's own eyes, to have gone somewhere by itself.
        # The difference between "in the way" and "on its way through".
        self.seen_moving: Set[int] = set()
        self.move_dir: Dict[int, Tuple[float, float]] = {}
        # Bumped whenever anything a cached plan was built against changes. The
        # planner puts it in its cache keys, so work costed against one state of
        # knowledge is never served up against another.
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
        """Is this body no longer where — or no longer what — the robot last saw?

        Only asked of the world, never of the belief: it is how the *simulator*
        decides whether the robot could have known about something, not a way for
        the robot to find out. A body it has never seen is not stale, it is
        unknown, and being run into is how that gets discovered.
        """
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
            # It is not where the robot left it, and the robot did not move it —
            # `relocate` keeps the belief in step when it does.
            self.seen_moving.add(known.oid)
        if reshaped:
            self._reconsider(known)

    def _reconsider(self, obs: MovableObstacle):
        """It is not the thing it was, so judgements about the old one lapse.

        Size and label are what the difficulty estimator and the risk model read,
        so a change the robot can see invalidates both, and what it learned by
        touching goes with them — that difficulty belonged to the old object. A
        change it *cannot* see, a difficulty rewritten behind an unchanged label,
        stays believed until it next takes hold of the thing. That is the price
        of not being told in advance.
        """
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
        """Drop memories the robot can see are wrong.

        A remembered footprint in plain view with nothing standing in it is a
        memory contradicted by what the robot is looking at. Without this, the
        ghost of anything that wandered off out of sight would go on blocking the
        roadmap for the rest of the run.
        """
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
        if region.area <= 1e-9:      # only floating-point fragments remain
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

    def force_reveal(self, world_obs: MovableObstacle) -> List[int]:  # reveal full obstacle properties on physical contact
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

    def _forget_edges(self, oid: int):  # forget edge-blocking info for this obstacle
        for blockers in self.edge_blockers.values():
            blockers.discard(oid)

    def record_move_direction(self, oid: int, from_xy, to_xy):
        """Remember which way this obstacle was last moved (pure rotations keep the old direction)"""
        dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
        n = math.hypot(dx, dy)
        if n > 1e-3:
            self.move_dir[oid] = (dx / n, dy / n)
            self._bump()

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):  # update obstacle blocking info after relocation
        self._forget_edges(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self.disturbed.add(obs.oid)
        self._update_edges_for(obs)
        self._bump()

    def invalidate_contact(self, oid: int):
        """The thing that was measured is not the thing that is standing there.

        Called when ground truth changed where the robot could not see it. What
        it believes it measured stays believed — it has no way of knowing the
        object was rewritten behind its back, and being quietly handed the new
        figure is exactly the free lunch this module exists to prevent. What
        lapses is the record of *having* measured it, so the next time the robot
        takes hold of the thing it reads it again instead of trusting a number
        that belonged to the old one. Having disturbed the old one lapses with
        it: shifting what is there now is a fresh decision, at a fresh price.
        """
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
                self._bump()
                return True
        return False

    def get_difficulty(self, oid: int, estimator) -> float:  # return formal difficulty, preserving any previously known true value
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    def others_union(self, oid: int, relocated=None):
        """Everything known to be in the way apart from obstacle *oid*.

        The perceived obstacles plus the anonymous contact regions, minus
        whatever part of a contact region is *oid* itself — a bump recorded
        against the obstacle being moved is not a second thing to steer around.
        Returns None when nothing else is known.

        `relocated` maps oids to poses they are about to be put in rather than
        the ones they are in. It is how a second obstacle on the same edge gets
        planned around where the first one is going, instead of around where it
        was before the plan moved it.
        """
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
        """The bodies the robot can see this one is propped against.

        Only the ones it has actually seen: a coupling to something it has never
        laid eyes on is not knowledge it has, however true it is.
        """
        known = self.perceived.get(oid)
        if known is None:
            return ()
        return tuple(p for p in known.interacts_with if p in self.perceived)

    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]
