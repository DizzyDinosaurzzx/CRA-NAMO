"""Online plan–execute–perceive–replan loop for NAMO."""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from config import Config
import contact
import dynamics
from obstacle import MovableObstacle, StaticObstacle
from roadmap import Roadmap
from perception import Belief
from llm_difficulty import DifficultyEstimator
import risk
from risk import RiskEstimator
from search import FailedMoves, Planner, move_signature
import cost
import geometry
import kinematics
import manipulation

# Candidate roadmap nodes considered when reanchoring.
_REANCHOR_CANDIDATES = 64

# Maximum reanchor attempts after dynamic interruptions.
_REANCHOR_ATTEMPTS = 4


def _driven_fraction(elapsed: float, turn_s: float, drive_s: float) -> float:

    if drive_s <= 0.0:
        return 1.0 if elapsed >= turn_s + drive_s else 0.0
    return min(1.0, max(0.0, (elapsed - turn_s) / drive_s))


def _disagree(a: float, b: float, ratio: float) -> bool:
    """Are these two figures more than `ratio` apart, whichever is the larger?"""
    lo, hi = sorted((abs(float(a)), abs(float(b))))
    if hi <= 0.0:
        return False
    return lo <= 0.0 or hi / lo >= ratio


def _lerp_xy(a, b, s: float):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


@dataclass
class RunResult:
    success: bool                           # Whether the goal was reached.
    C: float = 0.0                          # Objective value.
    J: float = 0.0                          # Energy cost.
    walk_cost: float = 0.0                  # Robot travel cost.
    manip_walk_cost: float = 0.0            # Travel cost while escorting.
    work_cost: float = 0.0                  # Obstacle manipulation work.
    risk_cost: float = 0.0                  # Risk surcharge.
    risk_levels: dict = field(default_factory=dict)   # Risk level charged per obstacle.
    T: float = 0.0                          # Simulated elapsed time.
    move_time: float = 0.0                  # Time spent moving.
    wait_time: float = 0.0                  # Time spent waiting.
    cycles: int = 0                         # Replanning cycles.
    plan_time: float = 0.0                  # Wall-clock planning time.
    first_plan_time: float = 0.0            # First planning duration.
    total_expansions: int = 0               # Total A* expansions.
    llm_calls: int = 0                      # LLM API calls.
    llm_mode: str = "heuristic"             # Estimator mode.
    removed: List[int] = field(default_factory=list)    # Moved obstacle IDs.
    robot_track: List[Tuple[float, float]] = field(     # Robot node coordinates.
        default_factory=list)
    frames: List[dict] = field(default_factory=list)    # Visualization snapshots.
    world_events: List[str] = field(default_factory=list)  # Autonomous world events.
    decisions: List[str] = field(default_factory=list)     # Authored decision records.
    message: str = ""                       # Result description.

class OnlineNAMO:
    """Run online NAMO while keeping ground truth separate from robot belief."""
    def __init__(self, workspace: Polygon,
                 static_obstacles: List[StaticObstacle],
                 movable_obstacles: List[MovableObstacle],
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 cfg: Config,
                 events=None,
                 decision_points=None):
        self.cfg = cfg
        self.workspace = workspace
        self.static_obstacles = static_obstacles
        self.world = movable_obstacles
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))

        self.roadmap = Roadmap(workspace, static_obstacles, cfg)
        self.start_node = self._add_terminal(self.start)
        self.goal_node = self._add_terminal(self.goal)
        self.start_point = self.roadmap.nodes[self.start_node]
        self.goal_point = self.roadmap.nodes[self.goal_node]

        self.estimator = DifficultyEstimator(cfg)
        self.risk = RiskEstimator(cfg)
        self.belief = Belief(self.roadmap, cfg, self.risk, self.estimator)
        # Dynamics updates ground truth without changing robot belief.
        self.dynamics = dynamics.WorldDynamics(
            self.world, static_obstacles, workspace, events, cfg)
        self._risk_charged: set = set()
        self.stranded = False
        self._plan_paths: List[dict] = []
        self.failed_moves = FailedMoves()
        # Stores a dynamic collision and the robot's stop position.
        self._world_hit: Optional[Tuple[List[int], Tuple[float, float]]] = None
        self._holding: Optional[int] = None
        # Keep authored decision points for reporting.
        self.decision_points = list(decision_points or ())
        # Accumulated wait time by blocker set.
        self.wait_budget: dict = {}
        # Manipulation can move the robot off its roadmap node.
        self.robot_xy: Tuple[float, float] = self.roadmap.nodes[self.start_node]
        # Planning wall time does not advance simulated time.
        self.clock: float = 0.0
        # Heading changes cost time even with a symmetric footprint.
        self.robot_heading: float = 0.0
        self._free_profile = cfg.free_profile()
        self._loaded_profile = cfg.loaded_profile()
        self._frame_node = self.start_node
        self._world_frame_t = 0.0
        self._waited = 0.0

    def _add_terminal(self, p: Tuple[float, float]) -> int:
        cfg = self.cfg
        if not self.roadmap.free_eroded_prep.contains(Point(p)):
            p = self.roadmap.nodes[self.roadmap.nearest_node(p)]
        return self.roadmap.add_terminal(p)

    def run(self) -> RunResult:
        """Run to completion and flush folded console output."""
        try:
            return self._run()
        finally:
            self.cfg.flush_log()

    def _run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
        node = self.start_node                           # Track the current roadmap node.
        res.robot_track.append(self.roadmap.nodes[node])
        
        self._perceive(res, self.roadmap.nodes[node])
        self._capture_frame(res, node, "start")

        planner = Planner(self.roadmap, self.belief, self.estimator, cfg,
                          self.failed_moves, self.risk, self.wait_budget)

        for cycle in range(cfg.max_replans):
            # Discard failures from older world versions.
            if self.failed_moves.drop_stale(self.dynamics.version):
                planner.forget_removals()
            t0 = time.time()
            plan = planner.plan(node, self.goal_node, self.robot_heading)
            dt = time.time() - t0
            res.plan_time += dt
            if cycle == 0:
                res.first_plan_time = dt
            self._plan_paths = self._plan_to_paths(plan)
            if plan is None:
                if self._wait_for_world(res, node):
                    continue
                res.message = "No feasible plan under current belief."
                res.cycles = cycle + 1
                return self._finalize(res, node)
            self._waited = 0.0
            res.total_expansions += plan.expansions

            if node == self.goal_node:
                break

            moves_done = 0
            reached_goal = False
            for step, act in enumerate(plan.actions):
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    # Contact information is revealed inside the escort.
                    move_success, hits, executed_dist, new_node = \
                        self._execute_move(act["oid"], obs, act, res, node, cfg)
                    if new_node is not None:
                        node = new_node
                    if executed_dist > 0.0:
                        true_diff = self._world_obstacle(act["oid"]).difficulty
                        executed_work = cost.manipulation_work(true_diff, executed_dist)
                        res.work_cost += executed_work
                        res.J += executed_work
                        self._charge_risk(res, act["oid"])
                        self.dynamics.note_moved(act["oid"])
                        if act["oid"] not in res.removed:
                            res.removed.append(act["oid"])
                    elif not move_success and hits is not None:
                        self.failed_moves.add((move_signature(obs), act["key"]))
                    if not move_success:
                        if hits is not None:
                            self._handle_move_collision(res, node, act["oid"], hits)
                        break
                    if new_node is not None or self.stranded:
                        break
                    if any(a["type"] == "remove"
                           for a in plan.actions[step + 1:]):
                        # Subsequent actions were planned for the old pose.
                        break
                elif act["type"] == "wait":
                    self._wait_on_edge(res, node, act)
                    break
                elif act["type"] == "move":
                    prev_node = node    # Remember where the move started.
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # Travel to the contact point and return before replanning.
                        leg = t_contact * act["dist"]
                        self._drive(res, from_pos, contact_pos, dist=leg)
                        self._drive(res, contact_pos, from_pos, dist=leg)
                        self._touch(res, cfg, contact_pos)
                        self._perceive(res)
                        self._capture_frame(
                            res, node, f"collision revealed {hit_oids} -> replan")
                        break
                    if not self._drive(res, from_pos, to_pos, dist=act["dist"]):
                        break       # Dynamic collision is recovered below.
                    node = act["v"]
                    moves_done += 1
                    touched = self._touch(res, cfg, self.roadmap.nodes[node])
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    self._perceive(res, self.roadmap.nodes[node])
                    self._capture_frame(res, node, f"move to node {node}")
                    if node == self.goal_node:
                        reached_goal = True
                        break
                    if moves_done >= cfg.step_execute_edges:
                        break

            if self._world_hit is not None:
                node = self._recover_from_world_hit(res, node)

            res.cycles = cycle + 1
            if self.stranded:
                res.message = "Robot ended a manipulation with no roadmap node in reach."
                break
            if reached_goal or node == self.goal_node:
                break

        return self._finalize(res, node)

    def _finalize(self, res: RunResult, node: int) -> RunResult:
        """Finalize and populate the run result."""
        res.success = (node == self.goal_node)
        res.C = round(cost.combine(self.cfg, res.J, self.clock) + res.risk_cost, 4)
        res.risk_cost = round(res.risk_cost, 4)
        res.J = round(res.J, 4)
        res.walk_cost = round(res.walk_cost, 4)
        res.manip_walk_cost = round(res.manip_walk_cost, 4)
        res.work_cost = round(res.work_cost, 4)
        res.T = round(self.clock, 4)
        res.move_time = round(res.move_time, 4)
        res.wait_time = round(res.wait_time, 4)
        res.plan_time = round(res.plan_time, 4)
        res.llm_calls = self.estimator.calls
        res.decisions = self._decisions_taken(res)
        if res.success and not res.message:
            res.message = "Reached goal."
        elif not res.success and not res.message:
            res.message = "Ran out of replan cycles."
        return res

    def _world_obstacle(self, oid: int) -> Optional[MovableObstacle]:
        for w in self.world:
            if w.oid == oid:
                return w
        return None

    def _world_collision(self, oid: int, nx: float, ny: float, theta: float,
                         tree=None, tree_items=None):
        mover = self._world_obstacle(oid)
        end_poly = mover.polygon_at(nx, ny, theta)
        swept = manipulation.swept_region(mover, nx, ny, theta)
        hits = []

        def overlapping(poly):
            overlap = swept.intersection(poly)
            if overlap.area > geometry.CONTACT_AREA_EPS:
                return overlap
            end_overlap = end_poly.intersection(poly)
            return end_overlap if end_overlap.area > geometry.CONTACT_AREA_EPS else None

        if tree is not None and tree_items is not None:
            candidates = tree.query(swept)
            for idx in candidates:
                oid_hit, poly = tree_items[idx]
                if oid_hit == oid:
                    continue
                overlap = overlapping(poly)
                if overlap is not None:
                    hits.append((oid_hit, overlap))
            return hits

        for w in self.world:
            if w.oid == oid:
                continue
            overlap = overlapping(w.polygon)
            if overlap is not None:
                hits.append((w.oid, overlap))
        for so in self.static_obstacles:
            overlap = overlapping(so.polygon)
            if overlap is not None:
                hits.append((None, overlap))
        return hits

    def _handle_move_collision(self, res: RunResult, node: int, oid: int, hits):
        """Register anonymous contact regions after an obstacle collision."""
        for oid_hit, region in hits:
            if oid_hit is None:
                continue      # Static wall contacts add no new belief information.
            self.belief.register_contact(region)
        hit_oids = sorted(o for o, _ in hits if o is not None)
        label = (f"move {oid} hit a wall -> replan" if not hit_oids
                 else f"move {oid} hit unknown obstruction -> replan")
        self._capture_frame(res, node, label)

    @staticmethod
    def _sample_move_path(move_path: list, max_frames: int) -> list:
        if len(move_path) <= max_frames:
            return list(range(len(move_path)))
        return [int(i) for i in np.linspace(0, len(move_path) - 1, max_frames)]

    def _charge_walk(self, res: RunResult, dist: float, in_contact: bool = False):
        """Charge robot travel in the objective."""
        charge = cost.motion_cost(self.cfg, dist)
        res.walk_cost += charge
        res.J += charge
        if in_contact:
            res.manip_walk_cost += charge

    def _charge_risk(self, res: RunResult, oid: int):
        """Charge the post-contact risk surcharge once per obstacle."""
        if oid in self._risk_charged:
            return
        self._risk_charged.add(oid)
        level = self.risk.level_of(oid, self.belief.partners_of(oid))
        charge = cost.risk_cost(self.cfg, level)
        if charge <= 0.0:
            res.risk_levels[oid] = level
            return
        res.risk_cost += charge
        res.risk_levels[oid] = level

    def _advance_clock(self, res: RunResult, seconds: float,
                       moving: bool = True) -> bool:
        """Advance simulated time for the robot and dynamic world."""
        if seconds <= 0.0:
            return False
        begins = self.clock
        self.clock += seconds
        if moving:
            res.move_time += seconds
        if not self.dynamics.active:
            return False
        before = self.dynamics.version
        for when, note in self.dynamics.advance(seconds, begins, self.robot_xy):
            res.world_events.append(f"t={when:,.1f}s  {note}")
            self.cfg.log(f"[world] t={when:,.1f}s {note}")
        self._absorb_world_changes()
        return self.dynamics.version != before

    def _wait_for_world(self, res: RunResult, node: int) -> bool:
        """Wait for dynamic obstacles when no plan is available."""
        if not self.dynamics.active or self._waited >= self.cfg.dynamic_max_wait:
            return False
        step = max(self.cfg.dynamic_wait_step, 1e-3)
        self._waited += step
        res.wait_time += step
        self._world_frame(res, self._advance_clock(res, step, moving=False))
        self._perceive(res)
        self._capture_frame(
            res, node, f"nowhere to go — waiting ({self._waited:,.0f}s)")
        return True

    def _perceive(self, res: RunResult, at=None) -> List[int]:
        """Update visual belief and measure its wall-clock time."""
        t0 = time.time()
        try:
            return self.belief.perceive(
                self.world, self.robot_xy if at is None else at)
        finally:
            res.plan_time += time.time() - t0

    def _touch(self, res: RunResult, cfg: Config, at=None) -> List[int]:
        """Update contact belief and measure its wall-clock time."""
        t0 = time.time()
        try:
            return self.belief.touch_check(
                self.robot_xy if at is None else at, self.world, cfg)
        finally:
            res.plan_time += time.time() - t0

    def _wait_on_edge(self, res: RunResult, node: int, act: dict):
        """Wait on a blocked edge and charge elapsed time."""
        seconds = float(act["seconds"])
        key = tuple(act["oids"])
        self.wait_budget[key] = self.wait_budget.get(key, 0.0) + seconds
        res.wait_time += seconds
        self._world_frame(res, self._advance_clock(res, seconds, moving=False))
        self._perceive(res)
        self._capture_frame(
            res, node,
            f"waiting {self.wait_budget[key]:,.0f}s for {act['oids']} to clear")

    def _decisions_taken(self, res: RunResult) -> List[str]:
        """Summarize authored decision points from moved obstacles."""
        moved = set(res.removed)
        taken = []
        for point in self.decision_points:
            name = point.get("name", "decision")
            risky, safer = point.get("risky"), point.get("safer_alternative")
            partners = [p for p in (point.get("partners") or ()) if p in moved]
            what = []
            if risky is not None and risky in moved:
                what.append(f"moved the risky one ({risky})")
            if safer is not None and safer in moved:
                what.append(f"took the safer one ({safer})")
            if partners:
                what.append(f"also disturbed {partners}")
            taken.append(f"{name}: " + ("; ".join(what) if what else "went round"))
        return taken

    def _world_frame(self, res: RunResult, moved: bool):
        """Draw the world in motion, at most one frame per animation step."""
        if not moved or self.clock - self._world_frame_t < self.cfg.gif_time_step:
            return
        self._world_frame_t = self.clock
        self._capture_frame(res, self._frame_node, "the world moves")

    def _drive(self, res: RunResult, a, b, in_contact: bool = False,
               loaded: bool = False, dist: Optional[float] = None):
        """Drive between two points and charge travel, energy, and time."""
        length = math.dist(a, b) if dist is None else dist
        profile = self._loaded_profile if loaded else self._free_profile
        if not self.dynamics.active:
            self._charge_walk(res, length, in_contact)
            seconds, self.robot_heading = kinematics.segment_time(
                profile, a, b, self.robot_heading)
            moved = self._advance_clock(res, seconds)
            self._set_robot(res, b)
            self._world_frame(res, moved)
            return True
        return self._drive_alongside_world(res, a, b, profile, length,
                                           in_contact)

    def _drive_alongside_world(self, res: RunResult, a, b, profile,
                               length: float, in_contact: bool) -> bool:
        """Drive while advancing and checking the dynamic world."""
        turn_s, drive_s, heading = kinematics.segment_legs(
            profile, a, b, self.robot_heading)
        self.robot_heading = heading
        total = turn_s + drive_s
        if total <= 0.0:
            self._charge_walk(res, length, in_contact)
            self._set_robot(res, b)
            return True
        step = max(self.cfg.dynamic_step, 1e-3)
        elapsed = 0.0
        done = 1.0
        while elapsed < total - 1e-9:
            dt = min(step, total - elapsed)
            here = _lerp_xy(a, b, _driven_fraction(elapsed, turn_s, drive_s))
            there = _lerp_xy(a, b, _driven_fraction(elapsed + dt, turn_s, drive_s))
            self.robot_xy = here          # Expose the robot position for this sub-step.
            was = {w.oid: (w.x, w.y, w.theta) for w in self.world}
            moved = self._advance_clock(res, dt)
            hits = self._crossed_by_world(here, there, was)
            if hits:
                done = _driven_fraction(elapsed, turn_s, drive_s)
                self._world_hit = (hits, here)
                self._set_robot(res, here)
                self._world_frame(res, True)
                break
            elapsed += dt
            self._world_frame(res, moved)
        self._charge_walk(res, length * done, in_contact)
        if done >= 1.0:
            self._set_robot(res, b)
            return True
        return False

    def _crossed_by_world(self, here, there, was: dict) -> List[int]:
        """Return dynamic obstacles intersecting the robot during a leg."""
        candidates = []
        for w in self.world:
            if w.oid == self._holding:
                continue
            then = was.get(w.oid, (w.x, w.y, w.theta))
            now = (w.x, w.y, w.theta)
            if then != now or self.belief.is_stale(w):
                candidates.append((w, then, now))
        if not candidates:
            return []
        corridor = LineString([here, there]).buffer(self.cfg.robot_radius,
                                                    cap_style=1)
        hits = []
        for w, then, now in candidates:
            swept = (w.polygon if then == now
                     else manipulation.swept_between(w, then, now))
            if swept.intersection(corridor).area > geometry.CONTACT_AREA_EPS:
                hits.append(w.oid)
                self.belief.force_reveal(w)
        return sorted(hits)

    def _recover_from_world_hit(self, res: RunResult, node: int) -> int:
        """Reperceive and reanchor after a dynamic collision."""
        hits, _stop = self._world_hit
        self._world_hit = None
        self._touch(res, self.cfg)
        self._perceive(res)
        new_node = self._reanchor(res, node)
        if new_node is not None:
            node = new_node
        self._capture_frame(res, node,
                            f"{hits} crossed the robot's path -> replan")
        return node

    def _absorb_world_changes(self):
        """Invalidate measurements after hidden ground-truth changes."""
        for oid in self.dynamics.drain_stale():
            self.belief.invalidate_contact(oid)
            self._risk_charged.discard(oid)

    def _set_robot(self, res: RunResult, p: Tuple[float, float]):
        self.robot_xy = (float(p[0]), float(p[1]))
        res.robot_track.append(self.robot_xy)

    def _walk_frames(self, pts, cfg: Config) -> set:
        """Return bounded walk segments for frame capture."""
        moving = [i for i in range(1, len(pts))
                  if math.dist(pts[i - 1], pts[i]) > 1e-9]
        return {moving[j] for j in
                self._sample_move_path(moving, cfg.manip_max_frames_per_action)}

    def _walk_robot(self, pts, res: RunResult, cfg: Config,
                    loaded: bool = False, node: Optional[int] = None,
                    label: str = "", move_oid: Optional[int] = None):
        """Drive through points and stop on unknown collisions."""
        frame_at = self._walk_frames(pts, cfg) if node is not None else ()
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._drive(res, a, stop, in_contact=True, loaded=loaded)
                self._touch(res, cfg)
                return i - 1, hits, self.robot_xy
            if not self._drive(res, a, b, in_contact=True, loaded=loaded):
                # Dynamic interruption leaves _world_hit for recovery.
                return i - 1, self._world_hit[0], self.robot_xy
            if i in frame_at:
                self._capture_frame(res, node, label, move_oid=move_oid)
        return len(pts) - 1, [], (pts[-1] if pts else self.robot_xy)

    def _retrace(self, res: RunResult, pts, node: Optional[int] = None,
                 label: str = "", move_oid: Optional[int] = None):
        """Retrace a recently completed path without collision checks."""
        frame_at = self._walk_frames(pts, self.cfg) if node is not None else ()
        for i in range(1, len(pts)):
            if not self._drive(res, pts[i - 1], pts[i], in_contact=True):
                return          # The caller handles an interrupted retrace.
            if i in frame_at:
                self._capture_frame(res, node, label, move_oid=move_oid)

    def _reanchor(self, res: RunResult, node: int) -> Optional[int]:
        """Return the robot to a reachable roadmap node after manipulation."""
        home = self.roadmap.nodes[node]
        for _ in range(_REANCHOR_ATTEMPTS):
            blocked = self._known_obstacles_inflated()
            if self.roadmap.can_drive(self.robot_xy, home, blocked):
                if self._drive(res, self.robot_xy, home, in_contact=True):
                    return None
            else:
                target = self.roadmap.nearest_reachable_node(
                    self.robot_xy, blocked, k=_REANCHOR_CANDIDATES)
                if target is None:
                    self.stranded = True
                    self.cfg.log(f"[reanchor] no roadmap node reachable from "
                                 f"({self.robot_xy[0]:,.2f}, {self.robot_xy[1]:,.2f})")
                    return None
                if self._drive(res, self.robot_xy,
                               self.roadmap.nodes[target], in_contact=True):
                    return target
            # Reperceive after a dynamic interruption.
            self._world_hit = None
            self._perceive(res)
        self.stranded = True
        self.cfg.log(f"[reanchor] the world would not let the robot back onto "
                     f"the roadmap from ({self.robot_xy[0]:,.2f}, "
                     f"{self.robot_xy[1]:,.2f})")
        return None

    def _known_obstacles_inflated(self):
        cfg = self.cfg
        polys = [ob.polygon for ob in self.belief.perceived.values()]
        if not polys:
            return None
        geom = unary_union(polys).buffer(
            max(cfg.robot_radius - cfg.contact_clearance, 1e-6))
        shapely.prepare(geom)
        return geom

    def _contact_plan(self, oid: int, obs, move_path, node: int, key):
        """Plan an escort from the robot's actual position and release options."""
        rm = self.roadmap
        exits = [(self.robot_xy, 0.0)]
        exit_nodes: List[Optional[int]] = [None]
        if key is not None:
            far = key[1] if key[0] == node else (key[0] if key[1] == node else None)
            if far is not None:
                exits = [(rm.nodes[far], 0.0), (self.robot_xy, rm.edge_len[key])]
                exit_nodes = [far, None]
        cplan = contact.plan_contact(
            obs, move_path, self.robot_xy, exits,
            self.roadmap.free_eroded_tol,
            contact.inflate_others(self.belief.others_union(oid), self.cfg),
            self.cfg)
        return cplan, exit_nodes

    def _execute_move(self, oid: int, obs, act: dict,
                      res: RunResult, node: int, cfg: Config):
        """Suspend dynamic motion while executing one manipulation."""
        self.dynamics.suspend(oid)
        self._holding = oid
        try:
            return self._escort(oid, obs, act, res, node, cfg)
        finally:
            self._holding = None
            self.dynamics.release(oid)

    def _escort(self, oid: int, obs, act: dict,
                res: RunResult, node: int, cfg: Config):
        """Execute an obstacle escort and return its outcome."""
        move_path = act["move_path"]
        frame_at = set(self._sample_move_path(move_path,
                                              cfg.manip_max_frames_per_action))
        n = len(move_path)
        start_xy = (obs.x, obs.y)
        home = self.robot_xy

        cplan = act.get("contact")
        exit_nodes: List[Optional[int]] = [None]
        if not cfg.contact_required or cplan is None or not cplan.feasible:
            # Without contact, the obstacle moves while the robot waits.
            cplan = contact.idle_plan(home, n)
        else:
            # Recompute approach and exit legs from the robot's actual position.
            cplan, exit_nodes = self._contact_plan(oid, obs, move_path, node,
                                                   act.get("key"))
            if not cplan.feasible:
                self.failed_moves.add((move_signature(obs), act["key"]))
                self._capture_frame(
                    res, node,
                    f"cannot escort {oid} from node {node}: {cplan.reason} -> replan")
                return (False, None, 0.0, None)
        if self.dynamics.active and not self._still_there(oid, move_path[0]):
            # A changed pose invalidates the escort plan.
            self._perceive(res)
            self._capture_frame(
                res, node, f"{oid} is no longer where it was -> replan")
            return (False, None, 0.0, None)
        rp = list(cplan.robot_path)
        off = cplan.move_offset

        # Rebuild the collision index only when the world changes.
        tree, tree_items = self._collision_index(oid)
        world_version = self.dynamics.version

        reached, hits, stop = self._walk_robot(
            rp[:off + 1], res, cfg, node=node, label=f"walk round to grip {oid}")
        if hits:
            self._retrace(res, [stop] + rp[reached::-1], node=node,
                          label=f"back off from {oid}")
            self._perceive(res)
            self._capture_frame(
                res, node, f"robot hit {sorted(hits)} approaching {oid} -> replan")
            return (False, None, 0.0, None)
        rethink = self._on_taking_hold(res, oid, node, cfg)
        if rethink is not None:
            return rethink
        self._capture_frame(res, node, f"grip obstacle {oid}", move_oid=oid)

        last_i = 0
        for i in range(1, n):
            if world_version != self.dynamics.version:
                tree, tree_items = self._collision_index(oid)
                world_version = self.dynamics.version
            wx, wy, wth = move_path[i]
            obs_hits = self._world_collision(oid, wx, wy, wth, tree=tree,
                                             tree_items=tree_items)
            if obs_hits:
                # Sync belief to the last completed obstacle pose.
                if last_i != 0:
                    self.belief.relocate(obs, *move_path[last_i])
                    self.belief.record_move_direction(oid, start_xy,
                                                      move_path[last_i])
                new_node = self._release_and_return(res, oid, rp, off, last_i,
                                                    False, node, cfg,
                                                    cplan, exit_nodes)
                return (False, obs_hits,
                        cost.se2_path_length(obs, move_path[:last_i + 1], cfg),
                        new_node)
            # The robot can collide while holding the obstacle.
            a, b = rp[off + i - 1], rp[off + i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                if last_i != 0:
                    self.belief.relocate(obs, *move_path[last_i])
                    self.belief.record_move_direction(oid, start_xy, move_path[last_i])
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._drive(res, a, stop, in_contact=True, loaded=True)
                self._touch(res, cfg)
                self._perceive(res)
                new_node = self._reanchor(res, node)
                self._capture_frame(
                    res, node if new_node is None else new_node,
                    f"robot hit {sorted(hits)} while moving {oid} -> replan")
                return (False, None,
                        cost.se2_path_length(obs, move_path[:last_i + 1], cfg),
                        new_node)
            self._relocate_world(oid, wx, wy, wth)
            if not self._drive(res, a, b, in_contact=True, loaded=True):
                # Roll back the incomplete leg before replanning.
                self._relocate_world(oid, *move_path[last_i])
                if last_i != 0:
                    self.belief.relocate(obs, *move_path[last_i])
                    self.belief.record_move_direction(oid, start_xy,
                                                      move_path[last_i])
                self._perceive(res)
                return (False, None,
                        cost.se2_path_length(obs, move_path[:last_i + 1], cfg),
                        None)
            last_i = i
            if i in frame_at:
                self._capture_frame(res, node, f"move {oid} step {i}/{n - 1}",
                                    move_oid=oid)

        # Sync belief to the final obstacle pose.
        self.belief.relocate(obs, *move_path[-1])
        self.belief.record_move_direction(oid, start_xy, move_path[-1])
        self._perceive(res)
        new_node = self._release_and_return(res, oid, rp, off, n - 1, True,
                                            node, cfg, cplan, exit_nodes)
        return (True, [], cost.se2_path_length(obs, move_path, cfg), new_node)

    def _on_taking_hold(self, res: RunResult, oid: int, node: int, cfg: Config):
        """Reassess difficulty and risk revealed when the robot takes hold."""
        before = self.risk.level_of(oid)
        costed = self.belief.get_difficulty(oid, self.estimator)
        touched = self._touch(res, cfg)
        if self.belief.reveal_by_interaction(oid, self.world):
            touched.append(oid)
        if touched:
            self._capture_frame(
                res, node, f"contact revealed difficulty of {sorted(touched)}")
        after = self.risk.level_of(oid)
        measured = self.belief.get_difficulty(oid, self.estimator)

        if before is not None and risk.higher(before, after) != before:
            why = f"contact re-rated {oid} {before} -> {after}"
        elif _disagree(costed, measured, cfg.contact_replan_ratio):
            why = (f"{oid} takes {measured:,.0f} N to shift, not the "
                   f"{costed:,.0f} N this route was costed at")
        else:
            return None

        new_node = self._reanchor(res, node)
        self._capture_frame(res, node if new_node is None else new_node,
                            f"{why} -> replan")
        return (False, None, 0.0, new_node)

    def _release_and_return(self, res: RunResult, oid: int, rp: list, off: int,
                            last_i: int, completed: bool, node: int,
                            cfg: Config, cplan, exit_nodes) -> Optional[int]:
        """Release the obstacle and return to the roadmap."""
        if not completed:
            return self._reanchor(res, node)
        _reached, hits, _stop = self._walk_robot(
            rp[off + last_i:], res, cfg, node=node,
            label=f"let go of {oid} and walk on")
        if hits:
            self._perceive(res)
            new_node = self._reanchor(res, node)
            self._capture_frame(res, node if new_node is None else new_node,
                                f"robot hit {sorted(hits)} backing out -> replan")
            return new_node
        i = cplan.exit_index
        landed = exit_nodes[i] if 0 <= i < len(exit_nodes) else None
        if landed is None or landed == node:
            return None
        # Perceive before planning from a new release node.
        self._touch(res, cfg)
        self._perceive(res)
        self._capture_frame(res, landed, f"let go of {oid} at node {landed}")
        return landed

    def _collision_index(self, oid: int):
        """Everything the obstacle being moved could run into, indexed for lookup."""
        polys, items = [], []
        for w in self.world:
            if w.oid != oid:
                polys.append(w.polygon)
                items.append((w.oid, w.polygon))
        for so in self.static_obstacles:
            polys.append(so.polygon)
            items.append((None, so.polygon))
        return (STRtree(polys), items) if polys else (None, None)

    def _still_there(self, oid: int, pose) -> bool:
        """Return whether the obstacle remains near its planned pose."""
        w = self._world_obstacle(oid)
        if w is None:
            return False
        return (math.dist((w.x, w.y), (pose[0], pose[1])) <= self.cfg.se2_cell
                and abs(geometry.wrap_dtheta(w.theta, pose[2]))
                <= math.pi / self.cfg.se2_n_theta)

    def _relocate_world(self, oid: int, x: float, y: float, theta: float):
        for w in self.world:
            if w.oid == oid:
                w.x, w.y, w.theta, w.removed = x, y, theta, True
                return

    def _plan_to_paths(self, plan) -> List[dict]:
        if plan is None:
            return []
        paths: List[dict] = []
        route = [self.roadmap.nodes[n] for n in plan.node_path]
        if len(route) >= 2:
            paths.append({"kind": "route", "pts": route})
        for act in plan.actions:
            if act["type"] != "remove":
                continue
            move_path = act.get("move_path") or []
            pts = [(p[0], p[1]) for p in move_path]
            if len(pts) >= 2:
                paths.append({"kind": "obstacle", "oid": act["oid"], "pts": pts})
            cplan = act.get("contact")
            if cplan is not None and cplan.feasible and len(cplan.robot_path) >= 2:
                paths.append({"kind": "contact", "oid": act["oid"],
                              "pts": list(cplan.robot_path)})
        return paths

    def _capture_frame(self, res: RunResult, node: int, label: str,
                       move_oid: Optional[int] = None):
        self._frame_node = node
        if not self.cfg.save_frames:
            return
        perceived = set(self.belief.perceived.keys())
        res.frames.append({
            "node": node,
            "move_oid": move_oid,   # Obstacle currently being moved.
            "plan_paths": list(self._plan_paths),   # Planned paths for this frame.
            "robot": self.robot_xy,     # True robot position, possibly off-node.
            "track": list(res.robot_track),
            "obstacles": [(w.oid, w.polygon, w.removed) for w in self.world],
            "world_moved": sorted(self.dynamics.moved_on_own, key=str),
            "perceived": perceived,
            "estimated_difficulty": {
                oid: value for oid, value in self.estimator.cache.items()
                if oid in perceived
            },
            "touched_difficulty": dict(self.belief.touched_difficulty),
            "risk": {oid: self.risk.level_of(oid, self.belief.partners_of(oid))
                     for oid in perceived},
            "J": round(res.J, 4),
            "t": round(self.clock, 3),      # Simulated seconds used by the animation.
            "move_t": round(res.move_time, 1),   # Displayed movement-time component.
            "plan_t": round(res.plan_time, 1),
            "label": label,
        })
