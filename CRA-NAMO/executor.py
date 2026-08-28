"""CA-NAMO online "plan–execute–perceive–replan" loop"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
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
from search import Planner, move_signature
import cost
import geometry
import kinematics
import manipulation

# How many of the nearest roadmap nodes to try when putting the robot back on the
# graph after a manipulation. At a 0.3 m grid step this reaches roughly 1.5 m out,
# far enough to clear the obstacle it has just let go of.
_REANCHOR_CANDIDATES = 64


@dataclass
class RunResult:
    success: bool                           # whether goal was reached
    C: float = 0.0                          # objective actually minimised: (1-w)J + w*time_value*T
    J: float = 0.0                          # energy cost = walk_cost + work_cost
    walk_cost: float = 0.0                  # motion cost = λ × total travel distance
    manip_walk_cost: float = 0.0            # part of walk_cost spent escorting obstacles
    work_cost: float = 0.0                  # manipulation cost = Σ(true difficulty × distance moved)
    risk_cost: float = 0.0                  # R: one surcharge per obstacle moved, by risk level
    risk_levels: dict = field(default_factory=dict)   # oid -> level charged for
    T: float = 0.0                          # simulated time elapsed = move_time + plan_time [s]
    move_time: float = 0.0                  # of which spent driving and turning [s]
    wait_time: float = 0.0                  # of which spent standing still for the world [s]
    cycles: int = 0                         # number of replan cycles
    plan_time: float = 0.0                  # total planning time (seconds)
    first_plan_time: float = 0.0            # first plan time (seconds) — cold-start cost measure
    total_expansions: int = 0               # total A* node expansions (all rounds combined)
    llm_calls: int = 0                      # LLM API call count
    llm_mode: str = "heuristic"             # LLM mode: heuristic / deepseek
    removed: List[int] = field(default_factory=list)    # list of moved obstacle IDs
    robot_track: List[Tuple[float, float]] = field(     # robot node coordinate sequence
        default_factory=list)
    frames: List[dict] = field(default_factory=list)    # per-frame snapshots (for render_sequence)
    world_events: List[str] = field(default_factory=list)  # what the world did on its own
    message: str = ""                       # result description

class OnlineNAMO:
    """Run online NAMO while keeping ground truth separate from robot belief."""
    def __init__(self, workspace: Polygon,
                 static_obstacles: List[StaticObstacle],
                 movable_obstacles: List[MovableObstacle],
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 cfg: Config,
                 events=None):
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
        # Ground truth moves on its own; the belief is never shown this object.
        self.dynamics = dynamics.WorldDynamics(
            self.world, static_obstacles, workspace, events, cfg)
        self._risk_charged: set = set()             # oids whose surcharge is already paid
        self.stranded = False       # robot ended a manipulation with no roadmap node in reach
        self._plan_paths: List[dict] = []           # all currently planned paths (for per-frame visualisation)
        self.failed_moves: set = set()
        # Manipulation may move the robot away from its current roadmap node.
        self.robot_xy: Tuple[float, float] = self.roadmap.nodes[self.start_node]
        # Driving, turning, and planning advance the same simulated clock.
        self.clock: float = 0.0
        # The disc footprint is symmetric, but turning still costs time.
        self.robot_heading: float = 0.0
        self._free_profile = cfg.free_profile()
        self._loaded_profile = cfg.loaded_profile()
        self._frame_node = self.start_node   # node the last frame was drawn against
        self._world_frame_t = 0.0            # clock at the last "world moves" frame
        self._waited = 0.0                   # time stood still in this stuck spell

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
        node = self.start_node                           # robot current roadmap node
        res.robot_track.append(self.roadmap.nodes[node])
        
        self.belief.perceive(self.world, self.roadmap.nodes[node])
        self._capture_frame(res, node, "start")

        planner = Planner(self.roadmap, self.belief, self.estimator, cfg,
                          self.failed_moves, self.risk)

        for cycle in range(cfg.max_replans):
            t0 = time.time()
            plan = planner.plan(node, self.goal_node)
            dt = time.time() - t0
            res.plan_time += dt
            if cfg.plan_time_in_clock:
                self._world_frame(res, self._advance_clock(res, dt, moving=False))
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
            for act in plan.actions:
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    # Moving requires contact, which reveals the true difficulty.
                    risk_before = self.risk.level_of(act["oid"])
                    touched = self.belief.touch_check(self.robot_xy, self.world, cfg)
                    if self.belief.reveal_by_interaction(act["oid"], self.world):
                        touched.append(act["oid"])
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # A higher contact risk invalidates the current plan.
                    risk_after = self.risk.level_of(act["oid"])
                    if (risk_before is not None
                            and risk.higher(risk_before, risk_after) != risk_before):
                        self._capture_frame(
                            res, node,
                            f"contact re-rated {act['oid']} {risk_before} -> "
                            f"{risk_after} -> replan")
                        break
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
                elif act["type"] == "move":
                    prev_node = node    # remember where the move started
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # advance to the contact point, then retreat to the
                        # node it set out from; both legs are real travel, real
                        # time, and leave it facing back the way it came
                        leg = t_contact * act["dist"]
                        self._drive(res, from_pos, contact_pos, dist=leg)
                        self._drive(res, contact_pos, from_pos, dist=leg)
                        self.belief.touch_check(contact_pos, self.world, cfg)
                        self.belief.perceive(self.world, from_pos)
                        self._capture_frame(
                            res, node, f"collision revealed {hit_oids} -> replan")
                        break
                    self._drive(res, from_pos, to_pos, dist=act["dist"])
                    node = act["v"]
                    moves_done += 1
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    self.belief.perceive(self.world, self.roadmap.nodes[node])
                    self._capture_frame(res, node, f"move to node {node}")
                    if node == self.goal_node:
                        reached_goal = True
                        break
                    if moves_done >= cfg.step_execute_edges:
                        break

            res.cycles = cycle + 1
            if self.stranded:
                res.message = "Robot ended a manipulation with no roadmap node in reach."
                break
            if reached_goal or node == self.goal_node:
                break

        return self._finalize(res, node)

    def _finalize(self, res: RunResult, node: int) -> RunResult:
        """Close the books. Both ways out of the loop come through here, so a run
        that gave up reports the same fields as one that reached the goal."""
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
        """Record what an obstacle ran into, without naming it.

        A collision reveals that *something* is there, not what it is: the region
        of overlap goes into the belief as an anonymous contact, and the obstacle
        behind it stays unidentified until the robot actually sees it.
        """
        for oid_hit, region in hits:
            if oid_hit is None:
                continue      # hit wall: walls are known static geometry, no new info to register
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
        """Bill λ × dist of robot travel. Manipulation travel is tracked separately
        for reporting but lands in the same λ·D term of J."""
        charge = cost.motion_cost(self.cfg, dist)
        res.walk_cost += charge
        res.J += charge
        if in_contact:
            res.manip_walk_cost += charge

    def _charge_risk(self, res: RunResult, oid: int):
        """Bill the risk surcharge for disturbing this obstacle, once and once only.

        By the time this runs the robot has necessarily touched the obstacle, so
        the level charged is the post-contact one — the first-sight verdict only
        ever steered the planner.
        """
        if oid in self._risk_charged:
            return
        self._risk_charged.add(oid)
        level = self.risk.level_of(oid)
        charge = cost.risk_cost(self.cfg, level)
        if charge <= 0.0:
            res.risk_levels[oid] = level
            return
        res.risk_cost += charge
        res.risk_levels[oid] = level

    def _advance_clock(self, res: RunResult, seconds: float,
                       moving: bool = True) -> bool:
        """Let `seconds` of simulated time pass, for the robot and for the world.

        One clock drives both, so an obstacle with somewhere to be covers ground
        while the robot drives, turns, and stands still thinking. Returns whether
        ground truth changed, which the caller turns into an animation frame.
        """
        if seconds <= 0.0:
            return False
        self.clock += seconds
        if moving:
            res.move_time += seconds
        if not self.dynamics.active:
            return False
        before = self.dynamics.version
        for note in self.dynamics.advance(seconds, self.clock, self.robot_xy):
            res.world_events.append(f"t={self.clock:,.1f}s  {note}")
            self.cfg.log(f"[world] t={self.clock:,.1f}s {note}")
        return self.dynamics.version != before

    def _wait_for_world(self, res: RunResult, node: int) -> bool:
        """Stand still and let the world move, when there is nowhere to go.

        On a map that changes, a way that is shut now may be open in a minute:
        the thing across it may be driving through rather than parked. So a
        moment with no plan in it is a reason to wait and look again rather than
        to stop — up to a point, after which the way really is shut. Waiting
        costs time and no distance, which is what standing still costs.
        """
        if not self.dynamics.active or self._waited >= self.cfg.dynamic_max_wait:
            return False
        step = max(self.cfg.dynamic_wait_step, 1e-3)
        self._waited += step
        res.wait_time += step
        self._world_frame(res, self._advance_clock(res, step, moving=False))
        self.belief.perceive(self.world, self.robot_xy)
        self._capture_frame(
            res, node, f"nowhere to go — waiting ({self._waited:,.0f}s)")
        return True

    def _world_frame(self, res: RunResult, moved: bool):
        """Draw the world in motion, at most one frame per animation step."""
        if not moved or self.clock - self._world_frame_t < self.cfg.gif_time_step:
            return
        self._world_frame_t = self.clock
        self._capture_frame(res, self._frame_node, "the world moves")

    def _drive(self, res: RunResult, a, b, in_contact: bool = False,
               loaded: bool = False, dist: Optional[float] = None):
        """Take the robot from *a* to *b*: bill the joules, spend the time, face the way it went.

        The single place the robot moves, so no journey can be charged in one
        currency and not the other. `loaded` selects the slower profile, which
        applies from the moment it grips an obstacle until it lets go.

        `dist` overrides the billed length for callers holding the authoritative
        figure. A roadmap edge is charged at the length the planner costed it
        with — `edge_len`, which is rounded — rather than at one recomputed from
        the endpoints, so what execution bills and what the plan predicted agree
        to the last decimal instead of drifting apart by a rounding step an edge.
        """
        self._charge_walk(res, math.dist(a, b) if dist is None else dist,
                          in_contact)
        profile = self._loaded_profile if loaded else self._free_profile
        seconds, self.robot_heading = kinematics.segment_time(
            profile, a, b, self.robot_heading)
        moved = self._advance_clock(res, seconds)
        self._set_robot(res, b)
        self._world_frame(res, moved)

    def _set_robot(self, res: RunResult, p: Tuple[float, float]):
        self.robot_xy = (float(p[0]), float(p[1]))
        res.robot_track.append(self.robot_xy)

    def _walk_frames(self, pts, cfg: Config) -> set:
        """Which legs of a walk to capture a frame after.

        Only the legs that actually move — a contact plan holds its grip point
        still for several steps at a time — and no more of those than one
        manipulation is allowed. Walking round a stationary obstacle to reach the
        far face is travel that takes time, and a frame list that skips it is
        what makes the robot appear to jump across the obstacle.
        """
        moving = [i for i in range(1, len(pts))
                  if math.dist(pts[i - 1], pts[i]) > 1e-9]
        return {moving[j] for j in
                self._sample_move_path(moving, cfg.manip_max_frames_per_action)}

    def _walk_robot(self, pts, res: RunResult, cfg: Config,
                    loaded: bool = False, node: Optional[int] = None,
                    label: str = "", move_oid: Optional[int] = None):
        """Drive the robot through *pts* (pts[0] is where it already stands).

        Stops at the first obstacle it did not know about. Returns
        (index reached, hit oids, stop position). Given *node*, the walk is
        animated as well as driven.
        """
        frame_at = self._walk_frames(pts, cfg) if node is not None else ()
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._drive(res, a, stop, in_contact=True, loaded=loaded)
                self.belief.touch_check(stop, self.world, cfg)
                return i - 1, hits, stop
            self._drive(res, a, b, in_contact=True, loaded=loaded)
            if i in frame_at:
                self._capture_frame(res, node, label, move_oid=move_oid)
        return len(pts) - 1, [], (pts[-1] if pts else self.robot_xy)

    def _retrace(self, res: RunResult, pts, node: Optional[int] = None,
                 label: str = "", move_oid: Optional[int] = None):
        """Back out along ground the robot has just covered — no collision check
        needed, it was clear a moment ago and nothing has moved since."""
        frame_at = self._walk_frames(pts, self.cfg) if node is not None else ()
        for i in range(1, len(pts)):
            self._drive(res, pts[i - 1], pts[i], in_contact=True)
            if i in frame_at:
                self._capture_frame(res, node, label, move_oid=move_oid)

    def _reanchor(self, res: RunResult, node: int) -> Optional[int]:
        """Put the robot back on the roadmap after a manipulation.

        Returns None when it made it back to the node the excursion started from
        — the rest of the plan is then still valid — and the new node otherwise.
        Every node it settles for is one it can reach in a straight line it has
        checked: driving to the nearest node regardless, which is what used to
        happen when none was reachable, put the robot through whatever stood in
        between. If nothing is reachable it stays put and the run ends there.
        """
        blocked = self._known_obstacles_inflated()
        home = self.roadmap.nodes[node]
        if self.roadmap.can_drive(self.robot_xy, home, blocked):
            self._drive(res, self.robot_xy, home, in_contact=True)
            return None
        target = self.roadmap.nearest_reachable_node(self.robot_xy, blocked,
                                                     k=_REANCHOR_CANDIDATES)
        if target is None:
            self.stranded = True
            self.cfg.log(f"[reanchor] no roadmap node reachable from "
                         f"({self.robot_xy[0]:,.2f}, {self.robot_xy[1]:,.2f})")
            return None
        self._drive(res, self.robot_xy, self.roadmap.nodes[target], in_contact=True)
        return target

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
        """Plan the escort for one manipulation from where the robot is standing.

        Same inputs the search used, with two differences that matter. The real
        robot position rather than the edge midpoint that stood in for it, which
        is what makes the walk to the grip point a route the robot could drive
        instead of a straight line through the obstacle it is about to push. And
        a *choice* of where to let go: the far end of the edge it has just
        cleared, or the end it set out from. The far end is worth the length of
        that edge, so it wins whenever the obstacle has moved far enough to leave
        a line to it — which is the whole point of having moved the obstacle. The
        robot then carries on from there instead of walking its grip back down
        the surface it just came along to restart from where it began.

        Returns the plan and, one per candidate release point, the roadmap node
        the robot ends up on — None for the position it is standing on now, which
        is `node` already.
        """
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
        """Move one obstacle, with the robot holding it and the world held off.

        From the moment the robot commits to moving something, that something is
        the robot's: whatever it was doing under its own steam is suspended for
        the length of the manipulation, and it starts again from wherever it is
        put down. Suspending on the decision rather than on the grip is what
        makes the escort plan mean anything — a body still under way would have
        drifted out from under its own grip points by the time the robot had
        walked over to them.
        """
        self.dynamics.suspend(oid)
        try:
            return self._escort(oid, obs, act, res, node, cfg)
        finally:
            self.dynamics.release(oid)

    def _escort(self, oid: int, obs, act: dict,
                res: RunResult, node: int, cfg: Config):
        """Move one obstacle along its SE2 path with the robot holding on to it.

        Returns (success, hits, obstacle distance moved, new node or None).
        ``hits`` is None when the *robot* — not the obstacle — ran into something
        unknown; that case is fully handled here and only needs a replan.
        """
        move_path = act["move_path"]
        frame_at = set(self._sample_move_path(move_path,
                                              cfg.manip_max_frames_per_action))
        n = len(move_path)
        start_xy = (obs.x, obs.y)
        home = self.robot_xy

        cplan = act.get("contact")
        exit_nodes: List[Optional[int]] = [None]
        if not cfg.contact_required or cplan is None or not cplan.feasible:
            # contact model disabled: the obstacle moves while the robot waits on
            # its node. Rebuilt around the robot's real position rather than
            # reusing the planned one, which is anchored to the edge midpoint.
            cplan = contact.idle_plan(home, n)
        else:
            # The planned escort was measured from the edge *midpoint*, which
            # stands in for either endpoint during the search. The robot is on
            # one of them, up to conn_radius away, so the approach and exit legs
            # that plan validated are not the ones about to be driven. Plan them
            # again from where the robot actually stands.
            cplan, exit_nodes = self._contact_plan(oid, obs, move_path, node,
                                                   act.get("key"))
            if not cplan.feasible:
                self.failed_moves.add((move_signature(obs), act["key"]))
                self._capture_frame(
                    res, node,
                    f"cannot escort {oid} from node {node}: {cplan.reason} -> replan")
                return (False, None, 0.0, None)
        if self.dynamics.active and not self._still_there(oid, move_path[0]):
            # It has gone somewhere else since the plan was drawn up, so the plan
            # is about a pose it is not in. Look again and think again — it is
            # standing still now, so the next plan will still be true when the
            # robot reaches it.
            self.belief.perceive(self.world, self.robot_xy)
            self._capture_frame(
                res, node, f"{oid} is no longer where it was -> replan")
            return (False, None, 0.0, None)
        rp = list(cplan.robot_path)
        off = cplan.move_offset

        # One STRtree for the whole manipulation instead of an O(N) polygon scan
        # per sub-step — rebuilt only if the world moves underneath it.
        tree, tree_items = self._collision_index(oid)
        world_version = self.dynamics.version

        reached, hits, stop = self._walk_robot(
            rp[:off + 1], res, cfg, node=node, label=f"walk round to grip {oid}")
        if hits:
            self._retrace(res, [stop] + rp[reached::-1], node=node,
                          label=f"back off from {oid}")
            self.belief.perceive(self.world, self.robot_xy)
            self._capture_frame(
                res, node, f"robot hit {sorted(hits)} approaching {oid} -> replan")
            return (False, None, 0.0, None)
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
                # obstacle stopped at move_path[last_i]; belief syncs to that pose
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
            # the robot is holding the obstacle, so it can run into things too
            a, b = rp[off + i - 1], rp[off + i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                if last_i != 0:
                    self.belief.relocate(obs, *move_path[last_i])
                    self.belief.record_move_direction(oid, start_xy, move_path[last_i])
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._drive(res, a, stop, in_contact=True, loaded=True)
                self.belief.touch_check(stop, self.world, cfg)
                self.belief.perceive(self.world, self.robot_xy)
                new_node = self._reanchor(res, node)
                self._capture_frame(
                    res, node if new_node is None else new_node,
                    f"robot hit {sorted(hits)} while moving {oid} -> replan")
                return (False, None,
                        cost.se2_path_length(obs, move_path[:last_i + 1], cfg),
                        new_node)
            self._relocate_world(oid, wx, wy, wth)
            self._drive(res, a, b, in_contact=True, loaded=True)
            last_i = i
            if i in frame_at:
                self._capture_frame(res, node, f"move {oid} step {i}/{n - 1}",
                                    move_oid=oid)

        # update belief with final pose
        self.belief.relocate(obs, *move_path[-1])
        self.belief.record_move_direction(oid, start_xy, move_path[-1])
        self.belief.perceive(self.world, self.robot_xy)
        new_node = self._release_and_return(res, oid, rp, off, n - 1, True,
                                            node, cfg, cplan, exit_nodes)
        return (True, [], cost.se2_path_length(obs, move_path, cfg), new_node)

    def _release_and_return(self, res: RunResult, oid: int, rp: list, off: int,
                            last_i: int, completed: bool, node: int,
                            cfg: Config, cplan, exit_nodes) -> Optional[int]:
        """Let go of the obstacle and get back onto the roadmap.

        After a completed move the planned exit walk applies: round the obstacle
        if need be, then out to the release point the escort plan chose. That is
        a roadmap node, and returning it is what lets the next cycle carry on
        from where the robot actually is — normally the far side of the edge it
        has just cleared — instead of restarting from the node it set out from.
        After an aborted move the robot is stranded at a grip point the plan
        never expected it to let go at — the way out may now be through the
        obstacle it was holding — so it drives to the nearest roadmap node it can
        actually reach and the caller replans from there.
        """
        if not completed:
            return self._reanchor(res, node)
        _reached, hits, _stop = self._walk_robot(
            rp[off + last_i:], res, cfg, node=node,
            label=f"let go of {oid} and walk on")
        if hits:
            self.belief.perceive(self.world, self.robot_xy)
            new_node = self._reanchor(res, node)
            self._capture_frame(res, node if new_node is None else new_node,
                                f"robot hit {sorted(hits)} backing out -> replan")
            return new_node
        i = cplan.exit_index
        landed = exit_nodes[i] if 0 <= i < len(exit_nodes) else None
        if landed is None or landed == node:
            return None
        # It let go somewhere it has never stood, so it looks around from there
        # before anything plans on its behalf — the same courtesy an ordinary
        # roadmap edge gets at the far end.
        self.belief.touch_check(self.robot_xy, self.world, cfg)
        self.belief.perceive(self.world, self.robot_xy)
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
        """Is the obstacle close enough to where the plan expects it?

        Judged at the resolution the SE(2) route was planned at, because that is
        the accuracy the route itself has: a body that has drifted less than one
        cell while the robot was thinking is put back on the planned route by the
        first sub-step, and one that has drifted further is somewhere else.
        """
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
            "move_oid": move_oid,   # obstacle currently being moved (drawn above the robot)
            "plan_paths": list(self._plan_paths),   # planned paths being executed at this frame
            "robot": self.robot_xy,     # true position — off-node while holding an obstacle
            "track": list(res.robot_track),
            "obstacles": [(w.oid, w.polygon, w.removed) for w in self.world],
            "world_moved": sorted(self.dynamics.actors),   # sent somewhere by the world
            "perceived": perceived,
            "estimated_difficulty": {
                oid: value for oid, value in self.estimator.cache.items()
                if oid in perceived
            },
            "touched_difficulty": dict(self.belief.touched_difficulty),
            "risk": {oid: self.risk.level_of(oid) for oid in perceived},
            "J": round(res.J, 4),
            "t": round(self.clock, 3),      # simulated seconds; the animation runs on this
            "move_t": round(res.move_time, 1),   # shown split rather than summed
            "plan_t": round(res.plan_time, 1),
            "label": label,
        })
