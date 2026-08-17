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
from obstacle import MovableObstacle, StaticObstacle
from roadmap import Roadmap
from perception import Belief
from llm_difficulty import DifficultyEstimator
from search import Planner, move_signature
import cost
import geometry
import manipulation

# Simulation result summary
@dataclass
class RunResult:
    success: bool                           # whether goal was reached
    J: float = 0.0                          # total cost = walk_cost + work_cost
    walk_cost: float = 0.0                  # motion cost = λ × total travel distance
    manip_walk_cost: float = 0.0            # part of walk_cost spent escorting obstacles
    work_cost: float = 0.0                  # manipulation cost = Σ(true difficulty × distance moved)
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
    message: str = ""                       # result description

# Online simulator
class OnlineNAMO:
    """
    Maintains two world models:
    - self.world (real world)
    - self.belief (robot belief)
    The planner searches against belief only and cannot access the real world
    """
    def __init__(self, workspace: Polygon,
                 static_obstacles: List[StaticObstacle],
                 movable_obstacles: List[MovableObstacle],
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 cfg: Config):
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
        self.belief = Belief(self.roadmap, cfg)     # robot partial-observability belief
        self._plan_paths: List[dict] = []           # all currently planned paths (for per-frame visualisation)
        self.failed_moves: set = set()
        # true robot position. It equals the current roadmap node between actions,
        # but during a manipulation the robot leaves the node to hold the obstacle.
        self.robot_xy: Tuple[float, float] = self.roadmap.nodes[self.start_node]

    def _add_terminal(self, p: Tuple[float, float]) -> int:
        # insert start/goal into roadmap; snap to nearest valid node when the robot disc does not fit
        cfg = self.cfg
        if not self.roadmap.free_eroded_prep.contains(Point(p)):
            # point is inside a wall or too close -> degrade to nearest valid roadmap node
            p = self.roadmap.nodes[self.roadmap.nearest_node(p)]
        return self.roadmap.add_terminal(p)

    # Main perception–action loop
    def run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
        node = self.start_node                           # robot current roadmap node
        res.robot_track.append(self.roadmap.nodes[node])
        
        # initial perception: scan visible obstacles around start
        self.belief.perceive(self.world, self.roadmap.nodes[node])
        self._capture_frame(res, node, "start")

        planner = Planner(self.roadmap, self.belief, self.estimator, cfg,
                          self.failed_moves)

        for cycle in range(cfg.max_replans):
            t0 = time.time()
            plan = planner.plan(node, self.goal_node)
            dt = time.time() - t0
            res.plan_time += dt
            if cycle == 0:
                res.first_plan_time = dt
            self._plan_paths = self._plan_to_paths(plan)
            if plan is None:
                res.message = "No feasible plan under current belief."
                res.cycles = cycle + 1
                return res
            res.total_expansions += plan.expansions

            if node == self.goal_node:
                break

            # --- execute planned edges ---
            moves_done = 0
            reached_goal = False
            for act in plan.actions:
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    # pre-move touch sensing: learn the true difficulty of this obstacle.
                    # Moving is itself contact, so the moved obstacle is always revealed.
                    touched = self.belief.touch_check(self.robot_xy, self.world, cfg)
                    if self.belief.reveal_by_interaction(act["oid"], self.world):
                        touched.append(act["oid"])
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # escort the obstacle along its SE2 path, staying in contact
                    move_success, hits, executed_dist, new_node = \
                        self._execute_move(act["oid"], obs, act, res, node, cfg)
                    # a new node means the robot could not walk back to the one it
                    # set out from, so the rest of this plan no longer applies
                    if new_node is not None:
                        node = new_node
                    # charge only for the segment actually moved, using the true difficulty
                    if executed_dist > 0.0:
                        true_diff = self._world_obstacle(act["oid"]).difficulty
                        executed_work = cost.manipulation_work(true_diff, executed_dist)
                        res.work_cost += executed_work
                        res.J += executed_work
                        if act["oid"] not in res.removed:
                            res.removed.append(act["oid"])
                    elif not move_success and hits is not None:
                        self.failed_moves.add((move_signature(obs), act["key"]))
                    if not move_success:
                        # hits is None when the robot itself ran into something
                        # unknown — that is already recorded and framed inside
                        if hits is not None:
                            # collision — invalidate this placement and replan a new location
                            self._handle_move_collision(res, node, act["oid"], hits)
                        break
                    if new_node is not None:
                        break
                elif act["type"] == "move":
                    prev_node = node    # remember where the move started
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    # collision sensing
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # advance to contact point then retreat to previous node; charge both legs
                        blocked_dist = 2.0 * t_contact * act["dist"]
                        self._charge_walk(res, blocked_dist)
                        # collision is physical contact; the true difficulty of the hit object is revealed
                        self.belief.touch_check(contact_pos, self.world, cfg)
                        self.belief.perceive(self.world, from_pos)
                        self._capture_frame(
                            res, node, f"collision revealed {hit_oids} -> replan")
                        break
                    self._charge_walk(res, act["dist"])
                    node = act["v"]
                    self._set_robot(res, self.roadmap.nodes[node])
                    moves_done += 1
                    # touch sensing: learn true difficulty of touched obstacles
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # perceive after each actual move (reveals exposed obstacles)
                    self.belief.perceive(self.world, self.roadmap.nodes[node])
                    self._capture_frame(res, node, f"move to node {node}")
                    if node == self.goal_node:
                        reached_goal = True
                        break
                    if moves_done >= cfg.step_execute_edges:
                        break

            res.cycles = cycle + 1
            if reached_goal or node == self.goal_node:
                break

        res.success = (node == self.goal_node)
        res.J = round(res.J, 4)
        res.walk_cost = round(res.walk_cost, 4)
        res.manip_walk_cost = round(res.manip_walk_cost, 4)
        res.work_cost = round(res.work_cost, 4)
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
        if not self.cfg.check_obstacle_collision:
            return []
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
        cfg = self.cfg
        for oid_hit, region in hits:
            if oid_hit is None:
                continue      # hit wall: walls are known static geometry, no new info to register
            if cfg.full_reveal_on_contact:
                self.belief.force_reveal(self._world_obstacle(oid_hit))
            else:
                self.belief.register_contact(region)
        hit_oids = sorted(o for o, _ in hits if o is not None)
        if not hit_oids:
            label = f"move {oid} hit a wall -> replan"
        elif cfg.full_reveal_on_contact:
            label = f"move {oid} blocked by {hit_oids} -> replan"
        else:
            label = f"move {oid} hit unknown obstruction -> replan"
        self._capture_frame(res, node, label)

    @staticmethod
    def _sample_move_path(move_path: list, max_frames: int) -> list:
        if len(move_path) <= max_frames:
            return list(range(len(move_path)))
        return [int(i) for i in np.linspace(0, len(move_path) - 1, max_frames)]

    # --- robot bookkeeping ---
    def _charge_walk(self, res: RunResult, dist: float, in_contact: bool = False):
        """Bill λ × dist of robot travel. Manipulation travel is tracked separately
        for reporting but lands in the same λ·D term of J."""
        charge = cost.motion_cost(self.cfg, dist)
        res.walk_cost += charge
        res.J += charge
        if in_contact:
            res.manip_walk_cost += charge

    def _set_robot(self, res: RunResult, p: Tuple[float, float]):
        self.robot_xy = (float(p[0]), float(p[1]))
        res.robot_track.append(self.robot_xy)

    def _walk_robot(self, pts, res: RunResult, cfg: Config):
        """Drive the robot through *pts* (pts[0] is where it already stands).

        Stops at the first obstacle it did not know about. Returns
        (index reached, hit oids, stop position).
        """
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._charge_walk(res, math.dist(a, stop), in_contact=True)
                self._set_robot(res, stop)
                self.belief.touch_check(stop, self.world, cfg)
                return i - 1, hits, stop
            self._charge_walk(res, math.dist(a, b), in_contact=True)
            self._set_robot(res, b)
        return len(pts) - 1, [], (pts[-1] if pts else self.robot_xy)

    def _retrace(self, res: RunResult, pts):
        """Back out along ground the robot has just covered — no collision check
        needed, it was clear a moment ago and nothing has moved since."""
        for i in range(1, len(pts)):
            self._charge_walk(res, math.dist(pts[i - 1], pts[i]), in_contact=True)
            self._set_robot(res, pts[i])

    def _reanchor(self, res: RunResult, node: int) -> Optional[int]:
        """Put the robot back on the roadmap after a manipulation.

        Returns None when it made it back to the node the excursion started from
        — the rest of the plan is then still valid — and the new node otherwise.
        """
        blocked = self._known_obstacles_inflated()
        home = self.roadmap.nodes[node]
        if self.roadmap.can_drive(self.robot_xy, home, blocked):
            self._charge_walk(res, math.dist(self.robot_xy, home), in_contact=True)
            self._set_robot(res, home)
            return None
        target = self.roadmap.nearest_reachable_node(self.robot_xy, blocked)
        if target is None:
            target = self.roadmap.nearest_node(self.robot_xy)
        self._charge_walk(res, math.dist(self.robot_xy, self.roadmap.nodes[target]),
                          in_contact=True)
        self._set_robot(res, self.roadmap.nodes[target])
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

    # --- manipulation ---
    def _execute_move(self, oid: int, obs, act: dict,
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
        if not cfg.contact_required or cplan is None or not cplan.feasible:
            # contact model disabled: the obstacle moves while the robot waits on
            # its node. Rebuilt around the robot's real position rather than
            # reusing the planned one, which is anchored to the edge midpoint.
            cplan = contact.idle_plan(home, n)
        rp = list(cplan.robot_path)
        # the plan measured the excursion from the edge midpoint; the robot is
        # standing on one of that edge's endpoints, so re-anchor both ends to it
        rp[0] = home
        rp[-1] = home
        off = cplan.move_offset

        # Build STRtree once for all steps in this manipulation — avoids O(N) polygon
        # scans on every sub-step of the manipulation trajectory.
        tree = None
        tree_items = None
        if cfg.check_obstacle_collision:
            polys = []
            items = []
            for w in self.world:
                if w.oid != oid:
                    polys.append(w.polygon)
                    items.append((w.oid, w.polygon))
            for so in self.static_obstacles:
                polys.append(so.polygon)
                items.append((None, so.polygon))
            if polys:
                tree = STRtree(polys)
                tree_items = items

        # --- approach ---
        reached, hits, stop = self._walk_robot(rp[:off + 1], res, cfg)
        if hits:
            self._retrace(res, [stop] + rp[reached::-1])
            self.belief.perceive(self.world, self.robot_xy)
            self._capture_frame(
                res, node, f"robot hit {sorted(hits)} approaching {oid} -> replan")
            return (False, None, 0.0, None)
        self._capture_frame(res, node, f"grip obstacle {oid}", move_oid=oid)

        # --- move the obstacle ---
        last_i = 0
        for i in range(1, n):
            wx, wy, wth = move_path[i]
            if cfg.check_obstacle_collision:
                obs_hits = self._world_collision(oid, wx, wy, wth, tree=tree,
                                                 tree_items=tree_items)
                if obs_hits:
                    # obstacle stopped at move_path[last_i]; belief syncs to that pose
                    if last_i != 0:
                        self.belief.relocate(obs, *move_path[last_i])
                        self.belief.record_move_direction(oid, start_xy,
                                                          move_path[last_i])
                    new_node = self._release_and_return(res, rp, off, last_i,
                                                        False, node, cfg)
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
                self._charge_walk(res, math.dist(a, stop), in_contact=True)
                self._set_robot(res, stop)
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
            self._charge_walk(res, math.dist(a, b), in_contact=True)
            self._set_robot(res, b)
            last_i = i
            if i in frame_at:
                self._capture_frame(res, node, f"move {oid} step {i}/{n - 1}",
                                    move_oid=oid)

        # update belief with final pose
        self.belief.relocate(obs, *move_path[-1])
        self.belief.record_move_direction(oid, start_xy, move_path[-1])
        self.belief.perceive(self.world, self.robot_xy)
        new_node = self._release_and_return(res, rp, off, n - 1, True, node, cfg)
        return (True, [], cost.se2_path_length(obs, move_path, cfg), new_node)

    def _release_and_return(self, res: RunResult, rp: list, off: int, last_i: int,
                            completed: bool, node: int, cfg: Config) -> Optional[int]:
        """Let go of the obstacle and get back onto the roadmap.

        After a completed move the planned exit walk applies: round the obstacle
        if need be, then back to the node the excursion started from, leaving the
        rest of the plan valid. After an aborted one the robot is stranded at a
        grip point the plan never expected it to let go at — the way home may now
        be through the obstacle it was holding — so it drives to the nearest
        roadmap node it can actually reach and the caller replans from there.
        """
        if not completed:
            return self._reanchor(res, node)
        _reached, hits, _stop = self._walk_robot(rp[off + last_i:], res, cfg)
        if hits:
            self.belief.perceive(self.world, self.robot_xy)
            new_node = self._reanchor(res, node)
            self._capture_frame(res, node if new_node is None else new_node,
                                f"robot hit {sorted(hits)} backing out -> replan")
            return new_node
        return None

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
            "perceived": perceived,
            "estimated_difficulty": {
                oid: value for oid, value in self.estimator.cache.items()
                if oid in perceived
            },
            "touched_difficulty": dict(self.belief.touched_difficulty),
            "J": round(res.J, 4),
            "label": label,
        })

