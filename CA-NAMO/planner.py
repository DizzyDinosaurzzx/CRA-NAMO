"""CA-NAMO online "plan–execute–perceive–replan" loop"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from config import Config
from obstacle import MovableObstacle, StaticObstacle
from roadmap import Roadmap
from perception import Belief
from llm_difficulty import DifficultyEstimator
from search import Planner, push_signature
import geometry

_CONTACT_AREA_EPS = 1e-9

# Simulation result summary
@dataclass
class RunResult:
    success: bool                           # whether goal was reached
    J: float = 0.0                          # total cost = walk_cost + work_cost
    walk_cost: float = 0.0                  # motion cost = λ × total travel distance
    work_cost: float = 0.0                  # manipulation cost = Σ(true difficulty × push distance)
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

        self.estimator = DifficultyEstimator(cfg)   # obstacle difficulty estimator
        self.belief = Belief(self.roadmap, cfg)     # robot partial-observability belief
        self._plan_paths: List[dict] = []           # all currently planned paths (for per-frame visualisation)
        self.failed_pushes: set = set()

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
                          self.failed_pushes)

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

            # ---- execute the planned lead edges (including corresponding obstacle removals) ----
            moves_done = 0
            reached_goal = False
            for act in plan.actions:
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    # pre-push touch sensing: learn the true difficulty of this obstacle
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # step through SE2 push path
                    push_success, hits, executed_dist = self._execute_push_path(
                        act["oid"], obs, act["push_path"], res, node, cfg)
                    # charge only for the segment actually pushed, using the true difficulty
                    if executed_dist > 0.0:
                        true_diff = self._world_obstacle(act["oid"]).difficulty
                        executed_work = geometry.push_work(true_diff, executed_dist)
                        res.work_cost += executed_work
                        res.J += executed_work
                        if act["oid"] not in res.removed:
                            res.removed.append(act["oid"])
                    elif not push_success:
                        self.failed_pushes.add((push_signature(obs), act["key"]))
                    if not push_success:
                        # collision — invalidate this placement and replan a new location
                        self._handle_push_collision(res, node, act["oid"], hits)
                        break
                elif act["type"] == "move":
                    prev_node = node    # req 3: record position before move
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    # req 3: collision sensing
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # advance to contact point then retreat to previous node; charge both legs
                        blocked_dist = 2.0 * t_contact * act["dist"]
                        res.walk_cost += cfg.lambda_distance * blocked_dist
                        res.J += cfg.lambda_distance * blocked_dist
                        # collision is physical contact; the true difficulty of the hit object is revealed
                        self.belief.touch_check(contact_pos, self.world, cfg)
                        self.belief.perceive(self.world, from_pos)
                        self._capture_frame(
                            res, node, f"collision revealed {hit_oids} -> replan")
                        break
                    res.walk_cost += cfg.lambda_distance * act["dist"]
                    res.J += cfg.lambda_distance * act["dist"]
                    node = act["v"]
                    res.robot_track.append(self.roadmap.nodes[node])
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

    def _world_collision(self, oid: int, nx: float, ny: float, theta: float):
        if not self.cfg.check_obstacle_collision:
            return []
        mover = self._world_obstacle(oid)
        end_poly = mover.polygon_at(nx, ny, theta)
        swept = geometry._swept_region(mover, nx, ny, theta)
        hits = []

        def overlapping(poly):
            contact = swept.intersection(poly)
            if contact.area > _CONTACT_AREA_EPS:
                return contact
            # swept region already contains the goal pose; normally no separate end_poly check needed
            # _swept_region can produce slightly undersized geometry on degenerate input.
            end_contact = end_poly.intersection(poly)
            return end_contact if end_contact.area > _CONTACT_AREA_EPS else None

        for w in self.world:
            if w.oid == oid:
                continue
            contact = overlapping(w.polygon)   # only "touch" the contact overlap region, not the whole obstacle
            if contact is not None:
                hits.append((w.oid, contact))
        for so in self.static_obstacles:
            contact = overlapping(so.polygon)
            if contact is not None:
                hits.append((None, contact))
        return hits

    def _handle_push_collision(self, res: RunResult, node: int, oid: int, hits):
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
            label = f"push {oid} hit a wall -> replan"
        elif cfg.full_reveal_on_contact:
            label = f"push {oid} blocked by {hit_oids} -> replan"
        else:
            label = f"push {oid} hit unknown obstruction -> replan"
        self._capture_frame(res, node, label)

    @staticmethod
    def _sample_push_path(push_path: list, max_frames: int) -> list:
        if len(push_path) <= max_frames:
            return list(range(len(push_path)))
        return [int(i) for i in np.linspace(0, len(push_path) - 1, max_frames)]

    def _execute_push_path(self, oid: int, obs, push_path: list,
                           res: RunResult, node: int, cfg: Config):
        frame_at = set(self._sample_push_path(push_path,
                                              cfg.push_max_frames_per_action))
        n = len(push_path)
        last_i = 0

        for i in range(1, n):
            wx, wy, wth = push_path[i]
            if cfg.check_obstacle_collision:
                hits = self._world_collision(oid, wx, wy, wth)
                if hits:
                    # obstacle was pushed to push_path[last_i] before being stopped; belief must sync to that actual pose
                    if last_i != 0:
                        self.belief.relocate(obs, *push_path[last_i])
                    return (False, hits,
                            geometry.se2_path_cost(obs, push_path[:last_i + 1], cfg))
            self._relocate_world(oid, wx, wy, wth)
            last_i = i
            if i in frame_at:
                self._capture_frame(res, node, f"push {oid} step {i}/{n - 1}")
        # update belief with final pose
        self.belief.relocate(obs, *push_path[-1])
        return (True, [], geometry.se2_path_cost(obs, push_path, cfg))

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
            paths.append({"kind": "move", "pts": route})
        for act in plan.actions:
            if act["type"] != "remove":
                continue
            push_path = act.get("push_path") or []
            pts = [(p[0], p[1]) for p in push_path]
            if len(pts) >= 2:
                paths.append({"kind": "push", "oid": act["oid"], "pts": pts})
        return paths

    def _capture_frame(self, res: RunResult, node: int, label: str):
        if not self.cfg.save_frames:
            return
        res.frames.append({
            "node": node,
            "plan_paths": list(self._plan_paths),   # planned paths being executed at this frame
            "robot": self.roadmap.nodes[node],
            "track": list(res.robot_track),
            "obstacles": [(w.oid, w.polygon, w.removed) for w in self.world],
            "perceived": set(self.belief.perceived.keys()),
            "J": round(res.J, 4),
            "label": label,
        })
