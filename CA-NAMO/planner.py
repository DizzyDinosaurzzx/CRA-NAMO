"""
在线“规划-执行-感知-重规划”循环（§感知与在线重规划）。

    plan  ：在当前信念状态下的最小代价规划（search.Planner）
    execute：执行该规划的第一条路网边（以及所需的障碍物移除），将真实代价累加到 J
    perceive：揭示新可见的障碍物（包括刚移动障碍物所暴露的物体），并增量更新路网信念
    replan ：重复上述过程，直到到达目标节点或不存在可行规划

对于未探索空间，一切都采用乐观假设（未知边视为畅通），因此执行过程可能揭示阻挡物，
使原先的乐观规划失效并触发重规划，这正是预期的闭环过程。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, Point

from config import Config
from obstacle import MovableObstacle, StaticObstacle
from roadmap import Roadmap
from perception import Belief
from llm_difficulty import DifficultyEstimator
from search import Planner, Plan


@dataclass
class RunResult:
    success: bool
    J: float = 0.0
    walk_cost: float = 0.0
    work_cost: float = 0.0
    cycles: int = 0
    plan_time: float = 0.0
    first_plan_time: float = 0.0
    total_expansions: int = 0
    llm_calls: int = 0
    llm_mode: str = "heuristic"
    removed: List[int] = field(default_factory=list)
    robot_track: List[Tuple[float, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)   # snapshots for visualisation
    message: str = ""


class OnlineNAMO:
    def __init__(self, workspace: Polygon,
                 static_obstacles: List[StaticObstacle],
                 movable_obstacles: List[MovableObstacle],   # ground-truth world
                 start: Tuple[float, float],
                 goal_region: Polygon,
                 cfg: Config):
        self.cfg = cfg
        self.workspace = workspace
        self.static_obstacles = static_obstacles
        self.world = movable_obstacles          # ground truth (true difficulty)
        self.start = start
        self.goal_region = goal_region

        self.roadmap = Roadmap(workspace, static_obstacles, cfg)
        self.start_node = self.roadmap.add_terminal(start)
        gp = goal_region.centroid
        gpt = (gp.x, gp.y)
        if not self.roadmap.static_free.contains(Point(gpt).buffer(cfg.robot_radius)):
            rp = goal_region.representative_point()
            gpt = (rp.x, rp.y)
        self.goal_node = self.roadmap.add_terminal(gpt)

        self.estimator = DifficultyEstimator(cfg)
        self.belief = Belief(self.roadmap, cfg)

    # -------------------------------------------------------------------- 运行
    def run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
        node = self.start_node
        res.robot_track.append(self.roadmap.nodes[node])

        self.belief.perceive(self.world, self.roadmap.nodes[node])

        planner = Planner(self.roadmap, self.belief, self.estimator, cfg)

        for cycle in range(cfg.max_replans):
            t0 = time.time()
            plan = planner.plan(node, self.goal_node)
            dt = time.time() - t0
            res.plan_time += dt
            if cycle == 0:
                res.first_plan_time = dt
            if plan is None:
                res.message = "No feasible plan under current belief."
                res.cycles = cycle + 1
                return res
            res.total_expansions += plan.expansions
            self._snapshot(res, node, plan)

            if node == self.goal_node:
                break

            # ---- 执行规划的前置边（包括对应的障碍物移除）----
            moves_done = 0
            reached_goal = False
            for act in plan.actions:
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    dx, dy, dth = act["drop"]
                    self.belief.relocate(obs, dx, dy, dth)
                    self._relocate_world(act["oid"], dx, dy, dth)
                    res.work_cost += cfg.lambda_w * act["work"]
                    res.J += cfg.lambda_w * act["work"]
                    if act["oid"] not in res.removed:
                        res.removed.append(act["oid"])
                elif act["type"] == "move":
                    res.walk_cost += cfg.lambda_d * act["dist"]
                    res.J += cfg.lambda_d * act["dist"]
                    node = act["v"]
                    res.robot_track.append(self.roadmap.nodes[node])
                    moves_done += 1
                    # 每次实际移动后感知（揭示被暴露的障碍物）
                    self.belief.perceive(self.world, self.roadmap.nodes[node])
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

    # -------------------------------------------------------------- 辅助方法
    def _relocate_world(self, oid: int, x: float, y: float, theta: float):
        for w in self.world:
            if w.oid == oid:
                w.x, w.y, w.theta, w.removed = x, y, theta, True
                return

    def _snapshot(self, res: RunResult, node: int, plan: Plan):
        res.history.append({
            "robot": self.roadmap.nodes[node],
            "node_path": list(plan.node_path),
            "perceived": [o.oid for o in self.belief.perceived.values()],
            "cost": plan.cost,
        })
