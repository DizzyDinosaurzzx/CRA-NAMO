"""
在线"规划-执行-感知-重规划"循环（§感知与在线重规划）。

    plan  ：在当前信念状态下的最小代价规划（search.Planner）
    execute：执行该规划的第一条路网边（以及所需的障碍物移除），按 J=lambda_d*D+W
             累加代价
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
import geometry


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
    work_source: str = "direct"
    removed: List[int] = field(default_factory=list)
    robot_track: List[Tuple[float, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)   # snapshots for visualisation
    frames: List[dict] = field(default_factory=list)    # per-step frames (逐步动画)
    message: str = ""


class OnlineNAMO:
    def __init__(self, workspace: Polygon,
                 static_obstacles: List[StaticObstacle],
                 movable_obstacles: List[MovableObstacle],   # ground-truth world
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 cfg: Config):
        self.cfg = cfg
        self.workspace = workspace
        self.static_obstacles = static_obstacles
        self.world = movable_obstacles          # ground truth (true difficulty)
        self.start = start
        self.goal = (float(goal[0]), float(goal[1]))

        self.roadmap = Roadmap(workspace, static_obstacles, cfg)
        self.start_node = self.roadmap.add_terminal(start)
        gpt = self.goal
        if not self.roadmap.static_free.contains(Point(gpt).buffer(cfg.robot_radius)):
            # 目标点处机器人圆盘放不下（贴墙/在墙内）-> 退化到最近的合法路网节点
            gpt = self.roadmap.nodes[self.roadmap.nearest_node(gpt)]
        self.goal_node = self.roadmap.add_terminal(gpt)
        self.goal_point = self.roadmap.nodes[self.goal_node]   # 实际到达点（用于绘图）

        self.estimator = DifficultyEstimator(cfg)
        self.belief = Belief(self.roadmap, cfg)

    # -------------------------------------------------------------------- 运行
    def run(self) -> RunResult:
        cfg = self.cfg
        llm_mode = ("not_used" if cfg.work_source == "direct"
                    else self.estimator.mode)
        res = RunResult(success=False, llm_mode=llm_mode,
                        work_source=cfg.work_source)
        node = self.start_node
        res.robot_track.append(self.roadmap.nodes[node])

        self.belief.perceive(self.world, self.roadmap.nodes[node])
        self._capture_frame(res, node, "start")

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
                    # 触碰事件保留；direct 模式下不会改写已知的 W。
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        touch_label = (
                            f"touch detected {touched}"
                            if cfg.work_source == "direct"
                            else f"touch revealed difficulty of {touched}"
                        )
                        self._capture_frame(res, node,
                                            touch_label)
                    # 物理校验：在真实世界里，这次推动是否会撞到别的障碍物（含尚未
                    # 感知的）？规划是乐观的、只避开已知障碍物；现实由此处兜底。
                    hits = self._world_collision(act["oid"], dx, dy, dth)
                    if hits:
                        # 撞上了 -> 作废本次放置并重规划换位置。撞到什么程度的"知情"取决于配置：
                        #   full_reveal_on_contact=True : 直接获知被撞障碍物全部信息
                        #   False(更真实)              : 只记录"此处有物"的匿名接触区域，
                        #                                身份/几何/难度要等遮挡移开后正常感知
                        for oid_hit, region in hits:
                            if cfg.full_reveal_on_contact:
                                self.belief.force_reveal(self._world_obstacle(oid_hit))
                            else:
                                self.belief.register_contact(region)
                        if cfg.full_reveal_on_contact:
                            label = (f"push {act['oid']} blocked by "
                                     f"{sorted(o for o, _ in hits)} -> replan")
                        else:
                            label = (f"push {act['oid']} hit unknown obstruction "
                                     f"-> replan")
                        self._capture_frame(res, node, label)
                        break
                    self.belief.relocate(obs, dx, dy, dth)
                    self._relocate_world(act["oid"], dx, dy, dth)
                    # direct 模式与搜索使用同一个给定 W；estimated 模式保留旧执行语义。
                    if cfg.work_source == "direct":
                        executed_work = act["work"]
                    else:
                        executed_work = geometry.push_work(obs, act["dist"])
                    res.work_cost += executed_work
                    res.J += executed_work
                    if act["oid"] not in res.removed:
                        res.removed.append(act["oid"])
                    self._capture_frame(res, node, f"push obstacle {act['oid']}")
                elif act["type"] == "move":
                    prev_node = node    # 需求3: 记录移动前位置
                    res.walk_cost += cfg.lambda_d * act["dist"]
                    res.J += cfg.lambda_d * act["dist"]
                    node = act["v"]
                    res.robot_track.append(self.roadmap.nodes[node])
                    moves_done += 1
                    # 需求3: 碰撞感知
                    hit_oids = self.belief.check_robot_collision(
                        self.roadmap.nodes[prev_node],
                        self.roadmap.nodes[node],
                        self.world, cfg)
                    if hit_oids:
                        self._capture_frame(res, node,
                            f"collision revealed {hit_oids}")
                    # 触碰感知保留；direct 模式只记录事件。
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        touch_label = (
                            f"touch detected {touched}"
                            if cfg.work_source == "direct"
                            else f"touch revealed difficulty of {touched}"
                        )
                        self._capture_frame(res, node,
                                            touch_label)
                    # 每次实际移动后感知（揭示被暴露的障碍物）
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

    # -------------------------------------------------------------- 辅助方法
    def _world_obstacle(self, oid: int) -> Optional[MovableObstacle]:
        for w in self.world:
            if w.oid == oid:
                return w
        return None

    def _world_collision(self, oid: int, nx: float, ny: float, theta: float):
        """真实世界物理校验：把 oid 推到 (nx, ny, theta) 是否与其他障碍物碰撞。

        对*全部*可移动障碍物（含机器人尚未感知的）检查终点位姿与推动扫掠路径。
        返回被撞到的障碍物列表 [(oid, contact_region), ...]，其中 contact_region 是
        被推物体扫掠区与该障碍物的重叠区（即"机器人实际感受到阻力"的那块区域）；
        空列表表示无碰撞。
        """
        if not self.cfg.check_obstacle_collision:
            return []
        mover = self._world_obstacle(oid)
        end_poly = mover.polygon_at(nx, ny, theta)
        swept = geometry._swept_region(mover, nx, ny)
        hits = []
        for w in self.world:
            if w.oid == oid:
                continue
            wp = w.polygon
            if end_poly.intersects(wp) or swept.intersects(wp):
                contact = swept.intersection(wp)   # 只"摸到"接触重叠区，非整个障碍物
                hits.append((w.oid, contact))
        return hits

    def _relocate_world(self, oid: int, x: float, y: float, theta: float):
        for w in self.world:
            if w.oid == oid:
                w.x, w.y, w.theta, w.removed = x, y, theta, True
                return

    def _capture_frame(self, res: RunResult, node: int, label: str):
        """记录一帧当前世界状态（机器人 + 所有障碍物位姿），用于逐步动画。"""
        if not self.cfg.save_frames:
            return
        res.frames.append({
            "node": node,
            "robot": self.roadmap.nodes[node],
            "track": list(res.robot_track),
            "obstacles": [(w.oid, w.polygon, w.removed) for w in self.world],
            "perceived": set(self.belief.perceived.keys()),
            "J": round(res.J, 4),
            "label": label,
        })

    def _snapshot(self, res: RunResult, node: int, plan: Plan):
        res.history.append({
            "robot": self.roadmap.nodes[node],
            "node_path": list(plan.node_path),
            "perceived": [o.oid for o in self.belief.perceived.values()],
            "cost": plan.cost,
        })
