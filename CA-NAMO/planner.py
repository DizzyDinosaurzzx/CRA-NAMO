"""在线"规划-执行-感知-重规划"循环"""

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
    removed: List[int] = field(default_factory=list)
    robot_track: List[Tuple[float, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    frames: List[dict] = field(default_factory=list)
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
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))

        self.roadmap = Roadmap(workspace, static_obstacles, cfg)
        # 起点与终点都是单点，插入路网的方式完全对称
        self.start_node = self._add_terminal(self.start)
        self.goal_node = self._add_terminal(self.goal)
        # 实际所在/到达的点（可能因退化而与声明值不同，绘图用）
        self.start_point = self.roadmap.nodes[self.start_node]
        self.goal_point = self.roadmap.nodes[self.goal_node]

        self.estimator = DifficultyEstimator(cfg)
        self.belief = Belief(self.roadmap, cfg)

    def _add_terminal(self, p: Tuple[float, float]) -> int:
        """把起点/终点插入路网；该点放不下机器人圆盘时退化到最近的合法节点。"""
        cfg = self.cfg
        if not self.roadmap.static_free.contains(Point(p).buffer(cfg.robot_radius)):
            # 点在墙内或贴墙太近 -> 退化到最近的合法路网节点
            p = self.roadmap.nodes[self.roadmap.nearest_node(p)]
        return self.roadmap.add_terminal(p)

    # ------------ 运行 ------------- #
    def run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
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
                    push_path = act.get("push_path")
                    # 推动前触摸感知: 获知该障碍物的真实 difficulty
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # ---- 如果有 SE2 推动路径则逐步执行，否则瞬移 ----
                    if push_path:
                        push_success, hits = self._execute_push_path(
                            act["oid"], obs, push_path, res, node, cfg)
                        if not push_success:
                            # 撞上了 -> 作废本次放置并重规划换位置
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
                    else:
                        # 瞬移模式: 物理校验
                        hits = self._world_collision(act["oid"], dx, dy, dth)
                        if hits:
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
                        self._capture_frame(res, node, f"push obstacle {act['oid']}")
                    # 执行时按真实 difficulty 结算，可能与规划时的估计值不同
                    executed_work = geometry.push_work(obs, act["dist"])
                    res.work_cost += executed_work
                    res.J += executed_work
                    if act["oid"] not in res.removed:
                        res.removed.append(act["oid"])
                elif act["type"] == "move":
                    prev_node = node    # 需求3: 记录移动前位置
                    res.walk_cost += cfg.lambda_distance * act["dist"]
                    res.J += cfg.lambda_distance * act["dist"]
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
                    # 触摸感知: 获知被触碰障碍物的真实 difficulty
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
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
        """真实世界物理校验: 把 oid 推到 (nx, ny, theta) 是否与其他障碍物碰撞。

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

    def _execute_push_path(self, oid: int, obs, push_path: list,
                           res: RunResult, node: int, cfg: Config):
        """逐步执行 SE2 推动路径，每步做碰撞检测并记录帧。
        返回 (success, hits)，success=False 表示中途撞到东西。"""
        max_frames = cfg.push_max_frames_per_action
        n_waypoints = len(push_path)
        if n_waypoints > max_frames:
            import numpy as np
            indices = np.linspace(0, n_waypoints - 1, max_frames).astype(int)
            sampled = [push_path[i] for i in indices]
            sampled[0] = push_path[0]
            sampled[-1] = push_path[-1]
        else:
            sampled = push_path

        for step_idx, (wx, wy, wth) in enumerate(sampled):
            if step_idx == 0:
                continue  # 跳过起点
            if cfg.check_obstacle_collision:
                hits = self._world_collision(oid, wx, wy, wth)
                if hits:
                    return (False, hits)
            self._relocate_world(oid, wx, wy, wth)
            step_label = (f"push {oid} step {step_idx + 1}/{len(sampled) - 1}"
                          if cfg.save_frames else f"push {oid}")
            self._capture_frame(res, node, step_label)
        # 更新信念中的最终位姿
        final_x, final_y, final_th = push_path[-1]
        self.belief.relocate(obs, final_x, final_y, final_th)
        return (True, [])

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
