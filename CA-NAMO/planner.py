"""
CA-NAMO的在线"规划-执行-感知-重规划"循环

本模块是 CA-NAMO 的核心调度器，负责将感知、规划、执行三个子系统串联成闭环：

    感知 (Belief)       规划 (Planner)        执行 (OnlineNAMO)
    ─────────────       ──────────────        ────────────────
    探测可见障碍物  ──→  在信念路网上搜索  ──→  沿规划边移动 /
    揭示物理接触障碍物    代价最小的路径        推动障碍物 /
                                               碰撞检测与回退
         ↑                                        │
         └──────────── 新信息触发重规划 ←──────────┘

一次完整的仿真运行流程：
1. 从起点出发，构建初始信念（扫描可见障碍物）
2. 规划器在信念路网上搜索一条考虑了"绕路 vs 推障碍物"权衡的最优路径
3. 执行规划的动作序列（移动 / 推动），每步做碰撞检测
4. 执行过程中持续感知：新的障碍物被发现或碰撞暴露了未知物体 → 更新信念 → 触发重规划
5. 重复 2–4，直到到达目标或超出最大重规划次数

"""

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

# 执行期碰撞判据的面积阈值：相切的接触面积恰好是 0，真实穿模远大于这个数量级。
# 见 _world_collision。
_CONTACT_AREA_EPS = 1e-9

# 仿真结果汇总
@dataclass
class RunResult:
    success: bool                           # 是否到达目标
    J: float = 0.0                          # 总代价 = walk_cost + work_cost
    walk_cost: float = 0.0                  # 运动代价 = λ × 总行驶距离
    work_cost: float = 0.0                  # 操作代价 = Σ(真实difficulty × 推动距离)
    cycles: int = 0                         # 重规划循环次数
    plan_time: float = 0.0                  # 总规划耗时（秒）
    first_plan_time: float = 0.0            # 首次规划耗时（秒）—— 衡量冷启动代价
    total_expansions: int = 0               # A* 搜索节点扩展总数（所有轮合计）
    llm_calls: int = 0                      # LLM API 调用次数
    llm_mode: str = "heuristic"             # LLM 模式：heuristic / deepseek
    removed: List[int] = field(default_factory=list)    # 被移开的障碍物 ID 列表
    robot_track: List[Tuple[float, float]] = field(     # 机器人途经的节点坐标序列
        default_factory=list)
    frames: List[dict] = field(default_factory=list)    # 逐帧快照（供 render_sequence 用）
    message: str = ""                       # 结果描述信息

# 在线仿真器
class OnlineNAMO:
    """
    维护两套世界模型：
    - self.world(现实世界)
    - self.belief(机器人信念)
    规划器只能基于信念搜索，不能访问真实世界
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

        self.estimator = DifficultyEstimator(cfg)   # 障碍物难度估算器
        self.belief = Belief(self.roadmap, cfg)     # 机器人的部分可观测信念
        self._plan_paths: List[dict] = []           # 当前规划出的所有路径（供逐帧可视化）
        # 执行期证实为"一步都推不动"的推动尝试。规划器只能读信念，而撞墙这类失败
        # 在信念里留不下痕迹，必须由执行侧单独记账，否则同一个死计划会被无限重放。
        self.failed_pushes: set = set()

    def _add_terminal(self, p: Tuple[float, float]) -> int:
        # 把起点/终点插入路网；该点放不下机器人圆盘时退化到最近的合法节点
        cfg = self.cfg
        if not self.roadmap.free_eroded_prep.contains(Point(p)):
            # 点在墙内或贴墙太近 -> 退化到最近的合法路网节点
            p = self.roadmap.nodes[self.roadmap.nearest_node(p)]
        return self.roadmap.add_terminal(p)

    # 主感知循环
    def run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
        node = self.start_node                           # 机器人当前所在路网节点
        res.robot_track.append(self.roadmap.nodes[node])
        
        # 初始感知：扫描起点周围的可见障碍物
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

            # ---- 执行规划的前置边（包括对应的障碍物移除）----
            moves_done = 0
            reached_goal = False
            for act in plan.actions:
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    # 推动前触摸感知: 获知该障碍物的真实 difficulty
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # 逐步执行 SE2 推动路径
                    push_success, hits, executed_dist = self._execute_push_path(
                        act["oid"], obs, act["push_path"], res, node, cfg)
                    # 按实际推动的那一段结算，difficulty 取真实值
                    if executed_dist > 0.0:
                        true_diff = self._world_obstacle(act["oid"]).difficulty
                        executed_work = geometry.push_work(true_diff, executed_dist)
                        res.work_cost += executed_work
                        res.J += executed_work
                        if act["oid"] not in res.removed:
                            res.removed.append(act["oid"])
                    elif not push_success:
                        # 一步都没推动就撞停：世界和信念都毫发未动，下一轮必然规划出
                        # 同一个计划。登记这次失败，逼规划器改走别的方案——最坏情况是
                        # 尽快报"无可行解"，而不是空转到 max_replans 耗尽。
                        self.failed_pushes.add((push_signature(obs), act["key"]))
                    if not push_success:
                        # 撞上了，作废本次放置并重规划换位置
                        self._handle_push_collision(res, node, act["oid"], hits)
                        break
                elif act["type"] == "move":
                    prev_node = node    # 需求3: 记录移动前位置
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    # 需求3: 碰撞感知
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # 推进到接触点再退回之前节点，这一去一回的路程照实计费
                        blocked_dist = 2.0 * t_contact * act["dist"]
                        res.walk_cost += cfg.lambda_distance * blocked_dist
                        res.J += cfg.lambda_distance * blocked_dist
                        # 撞上就是物理接触，被撞者的真实difficulty随之暴露
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

    def _world_obstacle(self, oid: int) -> Optional[MovableObstacle]:
        for w in self.world:
            if w.oid == oid:
                return w
        return None

    def _world_collision(self, oid: int, nx: float, ny: float, theta: float):
        """真实世界物理校验: 把 oid 推到 (nx, ny, theta) 是否撞上东西

        判据是【重叠面积】而不是 shapely 的 `.intersects()`：后者对**恰好贴合**也返回
        True，而扫掠区永远包含起点位姿，所以只要有邻居紧挨着当前障碍物，它的第一步
        无论往哪个方向推都会被判成撞车（接触面积 0.0），于是这一对障碍物互相锁死、
        谁也动不了。maze_doors 里 oid 7 和 8 就是这样一对（底边/顶边同为 y=4.05）。

        这与 push_planner._START_COLLISION_EPS 是同一个取舍：相切不算碰撞，必须真正
        重叠才算。真实的擦碰重叠面积远在阈值之上（实测撞墙那次是 0.0063 m²）。
        """
        if not self.cfg.check_obstacle_collision:
            return []
        mover = self._world_obstacle(oid)
        end_poly = mover.polygon_at(nx, ny, theta)
        swept = geometry._swept_region(mover, nx, ny, theta)
        hits = []

        def overlapping(poly):
            """返回接触区；只贴合不重叠时返回 None。"""
            contact = swept.intersection(poly)
            if contact.area > _CONTACT_AREA_EPS:
                return contact
            # 扫掠区已经包含终点位姿，正常情况下不必再单独判 end_poly；这里兜住
            # _swept_region 在退化输入下可能给出偏小几何的情形。
            end_contact = end_poly.intersection(poly)
            return end_contact if end_contact.area > _CONTACT_AREA_EPS else None

        for w in self.world:
            if w.oid == oid:
                continue
            contact = overlapping(w.polygon)   # 只"摸到"接触重叠区，非整个障碍物
            if contact is not None:
                hits.append((w.oid, contact))
        for so in self.static_obstacles:
            contact = overlapping(so.polygon)
            if contact is not None:
                hits.append((None, contact))
        return hits

    def _handle_push_collision(self, res: RunResult, node: int, oid: int, hits):
        """推动撞上东西后的统一善后：更新信念 + 记录一帧。"""
        cfg = self.cfg
        for oid_hit, region in hits:
            if oid_hit is None:
                continue      # 撞墙：墙体本就是已知的静态几何，没有新信息可登记
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
        """逐步执行 SE2 推动路径，每步做碰撞检测并记录帧"""
        frame_at = set(self._sample_push_path(push_path,
                                              cfg.push_max_frames_per_action))
        n = len(push_path)
        last_i = 0

        for i in range(1, n):
            wx, wy, wth = push_path[i]
            if cfg.check_obstacle_collision:
                hits = self._world_collision(oid, wx, wy, wth)
                if hits:
                    # 障碍物已经被推到 push_path[last_i] 才撞停，信念必须同步到该实际位姿：否则世界里它在半路、信念以为它没动过
                    if last_i != 0:
                        self.belief.relocate(obs, *push_path[last_i])
                    return (False, hits,
                            geometry.se2_path_cost(obs, push_path[:last_i + 1], cfg))
            self._relocate_world(oid, wx, wy, wth)
            last_i = i
            if i in frame_at:
                self._capture_frame(res, node, f"push {oid} step {i}/{n - 1}")
        # 更新信念中的最终位姿
        self.belief.relocate(obs, *push_path[-1])
        return (True, [], geometry.se2_path_cost(obs, push_path, cfg))

    def _relocate_world(self, oid: int, x: float, y: float, theta: float):
        """将真实世界中指定障碍物的位姿更新到新位置，并标记为已移除 (removed=True)"""
        for w in self.world:
            if w.oid == oid:
                w.x, w.y, w.theta, w.removed = x, y, theta, True
                return

    def _plan_to_paths(self, plan) -> List[dict]:
        """把一个计划拆成可视化用的折线集合：机器人路线 + 每个障碍物的推动路径"""
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
        """记录一帧当前世界状态（机器人 + 所有障碍物位姿），用于逐步动画"""
        if not self.cfg.save_frames:
            return
        res.frames.append({
            "node": node,
            "plan_paths": list(self._plan_paths),   # 该帧时刻正在执行的规划路径
            "robot": self.roadmap.nodes[node],
            "track": list(res.robot_track),
            "obstacles": [(w.oid, w.polygon, w.removed) for w in self.world],
            "perceived": set(self.belief.perceived.keys()),
            "J": round(res.J, 4),
            "label": label,
        })
