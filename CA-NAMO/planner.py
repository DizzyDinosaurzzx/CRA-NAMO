"""在线"规划-执行-感知-重规划"循环"""

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
from search import Planner
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

        # 路网只由工作空间与墙体决定。可移动障碍物【不得】参与建图：它们是部分可
        # 观测的，任何依赖其真实位姿的图结构都会把隐藏信息泄露给规划器。
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
        if not self.roadmap.free_eroded_prep.contains(Point(p)):
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
                    # 逐步执行 SE2 推动路径。搜索侧只在规划出非空路径时才判该障碍物
                    # 可推动，因此这里必定拿得到一条连续路径，不存在"瞬移到落点"。
                    push_success, hits, executed_dist = self._execute_push_path(
                        act["oid"], obs, act["push_path"], res, node, cfg)
                    # 按【实际推动的那一段】结算，difficulty 取【真实】值（可能与规划
                    # 时的估计不同）。必须从世界取：信念里的副本 difficulty 是 NaN。
                    # 中途撞停时 executed_dist 只覆盖已走完的前缀——障碍物确实动了、
                    # 功确实做了，直接 break 会把这笔功漏记，让 J 系统性偏低。
                    if executed_dist > 0.0:
                        true_diff = self._world_obstacle(act["oid"]).difficulty
                        executed_work = geometry.push_work(true_diff, executed_dist)
                        res.work_cost += executed_work
                        res.J += executed_work
                        if act["oid"] not in res.removed:
                            res.removed.append(act["oid"])
                    if not push_success:
                        # 撞上了 -> 作废本次放置并重规划换位置
                        self._handle_push_collision(res, node, act["oid"], hits)
                        break
                elif act["type"] == "move":
                    prev_node = node    # 需求3: 记录移动前位置
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    # 需求3: 碰撞感知。必须在记账与落子【之前】做：撞上未感知障碍物
                    # 时机器人只能推进到接触点，不可能穿过去。原先先付钱、先把 node
                    # 挪到 v、再检测，命中后只记一帧就继续，机器人会停在障碍物内部
                    # 甚至另一侧，这条边却按畅通计费。停在内部时它的所有出边随即被
                    # 新揭示的障碍物封死，下一轮还会伪装成"无可行解"的失败。
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # 推进到接触点再退回 prev_node：机器人始终留在路网节点上，
                        # 这一去一回的路程照实计费（绝不低估）。退回后该边已被新揭示
                        # 的障碍物封死，同一方案不会被再次选中，因此不会死循环。
                        blocked_dist = 2.0 * t_contact * act["dist"]
                        res.walk_cost += cfg.lambda_distance * blocked_dist
                        res.J += cfg.lambda_distance * blocked_dist
                        # 撞上就是物理接触，被撞者的真实 difficulty 随之暴露
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

    # -------------------------------------------------------------- 辅助方法
    def _world_obstacle(self, oid: int) -> Optional[MovableObstacle]:
        for w in self.world:
            if w.oid == oid:
                return w
        return None

    def _world_collision(self, oid: int, nx: float, ny: float, theta: float):
        """真实世界物理校验: 把 oid 推到 (nx, ny, theta) 是否撞上东西。

        检查终点位姿与推动扫掠路径，对手包括*全部*可移动障碍物（含机器人尚未感知的）
        **以及墙体**。返回 [(oid_hit, contact_region), ...]，墙体的 oid_hit 为 None；
        contact_region 是扫掠区与对方的重叠区（即"机器人实际感受到阻力"的那块）；
        空列表表示无碰撞。

        墙体必须一并检查。SE2 规划器的"无碰撞"只在网格分辨率意义下成立（构型空间是
        离散采样，起点还带 _unstick_start 的局部豁免），执行期漏掉墙体就等于把这层
        近似当成了物理事实：一旦规划器给出擦墙的路径，障碍物会被真的推进墙里，此后
        plan_anywhere 永远返回"起点与墙壁碰撞"，这个障碍物就彻底废掉，机器人只能围着
        它无休止地重规划。
        """
        if not self.cfg.check_obstacle_collision:
            return []
        mover = self._world_obstacle(oid)
        end_poly = mover.polygon_at(nx, ny, theta)
        swept = geometry._swept_region(mover, nx, ny, theta)
        hits = []
        for w in self.world:
            if w.oid == oid:
                continue
            wp = w.polygon
            if end_poly.intersects(wp) or swept.intersects(wp):
                contact = swept.intersection(wp)   # 只"摸到"接触重叠区，非整个障碍物
                hits.append((w.oid, contact))
        for so in self.static_obstacles:
            if end_poly.intersects(so.polygon) or swept.intersects(so.polygon):
                hits.append((None, swept.intersection(so.polygon)))
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
        """选出至多 max_frames 个【出图】的途经点下标（始终保留首尾）。

        只用于抽稀动画帧，**绝不能**同时拿来抽稀碰撞检测的步长：见
        `_execute_push_path` 的说明。
        """
        if len(push_path) <= max_frames:
            return list(range(len(push_path)))
        return [int(i) for i in np.linspace(0, len(push_path) - 1, max_frames)]

    def _execute_push_path(self, oid: int, obs, push_path: list,
                           res: RunResult, node: int, cfg: Config):
        """逐步执行 SE2 推动路径，每步做碰撞检测并记录帧。

        返回 `(success, hits, executed_dist)`：
        `executed_dist` 是【实际走完的那段路径】的 SE2 代价（平移弧长 + r̄·转角），
        中途撞停时只统计已执行的前缀，可直接喂给 `geometry.push_work()`。

        碰撞检测走【全分辨率】的每一步，抽稀只用来决定哪几步出图。这两件事必须分开：
        `_world_collision` 的扫掠区是"上一次落子位姿 -> 本次候选位姿"的分段凸包，
        若按抽稀后的航点推进，这个凸包跨越的是几十个规划步——
          * 路径拐弯时真实轨迹会鼓出凸包之外，中间的碰撞被整段跳过（切角穿墙）；
          * 反过来凸包又覆盖了物体根本没经过的区域，把没发生的碰撞报成撞停，
            推动被无故中断、还留下一块假的接触区把后续规划一起锁死。
        规划器相邻两步之间最多平移一格、旋转一个 theta 步，逐步检测时凸包足够贴合。
        """
        frame_at = set(self._sample_push_path(push_path,
                                              cfg.push_max_frames_per_action))
        n = len(push_path)
        last_i = 0

        for i in range(1, n):
            wx, wy, wth = push_path[i]
            if cfg.check_obstacle_collision:
                hits = self._world_collision(oid, wx, wy, wth)
                if hits:
                    # 障碍物已经被推到 push_path[last_i] 才撞停，信念必须同步到该
                    # 实际位姿：否则世界里它在半路、信念以为它没动过，二者永久脱节。
                    # 第一步就撞停时世界也没动过，此时不能调用 relocate——那会把
                    # 它标记成 removed。
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
