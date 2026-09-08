"""NAMO 在线规划、执行、感知和重规划闭环。"""

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
from llm_choice import ActionChooser
import risk
from risk import RiskEstimator
from search import FailedMoves, Planner, move_signature
import cost
import geometry
import kinematics
import manipulation

# 重新锚定时考虑的候选路线图节点数。
_REANCHOR_CANDIDATES = 64

# 动态中断后的最大重新锚定次数。
_REANCHOR_ATTEMPTS = 4


def _driven_fraction(elapsed: float, turn_s: float, drive_s: float) -> float:

    if drive_s <= 0.0:
        return 1.0 if elapsed >= turn_s + drive_s else 0.0
    return min(1.0, max(0.0, (elapsed - turn_s) / drive_s))


def _disagree(a: float, b: float, ratio: float) -> bool:
    """两数之差是否达到较大绝对值的 ratio 倍？"""
    lo, hi = sorted((abs(float(a)), abs(float(b))))
    if hi <= 0.0:
        return False
    return lo <= 0.0 or hi / lo >= ratio


def _lerp_xy(a, b, s: float):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


@dataclass
class RunResult:
    success: bool                           # 是否到达目标。
    C: float = 0.0                          # 目标函数值。
    J: float = 0.0                          # 能量代价。
    walk_cost: float = 0.0                  # 机器人行驶代价。
    manip_walk_cost: float = 0.0            # 伴随搬移时的行驶代价。
    work_cost: float = 0.0                  # 障碍物搬移功。
    risk_cost: float = 0.0                  # 风险附加代价。
    risk_levels: dict = field(default_factory=dict)   # 各障碍物计入的风险等级。
    T: float = 0.0                          # 模拟经过时间。
    move_time: float = 0.0                  # 移动时间。
    wait_time: float = 0.0                  # 等待时间。
    cycles: int = 0                         # 重规划次数。
    plan_time: float = 0.0                  # 墙钟规划时间。
    first_plan_time: float = 0.0            # 首次规划时间。
    total_expansions: int = 0               # A* 总扩展数。
    llm_calls: int = 0                      # LLM API 调用总数。
    llm_calls_cost: int = 0                 # 其中：难度估计。
    llm_calls_risk: int = 0                 # 其中：风险评估。
    llm_calls_choice: int = 0               # 其中：候选决策。
    llm_mode: str = "heuristic"             # 难度估计器模式。
    removed: List[int] = field(default_factory=list)    # 搬移过的障碍物 ID。
    robot_track: List[Tuple[float, float]] = field(     # 机器人节点坐标。
        default_factory=list)
    frames: List[dict] = field(default_factory=list)    # 可视化快照。
    world_events: List[str] = field(default_factory=list)  # 世界自主事件。
    decisions: List[str] = field(default_factory=list)     # 预设决策记录。
    choices: List[str] = field(default_factory=list)       # LLM 候选决策记录。
    message: str = ""                       # 结果说明。

class OnlineNAMO:
    """运行在线 NAMO，并将真实状态与机器人 belief 分离。"""
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
        self.chooser = ActionChooser(cfg)
        self.belief = Belief(self.roadmap, cfg, self.risk, self.estimator)
        # 动力学更新真实状态，不直接改变机器人 belief。
        self.dynamics = dynamics.WorldDynamics(
            self.world, static_obstacles, workspace, events, cfg)
        self._risk_charged: set = set()
        self.stranded = False
        self._plan_paths: List[dict] = []
        self.failed_moves = FailedMoves()
        # 保存动态碰撞信息和机器人停止位置。
        self._world_hit: Optional[Tuple[List[int], Tuple[float, float]]] = None
        self._holding: Optional[int] = None
        # 保存预设决策点用于报告。
        self.decision_points = list(decision_points or ())
        # 按阻挡物集合累计等待时间。
        self.wait_budget: dict = {}
        # 搬移可能使机器人离开路线图节点。
        self.robot_xy: Tuple[float, float] = self.roadmap.nodes[self.start_node]
        # 规划墙钟时间不推进模拟时间。
        self.clock: float = 0.0
        # 即使足迹对称，航向变化也计入时间。
        self.robot_heading: float = 0.0
        self._free_profile = cfg.free_profile()
        self._loaded_profile = cfg.loaded_profile()
        self._frame_node = self.start_node
        self._world_frame_t = 0.0
        self._waited = 0.0
        # 只在候选结构或选择发生变化时记一笔，避免每条边都刷一行。
        self._last_choice: Optional[tuple] = None

    def _add_terminal(self, p: Tuple[float, float]) -> int:
        cfg = self.cfg
        if not self.roadmap.free_eroded_prep.contains(Point(p)):
            p = self.roadmap.nodes[self.roadmap.nearest_node(p)]
        return self.roadmap.add_terminal(p)

    def run(self) -> RunResult:
        """运行至结束，并刷新折叠后的控制台输出。"""
        try:
            return self._run()
        finally:
            self.cfg.flush_log()

    def _run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
        node = self.start_node                           # 跟踪当前路线图节点。
        res.robot_track.append(self.roadmap.nodes[node])
        
        self._perceive(res, self.roadmap.nodes[node])
        self._capture_frame(res, node, "start")

        planner = Planner(self.roadmap, self.belief, self.estimator, cfg,
                          self.failed_moves, self.risk, self.wait_budget)

        for cycle in range(cfg.max_replans):
            # 丢弃旧世界版本中的失败记录。
            if self.failed_moves.drop_stale(self.dynamics.version):
                planner.forget_removals()
            t0 = time.time()
            plan = (self._choose_plan(res, planner, node) if cfg.llm_choice
                    else planner.plan(node, self.goal_node, self.robot_heading))
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
                    # 接触信息在伴随搬移过程中揭示。
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
                        # 后续动作是按旧姿态规划的。
                        break
                elif act["type"] == "wait":
                    self._wait_on_edge(res, node, act)
                    break
                elif act["type"] == "move":
                    prev_node = node    # 记录移动起点。
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # 先到接触点再返回，然后重新规划。
                        leg = t_contact * act["dist"]
                        self._drive(res, from_pos, contact_pos, dist=leg)
                        self._drive(res, contact_pos, from_pos, dist=leg)
                        self._touch(res, cfg, contact_pos)
                        self._perceive(res)
                        self._capture_frame(
                            res, node, f"collision revealed {hit_oids} -> replan")
                        break
                    if not self._drive(res, from_pos, to_pos, dist=act["dist"]):
                        break       # 动态碰撞在下方恢复。
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

    def _choose_plan(self, res: RunResult, planner: Planner, node: int):
        """枚举几何层已验证的候选，交由 LLM 选择；无 LLM 时按代价取最优。"""
        options, fallback, expansions = planner.options(
            node, self.goal_node, self.robot_heading)
        if not options:
            return fallback

        here = self.roadmap.nodes[node]
        rows = self.chooser.describe(options, self.belief, here, self.goal_point)
        goal_distance = math.hypot(self.goal_point[0] - here[0],
                                   self.goal_point[1] - here[1])
        pick, source = self.chooser.choose(options, rows, goal_distance)
        if pick is None:
            # 没有 LLM 判断时按目标代价取最优，与 A* 的选择一致。
            pick = min(range(len(options)),
                       key=lambda i: options[i]["plan"].cost)
        chosen = options[pick]

        # 候选枚举跑了多次 A*，把额外的扩展数一并计入报告。
        res.total_expansions += max(0, expansions - chosen["plan"].expansions)

        key = (self.chooser._key(options), pick, source)
        if len(options) > 1 and key != self._last_choice:
            self._last_choice = key
            note = (f"t={self.clock:,.1f}s  {len(options)} options -> "
                    f"#{pick} {rows[pick]['detail']}  [{source}]")
            res.choices.append(note)
            self.cfg.log(f"[choice] {note}")
        return chosen["plan"]

    def _finalize(self, res: RunResult, node: int) -> RunResult:
        """完成运行并填充结果。"""
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
        res.llm_calls_cost = self.estimator.calls
        res.llm_calls_risk = self.risk.calls
        res.llm_calls_choice = self.chooser.calls
        res.llm_calls = (res.llm_calls_cost + res.llm_calls_risk
                         + res.llm_calls_choice)
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
        """障碍物碰撞后登记未知接触区域。"""
        for oid_hit, region in hits:
            if oid_hit is None:
                continue      # 静态墙体接触不会增加 belief 信息。
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
        """将机器人行驶计入目标函数。"""
        charge = cost.motion_cost(self.cfg, dist)
        res.walk_cost += charge
        res.J += charge
        if in_contact:
            res.manip_walk_cost += charge

    def _charge_risk(self, res: RunResult, oid: int):
        """每个障碍物只计入一次接触后的风险附加代价。"""
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
        """推进机器人和动态世界的模拟时间。"""
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
        """无可行计划时等待动态障碍物变化。"""
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
        """更新视觉 belief，并测量墙钟时间。"""
        t0 = time.time()
        try:
            return self.belief.perceive(
                self.world, self.robot_xy if at is None else at)
        finally:
            res.plan_time += time.time() - t0

    def _touch(self, res: RunResult, cfg: Config, at=None) -> List[int]:
        """更新接触 belief，并测量墙钟时间。"""
        t0 = time.time()
        try:
            return self.belief.touch_check(
                self.robot_xy if at is None else at, self.world, cfg)
        finally:
            res.plan_time += time.time() - t0

    def _wait_on_edge(self, res: RunResult, node: int, act: dict):
        """在被阻挡的边上等待，并计入经过时间。"""
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
        """根据已搬移障碍物汇总预设决策点。"""
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
        """绘制运动中的世界，每个动画步最多一帧。"""
        if not moved or self.clock - self._world_frame_t < self.cfg.gif_time_step:
            return
        self._world_frame_t = self.clock
        self._capture_frame(res, self._frame_node, "the world moves")

    def _drive(self, res: RunResult, a, b, in_contact: bool = False,
               loaded: bool = False, dist: Optional[float] = None):
        """在两点之间行驶，并计入距离、能量和时间。"""
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
        """行驶时推进并检查动态世界。"""
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
            self.robot_xy = here          # 暴露该子步的机器人位置。
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
        """返回一段行驶过程中与机器人相交的动态障碍物。"""
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
        """动态碰撞后重新感知并重新锚定。"""
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
        """隐藏真实状态变化后使相关测量失效。"""
        for oid in self.dynamics.drain_stale():
            self.belief.invalidate_contact(oid)
            self._risk_charged.discard(oid)

    def _set_robot(self, res: RunResult, p: Tuple[float, float]):
        self.robot_xy = (float(p[0]), float(p[1]))
        res.robot_track.append(self.robot_xy)

    def _walk_frames(self, pts, cfg: Config) -> set:
        """返回用于采集帧的有界行走片段。"""
        moving = [i for i in range(1, len(pts))
                  if math.dist(pts[i - 1], pts[i]) > 1e-9]
        return {moving[j] for j in
                self._sample_move_path(moving, cfg.manip_max_frames_per_action)}

    def _walk_robot(self, pts, res: RunResult, cfg: Config,
                    loaded: bool = False, node: Optional[int] = None,
                    label: str = "", move_oid: Optional[int] = None):
        """依次通过各点，遇到未知碰撞时停止。"""
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
                # 动态中断会留下 _world_hit，供后续恢复。
                return i - 1, self._world_hit[0], self.robot_xy
            if i in frame_at:
                self._capture_frame(res, node, label, move_oid=move_oid)
        return len(pts) - 1, [], (pts[-1] if pts else self.robot_xy)

    def _retrace(self, res: RunResult, pts, node: Optional[int] = None,
                 label: str = "", move_oid: Optional[int] = None):
        """不做碰撞检查，回退最近完成的路径。"""
        frame_at = self._walk_frames(pts, self.cfg) if node is not None else ()
        for i in range(1, len(pts)):
            if not self._drive(res, pts[i - 1], pts[i], in_contact=True):
                return          # 由调用方处理被中断的回退。
            if i in frame_at:
                self._capture_frame(res, node, label, move_oid=move_oid)

    def _reanchor(self, res: RunResult, node: int) -> Optional[int]:
        """搬移后将机器人返回可到达的路线图节点。"""
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
            # 动态中断后重新感知。
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
        """根据机器人实际位置和释放选项规划伴随搬移。"""
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
        """执行一次搬移时暂停动态运动。"""
        self.dynamics.suspend(oid)
        self._holding = oid
        try:
            return self._escort(oid, obs, act, res, node, cfg)
        finally:
            self._holding = None
            self.dynamics.release(oid)

    def _escort(self, oid: int, obs, act: dict,
                res: RunResult, node: int, cfg: Config):
        """执行障碍物伴随搬移并返回结果。"""
        move_path = act["move_path"]
        frame_at = set(self._sample_move_path(move_path,
                                              cfg.manip_max_frames_per_action))
        n = len(move_path)
        start_xy = (obs.x, obs.y)
        home = self.robot_xy

        cplan = act.get("contact")
        exit_nodes: List[Optional[int]] = [None]
        if not cfg.contact_required or cplan is None or not cplan.feasible:
            # 无接触时机器人等待，障碍物自行移动。
            cplan = contact.idle_plan(home, n)
        else:
            # 根据机器人实际位置重新计算接近和离开路径。
            cplan, exit_nodes = self._contact_plan(oid, obs, move_path, node,
                                                   act.get("key"))
            if not cplan.feasible:
                self.failed_moves.add((move_signature(obs), act["key"]))
                self._capture_frame(
                    res, node,
                    f"cannot escort {oid} from node {node}: {cplan.reason} -> replan")
                return (False, None, 0.0, None)
        if self.dynamics.active and not self._still_there(oid, move_path[0]):
            # 姿态变化会使伴随计划失效。
            self._perceive(res)
            self._capture_frame(
                res, node, f"{oid} is no longer where it was -> replan")
            return (False, None, 0.0, None)
        rp = list(cplan.robot_path)
        off = cplan.move_offset

        # 只有世界变化时才重建碰撞索引。
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
                # 将 belief 同步到障碍物最近一次完成的姿态。
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
            # 机器人握持障碍物时仍可能碰撞。
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
                # 重新规划前回滚未完成的路径段。
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

        # 将 belief 同步到障碍物最终姿态。
        self.belief.relocate(obs, *move_path[-1])
        self.belief.record_move_direction(oid, start_xy, move_path[-1])
        self._perceive(res)
        new_node = self._release_and_return(res, oid, rp, off, n - 1, True,
                                            node, cfg, cplan, exit_nodes)
        return (True, [], cost.se2_path_length(obs, move_path, cfg), new_node)

    def _on_taking_hold(self, res: RunResult, oid: int, node: int, cfg: Config):
        """重新评估机器人握持时揭示的难度和风险。"""
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
        """释放障碍物并返回路线图。"""
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
        # 从新的释放节点规划前先进行感知。
        self._touch(res, cfg)
        self._perceive(res)
        self._capture_frame(res, landed, f"let go of {oid} at node {landed}")
        return landed

    def _collision_index(self, oid: int):
        """建立被搬移障碍物可能撞到的所有物体索引。"""
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
        """返回障碍物是否仍接近计划姿态。"""
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
            "move_oid": move_oid,   # 当前正在搬移的障碍物。
            "plan_paths": list(self._plan_paths),   # 当前帧的计划路径。
            "robot": self.robot_xy,     # 真实机器人位置，可能不在节点上。
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
            "t": round(self.clock, 3),      # 动画使用的模拟秒数。
            "move_t": round(res.move_time, 1),   # 显示的移动时间部分。
            "plan_t": round(res.plan_time, 1),
            "label": label,
        })
