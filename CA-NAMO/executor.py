"""CA-NAMO 在线"规划—执行—感知—重规划"主循环。"""

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
import timing

# 仿真结果汇总
@dataclass
class RunResult:
    success: bool                           # 是否到达目标
    J: float = 0.0                          # 总代价 = 行走代价 + 搬运代价
    walk_cost: float = 0.0                  # 行走代价 = λ × 总移动距离
    manip_walk_cost: float = 0.0            # 其中护送障碍物所占的行走代价
    work_cost: float = 0.0                  # 搬运代价 = Σ(真实难度 × 移动距离)
    cycles: int = 0                         # 重规划轮数
    # --- 任务时间：T = 运动时间 + 规划时间（见 timing.py）---
    motion_time: float = 0.0                # 机器人运动的仿真秒数
    manip_motion_time: float = 0.0          # 其中处理障碍物所占时间
    plan_time: float = 0.0                  # 决策计算（A* + LLM）实测秒数
    mission_time: float = 0.0               # 运动时间 + 规划时间
    first_plan_time: float = 0.0            # 首次规划耗时（秒）——冷启动开销
    total_expansions: int = 0               # A* 节点扩展总数（各轮累计）
    llm_calls: int = 0                      # LLM API 调用次数
    llm_mode: str = "heuristic"             # LLM 模式：heuristic / deepseek
    removed: List[int] = field(default_factory=list)    # 已移动障碍物 ID 列表
    robot_track: List[Tuple[float, float]] = field(     # 机器人途经节点坐标序列
        default_factory=list)
    frames: List[dict] = field(default_factory=list)    # 逐帧快照（供 render_sequence 用）
    message: str = ""                       # 结果描述

# 在线仿真器
class OnlineNAMO:
    """维护真实世界与机器人信念两套模型，规划器只基于信念搜索、无法访问真实世界。"""
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
        self.belief = Belief(self.roadmap, cfg)     # 机器人部分可观测信念
        self._plan_paths: List[dict] = []           # 当前全部规划路径（用于逐帧可视化）
        self.failed_moves: set = set()
        # 机器人真实位置：动作间等于当前路网节点，搬运时离开节点贴住障碍物
        self.robot_xy: Tuple[float, float] = self.roadmap.nodes[self.start_node]
        # 任务时钟：初始朝向对准目标，避免开局出现依赖坐标系的任意转向
        self.timer = timing.MotionTimer(
            timing.MotionProfile.from_config(cfg),
            heading=timing.heading_of(self.start_point, self.goal_point) or 0.0)

    def _add_terminal(self, p: Tuple[float, float]) -> int:
        # 将起点/终点插入路网；机器人圆盘放不下时吸附到最近有效节点
        cfg = self.cfg
        if not self.roadmap.free_eroded_prep.contains(Point(p)):
            # 点在墙内或离墙太近 -> 退化为最近的有效路网节点
            p = self.roadmap.nodes[self.roadmap.nearest_node(p)]
        return self.roadmap.add_terminal(p)

    # 主感知—动作循环
    def run(self) -> RunResult:
        cfg = self.cfg
        res = RunResult(success=False, llm_mode=self.estimator.mode)
        node = self.start_node                           # 机器人当前路网节点
        res.robot_track.append(self.roadmap.nodes[node])
        
        # 初始感知：扫描起点周围可见障碍物
        self.belief.perceive(self.world, self.roadmap.nodes[node])
        self._capture_frame(res, node, "start")

        planner = Planner(self.roadmap, self.belief, self.estimator, cfg,
                          self.failed_moves)

        for cycle in range(cfg.max_replans):
            t0 = time.time()
            # 传入机器人真实朝向：step_execute_edges=1 时只有首边会被真正
            # 行驶，其转向代价应由规划来承担
            plan = planner.plan(node, self.goal_node, self.timer.heading)
            dt = time.time() - t0
            res.plan_time += dt
            if cycle == 0:
                res.first_plan_time = dt
            self._plan_paths = self._plan_to_paths(plan)
            if plan is None:
                res.message = "No feasible plan under current belief."
                res.cycles = cycle + 1
                self._settle_time(res)
                return res
            res.total_expansions += plan.expansions

            if node == self.goal_node:
                break

            # --- 执行规划边 ---
            moves_done = 0
            reached_goal = False
            for act in plan.actions:
                if act["type"] == "remove":
                    obs = self.belief.obstacle(act["oid"])
                    # 搬运前触觉感知获取真实难度；推动本身就是接触，被搬障碍必被揭示
                    touched = self.belief.touch_check(self.robot_xy, self.world, cfg)
                    if self.belief.reveal_by_interaction(act["oid"], self.world):
                        touched.append(act["oid"])
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # 贴身护送障碍物沿其 SE2 路径移动
                    move_success, hits, executed_dist, new_node = \
                        self._execute_move(act["oid"], obs, act, res, node, cfg)
                    # 出现新节点说明机器人回不到出发节点，本计划其余部分已失效
                    if new_node is not None:
                        node = new_node
                    # 只按实际移动的路段、以真实难度计费
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
                        # hits 为 None 表示机器人自身撞上未知物，已在内部记录并截图
                        if hits is not None:
                            # 碰撞：作废该放置位置，重规划新位置
                            self._handle_move_collision(res, node, act["oid"], hits)
                        break
                    if new_node is not None:
                        break
                elif act["type"] == "move":
                    prev_node = node    # 记录移动起点
                    from_pos = self.roadmap.nodes[prev_node]
                    to_pos = self.roadmap.nodes[act["v"]]
                    # 碰撞感知
                    hit_oids, t_contact = self.belief.check_robot_collision(
                        from_pos, to_pos, self.world, cfg)
                    if hit_oids:
                        contact_pos = (
                            from_pos[0] + (to_pos[0] - from_pos[0]) * t_contact,
                            from_pos[1] + (to_pos[1] - from_pos[1]) * t_contact)
                        # 前进到接触点再退回上一节点；两段都计费，
                        # 退回是倒车，只付停车不付转向
                        leg = t_contact * act["dist"]
                        self._charge_walk(res, leg,
                                          heading=timing.heading_of(from_pos, to_pos))
                        self._charge_walk(res, leg)
                        # 碰撞即物理接触，被撞物的真实难度被揭示
                        self.belief.touch_check(contact_pos, self.world, cfg)
                        self.belief.perceive(self.world, from_pos)
                        self._capture_frame(
                            res, node, f"collision revealed {hit_oids} -> replan")
                        break
                    self._charge_walk(res, act["dist"],
                                      heading=timing.heading_of(from_pos, to_pos))
                    node = act["v"]
                    self._set_robot(res, self.roadmap.nodes[node])
                    moves_done += 1
                    # 触觉感知：获取所触障碍物的真实难度
                    touched = self.belief.touch_check(
                        self.roadmap.nodes[node], self.world, cfg)
                    if touched:
                        self._capture_frame(
                            res, node, f"touch revealed difficulty of {touched}")
                    # 每次实际移动后感知（揭示暴露出的障碍物）
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
        self._settle_time(res)
        res.llm_calls = self.estimator.calls
        if res.success and not res.message:
            res.message = "Reached goal."
        elif not res.success and not res.message:
            res.message = "Ran out of replan cycles."
        return res

    def _settle_time(self, res: RunResult):
        """结算任务时间：T = 运动(仿真) + 决策(实测)。"""
        self.timer.flush()
        res.motion_time = round(self.timer.total, 4)
        res.manip_motion_time = round(self.timer.contact_total, 4)
        res.plan_time = round(res.plan_time, 4)
        res.mission_time = round(self.timer.total + res.plan_time, 4)

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
                continue      # 撞墙：墙是已知静态几何，无新信息可登记
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

    # --- 机器人簿记 ---
    def _charge_walk(self, res: RunResult, dist: float, in_contact: bool = False,
                     heading: Optional[float] = None):
        """计收 λ × dist 的行走代价与耗时；搬运路程单独统计但同入 J 的 λ·D 项。

        heading 为行进方向，计时器据此计转向；倒车重走刚走过的路时传 None。
        """
        charge = cost.motion_cost(self.cfg, dist)
        res.walk_cost += charge
        res.J += charge
        if in_contact:
            res.manip_walk_cost += charge
        self.timer.travel(dist, heading, in_contact)

    def _set_robot(self, res: RunResult, p: Tuple[float, float]):
        self.robot_xy = (float(p[0]), float(p[1]))
        res.robot_track.append(self.robot_xy)

    def _walk_robot(self, pts, res: RunResult, cfg: Config):
        """驱动机器人依次通过 pts（pts[0] 为当前站位），遇到未知障碍即停。

        返回 (到达下标, 命中障碍 oid, 停止位置)。
        """
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._charge_walk(res, math.dist(a, stop), in_contact=True,
                                  heading=timing.heading_of(a, b))
                self._set_robot(res, stop)
                self.belief.touch_check(stop, self.world, cfg)
                return i - 1, hits, stop
            self._charge_walk(res, math.dist(a, b), in_contact=True,
                              heading=timing.heading_of(a, b))
            self._set_robot(res, b)
        return len(pts) - 1, [], (pts[-1] if pts else self.robot_xy)

    def _retrace(self, res: RunResult, pts):
        """沿刚走过的路倒回，无需碰撞检测——刚才畅通且此后无物移动。"""
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            self._charge_walk(res, math.dist(a, b), in_contact=True,
                              heading=timing.heading_of(a, b))
            self._set_robot(res, pts[i])

    def _reanchor(self, res: RunResult, node: int) -> Optional[int]:
        """搬运结束后让机器人回到路网；返回 None 表示回到出发节点（计划仍有效），否则返回新节点。"""
        blocked = self._known_obstacles_inflated()
        home = self.roadmap.nodes[node]
        if self.roadmap.can_drive(self.robot_xy, home, blocked):
            self._charge_walk(res, math.dist(self.robot_xy, home), in_contact=True,
                              heading=timing.heading_of(self.robot_xy, home))
            self._set_robot(res, home)
            return None
        target = self.roadmap.nearest_reachable_node(self.robot_xy, blocked)
        if target is None:
            target = self.roadmap.nearest_node(self.robot_xy)
        dest = self.roadmap.nodes[target]
        self._charge_walk(res, math.dist(self.robot_xy, dest), in_contact=True,
                          heading=timing.heading_of(self.robot_xy, dest))
        self._set_robot(res, dest)
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

    # --- 搬运 ---
    def _execute_move(self, oid: int, obs, act: dict,
                           res: RunResult, node: int, cfg: Config):
        """机器人抓着障碍物沿其 SE2 路径移动。

        返回 (成功, hits, 障碍物移动距离, 新节点或 None)；hits 为 None 表示
        是机器人自身撞上未知物，该情况此处已处理完，只需重规划。
        """
        move_path = act["move_path"]
        frame_at = set(self._sample_move_path(move_path,
                                              cfg.manip_max_frames_per_action))
        n = len(move_path)
        start_xy = (obs.x, obs.y)
        home = self.robot_xy

        cplan = act.get("contact")
        # 贴身护送时机器人自身路程已含运输时间；否则障碍物走 SE(2) 路径期间
        # 机器人原地等待，需在下面另行计费
        escorting = cfg.contact_required and cplan is not None and cplan.feasible
        if not escorting:
            # 接触模型关闭：障碍物移动时机器人在节点上等待；以机器人真实位置
            # 重建，而非复用锚定在边中点的规划位置
            cplan = contact.idle_plan(home, n)
        rp = list(cplan.robot_path)
        # 规划以边中点为基准测量往返程，机器人实际站在边的端点上，两端需重新锚定
        rp[0] = home
        rp[-1] = home
        off = cplan.move_offset

        # 整个搬运只建一次 STRtree，避免轨迹每个子步都做 O(N) 多边形扫描
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

        # --- 接近 ---
        reached, hits, stop = self._walk_robot(rp[:off + 1], res, cfg)
        if hits:
            self._retrace(res, [stop] + rp[reached::-1])
            self.belief.perceive(self.world, self.robot_xy)
            self._capture_frame(
                res, node, f"robot hit {sorted(hits)} approaching {oid} -> replan")
            return (False, None, 0.0, None)
        # 此处抓取、稍后松开，一次性计费，以下所有退出路径都要付这笔
        self.timer.grip()
        self._capture_frame(res, node, f"grip obstacle {oid}", move_oid=oid)

        # --- 移动障碍物 ---
        last_i = 0
        for i in range(1, n):
            wx, wy, wth = move_path[i]
            if cfg.check_obstacle_collision:
                obs_hits = self._world_collision(oid, wx, wy, wth, tree=tree,
                                                 tree_items=tree_items)
                if obs_hits:
                    # 障碍物停在 move_path[last_i]，信念同步到该位姿
                    if last_i != 0:
                        self.belief.relocate(obs, *move_path[last_i])
                        self.belief.record_move_direction(oid, start_xy,
                                                          move_path[last_i])
                    new_node = self._release_and_return(res, rp, off, last_i,
                                                        False, node, cfg)
                    return (False, obs_hits,
                            cost.se2_path_length(obs, move_path[:last_i + 1], cfg),
                            new_node)
            # 机器人抓着障碍物，自身也可能撞上东西
            a, b = rp[off + i - 1], rp[off + i]
            hits, t = self.belief.check_robot_collision(a, b, self.world, cfg)
            if hits:
                if last_i != 0:
                    self.belief.relocate(obs, *move_path[last_i])
                    self.belief.record_move_direction(oid, start_xy, move_path[last_i])
                stop = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._charge_walk(res, math.dist(a, stop), in_contact=True,
                                  heading=timing.heading_of(a, b))
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
            self._charge_walk(res, math.dist(a, b), in_contact=True,
                              heading=timing.heading_of(a, b))
            self._set_robot(res, b)
            if not escorting:
                # 障碍物走过该子步期间机器人在节点上等待
                self.timer.transport(
                    cost.se2_path_length(obs, move_path[i - 1:i + 1], cfg))
            last_i = i
            if i in frame_at:
                self._capture_frame(res, node, f"move {oid} step {i}/{n - 1}",
                                    move_oid=oid)

        # 以最终位姿更新信念
        self.belief.relocate(obs, *move_path[-1])
        self.belief.record_move_direction(oid, start_xy, move_path[-1])
        self.belief.perceive(self.world, self.robot_xy)
        new_node = self._release_and_return(res, rp, off, n - 1, True, node, cfg)
        return (True, [], cost.se2_path_length(obs, move_path, cfg), new_node)

    def _release_and_return(self, res: RunResult, rp: list, off: int, last_i: int,
                            completed: bool, node: int, cfg: Config) -> Optional[int]:
        """松开障碍物并回到路网。

        完成的搬运按规划退出路径走：必要时绕过障碍物回到出发节点，计划其余
        部分仍有效。中断的搬运中机器人被困在规划外的抓持点——回家路可能被
        手中障碍物挡住——故驶向实际可达的最近路网节点，由调用方重规划。
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
            "move_oid": move_oid,   # 当前被搬运的障碍物（画在机器人之上）
            "plan_paths": list(self._plan_paths),   # 本帧正在执行的规划路径
            "robot": self.robot_xy,     # 真实位置——抓着障碍物时不在节点上
            "track": list(res.robot_track),
            "obstacles": [(w.oid, w.polygon, w.removed) for w in self.world],
            "perceived": perceived,
            "estimated_difficulty": {
                oid: value for oid, value in self.estimator.cache.items()
                if oid in perceived
            },
            "touched_difficulty": dict(self.belief.touched_difficulty),
            # 累计值，使帧标题与最终汇总显示相同数字而非缩略版
            "J": round(res.J, 4),
            "walk_cost": round(res.walk_cost, 4),
            "work_cost": round(res.work_cost, 4),
            "motion_time": round(self.timer.elapsed, 4),
            "plan_time": round(res.plan_time, 4),
            "cycles": res.cycles,
            "label": label,
        })

