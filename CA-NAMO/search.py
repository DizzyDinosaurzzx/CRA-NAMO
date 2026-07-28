"""在增广路网上使用分支限界的最佳优先搜索"""

from __future__ import annotations
import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from shapely.ops import unary_union
from config import Config
from roadmap import Roadmap, EdgeKey
from perception import Belief
from llm_difficulty import DifficultyEstimator
import geometry

@dataclass
class Plan:
    cost: float
    node_path: List[int]
    actions: List[dict]                  # 有序的 'move' / 'remove' 动作序列
    expansions: int

class Planner:
    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.cfg = cfg
        # 本次规划中机器人所在位置，供推动规划器确定工作圆。整个 plan() 期间固定，
        # 见 plan() 里的说明。
        self._robot_pos: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ 规划
    def plan(self, start_node: int, goal_node: int) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]

        # 所有当前畅通边的通道：放置障碍物时必须避开这些通道（避免反效果）
        # 每次规划只计算一次。
        
        free_corridors = [rm.edge_corridor[k] for k in rm.edge_len
                          if not self.belief.blockers_of(k)]
        self.free_union = unary_union(free_corridors) if free_corridors else None
        # 放置障碍物时需要知道目标在哪，避免把它挪到机器人剩余路线上
        self._goal_xy = (gx, gy)
        # 推动可行性一律按【本次规划的出发点】评估，整个 plan() 期间固定。
        # 曾经用 A* 当前弹出的节点，它每扩展一步就变一次，而 _removal 的缓存键
        # 只有 (oid, edge)——于是从远处评估某次推动时，会沿用近处节点算出的
        # “可行”结论，实际上那时障碍物根本不在工作圆内，代价被系统性低估。
        self._robot_pos = rm.nodes[start_node]
        # 缓存键是 (oid, edge_key)：放置方案现在依赖具体要打通哪一条边
        self._removal_cache: Dict[
            Tuple[int, EdgeKey], Tuple[bool, float, Optional[tuple], float, Optional[list]]
        ] = {}
        # oid -> 该障碍物当前阻挡的全部走廊并集（软罚分用），与边无关故按 oid 缓存
        self._blocked_union_cache: Dict[int, Optional[object]] = {}

        def h(node):
            x, y = rm.nodes[node]
            return cfg.lambda_distance * math.hypot(x - gx, y - gy)

        counter = itertools.count()
        start_state = (start_node, frozenset())
        # 优先级：(f, llm_bias, tie)；父节点映射：state -> (prev_state, actions, gcost)
        open_heap = [(h(start_node), 0.0, next(counter), start_state)]
        g_best: Dict[Tuple[int, frozenset], float] = {start_state: 0.0}
        parent: Dict[Tuple[int, frozenset], Tuple] = {start_state: (None, [], 0.0)}
        incumbent = math.inf
        goal_state = None
        expansions = 0

        while open_heap:
            f, bias, _, state = heapq.heappop(open_heap)
            node, removed = state
            g = g_best.get(state, math.inf)
            if f > incumbent + 1e-9:            # 分支限界剪枝
                continue
            if node == goal_node:
                incumbent = g
                goal_state = state
                # 注意：这【不是】最优性保证。h 本身可采纳，但 _edge_cost 对
                # removed_in_plan 的处理是乐观的（一次推动只保证腾空了那一条边，
                # 却让该 oid 在后续所有边上都当作已让开），所以 g 本身就是下界。
                # 取第一个弹出的目标，与"对未知空间保持乐观、靠重规划纠正"一致。
                break
            expansions += 1
            if expansions > cfg.max_expansions:
                break

            for v, key, length in rm.neighbors(node):
                step_cost, removals = self._edge_cost(key, removed)
                if step_cost == math.inf:
                    continue
                ng = g + step_cost
                new_removed = removed
                for (oid, _drop, _dist, _work, _push_path) in removals:
                    new_removed = new_removed | {oid}
                nstate = (v, new_removed)
                if ng >= g_best.get(nstate, math.inf) - 1e-12:
                    continue
                nf = ng + h(v)
                if nf >= incumbent - 1e-9:
                    continue
                g_best[nstate] = ng
                acts = [{"type": "remove", "oid": oid, "drop": drop,
                         "dist": push_dist, "work": work, "push_path": push_path}
                        for (oid, drop, push_dist, work, push_path) in removals]
                acts.append({"type": "move", "u": node, "v": v,
                             "dist": length, "cost": step_cost})
                parent[nstate] = (state, acts, ng)
                nbias = bias + self._llm_bias(removals)
                heapq.heappush(open_heap, (nf, nbias, next(counter), nstate))

        if goal_state is None:
            return None

        # 重建路径
        actions: List[dict] = []
        node_path: List[int] = []
        s = goal_state
        while s is not None:
            prev, acts, _ = parent[s]
            node_path.append(s[0])
            if prev is not None:
                actions = acts + actions
            s = prev
        node_path.reverse()
        return Plan(cost=round(incumbent, 4), node_path=node_path,
                    actions=actions, expansions=expansions)

    # ------------------------------------------------------------- 边代价
    def _edge_cost(self, key: EdgeKey, removed_in_plan) -> Tuple[float, list]:
        cfg = self.cfg
        base = cfg.lambda_distance * self.roadmap.edge_len[key]
        blockers = self.belief.blockers_of(key)
        if cfg.push_clear_scope != "repeat":
            # "edge"/"union"：该 oid 一旦被推过，后续边一律免费。对 "union" 是自洽的
            # （一次推动确实清空了它挡住的全部通道）；对 "edge" 则是那条已知的低估。
            blockers = blockers - set(removed_in_plan)
        if not blockers:
            return base, []
        removals = []
        extra = 0.0
        for oid in blockers:
            feasible, work, drop, push_dist, push_path = self._removal(oid, key)
            if not feasible:
                return math.inf, []
            extra += work
            removals.append((oid, drop, push_dist, work, push_path))
        return base + extra, removals

    def _removal(self, oid: int, key: EdgeKey):
        """计算把障碍物推开所需的做功和放置位姿，同时规划 SE2 推动路径"""
        cache_key = (oid, key)
        if cache_key in self._removal_cache:
            return self._removal_cache[cache_key]
        obs = self.belief.obstacle(oid)
        # 该障碍物当前阻挡的全部通道并集。每个 oid 只算一次并缓存——与具体打通哪条边无关。
        own = [self.roadmap.edge_corridor[k]
               for k, bs in self.belief.edge_blockers.items() if oid in bs]
        residual = self._blocked_union_cache.get(oid, False)
        if residual is False:
            residual = unary_union(own) if own else None
            self._blocked_union_cache[oid] = residual

        # SE2 规划器的 set_corridor() 用 inside_convex() 逐块判定，要求每块【凸】；
        # 单条边的走廊是 buffer(cap_style=2) 出来的矩形，天然凸，而它们的并集
        # 既非凸也可能是 MultiPolygon。所以这里保留两种形式：
        #   clear_polys —— 凸块列表，喂给 SE2 规划器
        #   must_clear  —— 单个几何（可非凸），喂给环形采样的 intersects 判定
        if self.cfg.push_clear_scope == "union" and own:
            clear_polys = own
            must_clear = residual
            residual = None      # 并集已升为硬约束，不必再作为软罚分重复计入
        else:
            must_clear = self.roadmap.edge_corridor[key]
            clear_polys = [must_clear]
        avoid = self.free_union
        others = None
        if self.cfg.check_obstacle_collision:
            polys = [ob.polygon for oid2, ob in self.belief.perceived.items()
                     if oid2 != oid]
            # 压在被推物体自身位置上的接触区要排除：扫掠区的起点就是它当前的
            # footprint，必然与之相交，否则该物体会被判定为永远不可推动。
            polys += [c for c in self.belief.contacts
                      if not c.intersects(obs.polygon)]
            others = unary_union(polys) if polys else None

        estimated_diff = self.belief.get_difficulty(oid, self.est)
        push_path = None
        drop = None
        push_dist = 0.0
        feasible = False

        robot_pos = self._robot_pos
        bounds = self.roadmap.workspace.bounds
        bounds_xy = (bounds[0], bounds[2], bounds[1], bounds[3])

        # SE2 全空间搜索
        if self.cfg.push_use_planner and must_clear is not None:
            se2_feasible, se2_path, se2_cost, se2_goal = geometry.push_plan_se2(
                obs, clear_polys, self.roadmap.static_obstacles, bounds_xy,
                robot_pos, self.cfg, others_polys=others)
            if se2_feasible and se2_path:
                feasible = True
                push_path = se2_path
                drop = se2_goal
                push_dist = se2_cost

        # 回退：环形采样 + SE2 路径规划
        if not feasible:
            sample_thetas = None
            if self.cfg.push_sample_theta:
                base_th = obs.theta
                sample_thetas = [base_th, base_th + math.pi/4,
                                 base_th + math.pi/2, base_th + 3*math.pi/4]
            feasible, push_dist, drop = geometry.push_plan(
                obs, self.roadmap.static_free_prep, must_clear, avoid, self.cfg,
                others=others, goal_xy=self._goal_xy, residual=residual,
                sample_thetas=sample_thetas)
            if feasible and self.cfg.push_use_planner:
                # others 必须一并传入：漏传会规划出穿过其他可移动障碍物的推动路径
                se2_feasible, se2_path, se2_cost = geometry.push_path_plan(
                    obs, drop, self.roadmap.static_obstacles, bounds_xy,
                    robot_pos, self.cfg, others_polys=others)
                if se2_feasible and se2_path:
                    push_path = se2_path
                    push_dist = se2_cost
                else:
                    feasible = False  # 无 SE2 路径 → 不可行，不瞬移

        if not feasible:
            work = math.inf
        else:
            # 规划期用估计难度：触碰过的是真实值，否则是 LLM/启发式估计
            work = geometry.push_work(estimated_diff, push_dist)

        res = (feasible, work, drop, push_dist, push_path)
        self._removal_cache[cache_key] = res
        return res

    def _llm_bias(self, removals) -> float:
        """Ordering-only term: sum of LLM-estimated difficulty of removed obstacles."""
        if not self.cfg.use_llm_ordering or not removals:
            return 0.0
        s = 0.0
        for (oid, _drop, _dist, _work, _push_path) in removals:
            s += self.est.estimate(self.belief.obstacle(oid).observation())
        return s
