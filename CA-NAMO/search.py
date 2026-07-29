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
        self._robot_pos: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ 规划
    def plan(self, start_node: int, goal_node: int) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]

        # 推动可行性一律按【本次规划的出发点】评估，整个 plan() 期间固定。
        # 曾经用 A* 当前弹出的节点，它每扩展一步就变一次，而 _removal 的缓存键
        # 只有 (oid, edge)——于是从远处评估某次推动时，会沿用近处节点算出的
        # “可行”结论，实际上那时障碍物根本不在工作圆内，代价被系统性低估。
        self._robot_pos = rm.nodes[start_node]
        # 缓存键是 (oid, edge_key)：放置方案现在依赖具体要打通哪一条边
        self._removal_cache: Dict[
            Tuple[int, EdgeKey], Tuple[bool, float, Optional[tuple], float, Optional[list]]
        ] = {}

        def h(node):
            x, y = rm.nodes[node]
            return cfg.lambda_distance * math.hypot(x - gx, y - gy)

        counter = itertools.count()
        # 状态就是路网节点本身。曾经带着一个"本方案中已推开的 oid 集合"，那是给
        # "推过一次之后该障碍物在后续边上免费"的记账方式用的；现在每条被挡的边都
        # 照实计一次推动费，_edge_cost 根本不看这个集合，留着只会把同一个节点按
        # 不同的已推集合拆成许多代价完全相同的重复状态，白白多扩展、多算推动。
        # 优先级：(f, llm_bias, tie)；父节点映射：node -> (prev_node, actions, gcost)
        open_heap = [(h(start_node), 0.0, next(counter), start_node)]
        g_best: Dict[int, float] = {start_node: 0.0}
        parent: Dict[int, Tuple] = {start_node: (None, [], 0.0)}
        incumbent = math.inf
        goal_reached = False
        expansions = 0

        while open_heap:
            f, bias, _, node = heapq.heappop(open_heap)
            g = g_best.get(node, math.inf)
            if f > incumbent + 1e-9:            # 分支限界剪枝
                continue
            if node == goal_node:
                incumbent = g
                goal_reached = True
                # 注意：这【不是】最优性保证。h 本身可采纳，但 _edge_cost 是按
                # 每条被挡的边各计一次推动费来记账的（保守、可能高估真实做功），
                # 取第一个弹出的目标，与"对未知空间保持乐观、靠重规划纠正"一致。
                break
            expansions += 1
            if expansions > cfg.max_expansions:
                break

            for v, key, length in rm.neighbors(node):
                step_cost, removals = self._edge_cost(key)
                if step_cost == math.inf:
                    continue
                ng = g + step_cost
                if ng >= g_best.get(v, math.inf) - 1e-12:
                    continue
                nf = ng + h(v)
                if nf >= incumbent - 1e-9:
                    continue
                g_best[v] = ng
                acts = [{"type": "remove", "oid": oid, "drop": drop,
                         "dist": push_dist, "work": work, "push_path": push_path}
                        for (oid, drop, push_dist, work, push_path) in removals]
                acts.append({"type": "move", "u": node, "v": v,
                             "dist": length, "cost": step_cost})
                parent[v] = (node, acts, ng)
                nbias = bias + self._llm_bias(removals)
                heapq.heappush(open_heap, (nf, nbias, next(counter), v))

        if not goal_reached:
            return None

        # 重建路径
        actions: List[dict] = []
        node_path: List[int] = []
        s = goal_node
        while s is not None:
            prev, acts, _ = parent[s]
            node_path.append(s)
            if prev is not None:
                actions = acts + actions
            s = prev
        node_path.reverse()
        return Plan(cost=round(incumbent, 4), node_path=node_path,
                    actions=actions, expansions=expansions)

    # ------------------------------------------------------------- 边代价
    def _edge_cost(self, key: EdgeKey) -> Tuple[float, list]:
        cfg = self.cfg
        base = cfg.lambda_distance * self.roadmap.edge_len[key]
        # 一次推动只算腾空了当前这一条边：同一个 oid 若还挡着后续的边，
        # 那些边会各自再计一次推动费（保守记账，不做"推过一次就永久免费"的假设）。
        blockers = self.belief.blockers_of(key)
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
        # 只要求腾空当前这一条边的走廊。SE2 规划器的 set_corridor() 用 inside_convex()
        # 逐块判定，要求每块【凸】——单条边的走廊是 buffer(cap_style=2) 出来的矩形，
        # 天然凸，所以这里直接包成单元素列表喂给它。
        clear_polys = [self.roadmap.edge_corridor[key]]
        others = None
        if self.cfg.check_obstacle_collision:
            polys = [ob.polygon for oid2, ob in self.belief.perceived.items()
                     if oid2 != oid]
            # 压在被推物体自身位置上的接触区要扣掉：扫掠区的起点就是它当前的
            # footprint，必然与之相交，否则该物体会被判定为永远不可推动。但只扣掉
            # 重叠的那一块——整块丢弃等于凭空忘掉一处已知的真实障碍，规划器会再次
            # 把物体推向那里、再撞一次，反复空转。
            for c in self.belief.contacts:
                part = c.difference(obs.polygon)
                if not part.is_empty and part.area > 1e-9:
                    polys.append(part)
            others = unary_union(polys) if polys else None

        estimated_diff = self.belief.get_difficulty(oid, self.est)
        push_path = None
        drop = None
        push_dist = 0.0
        feasible = False

        robot_pos = self._robot_pos
        bounds = self.roadmap.workspace.bounds
        bounds_xy = (bounds[0], bounds[2], bounds[1], bounds[3])

        # SE2 全空间搜索：落点与到达它的连续推动路径由同一次搜索给出。
        # 搜不出来就是本轮不可推动——没有"直接把障碍物挪到落点"的兜底。
        if self.cfg.push_use_planner:
            se2_feasible, se2_path, se2_cost, se2_goal = geometry.push_plan_se2(
                obs, clear_polys, self.roadmap.static_obstacles, bounds_xy,
                robot_pos, self.cfg, others_polys=others)
            if se2_feasible and se2_path:
                feasible = True
                push_path = se2_path
                drop = se2_goal
                push_dist = se2_cost

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
