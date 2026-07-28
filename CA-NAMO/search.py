"""在增广路网上使用分支限界的最佳优先搜索"""

from __future__ import annotations
import heapq
import itertools
import math
from dataclasses import dataclass, field
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
    def removals(self):
        return [a for a in self.actions if a["type"] == "remove"]


class Planner:
    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.cfg = cfg

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
        self._removal_cache: Dict[
            int, Tuple[bool, float, Optional[tuple], float]
        ] = {}

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
                break                            # h 可采纳 => 第一个弹出的即为最优
            expansions += 1
            if expansions > cfg.max_expansions:
                break

            for v, key, length in rm.neighbors(node):
                step_cost, removals = self._edge_cost(key, removed)
                if step_cost == math.inf:
                    continue
                ng = g + step_cost
                new_removed = removed
                for (oid, _drop, _dist, _work) in removals:
                    new_removed = new_removed | {oid}
                nstate = (v, new_removed)
                if ng >= g_best.get(nstate, math.inf) - 1e-12:
                    continue
                nf = ng + h(v)
                if nf >= incumbent - 1e-9:
                    continue
                g_best[nstate] = ng
                acts = [{"type": "remove", "oid": oid, "drop": drop,
                         "dist": push_dist, "work": work}
                        for (oid, drop, push_dist, work) in removals]
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
        blockers = self.belief.blockers_of(key) - set(removed_in_plan)
        if not blockers:
            return base, []
        removals = []
        extra = 0.0
        for oid in blockers:
            feasible, work, drop, push_dist = self._removal(oid)
            if not feasible:
                return math.inf, []
            extra += work
            removals.append((oid, drop, push_dist, work))
        return base + extra, removals

    def _removal(self, oid: int):
        """计算把障碍物移出其全部阻挡边所需的做功和放置位姿。"""
        if oid in self._removal_cache:
            return self._removal_cache[oid]
        obs = self.belief.obstacle(oid)
        own = [self.roadmap.edge_corridor[k]
               for k, bs in self.belief.edge_blockers.items() if oid in bs]
        must_clear = unary_union(own) if own else None
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
        feasible, dist, drop = geometry.push_plan(
            obs, self.roadmap.static_free, must_clear, avoid, self.cfg,
            others=others, goal_xy=self._goal_xy)
        if not feasible:
            work = math.inf
        else:
            # W = difficulty * 推动距离；未触碰过的障碍物用 LLM/启发式估计的 difficulty
            estimated_diff = self.belief.get_difficulty(oid, self.est)
            work = estimated_diff * dist
        res = (feasible, work, drop, dist)
        self._removal_cache[oid] = res
        return res

    def _llm_bias(self, removals) -> float:
        """Ordering-only term: sum of LLM-estimated difficulty of removed obstacles."""
        if not self.cfg.use_llm_ordering or not removals:
            return 0.0
        s = 0.0
        for (oid, _drop, _dist, _work) in removals:
            s += self.est.estimate(self.belief.obstacle(oid).observation())
        return s
