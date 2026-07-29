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


def push_signature(obs) -> tuple:
    """一次推动尝试的登记标识：障碍物 + 它当时的位姿。

    位姿量化到毫米 / 毫弧度纯粹是为了让 key 稳定可哈希——信念里的位姿只会被
    `Belief.relocate` 整体改写，不存在逐步累积的浮点漂移。
    """
    return (obs.oid, round(obs.x, 3), round(obs.y, 3), round(obs.theta, 4))


class Planner:
    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config,
                 failed_pushes: Optional[set] = None):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.cfg = cfg
        # 执行期实测为"一步都推不动"的推动尝试，元素是 (push_signature, EdgeKey)。
        # 由 OnlineNAMO 持有并跨重规划周期累积，见 _removal()。
        self.failed_pushes = set() if failed_pushes is None else failed_pushes
        self._robot_pos: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ 规划
    def plan(self, start_node: int, goal_node: int) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]
        self._robot_pos = rm.nodes[start_node]
        self._removal_cache: Dict[
            Tuple[int, EdgeKey], Tuple[bool, float, Optional[tuple], float, Optional[list]]
        ] = {}

        def h(node):
            x, y = rm.nodes[node]
            return cfg.lambda_distance * math.hypot(x - gx, y - gy)

        counter = itertools.count()
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
                # 带上 key：执行期推动失败时要按【这条边的走廊】登记，
                # 同一个障碍物为另一条边让路是另一回事，不该一并封掉。
                acts = [{"type": "remove", "oid": oid, "drop": drop,
                         "dist": push_dist, "work": work, "push_path": push_path,
                         "key": key}
                        for (oid, drop, push_dist, work, push_path) in removals]
                acts.append({"type": "move", "u": node, "v": v,
                             "dist": length, "cost": step_cost})
                parent[v] = (node, acts, ng)
                nbias = bias + self._llm_bias(removals)
                heapq.heappush(open_heap, (nf, nbias, next(counter), v))

        if not goal_reached:
            return None

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
        if (push_signature(obs), key) in self.failed_pushes:
            # 这条推动上一轮真去执行过，一步都没推动就撞停了。撞的若是墙，信念里不会
            # 留下任何痕迹（墙本就是已知几何），于是不拦住它的话，下一轮会拿完全相同
            # 的信念规划出逐字节相同的计划，如此空转到 max_replans 耗尽。
            # 登记的是"该位姿 + 该走廊"这一组合：障碍物一旦真被推动过、位姿变了，
            # 签名自然不再匹配，封禁随之失效，不会误伤后续的重试。
            res = (False, math.inf, None, 0.0, None)
            self._removal_cache[cache_key] = res
            return res
        clear_polys = [self.roadmap.edge_corridor[key]]
        others = None
        if self.cfg.check_obstacle_collision:
            polys = [ob.polygon for oid2, ob in self.belief.perceived.items()
                     if oid2 != oid]
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
            work = geometry.push_work(estimated_diff, push_dist)

        res = (feasible, work, drop, push_dist, push_path)
        self._removal_cache[cache_key] = res
        return res

    def _llm_bias(self, removals) -> float:
        if not self.cfg.use_llm_ordering or not removals:
            return 0.0
        s = 0.0
        for (oid, _drop, _dist, _work, _push_path) in removals:
            s += self.est.estimate(self.belief.obstacle(oid).observation())
        return s
