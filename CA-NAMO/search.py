"""Branch-and-bound best-first search on the augmented roadmap"""

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
    actions: List[dict]                  # ordered sequence of 'move' / 'remove' actions
    expansions: int


def push_signature(obs) -> tuple:
    return (obs.oid, round(obs.x, 3), round(obs.y, 3), round(obs.theta, 4))


class Planner:
    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config,
                 failed_pushes: Optional[set] = None):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.cfg = cfg
        self.failed_pushes = set() if failed_pushes is None else failed_pushes
        self._robot_pos: Tuple[float, float] = (0.0, 0.0)
        self._persistent_removal_cache: Dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ planning
    def plan(self, start_node: int, goal_node: int) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]
        self._robot_pos = rm.nodes[start_node]

        # Clear persistent cache if new obstacles were revealed since the last
        # plan call — that changes the others_polys geometry for every removal
        # and would invalidate previously cached push plans.
        if self.belief.newly_revealed:
            self._persistent_removal_cache.clear()

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
            if f > incumbent + 1e-9:            # branch-and-bound pruning
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

    # ------------------------------------------------------------- edge cost
    def _edge_cost(self, key: EdgeKey) -> Tuple[float, list]:
        cfg = self.cfg
        base = cfg.lambda_distance * self.roadmap.edge_len[key]
        blockers = self.belief.blockers_of(key)
        if not blockers:
            return base, []

        # "shortest" — always push through blockers regardless of work cost.
        # Push plans are still computed (needed for execution), but the work
        # penalty is omitted from the edge cost so the search prefers the
        # geometrically shortest path.
        # "easiest"  — heavily penalise pushing so detours are preferred,
        # but obstacles can still be pushed when no detour exists.
        # "normal"   — full J = λ·D + W; both path length and push work matter.
        if cfg.strategy == "shortest":
            work_mult = 0.0
            work_bias = 0.0
        elif cfg.strategy == "easiest":
            work_mult = 1.0
            work_bias = 50.0    # per-obstacle surcharge — prefer any reasonable detour
        else:
            work_mult = 1.0
            work_bias = 0.0

        removals = []
        extra = 0.0
        for oid in blockers:
            feasible, work, drop, push_dist, push_path = self._removal(oid, key)
            if not feasible:
                return math.inf, []
            extra += work * work_mult + work_bias
            removals.append((oid, drop, push_dist, work, push_path))
        return base + extra, removals

    def _removal(self, oid: int, key: EdgeKey):
        """Compute work and drop pose needed to push an obstacle aside, also plan SE2 push path"""
        obs = self.belief.obstacle(oid)
        cache_key = (push_signature(obs), key)
        if cache_key in self._persistent_removal_cache:
            return self._persistent_removal_cache[cache_key]
        if cache_key in self.failed_pushes:
            res = (False, math.inf, None, 0.0, None)
            self._persistent_removal_cache[cache_key] = res
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
        self._persistent_removal_cache[cache_key] = res
        return res

    def _llm_bias(self, removals) -> float:
        if not self.cfg.use_llm_ordering or not removals:
            return 0.0
        s = 0.0
        for (oid, _drop, _dist, _work, _push_path) in removals:
            s += self.est.estimate(self.belief.obstacle(oid).observation())
        return s
