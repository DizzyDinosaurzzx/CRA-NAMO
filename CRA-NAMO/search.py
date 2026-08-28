"""Run branch-and-bound best-first search on the augmented roadmap."""

from __future__ import annotations
import heapq
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from config import Config
from roadmap import Roadmap, EdgeKey
from perception import Belief
from llm_difficulty import DifficultyEstimator
import contact
import cost
import manipulation

@dataclass
class Plan:
    cost: float
    node_path: List[int]
    actions: List[dict]                  # ordered sequence of 'move' / 'remove' actions
    expansions: int


_MOVE_DIR_EPS = 1e-3    # below this displacement a drop pose carries no move direction


def move_signature(obs) -> tuple:
    return (obs.oid, round(obs.x, 3), round(obs.y, 3), round(obs.theta, 4))


class Planner:
    """Plan robot motion and obstacle-removal actions against current belief."""

    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config,
                 failed_moves: Optional[set] = None,
                 risk_estimator=None):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.risk = risk_estimator
        self.cfg = cfg
        self.failed_moves = set() if failed_moves is None else failed_moves
        self._persistent_removal_cache: Dict[tuple, tuple] = {}

    def plan(self, start_node: int, goal_node: int) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]

        # Clear persistent cache if new obstacles were revealed since the last
        # plan call — that changes the others_polys geometry for every removal
        # and would invalidate previously cached move plans.
        if self.belief.changed:
            self._persistent_removal_cache.clear()

        def h(node):
            x, y = rm.nodes[node]
            return cost.heuristic(cfg, math.hypot(x - gx, y - gy))

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
                         "dist": move_dist, "work": work, "move_path": move_path,
                         "contact": cplan, "key": key}
                        for (oid, drop, move_dist, work, move_path, cplan) in removals]
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

    def _edge_cost(self, key: EdgeKey) -> Tuple[float, list]:
        """Cost of traversing one edge, including clearing whatever blocks it.

        Returns C, not J: how energy and time are weighed against each other
        lives in `cost.combine`. This function only decides *what* has to be
        moved.
        """
        base = cost.edge_cost(self.cfg, self.roadmap.edge_len[key])
        blockers = self.belief.blockers_of(key)
        if not blockers:
            return base, []

        removals = []
        extra = 0.0
        for oid in blockers:
            feasible, work, drop, move_dist, move_path, cplan = self._removal(oid, key)
            if not feasible:
                return math.inf, []
            seconds = cost.manipulation_time(self.cfg, cplan, len(move_path or []),
                                             move_dist)
            extra += cost.removal_cost(self.cfg, work, cplan.travel, seconds,
                                       self._risk_to_charge(oid))
            removals.append((oid, drop, move_dist, work, move_path, cplan))
        return base + extra, removals

    def _risk_to_charge(self, oid: int):
        """This obstacle's risk level, or None if it would be a second charge.

        The surcharge buys the decision to disturb something at all, so it is paid
        once. An obstacle already shifted has paid; shifting it again is only the
        work of shifting it.
        """
        if self.risk is None or self.belief.obstacle(oid).removed:
            return None
        return self.risk.level_of(oid)

    def _removal(self, oid: int, key: EdgeKey):
        """Compute work and drop pose needed to move an obstacle aside, also plan its SE(2) route"""
        obs = self.belief.obstacle(oid)
        cache_key = (move_signature(obs), key)
        # A move the executor has since given up on comes first: what is cached
        # is what looked possible before it was tried, and re-serving that would
        # have the planner keep proposing the move that just failed.
        if cache_key in self.failed_moves:
            res = (False, math.inf, None, 0.0, None, None)
            self._persistent_removal_cache[cache_key] = res
            return res
        if cache_key in self._persistent_removal_cache:
            return self._persistent_removal_cache[cache_key]
        clear_polys = [self.roadmap.edge_corridor[key]]
        others = self.belief.others_union(oid)

        estimated_diff = self.belief.get_difficulty(oid, self.est)
        move_path = None
        drop = None
        move_dist = 0.0
        feasible = False
        cplan = None

        bounds = self.roadmap.workspace.bounds
        bounds_xy = (bounds[0], bounds[2], bounds[1], bounds[3])

        # The robot performs the manipulation as an excursion from the edge it is
        # about to traverse. The edge midpoint stands in for whichever of its two
        # endpoints the robot is actually on — they are at most conn_radius apart —
        # which keeps this result cacheable per edge instead of per (edge,
        # direction). It lets go at whichever endpoint is cheaper to reach from
        # where the obstacle ends up: both are places the robot has to be to use
        # the edge, so neither choice hides any travel from the cost. Which one it
        # picks is the search's guess; the executor plans the same excursion again
        # from the endpoint the robot is really standing on.
        # The midpoint stands in for the robot everywhere in this function, not
        # only in the escort: the manipulation is reachable from it, and drop
        # poses are biased away from it. Feeding the robot's actual position in
        # instead would make the answer depend on which end of the edge it
        # happened to be standing on — which is exactly what the cache below
        # asserts it does not, and what had the robot stepping back and forth
        # between two nodes, each one preferring the plan of the other.
        u, v = key
        mid = tuple((a + b) / 2.0 for a, b in
                    zip(self.roadmap.nodes[u], self.roadmap.nodes[v]))
        robot_pos = mid
        exits = [(self.roadmap.nodes[u], 0.0), (self.roadmap.nodes[v], 0.0)]
        contact_memo: Dict[int, tuple] = {}

        def _contact_for(poses):
            # plan_anywhere hands the same list object to the validator and back
            # out in the result; hold on to it so the id stays unique
            hit = contact_memo.get(id(poses))
            if hit is not None:
                return hit[1]
            plan = contact.plan_contact(obs, poses, mid, exits,
                                        self.roadmap.free_eroded_tol,
                                        contact.inflate_others(others, self.cfg),
                                        self.cfg)
            contact_memo[id(poses)] = (poses, plan)
            return plan

        path_accept = None
        rejected: List[str] = []
        if self.cfg.contact_required:
            def path_accept(poses):
                plan = _contact_for(poses)
                if not plan.feasible:
                    rejected.append(plan.reason)
                return plan.feasible
        se2_feasible, se2_path, se2_cost, se2_goal = manipulation.plan_move_se2(
            obs, clear_polys, self.roadmap.static_obstacles, bounds_xy,
            robot_pos, self.cfg, others_polys=others,
            goal_accept=self._goal_filter(obs), path_accept=path_accept)
        if se2_feasible and se2_path:
            cplan = (_contact_for(se2_path) if self.cfg.contact_required
                     else contact.idle_plan(mid, len(se2_path)))
            if cplan.feasible:
                feasible = True
                move_path = se2_path
                drop = se2_goal
                move_dist = se2_cost
            else:
                self.cfg.log(f"[contact] oid={oid} {cplan.reason}")
        elif rejected:
            # the obstacle could go somewhere, the robot just could not
            # escort it there — say which half of the constraint bit
            counts = Counter(rejected).most_common(2)
            self.cfg.log(f"[contact] oid={oid} rejected {len(rejected):,} path(s): "
                         + "; ".join(f"{why} x{n:,}" for why, n in counts))

        if not feasible:
            work = math.inf
        else:
            work = cost.manipulation_work(estimated_diff, move_dist)

        res = (feasible, work, drop, move_dist, move_path, cplan)
        self._persistent_removal_cache[cache_key] = res
        return res

    def _goal_filter(self, obs):
        """Reject drop poses that leave the obstacle blocking more edges than it does
        now, or that reverse the direction this obstacle was last moved in."""
        base_blocked = self.roadmap.count_blocked_edges(obs.polygon)
        last_dir = self.belief.move_dir.get(obs.oid)

        def accept(goal):
            if self.roadmap.count_blocked_edges(obs.polygon_at(*goal)) > base_blocked:
                return False
            if last_dir is None:
                return True
            dx, dy = goal[0] - obs.x, goal[1] - obs.y
            if math.hypot(dx, dy) <= _MOVE_DIR_EPS:     # pure rotation has no direction
                return True
            return dx * last_dir[0] + dy * last_dir[1] >= 0.0

        return accept

    def _llm_bias(self, removals) -> float:
        if not self.cfg.use_llm_ordering or not removals:
            return 0.0
        s = 0.0
        for (oid, _drop, _dist, _work, _move_path, _cplan) in removals:
            s += self.est.estimate(self.belief.obstacle(oid).observation())
        return s
