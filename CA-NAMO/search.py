"""Branch-and-bound best-first search on the augmented roadmap"""

from __future__ import annotations
import heapq
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import shapely
from shapely.ops import unary_union
from config import Config
from roadmap import Roadmap, EdgeKey
from perception import Belief
from llm_difficulty import DifficultyEstimator
import contact
import cost
import manipulation
import timing

@dataclass
class Plan:
    cost: float
    node_path: List[int]
    actions: List[dict]                  # ordered sequence of 'move' / 'remove' actions
    expansions: int


_MOVE_DIR_EPS = 1e-3    # below this displacement a drop pose carries no move direction
_INFLATED_CACHE_MAX = 64


def move_signature(obs) -> tuple:
    return (obs.oid, round(obs.x, 3), round(obs.y, 3), round(obs.theta, 4))


class Planner:
    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config,
                 failed_moves: Optional[set] = None):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.cfg = cfg
        self.failed_moves = set() if failed_moves is None else failed_moves
        self._robot_pos: Tuple[float, float] = (0.0, 0.0)
        self._persistent_removal_cache: Dict[tuple, tuple] = {}
        self._inflated_cache: Dict[bytes, object] = {}

    # --- planning ---
    def plan(self, start_node: int, goal_node: int,
             heading: Optional[float] = None) -> Optional[Plan]:
        """Best route from `start_node`, given the robot is facing `heading`.

        `heading` only matters when `time_importance > 0`, where it prices the
        pivot the robot would have to make to set off down each first edge.
        """
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]
        self._robot_pos = rm.nodes[start_node]

        # Clear persistent cache if new obstacles were revealed since the last
        # plan call — that changes the others_polys geometry for every removal
        # and would invalidate previously cached move plans.
        if self.belief.newly_revealed:
            self._persistent_removal_cache.clear()

        def h(node):
            # Admissible at every time_importance: the straight line is a lower
            # bound on the remaining distance, and no route covers it faster
            # than v_max, so it also lower-bounds the remaining time.
            x, y = rm.nodes[node]
            return cost.search_motion_cost(cfg, math.hypot(x - gx, y - gy))

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

            # Heading the robot would arrive on. At the start node that is where
            # it is really pointing right now, which matters more than anything
            # else here: `step_execute_edges` is 1, so the first edge is the only
            # one that ever gets driven, and pricing its corner against the real
            # heading is what stops consecutive plans from zig-zagging.
            # Deeper in the tree it comes from whichever parent is recorded —
            # `g_best` stays keyed on the node alone rather than on
            # (node, heading), the same trade the edge midpoint already makes for
            # the manipulation excursion. Only ever adds non-negative cost, so h
            # stays admissible, and it is identically zero at time_importance=0.
            here = rm.nodes[node]
            if node == start_node:
                h_in = heading
            else:
                prev = parent[node][0]
                h_in = (timing.heading_of(rm.nodes[prev], here)
                        if prev is not None else None)

            for v, key, length in rm.neighbors(node):
                step_cost, removals = self._edge_cost(key)
                if step_cost == math.inf:
                    continue
                if h_in is not None:
                    step_cost += cost.search_turn_cost(cfg, timing.turn_between(
                        h_in, timing.heading_of(here, rm.nodes[v])))
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

    # --- edge cost ---
    def _edge_cost(self, key: EdgeKey) -> Tuple[float, list]:
        """Cost of traversing one edge, including clearing whatever blocks it.

        How energy and time are weighed against each other lives in `cost.py`;
        this function only decides *what* has to be moved.
        """
        base = cost.search_motion_cost(self.cfg, self.roadmap.edge_len[key])
        blockers = self.belief.blockers_of(key)
        if not blockers:
            return base, []

        removals = []
        extra = 0.0
        for oid in blockers:
            feasible, work, drop, move_dist, move_path, cplan = self._removal(oid, key)
            if not feasible:
                return math.inf, []
            extra += cost.removal_cost(self.cfg, work, cplan.travel, move_dist)
            removals.append((oid, drop, move_dist, work, move_path, cplan))
        return base + extra, removals

    def _removal(self, oid: int, key: EdgeKey):
        """Compute work and drop pose needed to move an obstacle aside, also plan its SE(2) route"""
        obs = self.belief.obstacle(oid)
        cache_key = (move_signature(obs), key)
        if cache_key in self._persistent_removal_cache:
            return self._persistent_removal_cache[cache_key]
        if cache_key in self.failed_moves:
            res = (False, math.inf, None, 0.0, None, None)
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
        move_path = None
        drop = None
        move_dist = 0.0
        feasible = False
        cplan = None

        robot_pos = self._robot_pos
        bounds = self.roadmap.workspace.bounds
        bounds_xy = (bounds[0], bounds[2], bounds[1], bounds[3])

        # The robot performs the manipulation as an excursion from the edge it is
        # about to traverse and returns to it afterwards. The edge midpoint stands
        # in for whichever of its two endpoints the robot is actually on — they are
        # at most conn_radius apart — which keeps this result cacheable per edge
        # instead of per (edge, direction).
        u, v = key
        mid = tuple((a + b) / 2.0 for a, b in
                    zip(self.roadmap.nodes[u], self.roadmap.nodes[v]))
        contact_memo: Dict[int, tuple] = {}

        def _contact_for(poses):
            # plan_anywhere hands the same list object to the validator and back
            # out in the result; hold on to it so the id stays unique
            hit = contact_memo.get(id(poses))
            if hit is not None:
                return hit[1]
            plan = contact.plan_contact(obs, poses, mid, mid,
                                        self.roadmap.free_eroded_tol,
                                        self._others_inflated(others), self.cfg)
            contact_memo[id(poses)] = (poses, plan)
            return plan

        if self.cfg.se2_use_planner:
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
                self.cfg.log(f"[contact] oid={oid} rejected {len(rejected)} path(s): "
                             + "; ".join(f"{why} x{n}" for why, n in counts))

        if not feasible:
            work = math.inf
        else:
            work = cost.manipulation_work(estimated_diff, move_dist)

        res = (feasible, work, drop, move_dist, move_path, cplan)
        self._persistent_removal_cache[cache_key] = res
        return res

    def _others_inflated(self, others):
        """Robot-centre positions ruled out by the *other* movable obstacles.

        Content-addressed by WKB: the union is rebuilt from scratch for every
        removal candidate, but the same set of obstacles recurs constantly across
        edges and cycles, and inflating it is the expensive half.
        """
        if others is None or others.is_empty:
            return None
        key = others.wkb
        cached = self._inflated_cache.get(key)
        if cached is not None:
            return cached
        geom = others.buffer(max(self.cfg.robot_radius - self.cfg.contact_clearance,
                                 1e-6))
        shapely.prepare(geom)
        self._inflated_cache[key] = geom
        while len(self._inflated_cache) > _INFLATED_CACHE_MAX:
            self._inflated_cache.pop(next(iter(self._inflated_cache)))
        return geom

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

