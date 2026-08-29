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
import kinematics
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


class FailedMoves:
    """Manipulations the executor tried and could not carry out.

    Keeping them is what stops the planner proposing, cycle after cycle, the
    move that has just failed. On a map that moves, though, a refusal is only
    good for the arrangement of the world it was collected in: the escort that
    could not be walked because something stood in the way is walkable once that
    something drives off. Each entry therefore carries the world version it
    failed under, and is dropped as soon as the world moves on from it.
    """

    def __init__(self):
        self._at: Dict[tuple, int] = {}

    def add(self, key: tuple, version: int = 0):
        self._at[key] = version

    def drop_stale(self, version: int) -> int:
        """Forget refusals collected before world version *version*."""
        stale = [k for k, v in self._at.items() if v < version]
        for k in stale:
            del self._at[k]
        return len(stale)

    def __contains__(self, key) -> bool:
        return key in self._at

    def __len__(self) -> int:
        return len(self._at)


class Planner:
    """Plan robot motion and obstacle-removal actions against current belief."""

    def __init__(self, roadmap: Roadmap, belief: Belief,
                 estimator: DifficultyEstimator, cfg: Config,
                 failed_moves: Optional[FailedMoves] = None,
                 risk_estimator=None,
                 wait_budget: Optional[Dict[tuple, float]] = None):
        self.roadmap = roadmap
        self.belief = belief
        self.est = estimator
        self.risk = risk_estimator
        self.cfg = cfg
        self.failed_moves = FailedMoves() if failed_moves is None else failed_moves
        # How long the robot has already stood waiting for each set of bodies to
        # get out of the way. Keyed by what it is waiting *for* and not by which
        # edge it is standing at: a door is several parallel edges, and a tally
        # per edge lets a robot wait out its patience on each of them in turn.
        self.wait_budget: Dict[tuple, float] = (
            {} if wait_budget is None else wait_budget)
        self._persistent_removal_cache: Dict[tuple, tuple] = {}

    def plan(self, start_node: int, goal_node: int,
             start_heading: Optional[float] = None) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]

        # Whether which way the robot is facing is part of where it is. Turning
        # costs time and no energy, so with the objective all energy it cannot
        # change which route is best and carrying it would buy nothing but
        # states. With time in the objective it can: a route of two long
        # straights beats one of ten short zigzags of the same length, and a
        # search that cannot tell them apart is not minimising what the executor
        # will be billed for.
        track = cfg.time_importance > 0.0
        profile = cfg.free_profile()

        def turn_cost(heading: Optional[float], course: float) -> float:
            if heading is None:
                return 0.0
            return cost.combine(cfg, 0.0, profile.rotate_time(
                kinematics.turn_between(heading, course)))

        # Clear persistent cache if new obstacles were revealed since the last
        # plan call — that changes the others_polys geometry for every removal
        # and would invalidate previously cached move plans.
        if self.belief.changed:
            self.forget_removals()

        def h(node):
            x, y = rm.nodes[node]
            return cost.heuristic(cfg, math.hypot(x - gx, y - gy))

        counter = itertools.count()
        start_state = (start_node, start_heading) if track else start_node
        open_heap = [(h(start_node), 0.0, next(counter), start_state)]
        g_best: Dict[object, float] = {start_state: 0.0}
        parent: Dict[object, Tuple] = {start_state: (None, [], 0.0)}
        incumbent = math.inf
        goal_state = None
        expansions = 0

        while open_heap:
            f, bias, _, state = heapq.heappop(open_heap)
            node = state[0] if track else state
            g = g_best.get(state, math.inf)
            if f > incumbent + 1e-9:            # branch-and-bound pruning
                continue
            if node == goal_node:
                incumbent = g
                goal_state = state
                break
            expansions += 1
            if expansions > cfg.max_expansions:
                break

            ux, uy = rm.nodes[node]
            for v, key, length in rm.neighbors(node):
                step_cost, edge_acts = self._edge_cost(key)
                if step_cost == math.inf:
                    continue
                if track:
                    vx, vy = rm.nodes[v]
                    course = math.atan2(vy - uy, vx - ux)
                    step_cost += turn_cost(state[1], course)
                    nstate = (v, course)
                else:
                    nstate = v
                ng = g + step_cost
                if ng >= g_best.get(nstate, math.inf) - 1e-12:
                    continue
                nf = ng + h(v)
                if nf >= incumbent - 1e-9:
                    continue
                g_best[nstate] = ng
                acts = list(edge_acts)
                acts.append({"type": "move", "u": node, "v": v,
                             "dist": length, "cost": step_cost})
                parent[nstate] = (state, acts, ng)
                nbias = bias + self._llm_bias(edge_acts)
                heapq.heappush(open_heap, (nf, nbias, next(counter), nstate))

        if goal_state is None:
            return None

        actions: List[dict] = []
        node_path: List[int] = []
        s = goal_state
        while s is not None:
            prev, acts, _ = parent[s]
            node_path.append(s[0] if track else s)
            if prev is not None:
                actions = acts + actions
            s = prev
        node_path.reverse()
        return Plan(cost=round(incumbent, 4), node_path=node_path,
                    actions=actions, expansions=expansions)

    def forget_removals(self):
        """Drop every cached manipulation; the world they were costed in is gone."""
        self._persistent_removal_cache.clear()

    def _edge_cost(self, key: EdgeKey) -> Tuple[float, list]:
        """Cost of traversing one edge, and what has to happen first.

        Returns C, not J: how energy and time are weighed against each other
        lives in `cost.combine`. This function only decides *what* has to be
        done — which is not always moving something. A body the robot has
        watched drive itself somewhere is a body that may drive itself away
        again, and standing still until it does is an option with a price, so it
        goes up against the price of shifting it and the price of going round.
        """
        base = cost.edge_cost(self.cfg, self.roadmap.edge_len[key])
        blockers = self.belief.blockers_of(key)
        if not blockers:
            return base, []

        wait = self._wait_option(key, blockers)
        removals = []
        extra = 0.0
        # Sorted, because the order two obstacles come off an edge is a decision
        # and not something to be left to how a set happens to iterate; and
        # sequential, because the second one has to be planned around where the
        # first one is going rather than around where it is standing now.
        moved_ahead: Dict[int, tuple] = {}
        for oid in sorted(blockers):
            feasible, work, drop, move_dist, move_path, cplan = self._removal(
                oid, key, moved_ahead)
            if not feasible:
                if wait is None:
                    return math.inf, []
                return base + wait, [{"type": "wait", "key": key,
                                      "seconds": self.cfg.dynamic_wait_step,
                                      "oids": sorted(blockers)}]
            if drop is not None:
                moved_ahead[oid] = tuple(drop)
            seconds = cost.manipulation_time(self.cfg, cplan, len(move_path or []),
                                             move_dist)
            extra += cost.removal_cost(self.cfg, work, cplan.travel, seconds,
                                       self._risk_to_charge(oid))
            removals.append((oid, drop, move_dist, work, move_path, cplan))
        if wait is not None and wait <= extra:
            return base + wait, [{"type": "wait", "key": key,
                                  "seconds": self.cfg.dynamic_wait_step,
                                  "oids": sorted(blockers)}]
        return base + extra, [
            {"type": "remove", "oid": oid, "drop": drop, "dist": move_dist,
             "work": work, "move_path": move_path, "contact": cplan, "key": key}
            for (oid, drop, move_dist, work, move_path, cplan) in removals]

    def _wait_option(self, key: EdgeKey, blockers) -> Optional[float]:
        """What standing still in front of this edge would cost, if it is an option.

        Only for edges blocked by things the robot has actually watched move on
        their own. Anything else is furniture, and waiting for furniture to walk
        away is not a plan. The wait is priced as what one step of standing
        still costs — no distance, no work, just the clock — and it is withdrawn
        once the robot has spent its patience on this edge, so a way that never
        opens cannot be waited on for ever.
        """
        moving = getattr(self.belief, "seen_moving", ())
        if not moving or self.cfg.dynamic_wait_step <= 0.0:
            return None
        if not all(oid in moving for oid in blockers):
            return None
        if self.wait_budget.get(tuple(sorted(blockers)),
                                0.0) >= self.cfg.dynamic_max_wait:
            return None
        return cost.combine(self.cfg, 0.0, self.cfg.dynamic_wait_step)

    def _risk_to_charge(self, oid: int):
        """This obstacle's risk level, or None if it would be a second charge.

        The surcharge buys the decision to disturb something at all, so it is paid
        once. An obstacle already shifted has paid; shifting it again is only the
        work of shifting it. What the world rewrites while nobody is looking is
        not the thing that was paid for, so `Belief.invalidate_contact` takes the
        receipt back and the next decision to disturb it is priced afresh.
        """
        if self.risk is None or oid in self.belief.disturbed:
            return None
        return self.risk.level_of(oid, self.belief.partners_of(oid))

    def _off_limits(self, oid: int, obs) -> str:
        """Why this body may not be moved at all, if it may not.

        Three ways a thing can be beyond the reach of a price. It is dangerous
        enough that no detour is worse than disturbing it — the surcharge alone
        left the search willing to bring a building down for want of another
        route, because a finite number is always worth paying when the
        alternative is failure. It needs more push than the robot has. Or it is
        narrow enough that pushing it at the height the robot pushes would put
        it on its side instead of moving it along: mass drops out of that
        comparison, so it is a question about shape, and the answer does not
        depend on what the thing is made of.
        """
        cfg = self.cfg
        if self.risk is not None:
            level = self.risk.level_of(oid, self.belief.partners_of(oid))
            if self.risk.forbids(level):
                return f"{level} risk is not something to be priced"
        force = self.belief.get_difficulty(oid, self.est)
        if cfg.robot_max_push_force > 0.0 and force > cfg.robot_max_push_force:
            return (f"needs {force:,.0f} N, the robot has "
                    f"{cfg.robot_max_push_force:,.0f} N")
        width = min(obs.l, obs.d)
        if (cfg.robot_push_height > 0.0 and cfg.push_friction_mu > 0.0
                and cfg.robot_push_height > width / (2.0 * cfg.push_friction_mu)):
            return (f"{width:,.2f} m wide: a push at {cfg.robot_push_height:,.2f} m "
                    "would tip it over")
        return ""

    def _removal(self, oid: int, key: EdgeKey, moved_ahead=None):
        """Compute work and drop pose needed to move an obstacle aside, also plan its SE(2) route

        `moved_ahead` is what the plan has already decided to do with the other
        obstacles on this edge — where they will be by the time this one is
        moved, which is not where they are now.
        """
        obs = self.belief.obstacle(oid)
        moved_ahead = moved_ahead or {}
        estimated_diff = self.belief.get_difficulty(oid, self.est)
        # What the executor refuses is a move of this body from this pose along
        # this edge, regardless of what it is believed to weigh.
        fail_key = (move_signature(obs), key)
        # What may be *reused*, though, is only work costed against the same
        # knowledge: the same arrangement of everything else (`belief.version`),
        # the same idea of how hard this thing is to shift, and the same risk
        # verdict. Touching an obstacle changes the second without moving
        # anything, which is exactly the case the pose-only key used to miss —
        # the search went on believing the estimate it had already been shown
        # was wrong.
        cache_key = (fail_key, self.belief.version,
                     round(estimated_diff, 6), self._risk_to_charge(oid),
                     tuple(sorted(moved_ahead.items())))
        # A move the executor has since given up on comes first: what is cached
        # is what looked possible before it was tried, and re-serving that would
        # have the planner keep proposing the move that just failed.
        if fail_key in self.failed_moves:
            res = (False, math.inf, None, 0.0, None, None)
            self._persistent_removal_cache[cache_key] = res
            return res
        # Some things are not moved for any price. Asked before the route is
        # planned, because there is no route worth planning.
        off_limits = self._off_limits(oid, obs)
        if off_limits:
            self.cfg.log(f"[refuse] oid={oid} {off_limits}")
            res = (False, math.inf, None, 0.0, None, None)
            self._persistent_removal_cache[cache_key] = res
            return res
        if cache_key in self._persistent_removal_cache:
            return self._persistent_removal_cache[cache_key]
        # Just this edge. Asking a move to clear the whole local cluster of
        # edges the obstacle lies across — so that one shove deals with the next
        # edge along too, instead of the robot walking back to shove it again —
        # sounds like the fix for that walking back, and measured on
        # strategy_demo it is not: J 40,961 -> 41,975 and 54 s -> 346 s, because
        # a corridor union that large leaves the body almost nowhere to go and
        # the search spends its time discovering that.
        clear_polys = [self.roadmap.edge_corridor[key]]
        others = self.belief.others_union(oid, moved_ahead)

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
            goal_accept=self._goal_filter(obs), goal_rank=self._goal_rank(obs),
            path_accept=path_accept)
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
        """Reject drop poses that undo the last move of this obstacle.

        The one thing that has to be a veto rather than a price. Without it the
        cheapest way to clear the edge in front of the robot is to shove the
        obstacle back across the edge behind it, and the cheapest way to clear
        *that* is to shove it back again: two moves that each look like progress
        and together are a robot pushing a crate up and down a corridor.
        """
        last_dir = self.belief.move_dir.get(obs.oid)

        def accept(goal):
            if last_dir is None:
                return True
            dx, dy = goal[0] - obs.x, goal[1] - obs.y
            if math.hypot(dx, dy) <= _MOVE_DIR_EPS:     # pure rotation has no direction
                return True
            return dx * last_dir[0] + dy * last_dir[1] >= 0.0

        return accept

    def _goal_rank(self, obs):
        """How much worse each drop pose leaves the rest of the map.

        Blocking edges the obstacle does not block now is a real cost — the next
        route has to go round them — but it is a cost, not a prohibition, and it
        used to be written as one. As a veto it is close to unusable: a body
        plugging a doorway sits half inside a wall, where there are no roadmap
        edges to block, so *every* pose it could legally be pushed to blocks more
        than where it stands, and the search would sooner report the door
        impassable than open it. Priced instead, in metres of obstacle travel per
        extra edge, it does what it was meant to do: among the poses that clear
        the way, prefer the one that leaves the map most open.
        """
        per_edge = self.cfg.manip_blocked_edge_penalty_m
        if per_edge <= 0.0:
            return None
        base = self.roadmap.count_blocked_edges(obs.polygon)

        def penalty(goal):
            extra = self.roadmap.count_blocked_edges(obs.polygon_at(*goal)) - base
            return max(0, extra) * per_edge

        return penalty

    def _llm_bias(self, acts) -> float:
        if not self.cfg.use_llm_ordering or not acts:
            return 0.0
        s = 0.0
        for act in acts:
            if act["type"] == "remove":
                s += self.est.estimate(self.belief.obstacle(act["oid"]).observation())
        return s
