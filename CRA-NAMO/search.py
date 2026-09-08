"""在增强路线图上运行分支定界最佳优先搜索。"""

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
    actions: List[dict]                  # 已排序的搬移和清除动作。
    expansions: int


_MOVE_DIR_EPS = 1e-3    # 过小的位移不定义方向。


def move_signature(obs) -> tuple:
    return (obs.oid, round(obs.x, 3), round(obs.y, 3), round(obs.theta, 4))


class FailedMoves:
    """跟踪失败的搬移，直到世界状态发生变化。"""

    def __init__(self):
        self._at: Dict[tuple, int] = {}

    def add(self, key: tuple, version: int = 0):
        self._at[key] = version

    def drop_stale(self, version: int) -> int:
        """忘记世界版本 version 之前记录的拒绝结果。"""
        stale = [k for k, v in self._at.items() if v < version]
        for k in stale:
            del self._at[k]
        return len(stale)

    def __contains__(self, key) -> bool:
        return key in self._at

    def __len__(self) -> int:
        return len(self._at)


class Planner:
    """根据当前 belief 规划机器人运动和障碍物清除动作。"""

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
        # 等待预算按阻挡物集合记录，而不是按单条边记录。
        self.wait_budget: Dict[tuple, float] = (
            {} if wait_budget is None else wait_budget)
        self._persistent_removal_cache: Dict[tuple, tuple] = {}

    def plan(self, start_node: int, goal_node: int,
             start_heading: Optional[float] = None) -> Optional[Plan]:
        rm = self.roadmap
        cfg = self.cfg
        gx, gy = rm.nodes[goal_node]

        # 只有目标函数计入时间时才跟踪航向。
        track = cfg.time_importance > 0.0
        profile = cfg.free_profile()

        def turn_cost(heading: Optional[float], course: float) -> float:
            if heading is None:
                return 0.0
            return cost.combine(cfg, 0.0, profile.rotate_time(
                kinematics.turn_between(heading, course)))

        # 新观测会使缓存的清除计划失效。
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
            if f > incumbent + 1e-9:
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
        """删除所有缓存搬移，因为其代价对应的世界已不存在。"""
        self._persistent_removal_cache.clear()

    def _edge_cost(self, key: EdgeKey) -> Tuple[float, list]:
        """返回边的目标代价及所需的准备动作。"""
        base = cost.edge_cost(self.cfg, self.roadmap.edge_len[key])
        blockers = self.belief.blockers_of(key)
        if not blockers:
            return base, []

        wait = self._wait_option(key, blockers)
        removals = []
        extra = 0.0
        # 按序规划清除动作，保证依赖关系确定。
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
        """返回一次允许等待的代价；不允许时返回 None。"""
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
        """返回风险附加等级；若已支付则不再计入。"""
        if self.risk is None or oid in self.belief.disturbed:
            return None
        return self.risk.level_of(oid, self.belief.partners_of(oid))

    def _off_limits(self, oid: int, obs) -> str:
        """返回该障碍物因物理或语义原因被禁止搬移的理由。"""
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
        """计算清除一个障碍物所需的功、路线和放置姿态。"""
        obs = self.belief.obstacle(oid)
        moved_ahead = moved_ahead or {}
        estimated_diff = self.belief.get_difficulty(oid, self.est)
        # 失败搬移按姿态和边记录，与代价 belief 无关。
        fail_key = (move_signature(obs), key)
        # 只复用与当前 belief、难度和风险绑定的结果。
        cache_key = (fail_key, self.belief.version,
                     round(estimated_diff, 6), self._risk_to_charge(oid),
                     tuple(sorted(moved_ahead.items())))
        # 不恢复执行器已经拒绝的搬移。
        if fail_key in self.failed_moves:
            res = (False, math.inf, None, 0.0, None, None)
            self._persistent_removal_cache[cache_key] = res
            return res
        # 规划路线前先拒绝禁止搬移的障碍物。
        off_limits = self._off_limits(oid, obs)
        if off_limits:
            self.cfg.log(f"[refuse] oid={oid} {off_limits}")
            res = (False, math.inf, None, 0.0, None, None)
            self._persistent_removal_cache[cache_key] = res
            return res
        if cache_key in self._persistent_removal_cache:
            return self._persistent_removal_cache[cache_key]
        # 只规划当前评估的边；清除相邻边是独立决策。
        clear_polys = [self.roadmap.edge_corridor[key]]
        others = self.belief.others_union(oid, moved_ahead)

        move_path = None
        drop = None
        move_dist = 0.0
        feasible = False
        cplan = None

        bounds = self.roadmap.workspace.bounds
        bounds_xy = (bounds[0], bounds[2], bounds[1], bounds[3])

        # 使用边中点，使缓存代价不依赖遍历方向。
        u, v = key
        mid = tuple((a + b) / 2.0 for a, b in
                    zip(self.roadmap.nodes[u], self.roadmap.nodes[v]))
        robot_pos = mid
        exits = [(self.roadmap.nodes[u], 0.0), (self.roadmap.nodes[v], 0.0)]
        contact_memo: Dict[int, tuple] = {}

        def _contact_for(poses):
            # 缓存对同一姿态列表对象的验证结果。
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
        # 按完整搬移代价评估候选，而不是只看推动距离。
        best = math.inf
        for path, push_cost, goal in manipulation.move_se2_options(
                obs, clear_polys, self.roadmap.static_obstacles, bounds_xy,
                robot_pos, self.cfg, others_polys=others,
                goal_accept=self._goal_filter(obs),
                goal_rank=self._goal_rank(obs), path_accept=path_accept):
            floor = cost.combine(
                self.cfg, cost.manipulation_work(estimated_diff, push_cost), 0.0)
            if floor >= best:
                break
            plan = (_contact_for(path) if self.cfg.contact_required
                    else contact.idle_plan(mid, len(path)))
            if not plan.feasible:
                self.cfg.log(f"[contact] oid={oid} {plan.reason}")
                continue
            seconds = cost.manipulation_time(self.cfg, plan, len(path), push_cost)
            total = cost.removal_cost(
                self.cfg, cost.manipulation_work(estimated_diff, push_cost),
                plan.travel, seconds, self._risk_to_charge(oid))
            if total < best:
                best = total
                feasible = True
                move_path, drop, move_dist, cplan = path, goal, push_cost, plan
        if not feasible and rejected:
            # 汇总其他条件有效的障碍物路径被拒绝的原因。
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
        """返回一个过滤器，防止立即反向搬移。"""
        last_dir = self.belief.move_dir.get(obs.oid)

        def accept(goal):
            if last_dir is None:
                return True
            dx, dy = goal[0] - obs.x, goal[1] - obs.y
            if math.hypot(dx, dy) <= _MOVE_DIR_EPS:
                return True
            return dx * last_dir[0] + dy * last_dir[1] >= 0.0

        return accept

    def _goal_rank(self, obs):
        """返回阻挡额外路线图边的放置姿态软惩罚。"""
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
