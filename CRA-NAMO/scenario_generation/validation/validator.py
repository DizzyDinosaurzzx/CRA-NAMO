"""Fast structural validation before a map reaches the expensive planner."""

from __future__ import annotations

import math
from collections import Counter

from shapely.geometry import Point
from shapely.ops import unary_union

from dynamics import AfterMoved, MoveTo
import risk
from scenarios._realism import check_layout
from scenario_generation.models import ValidationIssue, ValidationReport


def _connected(candidate, polygons) -> bool:
    blocked = unary_union(list(polygons))
    free = candidate.workspace.difference(blocked).buffer(
        -candidate.cfg.robot_radius)
    return any(
        geom.covers(Point(candidate.start)) and geom.covers(Point(candidate.goal))
        for geom in (getattr(free, "geoms", None) or [free]))


class ScenarioValidator:
    def validate(self, candidate, *, min_decisions: int) -> ValidationReport:
        issues: list[ValidationIssue] = []
        try:
            check_layout(
                "seeded_random", workspace=candidate.workspace,
                static=candidate.static, movable=candidate.movable,
                start=candidate.start, goal=candidate.goal, cfg=candidate.cfg)
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue("geometry", "invalid_layout", str(exc)))
            return ValidationReport(False, issues)

        oids = {obs.oid for obs in candidate.movable}
        if len(candidate.decision_points) < min_decisions:
            issues.append(ValidationIssue(
                "decision", "too_few_decisions",
                f"expected at least {min_decisions}, got {len(candidate.decision_points)}"))

        for point in candidate.decision_points:
            missing = set(point.involved_oids) - oids
            if missing:
                issues.append(ValidationIssue(
                    "decision", "unknown_oid",
                    f"{point.name} references missing obstacles {sorted(missing)}"))
            if len(point.alternatives) < 2:
                issues.append(ValidationIssue(
                    "decision", "no_tradeoff",
                    f"{point.name} has fewer than two alternatives"))

        oracle_margins = []
        oracle_actions = Counter()
        belief_flip_count = 0
        for index, point in enumerate(candidate.decision_points):
            oracle = point.metadata.get("oracle")
            if not oracle:
                issues.append(ValidationIssue(
                    "calibration", "missing_oracle",
                    f"{point.name} has no calibrated oracle label"))
                continue
            best_action = oracle.get("best_action", "unknown")
            oracle_actions[best_action] += 1
            if oracle.get("qualitative"):
                continue
            margin = oracle.get("relative_margin")
            costs = [value for value in oracle.get("option_costs", {}).values()
                     if isinstance(value, (int, float)) and math.isfinite(value)]
            if not isinstance(margin, (int, float)) or len(costs) < 2:
                issues.append(ValidationIssue(
                    "calibration", "invalid_oracle",
                    f"{point.name} oracle needs a margin and two finite options"))
                continue
            if not 0.05 - 1e-6 <= margin <= 0.30 + 1e-6:
                issues.append(ValidationIssue(
                    "calibration", "uncalibrated_margin",
                    f"{point.name} relative margin {margin:.4f} is outside [0.05, 0.30]"))
            oracle_margins.append(float(margin))
            belief_action = oracle.get("belief_action_before_contact")
            if point.kind == "hidden_difficulty" and not belief_action:
                issues.append(ValidationIssue(
                    "calibration", "missing_belief_oracle",
                    f"{point.name} has no pre-contact belief label"))
            if belief_action and belief_action != best_action:
                belief_flip_count += 1

        if sum(1 for point in candidate.decision_points
               if not point.metadata.get("oracle", {}).get("qualitative")) < 3:
            issues.append(ValidationIssue(
                "calibration", "too_few_quantitative_oracles",
                "expected at least three quantitative decision labels"))

        by_oid = {obs.oid: obs for obs in candidate.movable}
        for event in candidate.events:
            effect_oid = getattr(event.effect, "oid", None)
            trigger_oid = (event.trigger.oid
                           if isinstance(event.trigger, AfterMoved) else None)
            for role, oid in (("effect", effect_oid), ("trigger", trigger_oid)):
                if oid is not None and oid not in oids:
                    issues.append(ValidationIssue(
                        "dynamics", "unknown_oid",
                        f"event {event.name!r} {role} references missing oid {oid}"))
            if isinstance(event.effect, MoveTo) and effect_oid in by_oid:
                obs = by_oid[effect_oid]
                target = obs.polygon_at(*event.effect.goal)
                if not candidate.workspace.covers(target):
                    issues.append(ValidationIssue(
                        "dynamics", "target_outside",
                        f"event {event.name!r} target is outside the workspace"))
                if any(target.intersection(w.polygon).area > 1e-7
                       for w in candidate.static):
                    issues.append(ValidationIssue(
                        "dynamics", "target_hits_wall",
                        f"event {event.name!r} target intersects a wall"))
                future_blockers = [w.polygon for w in candidate.static]
                future_blockers += [
                    obs.polygon for obs in candidate.movable
                    if obs.oid != effect_oid
                ]
                future_blockers.append(target)
                if not _connected(candidate, future_blockers):
                    issues.append(ValidationIssue(
                        "dynamics", "target_closes_all_routes",
                        f"event {event.name!r} removes every fallback route"))

        # Static reachability is a cheap necessary condition.
        reachable = _connected(candidate, [w.polygon for w in candidate.static])
        if not reachable:
            issues.append(ValidationIssue(
                "reachability", "static_disconnect",
                "start and goal are disconnected even before movable blockers"))

        # Every generated map retains a deliberately long emergency route. This
        # makes the corpus robustly solvable while the shorter routes still
        # exercise move/risk/wait decisions.
        all_blocked = [
            *[w.polygon for w in candidate.static],
            *[obs.polygon for obs in candidate.movable],
        ]
        movable_reachable = _connected(candidate, all_blocked)
        if not movable_reachable:
            issues.append(ValidationIssue(
                "reachability", "emergency_route_blocked",
                "no obstacle-free fallback route remains"))

        topology_metrics = candidate.metadata.get("topology_metrics", {})
        if topology_metrics:
            detours = topology_metrics.get("decision_detour_hops", ())
            if len(detours) < min_decisions or any(value < 0 for value in detours):
                issues.append(ValidationIssue(
                    "decision", "missing_counterfactual_detour",
                    "a selected graph decision edge has no counterfactual route"))
            if topology_metrics.get("cycle_rank", 0) < 1:
                issues.append(ValidationIssue(
                    "topology", "no_cycle",
                    "room graph contains no route cycle"))
            if topology_metrics.get("junction_count", 0) < 1:
                issues.append(ValidationIssue(
                    "topology", "no_junction",
                    "room graph contains no branching junction"))

        metrics = {
            "wall_count": len(candidate.static),
            "obstacle_count": len(candidate.movable),
            "dynamic_obstacle_count": len({
                event.effect.oid for event in candidate.events
                if isinstance(event.effect, MoveTo)
            }),
            "event_count": len(candidate.events),
            "decision_count": len(candidate.decision_points),
            "decision_kind_count": len({p.kind for p in candidate.decision_points}),
            "static_reachable": int(reachable),
            "fallback_reachable": int(movable_reachable),
            "difficulty_span": round(
                max(obs.difficulty for obs in candidate.movable)
                / min(obs.difficulty for obs in candidate.movable), 3),
            "highest_risk": max(
                (risk.LEVELS.index(risk.keyword_level(obs.material))
                 for obs in candidate.movable), default=0),
            "calibrated_decision_count": len(oracle_margins),
            "oracle_margin_min": round(min(oracle_margins), 4)
            if oracle_margins else None,
            "oracle_margin_max": round(max(oracle_margins), 4)
            if oracle_margins else None,
            "oracle_margin_mean": round(
                sum(oracle_margins) / len(oracle_margins), 4)
            if oracle_margins else None,
            "oracle_action_counts": dict(sorted(oracle_actions.items())),
            "belief_flip_count": belief_flip_count,
        }
        metrics.update(topology_metrics)
        metrics["topology_family"] = candidate.metadata.get(
            "topology_family", candidate.metadata.get("topology", "unknown"))
        return ValidationReport(not any(issue.fatal for issue in issues),
                                issues, metrics)
