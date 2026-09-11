"""Calibrate decision difficulty after topology generation.

Only physical ground truth is adjusted. The topology stays fixed, so the same
seed still describes the same layout while its choices remain informative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dynamics import AtTime
from llm_difficulty import friction_force, material_mu_rho


@dataclass(frozen=True)
class CalibrationReport:
    calibrated: int
    quantitative: int
    margins: tuple[float, ...]
    oracle_actions: tuple[str, ...]


def _travel_objective(cfg, distance: float) -> float:
    distance = max(0.0, float(distance))
    seconds = distance / max(cfg.robot_v_max, 1e-6)
    return ((1.0 - cfg.time_importance) * cfg.lambda_distance * distance
            + cfg.time_importance * cfg.time_value * seconds)


def _push_objective(cfg, difficulty: float, distance: float) -> float:
    distance = max(0.05, float(distance))
    seconds = distance / max(cfg.robot_v_max_loaded, 1e-6)
    return ((1.0 - cfg.time_importance) * difficulty * distance
            + cfg.time_importance * cfg.time_value * seconds)


def _difficulty_for_cost(cfg, target_cost: float, push_distance: float) -> float:
    time_cost = (cfg.time_importance * cfg.time_value
                 * push_distance / max(cfg.robot_v_max_loaded, 1e-6))
    denominator = (1.0 - cfg.time_importance) * push_distance
    if denominator <= 1e-9:
        return 50.0
    # Keep the force physically feasible for the configured robot while still
    # allowing long graph detours to be matched without flattening the choice.
    return min(20_000.0, max(5.0, (target_cost - time_cost) / denominator))


def _attach_oracle(point, option_costs: dict[str, float], *, stage: str,
                   extra=None) -> float:
    finite = sorted((cost, action) for action, cost in option_costs.items()
                    if math.isfinite(cost))
    best_cost, best_action = finite[0]
    second_cost = finite[1][0] if len(finite) > 1 else best_cost
    margin = ((second_cost - best_cost) / max(second_cost, 1e-9)
              if second_cost > best_cost else 0.0)
    oracle = {
        "stage": stage,
        "best_action": best_action,
        "best_cost": round(best_cost, 4),
        "second_cost": round(second_cost, 4),
        "relative_margin": round(margin, 4),
        "target_margin": [0.05, 0.30],
        "option_costs": {
            action: (None if not math.isfinite(cost) else round(cost, 4))
            for action, cost in option_costs.items()
        },
    }
    if extra:
        oracle.update(extra)
    point.metadata["oracle"] = oracle
    return margin


class DecisionCalibrator:
    """Tune local choices to a seeded 8--24 percent cost separation."""

    def calibrate(self, candidate, topology, rng) -> CalibrationReport:
        by_oid = {obs.oid: obs for obs in candidate.movable}
        gates = topology.anchors["gates"]
        graph_metrics = topology.anchors.get("topology_metrics", {})
        graph_detours = graph_metrics.get("decision_graph_detour_m", ())
        mean_edge = float(graph_metrics.get("mean_edge_length_m", 3.0))
        margins = []
        actions = []
        quantitative = 0

        for index, point in enumerate(candidate.decision_points):
            # build_gate_decisions owns exactly the first three records. Some
            # legacy topologies expose extra gate anchors for event placement,
            # but those anchors are not additional static decisions.
            if index >= 3:
                timed = sorted(
                    (event for event in candidate.events
                     if isinstance(event.trigger, AtTime)),
                    key=lambda event: event.trigger.t)
                if (point.kind == "wait_or_replan" and len(timed) >= 2
                        and candidate.cfg.time_importance > 0.0):
                    replan_distance = max(3.0, 2.0 * mean_edge)
                    replan_cost = _travel_objective(
                        candidate.cfg, replan_distance)
                    target_margin = rng.uniform(0.08, 0.24)
                    wait_ratio = ((1.0 - target_margin)
                                  if rng.random() < 0.5
                                  else (1.0 + target_margin))
                    wait_cost = replan_cost * wait_ratio
                    wait_seconds = wait_cost / (
                        candidate.cfg.time_importance
                        * candidate.cfg.time_value)
                    timed[-1].trigger.t = timed[0].trigger.t + wait_seconds
                    costs = {"wait": wait_cost, "replan": replan_cost}
                    margin = _attach_oracle(
                        point, costs, stage="dynamic",
                        extra={
                            "wait_window_s": round(wait_seconds, 4),
                            "replan_distance_m": round(replan_distance, 4),
                        })
                    quantitative += 1
                    margins.append(margin)
                    actions.append(point.metadata["oracle"]["best_action"])
                else:
                    point.metadata["oracle"] = {
                        "stage": "dynamic",
                        "best_action": "replan_on_change",
                        "qualitative": True,
                    }
                    actions.append("replan_on_change")
                continue

            gate = gates[index]
            portal_gap = math.dist(gate["direct"][:2], gate["bypass"][:2])
            local_detour = max(1.0, 2.0 * portal_gap + 0.4)
            graph_detour = (float(graph_detours[index])
                            if index < len(graph_detours) else local_detour)
            push_distance = max(0.55, gate["wall_thickness"] + 0.35)
            target_margin = rng.uniform(0.08, 0.24)

            if point.kind == "risk_or_distance":
                safe = by_oid[point.metadata["safer_alternative"]]
                detour_distance = max(local_detour, graph_detour, mean_edge)
                detour_cost = _travel_objective(candidate.cfg, detour_distance)
                safe_ratio = ((1.0 - target_margin) if rng.random() < 0.65
                              else (1.0 + target_margin))
                safe.difficulty = _difficulty_for_cost(
                    candidate.cfg, detour_cost * safe_ratio, push_distance)
                costs = {
                    "risky_move": math.inf,
                    "safe_move": _push_objective(
                        candidate.cfg, safe.difficulty, push_distance),
                    "detour": detour_cost,
                }
                margin = _attach_oracle(point, costs, stage="visible")
            else:
                obstacle = by_oid[point.involved_oids[0]]
                detour_cost = _travel_objective(candidate.cfg, local_detour)
                if point.kind == "hidden_difficulty":
                    ratio = 1.0 + target_margin
                else:
                    ratio = ((1.0 - target_margin) if rng.random() < 0.55
                             else (1.0 + target_margin))
                obstacle.difficulty = _difficulty_for_cost(
                    candidate.cfg, detour_cost * ratio, push_distance)
                costs = {
                    "move": _push_objective(
                        candidate.cfg, obstacle.difficulty, push_distance),
                    "detour": detour_cost,
                }
                extra = None
                if point.kind == "hidden_difficulty":
                    perceived = friction_force(
                        material_mu_rho(obstacle.material), obstacle.volume)
                    perceived_costs = {
                        "move": _push_objective(
                            candidate.cfg, perceived, push_distance),
                        "detour": detour_cost,
                    }
                    extra = {
                        "belief_action_before_contact": min(
                            perceived_costs, key=perceived_costs.get),
                        "belief_option_costs": {
                            key: round(value, 4)
                            for key, value in perceived_costs.items()
                        },
                    }
                margin = _attach_oracle(
                    point, costs, stage="contact", extra=extra)

            quantitative += 1
            margins.append(margin)
            actions.append(point.metadata["oracle"]["best_action"])

        return CalibrationReport(
            calibrated=len(candidate.decision_points), quantitative=quantitative,
            margins=tuple(round(value, 4) for value in margins),
            oracle_actions=tuple(actions))
