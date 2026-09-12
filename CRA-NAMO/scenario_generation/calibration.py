"""Calibrate decision difficulty after topology generation.

Only physical ground truth is adjusted. The topology stays fixed, so the same
seed still describes the same layout while its choices remain informative.

Two costs are tracked for every gate. The *truth* is what the world will
charge: the push work plus, where the label names something that should not be
disturbed, the risk surcharge that lands once contact reveals it. The *belief*
is what an arm reading the offline tables would compute from the visible label
alone. Calibration tunes the truth into a narrow band around the detour, so the
decision is genuinely close; the gap between truth and belief is what a
semantic reader is being tested on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cost
import risk as risk_model
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


def _believed_difficulty(obstacle) -> float:
    """The push force an arm reading only the offline tables would expect."""
    return friction_force(material_mu_rho(obstacle.material), obstacle.volume)


def _attach_oracle(point, option_costs: dict[str, float], *, stage: str,
                   extra=None) -> float:
    finite = sorted((cost_value, action)
                    for action, cost_value in option_costs.items()
                    if math.isfinite(cost_value))
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
            action: (None if not math.isfinite(value) else round(value, 4))
            for action, value in option_costs.items()
        },
    }
    if extra:
        oracle.update(extra)
    point.metadata["oracle"] = oracle
    return margin


def _attach_qualitative(point, *, stage: str, best_action: str, extra=None):
    """Label a decision whose answer is not a close call.

    A cylinder bank behind a door is not a 10 percent question: avoiding it is
    worth hundreds of metres. Such a gate still belongs on the map, because the
    offline tables cannot see it at all, but it is not asked to sit inside the
    calibrated margin band.
    """
    oracle = {"stage": stage, "best_action": best_action, "qualitative": True}
    if extra:
        oracle.update(extra)
    point.metadata["oracle"] = oracle


def _belief_extra(cfg, obstacle, push_distance: float,
                  detour_cost: float) -> dict:
    """What the offline tables would have chosen, and at what price."""
    believed = _believed_difficulty(obstacle)
    believed_costs = {
        "move": _push_objective(cfg, believed, push_distance),
        "detour": detour_cost,
    }
    return {
        "belief_action_before_contact": min(believed_costs,
                                            key=believed_costs.get),
        "belief_option_costs": {key: round(value, 4)
                                for key, value in believed_costs.items()},
        "believed_difficulty_n": round(believed, 4),
    }


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

        for point in candidate.decision_points:
            gate_index = point.metadata.get("gate_index")
            if gate_index is None or gate_index >= len(gates):
                # Dynamic decisions own an event, not a doorway.
                margin = self._calibrate_dynamic(candidate, point, rng,
                                                 mean_edge)
                actions.append(point.metadata["oracle"]["best_action"])
                if margin is not None:
                    quantitative += 1
                    margins.append(margin)
                continue

            gate = gates[gate_index]
            bypass = gate.get("bypass")
            push_distance = max(0.55, gate["wall_thickness"] + 0.35)
            target_margin = rng.uniform(0.08, 0.24)
            graph_detour = (float(graph_detours[gate_index])
                            if gate_index < len(graph_detours) else mean_edge)
            # Price the alternative the robot will actually take. A gate with
            # a second, unblocked doorway is dodged by walking a few metres;
            # a gate whose wall has one doorway, or whose doorways are both
            # blocked, is dodged only by routing through another room.
            walk_around = (point.kind not in ("risk_or_distance",
                                              "risk_or_risk")
                           and bypass is not None)
            detour_distance = (max(1.0, 2.0 * math.dist(
                gate["direct"][:2], bypass[:2]) + 0.4) if walk_around
                else max(graph_detour, mean_edge))

            margin = self._calibrate_gate(
                candidate, point, by_oid, rng,
                push_distance=push_distance,
                detour_distance=detour_distance,
                target_margin=target_margin)
            actions.append(point.metadata["oracle"]["best_action"])
            if margin is not None:
                quantitative += 1
                margins.append(margin)

        return CalibrationReport(
            calibrated=len(candidate.decision_points), quantitative=quantitative,
            margins=tuple(round(value, 4) for value in margins),
            oracle_actions=tuple(actions))

    # ----------------------------------------------------------------------

    def _calibrate_gate(self, candidate, point, by_oid, rng, *,
                        push_distance: float, detour_distance: float,
                        target_margin: float):
        cfg = candidate.cfg
        kind = point.kind

        if kind == "risk_or_distance":
            safe = by_oid[point.metadata["safer_alternative"]]
            detour_cost = _travel_objective(cfg, detour_distance)
            safe_ratio = ((1.0 - target_margin) if rng.random() < 0.65
                          else (1.0 + target_margin))
            safe.difficulty = _difficulty_for_cost(
                cfg, detour_cost * safe_ratio, push_distance)
            costs = {
                "risky_move": math.inf,
                "safe_move": _push_objective(cfg, safe.difficulty,
                                             push_distance),
                "detour": detour_cost,
            }
            return _attach_oracle(point, costs, stage="visible")

        if kind == "risk_or_risk":
            return self._calibrate_risk_or_risk(
                candidate, point, by_oid, rng, push_distance=push_distance,
                detour_distance=detour_distance, target_margin=target_margin)

        if kind == "false_alarm":
            return self._calibrate_false_alarm(
                candidate, point, by_oid, push_distance=push_distance,
                detour_distance=detour_distance, target_margin=target_margin)

        if kind == "blind_risk":
            return self._calibrate_blind_risk(
                candidate, point, by_oid, push_distance=push_distance,
                detour_distance=detour_distance, target_margin=target_margin)

        obstacle = by_oid[point.involved_oids[0]]
        detour_cost = _travel_objective(cfg, detour_distance)
        if kind in ("hidden_difficulty", "blind_weight"):
            # The truth must make the detour the better call, so that an arm
            # which trusts the light-sounding label is measurably wrong.
            ratio = 1.0 + target_margin
        else:
            ratio = ((1.0 - target_margin) if rng.random() < 0.55
                     else (1.0 + target_margin))
        obstacle.difficulty = _difficulty_for_cost(
            cfg, detour_cost * ratio, push_distance)
        costs = {
            "move": _push_objective(cfg, obstacle.difficulty, push_distance),
            "detour": detour_cost,
        }
        extra = None
        if kind in ("hidden_difficulty", "blind_weight"):
            extra = _belief_extra(cfg, obstacle, push_distance, detour_cost)
        stage = "contact" if kind != "move_or_detour" else "visible"
        return _attach_oracle(point, costs, stage=stage, extra=extra)

    def _calibrate_blind_risk(self, candidate, point, by_oid, *,
                              push_distance: float, detour_distance: float,
                              target_margin: float):
        """Price the push so only the risk surcharge makes it the wrong call."""
        cfg = candidate.cfg
        obstacle = by_oid[point.involved_oids[0]]
        detour_cost = _travel_objective(cfg, detour_distance)
        surcharge = cost.risk_cost(cfg, point.metadata.get("true_risk"))
        target_total = detour_cost * (1.0 + target_margin)

        if surcharge >= target_total - 1e-9:
            # Avoiding it is worth far more than the detour; the decision is
            # not close, it is simply invisible to a lexical reader.
            obstacle.difficulty = _difficulty_for_cost(
                cfg, detour_cost * (1.0 - target_margin), push_distance)
            believed = _belief_extra(cfg, obstacle, push_distance, detour_cost)
            _attach_qualitative(
                point, stage="visible", best_action="detour",
                extra={**believed,
                       "risk_surcharge": round(surcharge, 4),
                       "option_costs": {
                           "move": round(_push_objective(
                               cfg, obstacle.difficulty, push_distance)
                               + surcharge, 4),
                           "detour": round(detour_cost, 4)}})
            return None

        obstacle.difficulty = _difficulty_for_cost(
            cfg, target_total - surcharge, push_distance)
        push_cost = _push_objective(cfg, obstacle.difficulty, push_distance)
        costs = {"move": push_cost + surcharge, "detour": detour_cost}
        extra = _belief_extra(cfg, obstacle, push_distance, detour_cost)
        extra["risk_surcharge"] = round(surcharge, 4)
        # Without the surcharge the push is the cheaper option, which is
        # exactly what a keyword-blind arm will compute.
        extra["belief_action_before_contact"] = (
            "move" if push_cost < detour_cost else "detour")
        return _attach_oracle(point, costs, stage="visible", extra=extra)

    def _calibrate_false_alarm(self, candidate, point, by_oid, *,
                               push_distance: float, detour_distance: float,
                               target_margin: float):
        """Moving it is right; only the phantom surcharge says otherwise."""
        cfg = candidate.cfg
        obstacle = by_oid[point.involved_oids[0]]
        detour_cost = _travel_objective(cfg, detour_distance)
        obstacle.difficulty = _difficulty_for_cost(
            cfg, detour_cost * (1.0 - target_margin), push_distance)
        push_cost = _push_objective(cfg, obstacle.difficulty, push_distance)
        costs = {"move": push_cost, "detour": detour_cost}

        # What an arm reading the keyword table computes: a surcharge that the
        # object does not actually carry.
        phantom = cost.risk_cost(cfg, risk_model.keyword_level(obstacle.material))
        believed_push = _push_objective(
            cfg, _believed_difficulty(obstacle), push_distance) + phantom
        extra = {
            "phantom_risk_surcharge": round(phantom, 4),
            "belief_option_costs": {"move": round(believed_push, 4),
                                    "detour": round(detour_cost, 4)},
            "belief_action_before_contact": (
                "move" if believed_push < detour_cost else "detour"),
        }
        return _attach_oracle(point, costs, stage="visible", extra=extra)

    def _calibrate_risk_or_risk(self, candidate, point, by_oid, rng, *,
                                push_distance: float, detour_distance: float,
                                target_margin: float):
        """Two blocked doors: the lexically obvious hazard is the safer one."""
        cfg = candidate.cfg
        blind = by_oid[point.metadata["risky"]]
        known = by_oid[point.metadata["safer_alternative"]]
        detour_cost = _travel_objective(cfg, detour_distance)
        blind_surcharge = cost.risk_cost(cfg, point.metadata.get("true_risk"))
        known_surcharge = cost.risk_cost(
            cfg, risk_model.keyword_level(known.material))

        # The known hazard is tuned to win outright once both surcharges are
        # counted, so an arm that reads both labels has somewhere to go.
        known.difficulty = _difficulty_for_cost(
            cfg, max(1.0, detour_cost * (1.0 - target_margin) - known_surcharge),
            push_distance)
        blind.difficulty = _difficulty_for_cost(
            cfg, detour_cost * rng.uniform(0.35, 0.6), push_distance)

        blind_cost = _push_objective(cfg, blind.difficulty, push_distance)
        known_cost = _push_objective(cfg, known.difficulty, push_distance)
        costs = {
            "move_blind_hazard": blind_cost + blind_surcharge,
            "move_known_hazard": known_cost + known_surcharge,
            "detour": detour_cost,
        }
        extra = {
            "risk_surcharge": round(blind_surcharge, 4),
            "known_risk_surcharge": round(known_surcharge, 4),
            # A lexical reader sees no surcharge on the blind door at all.
            "belief_option_costs": {
                "move_blind_hazard": round(blind_cost, 4),
                "move_known_hazard": round(known_cost + known_surcharge, 4),
                "detour": round(detour_cost, 4)},
        }
        believed = {"move_blind_hazard": blind_cost,
                    "move_known_hazard": known_cost + known_surcharge,
                    "detour": detour_cost}
        extra["belief_action_before_contact"] = min(believed,
                                                    key=believed.get)
        if blind_surcharge >= detour_cost:
            _attach_qualitative(point, stage="visible",
                                best_action=min(
                                    ("move_known_hazard", "detour"),
                                    key=lambda action: costs[action]),
                                extra={**extra, "option_costs": {
                                    key: round(value, 4)
                                    for key, value in costs.items()}})
            return None
        return _attach_oracle(point, costs, stage="visible", extra=extra)

    def _calibrate_dynamic(self, candidate, point, rng, mean_edge: float):
        """Tune the wait window of a temporary closure, when there is one."""
        timed = sorted(
            (event for event in candidate.events
             if isinstance(event.trigger, AtTime)),
            key=lambda event: event.trigger.t)
        if (point.kind == "wait_or_replan" and len(timed) >= 2
                and candidate.cfg.time_importance > 0.0):
            replan_distance = max(3.0, 2.0 * mean_edge)
            replan_cost = _travel_objective(candidate.cfg, replan_distance)
            target_margin = rng.uniform(0.08, 0.24)
            wait_ratio = ((1.0 - target_margin) if rng.random() < 0.5
                          else (1.0 + target_margin))
            wait_cost = replan_cost * wait_ratio
            wait_seconds = wait_cost / (candidate.cfg.time_importance
                                        * candidate.cfg.time_value)
            timed[-1].trigger.t = timed[0].trigger.t + wait_seconds
            costs = {"wait": wait_cost, "replan": replan_cost}
            return _attach_oracle(
                point, costs, stage="dynamic",
                extra={"wait_window_s": round(wait_seconds, 4),
                       "replan_distance_m": round(replan_distance, 4)})
        _attach_qualitative(point, stage="dynamic",
                            best_action="replan_on_change")
        return None
