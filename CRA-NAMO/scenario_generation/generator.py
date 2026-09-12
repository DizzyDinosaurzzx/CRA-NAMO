"""Orchestrate deterministic scenario generation and rejection sampling."""

from __future__ import annotations

from shapely.geometry import LineString

from config import Config
from scenario_generation.calibration import DecisionCalibrator
from scenario_generation.decisions import build_gate_decisions
from scenario_generation.decisions.builders import (OidAllocator, doors_needed,
                                                    plan_gate_kinds)
from scenario_generation.models import (
    CandidateScenario,
    GenerationResult,
    RandomScenarioRequest,
    ScenarioGenerationError,
)
from scenario_generation.materials import get_theme
from scenario_generation.profiles import get_profile, pick_theme, weighted_choice
from scenario_generation.rng import SeedStreams
from scenario_generation.sampling import (build_additional_dynamic_event,
                                           build_dynamic_event, fill_background)
from scenario_generation.serialization import manifest_from_candidate
from scenario_generation.topology import BUILDERS
from scenario_generation.validation import ScenarioValidator


class ScenarioGenerator:
    def __init__(self, validator: ScenarioValidator | None = None):
        self.validator = validator or ScenarioValidator()
        self.calibrator = DecisionCalibrator()

    def generate(self, request: RandomScenarioRequest) -> GenerationResult:
        profile = get_profile(request.profile)
        rejected = []
        for attempt in range(request.max_generation_attempts):
            streams = SeedStreams(request.seed, attempt)
            topology_name = request.topology or weighted_choice(
                profile.topology_weights, streams.rng("topology.choice"))
            if topology_name not in BUILDERS:
                raise ValueError(f"unknown topology {topology_name!r}")
            # Each gate's kind decides how many doorways its wall needs, so
            # it has to be settled before the walls exist.
            kinds = plan_gate_kinds(
                profile.gate_decisions, streams.rng("decisions.kinds"),
                include_hidden=(streams.rng("hidden_state").random()
                                < profile.hidden_probability),
                trap_probability=profile.trap_probability)
            try:
                topology = BUILDERS[topology_name](
                    request, streams.rng("topology.geometry"), profile,
                    door_counts=doors_needed(kinds))
            except ValueError as exc:
                # A layout that cannot carry every gate is rejected like any
                # other unusable candidate; the next attempt reshapes it.
                rejected.append(f"topology {topology_name}: {exc}")
                continue
            theme = get_theme(pick_theme(
                profile, streams.rng("materials.theme")))
            allocator = OidAllocator()
            critical, decisions = build_gate_decisions(
                topology, allocator, streams.rng("decisions"), theme, kinds)

            events = []
            event_rng = streams.rng("events")
            exact_event_count = isinstance(request.event_count, int)
            event_range = ((request.event_count, request.event_count)
                           if exact_event_count else request.event_count)
            requested_events = event_rng.randint(*event_range)
            if (event_range[1] > 0
                    and profile.name in ("dynamic", "showcase")):
                requested_events = max(1, requested_events)
            if (requested_events and len(decisions) < request.max_decision_points
                    and (exact_event_count
                         or event_rng.random() < profile.dynamic_probability)):
                actors, event_batch, decision, swept_segment = build_dynamic_event(
                    topology, critical, allocator, event_rng,
                    allow_state_mutation=not exact_event_count)
                critical.extend(actors)
                events.extend(event_batch)
                decisions.append(decision)
                if swept_segment is not None:
                    topology.reserved_regions.append(
                        LineString(swept_segment).buffer(0.8))
                if exact_event_count:
                    for _ in range(requested_events - 1):
                        actor, event_batch = build_additional_dynamic_event(
                            topology, critical, allocator, event_rng)
                        if actor is None:
                            break
                        critical.append(actor)
                        events.extend(event_batch)

            exact_obstacle_count = isinstance(request.obstacle_count, int)
            lo, hi = ((request.obstacle_count, request.obstacle_count)
                      if exact_obstacle_count else request.obstacle_count)
            base_target = streams.rng("obstacles.count").randint(lo, hi)
            target = max(lo, len(critical),
                         (base_target if exact_obstacle_count else
                          round(base_target * profile.background_density)))
            background = fill_background(
                topology, critical, allocator,
                streams.rng("obstacles.background"), target, theme)

            cfg = Config(
                robot_radius=0.10,
                grid_step=0.45,
                conn_radius=0.90,
                se2_cell=0.30,
                R_perc=7.0,
                R_manip=4.5,
                time_importance=profile.time_importance,
                strategy="no-llm",
                use_llm_ordering=False,
                # Random scenarios are primarily inspected as robustness runs;
                # keep their world evolution visible unless --no-frames overrides it.
                save_frames=True,
                show_global_obstacles=True,
                deepseek_api_key="",
            )
            candidate = CandidateScenario(
                workspace=topology.workspace, static=topology.walls,
                movable=[*critical, *background], start=topology.start,
                goal=topology.goal, cfg=cfg, events=events,
                decision_points=decisions,
                metadata={
                    "seed": request.seed, "profile": profile.name,
                    "theme": theme.name,
                    "topology": topology_name, "attempt": attempt,
                    "topology_family": topology.anchors.get(
                        "topology_family", topology_name),
                    "topology_metrics": dict(topology.anchors.get(
                        "topology_metrics", {})),
                })
            calibration = self.calibrator.calibrate(
                candidate, topology, streams.rng("decision.calibration"))
            candidate.metadata["calibration"] = {
                "calibrated": calibration.calibrated,
                "quantitative": calibration.quantitative,
                "margins": list(calibration.margins),
                "oracle_actions": list(calibration.oracle_actions),
            }
            report = self.validator.validate(
                candidate, min_decisions=request.min_decision_points)
            if not report.accepted:
                rejected.append("; ".join(issue.message for issue in report.issues))
                continue

            manifest = manifest_from_candidate(
                candidate, seed=request.seed, profile=profile.name,
                attempt=attempt, validation=report.metrics)
            fingerprint = manifest["fingerprint"]
            candidate.metadata["fingerprint"] = fingerprint
            return GenerationResult(candidate, manifest, report,
                                    attempt + 1, fingerprint)

        detail = rejected[-1] if rejected else "no candidate produced"
        raise ScenarioGenerationError(
            f"seed {request.seed} failed after {request.max_generation_attempts} "
            f"attempts; last rejection: {detail}")
