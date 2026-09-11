"""Generate a seed corpus and optionally compare planner strategies."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from executor import OnlineNAMO
from scenario_generation import RandomScenarioRequest, ScenarioGenerator
from scenario_generation.naming import experiment_id, map_stem, run_stem
from scenario_generation.serialization import candidate_from_manifest, save_manifest
import viz


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", 1))
        if stop < start:
            raise ValueError("seed range stop must be >= start")
        return list(range(start, stop + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def _decision_metrics(scenario, result) -> dict:
    moved = set(result.removed)
    decisions = correct = forbidden = 0
    regret = 0.0
    for point in scenario.decision_points:
        oracle = point.metadata.get("oracle", {})
        if not oracle or oracle.get("qualitative"):
            continue
        costs = oracle.get("option_costs", {})
        if point.kind == "risk_or_distance":
            if point.metadata.get("risky") in moved:
                actual = "risky_move"
            elif point.metadata.get("safer_alternative") in moved:
                actual = "safe_move"
            else:
                actual = "detour"
        elif point.kind == "wait_or_replan":
            actual = "wait" if result.wait_time > 0.0 else "replan"
        else:
            actual = ("move" if any(oid in moved for oid in point.involved_oids)
                      else "detour")
        decisions += 1
        correct += int(actual == oracle.get("best_action"))
        actual_cost = costs.get(actual)
        if not isinstance(actual_cost, (int, float)):
            forbidden += 1
            continue
        regret += max(0.0, float(actual_cost) - float(oracle["best_cost"]))
    return {
        "oracle_decisions": decisions,
        "oracle_correct": correct,
        "oracle_accuracy": round(correct / decisions, 4) if decisions else None,
        "oracle_regret": round(regret, 4),
        "oracle_forbidden_choices": forbidden,
    }


def _run(manifest: dict, strategy: str, image_path: Path | None = None) -> dict:
    scenario = candidate_from_manifest(manifest)
    scenario.cfg.strategy = strategy
    scenario.cfg.save_frames = False
    scenario.cfg.verbose = False
    sim = OnlineNAMO(
        scenario.workspace, scenario.static, scenario.movable,
        scenario.start, scenario.goal, scenario.cfg,
        events=scenario.events,
        decision_points=[p.runtime_record() for p in scenario.decision_points])
    result = sim.run()
    if image_path is not None:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        original_poses = {obs.oid: obs.polygon for obs in scenario.movable}
        viz.visualize(sim, result, original_poses, str(image_path))
    metrics = {
        "strategy": strategy, "success": result.success,
        "C": result.C, "J": result.J, "risk_cost": result.risk_cost,
        "T": result.T, "wait_time": result.wait_time,
        "cycles": result.cycles, "expansions": result.total_expansions,
        "moved": len(result.removed), "message": result.message,
    }
    metrics.update(_decision_metrics(scenario, result))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=None,
                        help="inclusive range (0:99) or comma-separated seeds; "
                        "by default uses random_map_experiment_count seeds")
    parser.add_argument("--profile", default="benchmark")
    parser.add_argument("--out", default="img/random_benchmark")
    parser.add_argument("--run", action="store_true",
                        help="also run no-llm and shortest baselines")
    args = parser.parse_args()

    generation_cfg = Config()
    seeds = (parse_seeds(args.seeds) if args.seeds is not None
             else list(range(generation_cfg.random_map_experiment_count)))
    out = Path(args.out)
    maps_dir = out / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    generator = ScenarioGenerator()
    rows = []
    topology_signatures = set()
    topology_families = Counter()
    decision_kinds = Counter()
    event_templates = Counter()
    oracle_actions = Counter()
    oracle_margins = []
    total = len(seeds)
    for index, seed in enumerate(seeds, 1):
        print(f"[{index}/{total}] generating seed={seed}...", flush=True)
        generated = generator.generate(RandomScenarioRequest(
            seed=seed, profile=args.profile,
            obstacle_count=generation_cfg.random_map_obstacle_count,
            event_count=generation_cfg.random_map_dynamic_obstacle_count))
        save_manifest(generated.manifest,
                      maps_dir / f"{map_stem(generated.manifest)}.json")
        base = {
            "seed": seed, "profile": args.profile,
            "fingerprint": generated.fingerprint,
            "attempts": generated.attempts,
            **generated.validation.metrics,
        }
        metrics = generated.validation.metrics
        topology_families[metrics.get("topology_family", "unknown")] += 1
        topology_signatures.add((
            metrics.get("topology_family"), metrics.get("room_count"),
            metrics.get("graph_edge_count"), metrics.get("cycle_rank"),
            metrics.get("dead_end_count"), metrics.get("junction_count"),
            metrics.get("shortest_hops")))
        decision_kinds.update(p.kind for p in generated.scenario.decision_points)
        event_templates.update(event.name for event in generated.scenario.events)
        for point in generated.scenario.decision_points:
            oracle = point.metadata.get("oracle", {})
            if oracle:
                oracle_actions[oracle.get("best_action", "unknown")] += 1
            if isinstance(oracle.get("relative_margin"), (int, float)):
                oracle_margins.append(float(oracle["relative_margin"]))
        if args.run:
            print(f"[{index}/{total}] running planners...", flush=True)
            for strategy in ("no-llm", "shortest"):
                image_path = None
                if generation_cfg.random_map_generate_images:
                    image_path = out / "images" / (
                        f"{run_stem(generated.manifest, strategy)}_summary.png")
                rows.append({**base, **_run(
                    generated.manifest, strategy, image_path=image_path)})
        else:
            rows.append(base)
        print(f"experiment={experiment_id(seed)} map={generated.fingerprint} "
              f"attempts={generated.attempts} done ({index}/{total})",
              flush=True)

    (out / "results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (out / "results.csv").write_text(buffer.getvalue(), encoding="utf-8")
    coverage = {
        "seed_count": len(seeds),
        "unique_map_count": len({row["fingerprint"] for row in rows}),
        "unique_topology_signatures": len(topology_signatures),
        "topology_families": dict(sorted(topology_families.items())),
        "decision_kinds": dict(sorted(decision_kinds.items())),
        "event_templates": dict(sorted(event_templates.items())),
        "oracle_actions": dict(sorted(oracle_actions.items())),
        "oracle_margin": {
            "min": round(min(oracle_margins), 4) if oracle_margins else None,
            "max": round(max(oracle_margins), 4) if oracle_margins else None,
            "mean": (round(sum(oracle_margins) / len(oracle_margins), 4)
                     if oracle_margins else None),
        },
    }
    (out / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
