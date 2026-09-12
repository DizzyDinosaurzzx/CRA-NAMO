"""One-click, resumable random-map benchmark driven entirely by config.py."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config, STRATEGIES
from executor import OnlineNAMO, RunResult
from scenario_generation import RandomScenarioRequest, ScenarioGenerator
from scenario_generation.serialization import candidate_from_manifest
from PIL import Image
import viz

PROFILE = "benchmark"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _dynamic_oids(manifest: dict) -> list[int]:
    return sorted({int(event["effect"]["oid"])
                   for event in manifest.get("events", ())
                   if event.get("effect", {}).get("type") == "move_to"})


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


def _run(manifest: dict, strategy: str, image_path: str | None,
         gif_path: str | None,
         benchmark_info: dict, planner_done_callback=None) -> dict:
    scenario = candidate_from_manifest(manifest)
    scenario.cfg.strategy = strategy
    scenario.cfg.save_frames = gif_path is not None
    scenario.cfg.verbose = False
    sim = OnlineNAMO(
        scenario.workspace, scenario.static, scenario.movable,
        scenario.start, scenario.goal, scenario.cfg, events=scenario.events,
        decision_points=[p.runtime_record() for p in scenario.decision_points])
    original_poses = {obs.oid: obs.polygon for obs in scenario.movable}
    started = time.monotonic()
    result = sim.run()
    wall_time = time.monotonic() - started
    status = "success" if result.success else "failed"
    if planner_done_callback is not None:
        planner_done_callback(wall_time)
    if image_path is not None:
        target = Path(image_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        viz.visualize(
            sim, result, original_poses,
            str(temporary), benchmark_info={
                **benchmark_info, "status": status,
                "wall_time_seconds": wall_time,
                "llm_modes": (f"cost={sim.estimator.mode}, "
                              f"risk={sim.risk.mode}"),
                "dynamic_oids": _dynamic_oids(manifest)})
        os.replace(temporary, target)
    if gif_path is not None:
        target = Path(gif_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        viz.render_sequence(sim, result, original_poses, str(temporary))
        os.replace(temporary, target)
    metrics = {
        "strategy": strategy, "status": status, "success": result.success,
        "C": result.C, "J": result.J, "risk_cost": result.risk_cost,
        "walk_cost": result.walk_cost, "work_cost": result.work_cost,
        "T": result.T, "move_time": result.move_time,
        "wait_time": result.wait_time,
        "wall_time_seconds": round(wall_time, 6),
        "plan_time_seconds": result.plan_time,
        "cycles": result.cycles, "expansions": result.total_expansions,
        "llm_cost_mode": sim.estimator.mode,
        "llm_risk_mode": sim.risk.mode,
        "llm_choice_mode": sim.chooser.mode,
        "llm_calls": result.llm_calls,
        "llm_calls_cost": result.llm_calls_cost,
        "llm_calls_risk": result.llm_calls_risk,
        "llm_calls_choice": result.llm_calls_choice,
        "moved": len(result.removed), "moved_obstacle_ids": result.removed,
        "robot_trajectory": [list(point) for point in result.robot_track],
        "dynamic_obstacle_trajectories": {
            str(oid): [list(point) for point in track]
            for oid, track in sim.dynamics.tracks.items()},
        "message": result.message,
    }
    metrics.update(_decision_metrics(scenario, result))
    return metrics


def _worker(manifest: dict, strategy: str, image_path: str | None,
            gif_path: str | None,
            benchmark_info: dict, result_queue) -> None:
    try:
        def planner_done(wall_time):
            result_queue.put({"kind": "planner_done",
                              "wall_time_seconds": wall_time})

        result_queue.put({"ok": True, "metrics": _run(
            manifest, strategy, image_path, gif_path, benchmark_info,
            planner_done_callback=planner_done)})
    except BaseException as exc:
        result_queue.put({"ok": False,
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()})


def _render_stopped(manifest: dict, strategy: str, image_path: Path,
                    gif_path: Path,
                    benchmark_info: dict, status: str, message: str,
                    wall_time: float) -> None:
    scenario = candidate_from_manifest(manifest)
    scenario.cfg.strategy = strategy
    scenario.cfg.save_frames = False
    scenario.cfg.verbose = False
    sim = OnlineNAMO(scenario.workspace, scenario.static, scenario.movable,
                     scenario.start, scenario.goal, scenario.cfg,
                     events=scenario.events)
    result = RunResult(success=False, message=message,
                       robot_track=[scenario.start])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = image_path.with_name(image_path.name + ".tmp")
    viz.visualize(
        sim, result, {obs.oid: obs.polygon for obs in scenario.movable},
        str(temporary), benchmark_info={
            **benchmark_info, "status": status,
            "wall_time_seconds": wall_time,
            "dynamic_oids": _dynamic_oids(manifest)})
    os.replace(temporary, image_path)
    gif_temporary = gif_path.with_name(gif_path.name + ".tmp")
    with Image.open(image_path) as frame:
        frame.convert("RGB").save(gif_temporary, format="GIF",
                                  duration=2000, loop=0)
    os.replace(gif_temporary, gif_path)


def _run_with_timeout(manifest: dict, strategy: str, image_path: Path | None,
                      gif_path: Path | None,
                      timeout: float, benchmark_info: dict,
                      progress_callback=None) -> dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_worker,
        args=(manifest, strategy, str(image_path) if image_path else None,
              str(gif_path) if gif_path else None,
              benchmark_info, result_queue),
        name=f"random-map-{manifest['seed']}-{strategy}")
    started = time.monotonic()
    process.start()
    # Drain the queue before joining.  A completed run can contain a sizeable
    # trajectory; joining first can deadlock while the queue's feeder waits for
    # the parent to read that payload.
    deadline = started + timeout
    reply = None
    last_progress = started
    phase = "planning"
    while reply is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            message = result_queue.get(timeout=min(0.2, remaining))
            if message.get("kind") == "planner_done":
                phase = "rendering"
                deadline = time.monotonic() + timeout
                continue
            reply = message
        except queue.Empty:
            now = time.monotonic()
            if progress_callback is not None and now - last_progress >= 1.0:
                progress_callback(now - started)
                last_progress = now
            if not process.is_alive():
                try:
                    reply = result_queue.get(timeout=min(0.2, remaining))
                except queue.Empty:
                    break
    process.join(1.0 if reply is not None else 0.0)
    elapsed = time.monotonic() - started
    if reply is None and process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join()
        payload = {
            "strategy": strategy,
            "status": "timeout" if phase == "planning" else "failed",
            "success": False,
            "wall_time_seconds": round(elapsed, 6),
            "message": ((f"Planner exceeded the {timeout:g}s timeout and was terminated.")
                        if phase == "planning" else
                        (f"PNG/GIF rendering exceeded {timeout:g}s and was terminated."))}
    else:
        if reply is None:
            reply = {"ok": False,
                     "error": f"worker exited with code {process.exitcode} without a result"}
        if reply.get("ok"):
            payload = reply["metrics"]
        else:
            payload = {
                "strategy": strategy, "status": "failed", "success": False,
                "wall_time_seconds": round(elapsed, 6),
                "message": reply.get("error", "unknown worker failure"),
                "traceback": reply.get("traceback", "")}
    if process.is_alive():
        process.terminate()
        process.join()
    result_queue.close()
    result_queue.join_thread()
    if (image_path is not None and gif_path is not None
            and (not image_path.exists() or not gif_path.exists())):
        _render_stopped(manifest, strategy, image_path, gif_path, benchmark_info,
                        payload["status"], payload["message"],
                        payload["wall_time_seconds"])
    return payload


def _base_row(generated, seed: int, experiment_name: str,
              map_path: Path) -> dict:
    return {
        "experiment": experiment_name, "seed": seed, "profile": PROFILE,
        "fingerprint": generated.fingerprint,
        "attempts": generated.attempts, "map_json": str(map_path),
        "obstacle_count": len(generated.scenario.movable),
        "dynamic_obstacle_count": len(_dynamic_oids(generated.manifest)),
        **generated.validation.metrics}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_aggregate(out: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (row["seed"], row["strategy"]))
    _atomic_json(out / "results.json", ordered)
    if not ordered:
        return
    fields = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in ordered:
        writer.writerow({key: (json.dumps(value, sort_keys=True)
                               if isinstance(value, (dict, list)) else value)
                         for key, value in row.items()})
    _atomic_text(out / "results.csv", buffer.getvalue())


def _coverage(manifests: list[dict]) -> dict:
    signatures, families = set(), Counter()
    decision_kinds, event_templates, oracle_actions = Counter(), Counter(), Counter()
    margins = []
    for manifest in manifests:
        metrics = manifest.get("validation", {})
        families[metrics.get("topology_family", "unknown")] += 1
        signatures.add((metrics.get("topology_family"), metrics.get("room_count"),
                        metrics.get("graph_edge_count"), metrics.get("cycle_rank"),
                        metrics.get("dead_end_count"), metrics.get("junction_count"),
                        metrics.get("shortest_hops")))
        for point in manifest.get("decision_points", ()):
            decision_kinds[point.get("kind", "unknown")] += 1
            oracle = point.get("metadata", {}).get("oracle", {})
            if oracle:
                oracle_actions[oracle.get("best_action", "unknown")] += 1
            if isinstance(oracle.get("relative_margin"), (int, float)):
                margins.append(float(oracle["relative_margin"]))
        event_templates.update(event.get("name", "unknown")
                               for event in manifest.get("events", ()))
    return {
        "seed_count": len(manifests),
        "unique_map_count": len({m.get("fingerprint") for m in manifests}),
        "unique_topology_signatures": len(signatures),
        "topology_families": dict(sorted(families.items())),
        "decision_kinds": dict(sorted(decision_kinds.items())),
        "event_templates": dict(sorted(event_templates.items())),
        "oracle_actions": dict(sorted(oracle_actions.items())),
        "oracle_margin": {
            "min": round(min(margins), 4) if margins else None,
            "max": round(max(margins), 4) if margins else None,
            "mean": round(sum(margins) / len(margins), 4) if margins else None}}


def _mean(values) -> float:
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else 0.0


# The final comparison is a table, not a chart: one row per strategy, one
# column per quantity the strategies are actually being compared on.
_SUMMARY_COLUMNS = (
    ("strategy", "strategy", "{}"),
    ("runs", "runs", "{:d}"),
    ("success_rate", "success %", "{:.1f}"),
    ("llm_calls", "LLM calls", "{:,.1f}"),
    ("J", "J", "{:,.1f}"),
    ("work_cost", "W (obstacle)", "{:,.1f}"),
    ("T", "T [s]", "{:,.2f}"),
    ("move_time", "move [s]", "{:,.2f}"),
    ("wait_time", "wait [s]", "{:,.2f}"),
)

_SUMMARY_NOTE = (
    "success %% and LLM calls cover all %d runs; "
    "J, W, T, move and wait are averaged over successful runs only "
    "(%s). W is the obstacle-manipulation work inside J."
)


def _summary_rows(rows: list[dict], strategies: tuple[str, ...]) -> list[dict]:
    """Aggregate one comparison row per strategy."""
    table = []
    for strategy in strategies:
        runs = [row for row in rows if row.get("strategy") == strategy]
        done = [row for row in runs if row.get("success")]
        table.append({
            "strategy": strategy,
            "runs": len(runs),
            "succeeded": len(done),
            "success_rate": 100.0 * len(done) / len(runs) if runs else 0.0,
            "llm_calls": _mean([row.get("llm_calls") for row in runs]),
            "J": _mean([row.get("J") for row in done]),
            "work_cost": _mean([row.get("work_cost") for row in done]),
            "T": _mean([row.get("T") for row in done]),
            "move_time": _mean([row.get("move_time") for row in done]),
            "wait_time": _mean([row.get("wait_time") for row in done]),
        })
    return table


# Columns that only make sense over runs that actually reached the goal.
_AVERAGED_OVER_SUCCESSES = ("J", "work_cost", "T", "move_time", "wait_time")


def _summary_cells(table: list[dict]) -> list[list[str]]:
    """Format every cell, leaving what has not been measured yet blank."""
    body = []
    for row in table:
        cells = []
        for key, _, spec in _SUMMARY_COLUMNS:
            blank = ((not row["runs"] and key != "strategy")
                     or (key in _AVERAGED_OVER_SUCCESSES and not row["succeeded"]))
            cells.append("-" if blank else spec.format(row[key]))
        body.append(cells)
    return body


def _summary_text(table: list[dict]) -> str:
    """Render the comparison as an aligned plain-text table."""
    headers = [header for _, header, _ in _SUMMARY_COLUMNS]
    body = _summary_cells(table)
    widths = [max(len(headers[i]), *(len(row[i]) for row in body))
              if body else len(headers[i]) for i in range(len(headers))]

    def line(cells):
        return "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells))

    out = [line(headers), "  ".join("-" * width for width in widths)]
    out.extend(line(row) for row in body)
    return "\n".join(out)


def _summary_markdown(table: list[dict]) -> str:
    headers = [header for _, header, _ in _SUMMARY_COLUMNS]
    aligns = ["---" if i == 0 else "---:" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(aligns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in _summary_cells(table))
    return "\n".join(lines)


def _write_summary(out: Path, rows: list[dict],
                   strategies: tuple[str, ...]) -> str:
    """Write the strategy comparison as Markdown and CSV; return the text form."""
    table = _summary_rows(rows, strategies)
    total_runs = sum(row["runs"] for row in table)
    succeeded = ", ".join(f"{row['strategy']} {row['succeeded']}"
                          for row in table)
    note = _SUMMARY_NOTE % (total_runs, succeeded)
    _atomic_text(out / "experiment_summary.md",
                 "# Strategy comparison\n\n" + _summary_markdown(table)
                 + "\n\n" + note + "\n")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=[key for key, _, _ in _SUMMARY_COLUMNS] + ["succeeded"])
    writer.writeheader()
    for row in table:
        writer.writerow({key: row[key] for key in writer.fieldnames})
    _atomic_text(out / "experiment_summary.csv", buffer.getvalue())
    return _summary_text(table) + "\n\n" + note


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculating"
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return ((f"{hours:d}h " if hours else "")
            + (f"{minutes:02d}m " if hours else f"{minutes:d}m ")
            + f"{seconds:02d}s")


def main() -> int:
    cfg = Config()
    out = Path(cfg.random_map_output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds = list(range(cfg.random_map_seed_start,
                       cfg.random_map_seed_start + cfg.random_map_experiment_count))
    strategies = cfg.random_map_run_strategies
    total_runs = len(seeds) * len(strategies)
    rows, manifests = [], []
    completed_maps = 0
    benchmark_started = time.monotonic()
    generator = ScenarioGenerator()

    print(f"Random-map benchmark: {len(seeds)} maps x {len(strategies)} strategies")
    print(f"Output: {out}  timeout: {cfg.random_map_timeout_seconds:g}s  "
          f"resume: {cfg.random_map_resume}", flush=True)
    needs_llm = any(STRATEGIES[strategy].llm_cost
                    or STRATEGIES[strategy].llm_risk
                    or STRATEGIES[strategy].llm_choice
                    for strategy in strategies)
    if needs_llm and not (cfg.deepseek_api_key
                          or os.getenv("DEEPSEEK_API_KEY", "")):
        print("WARNING: an LLM strategy is configured but no DeepSeek API key "
              "was found; it will use the heuristic fallback. Set "
              "deepseek_api_key in config.py or DEEPSEEK_API_KEY to run a true "
              "LLM experiment.", flush=True)

    def save_progress(status: str, *, seed=None, strategy=None,
                      run_elapsed=None) -> float:
        elapsed = time.monotonic() - benchmark_started
        completed_runs = len(rows)
        known_times = [row.get("wall_time_seconds") for row in rows
                       if isinstance(row.get("wall_time_seconds"), (int, float))]
        eta = (_mean(known_times) * (total_runs - completed_runs)
               if known_times else elapsed * total_runs)
        _atomic_json(out / "progress.json", {
            "status": status, "completed_maps": completed_maps,
            "total_maps": len(seeds), "completed_runs": completed_runs,
            "total_runs": total_runs, "current_seed": seed,
            "current_strategy": strategy,
            "current_run_elapsed_seconds": (round(run_elapsed, 3)
                                            if run_elapsed is not None else None),
            "elapsed_seconds": round(elapsed, 3),
            "estimated_remaining_seconds": round(eta, 3) if eta is not None else None,
            "updated_at": _utc_now()})
        return eta

    save_progress("running")
    for index, seed in enumerate(seeds, 1):
        map_started = time.monotonic()
        experiment_name = f"experiment_{index:04d}"
        experiment_dir = out / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{index}/{len(seeds)}] {experiment_name}, seed={seed}: "
              "generating map...", flush=True)
        generated = generator.generate(RandomScenarioRequest(
            seed=seed, profile=PROFILE,
            obstacle_count=cfg.random_map_obstacle_count,
            event_count=cfg.random_map_dynamic_obstacle_count))
        manifest = generated.manifest
        manifests.append(manifest)
        map_path = experiment_dir / f"{experiment_name}_map.json"
        saved_manifest = _read_json(map_path) if cfg.random_map_resume else None
        if (saved_manifest is None
                or saved_manifest.get("fingerprint") != generated.fingerprint):
            _atomic_json(map_path, manifest)
        base = _base_row(generated, seed, experiment_name, map_path)
        map_rows = []

        for strategy in strategies:
            artifact_stem = f"{experiment_name}_{strategy}"
            run_path = experiment_dir / f"{artifact_stem}_result.json"
            image_path = (experiment_dir / f"{artifact_stem}.png"
                          if cfg.random_map_generate_images else None)
            gif_path = (experiment_dir / f"{artifact_stem}.gif"
                        if cfg.random_map_generate_images else None)
            existing = _read_json(run_path) if cfg.random_map_resume else None
            resumable = (existing is not None
                         and existing.get("seed") == seed
                         and existing.get("fingerprint") == generated.fingerprint
                         and existing.get("strategy") == strategy
                         and (image_path is None or image_path.exists())
                         and (gif_path is None or gif_path.exists()))
            if resumable:
                row = existing
                print(f"  {strategy}: resumed ({row.get('status', 'done')})", flush=True)
            else:
                save_progress("running", seed=seed, strategy=strategy)
                print(f"  {strategy}: running (timeout {cfg.random_map_timeout_seconds:g}s)...",
                      flush=True)
                last_report = [-5.0]

                def report_running(run_elapsed):
                    eta = save_progress("running", seed=seed, strategy=strategy,
                                        run_elapsed=run_elapsed)
                    if run_elapsed - last_report[0] >= 5.0:
                        last_report[0] = run_elapsed
                        progress = f"{len(rows)}/{total_runs} runs"
                        print(f"    still running: {_format_duration(run_elapsed)}; "
                              f"{progress}; ETA {_format_duration(eta)}",
                              flush=True)

                metrics = _run_with_timeout(
                    manifest, strategy, image_path, gif_path,
                    cfg.random_map_timeout_seconds, {
                        "seed": seed, "strategy": strategy,
                        "experiment_index": index,
                        "experiment_total": len(seeds)},
                    progress_callback=report_running)
                row = {**base, **metrics, "result_json": str(run_path),
                       "image_png": str(image_path) if image_path else None,
                       "animation_gif": str(gif_path) if gif_path else None,
                       "completed_at": _utc_now()}
                _atomic_json(run_path, row)
                print(f"  {strategy}: {row['status']} in "
                      f"{row['wall_time_seconds']:.2f}s", flush=True)
            rows.append(row)
            map_rows.append(row)
            _write_aggregate(out, rows)
            save_progress("running", seed=seed, strategy=strategy)

        completed_maps += 1
        _atomic_json(experiment_dir / f"{experiment_name}_results.json", {
            "experiment": experiment_name,
            "experiment_index": index, "seed": seed,
            "map_json": str(map_path), "fingerprint": generated.fingerprint,
            "strategies": map_rows, "completed_at": _utc_now()})
        _atomic_json(out / "coverage.json", _coverage(manifests))
        _write_summary(out, rows, strategies)
        save_progress("running", seed=seed)
        elapsed = time.monotonic() - benchmark_started
        eta = elapsed / completed_maps * (len(seeds) - completed_maps)
        print(f"[{index}/{len(seeds)}] seed={seed} saved; "
              f"map time {_format_duration(time.monotonic() - map_started)}, "
              f"ETA {_format_duration(eta)}", flush=True)

    summary = _write_summary(out, rows, strategies)
    save_progress("completed")
    print(f"\nCompleted {len(seeds)} maps.\n")
    print(summary, flush=True)
    print(f"\nSummary -> {out / 'experiment_summary.md'}, "
          f"{out / 'experiment_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
