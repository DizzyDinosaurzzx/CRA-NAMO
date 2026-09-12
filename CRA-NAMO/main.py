"""命令行入口。"""

from __future__ import annotations
import argparse
import os
from typing import Optional


_pending_log: Optional[str] = None
_log_repeats = 0


def emit_log(line: str) -> None:
    """输出一行，并将连续重复项折叠为重复次数。"""
    global _pending_log, _log_repeats
    if line == _pending_log:
        _log_repeats += 1
        return
    flush_log()
    print(line)
    _pending_log = line


def flush_log() -> None:
    """结束当前一组折叠的控制台消息。"""
    global _pending_log, _log_repeats
    if _log_repeats:
        times = "time" if _log_repeats == 1 else "times"
        print(f"  ... last line repeated {_log_repeats} more {times}")
    _pending_log = None
    _log_repeats = 0

import config
import scenarios
import viz
from executor import OnlineNAMO
from scenario_generation.naming import map_stem, run_stem
from scenario_generation.profiles import PROFILES
from scenario_generation.serialization import save_manifest

def main():
    """解析命令行选项并运行一个场景。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO,
                    choices=scenarios.names())
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed used by --scenario seeded_random")
    ap.add_argument("--map-profile", choices=sorted(PROFILES), default="balanced",
                    help="Generation profile used by --scenario seeded_random")
    ap.add_argument("--map-manifest", default=None,
                    help="Replay a previously saved seeded-random JSON manifest")
    ap.add_argument("--save-map-manifest", nargs="?", const="auto", default=None,
                    metavar="PATH", help="Save the generated map manifest; an omitted "
                    "PATH writes it under the output directory")
    ap.add_argument("--generation-attempts", type=int, default=40,
                    help="Maximum deterministic candidates tried for a random map")
    ap.add_argument("--lambda", "--lambda_distance", dest="lambda_distance",
                    type=float, default=None,
                    help="Motion cost λ weight (larger values favour moving obstacles rather than detouring)")
    ap.add_argument("--time-importance", "-w", dest="time_importance",
                    type=float, default=None,
                    help="w in [0, 1] for C = (1-w)J + w*(time_value*T): 0 minimises "
                         "energy alone (default), 1 minimises time alone")
    ap.add_argument("--strategy", default=None,
                    choices=sorted(config.STRATEGIES),
                    help="What the LLM is asked to estimate. "
                         "cra-namo (alias: llm-cost-risk): LLM estimates both "
                         "push cost and risk; "
                         "llm-cost: LLM estimates cost, risk falls back to the "
                         "offline keyword table; "
                         "llm-risk: LLM estimates risk, cost falls back to the "
                         "offline material table; "
                         "no-llm: both from the offline heuristics; "
                         "shortest: no LLM and no NAMO trade-off at all — take the "
                         "shortest path to the goal and clear whatever stands on it "
                         "(J, R and T are still measured normally)")
    ap.add_argument("--no-llm-order", action="store_true",
                    help="Disable LLM-based intelligent ordering of obstacle processing")
    frames = ap.add_mutually_exclusive_group()
    frames.add_argument("--frames", dest="save_frames", action="store_true",
                    default=None,
                    help="Save the per-step robot motion as an animated GIF: "
                         "img/experiment_<number>_<profile>_<strategy>_<id>_animation.gif")
    frames.add_argument("--no-frames", dest="save_frames", action="store_false",
                        help="Do not save per-step frames or an animated GIF")
    ap.add_argument("--no-contact", action="store_true",
                    help="Drop the requirement that the robot stays in contact with "
                         "an obstacle while moving it (obstacles then move while the "
                         "robot waits on its node, and its escort travel is not charged)")
    ap.add_argument("--no-lookahead", action="store_true",
                    help="Score a drop pose by the move alone, without asking what "
                         "the robot's remaining route costs once the obstacle sits "
                         "there (and without refusing poses that shut the way)")
    ap.add_argument("--forward-penalty", type=float, default=None,
                    help="Soft bias towards dropping obstacles ahead of the robot "
                         "rather than behind it; 0 removes the bias entirely")
    args = ap.parse_args()

    if args.map_manifest and args.scenario != "seeded_random":
        ap.error("--map-manifest requires --scenario seeded_random")
    scenario_options = {}
    if args.scenario == "seeded_random":
        scenario_options = {
            "seed": args.seed,
            "profile": args.map_profile,
            "manifest_path": args.map_manifest,
            "generation_attempts": args.generation_attempts,
        }
    s = scenarios.load(args.scenario, **scenario_options)
    cfg = s["cfg"]
    cfg.set_logger(emit_log, flush_log)

    if args.lambda_distance is not None:
        try:
            cfg.lambda_distance = config.validate_lambda(args.lambda_distance)
        except ValueError as e:
            ap.error(str(e))

    if args.time_importance is not None:
        try:
            cfg.time_importance = config.validate_time_importance(args.time_importance)
        except ValueError as e:
            ap.error(str(e))

    if args.strategy is not None:
        try:
            cfg.strategy = config.validate_strategy(args.strategy)
        except ValueError as e:
            ap.error(str(e))

    if args.no_llm_order:
        cfg.use_llm_ordering = False

    if args.no_contact:
        cfg.contact_required = False
    if args.no_lookahead:
        cfg.manip_lookahead = False
    if args.forward_penalty is not None:
        cfg.manip_forward_penalty = max(0.0, args.forward_penalty)

    if args.save_frames is not None:
        cfg.save_frames = args.save_frames

    os.makedirs(cfg.out_dir, exist_ok=True)

    if args.save_map_manifest is not None:
        if "manifest" not in s:
            ap.error("--save-map-manifest requires --scenario seeded_random")
        manifest_path = args.save_map_manifest
        if manifest_path == "auto":
            manifest_path = os.path.join(
                cfg.out_dir, "maps",
                f"{map_stem(s['manifest'])}.json")
        saved = save_manifest(s["manifest"], manifest_path)
        print(f"Saved map manifest -> {saved}")

    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal"], cfg, events=s.get("dynamics"),
                     decision_points=s.get("decision_points"))
    
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    def _mode(estimator) -> str:
        return (estimator.mode
                + ("" if estimator.mode == "heuristic" else " (DeepSeek)"))

    print(f"Scenario: {s['name']}   {sim.roadmap}")
    note = ""
    if cfg.shortest_path_mode:
        note = "   (shortest path, obstacle cost ignored while planning)"
    elif cfg.llm_choice:
        note = "   (LLM picks among geometrically validated options)"
    print(f"Strategy: {cfg.strategy}{note}")
    print(f"Difficulty estimator: {_mode(sim.estimator)}"
          f"   Risk estimator: {_mode(sim.risk)}"
          + (f"   Action chooser: {_mode(sim.chooser)}" if cfg.llm_choice else ""))
    print("-" * 60)

    res = sim.run()
    print("=" * 60)

    W = 22
    print(f"{'Success':<{W}} : {res.success}   ({res.message})")
    print(f"{'Objective C':<{W}} : {res.C:,}"
          f"   (w={cfg.time_importance:g}: C = (1-w)J + w*{cfg.time_value:,g}*T)")
    print(f"{'Total cost J':<{W}} : {res.J:,}")
    print(f"{'motion lambda*D':<{W}} : {res.walk_cost:,}")
    print(f"{'  of which in contact':<{W}} : {res.manip_walk_cost:,}"
          f"   (robot travel while holding an obstacle)")
    print(f"{'obstacle work W':<{W}} : {res.work_cost:,}")
    charged = ", ".join(f"{oid}:{level}" for oid, level in sorted(res.risk_levels.items()))
    print(f"{'risk surcharge R':<{W}} : {res.risk_cost:,}"
          + (f"   (moved {charged})" if charged else ""))
    print(f"{'Obstacles moved':<{W}} : {res.removed}")
    print(f"{'Replan cycles':<{W}} : {res.cycles:,}")
    print(f"{'Total time T (s)':<{W}} : {res.T:,}   (simulated clock: moving + waiting)")
    print(f"{'  of which moving':<{W}} : {res.move_time:,}   (driving and turning)")
    print(f"{'  of which waiting':<{W}} : {res.wait_time:,}   (standing still for the world)")
    print(f"{'Total plan time (s)':<{W}} : {res.plan_time:,}")
    print(f"{'A* expansions':<{W}} : {res.total_expansions:,}")
    breakdown = [f"cost {res.llm_calls_cost:,}", f"risk {res.llm_calls_risk:,}"]
    if cfg.llm_choice:
        breakdown.append(f"choice {res.llm_calls_choice:,}")
    print(f"{'LLM calls':<{W}} : {res.llm_calls:,}  ({', '.join(breakdown)})")
    print(f"{'Risk assessments':<{W}} : {len(sim.risk.level):,} seen, "
          f"{len(sim.risk.on_contact):,} revised on contact"
          f"  (mode={sim.risk.mode})")
    if cfg.llm_choice:
        print(f"{'Action choices':<{W}} : {len(res.choices):,} logged, "
              f"{sim.chooser.reused:,} reused  (mode={sim.chooser.mode})")
        for line in res.choices:
            print(f"{'':<{W}}   {line}")
    if res.decisions:
        print(f"{'Decision points':<{W}} : {len(res.decisions):,}")
        for line in res.decisions:
            print(f"{'':<{W}}   {line}")
    if res.world_events:
        print(f"{'World events':<{W}} : {len(res.world_events):,}")
        for line in res.world_events:
            print(f"{'':<{W}}   {line}")

    stem = (run_stem(s["manifest"], cfg.strategy)
            if "manifest" in s else f"{s['name']}_{cfg.strategy}")
    out = os.path.join(cfg.out_dir, f"{stem}_summary.png")
    viz.visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")

    if cfg.save_frames:
        gif_path = os.path.join(cfg.out_dir, f"{stem}_animation.gif")
        n, step = viz.render_sequence(sim, res, original_poses, gif_path)
        print(f"Saved {n:,}-frame animation ({step:g}s of simulated time per frame, "
              f"{cfg.gif_fps:g} fps) -> {gif_path}")
    return res

if __name__ == "__main__":
    main()
