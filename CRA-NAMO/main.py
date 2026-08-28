"""Command-line entry point."""

from __future__ import annotations
import argparse
import os
from typing import Optional


_pending_log: Optional[str] = None
_log_repeats = 0


def emit_log(line: str) -> None:
    """Print a line, folding consecutive duplicates into a repeat count."""
    global _pending_log, _log_repeats
    if line == _pending_log:
        _log_repeats += 1
        return
    flush_log()
    print(line)
    _pending_log = line


def flush_log() -> None:
    """Finish the current run of folded console messages."""
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

def main():
    """Parse command-line options and run one scenario."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO,
                    choices=scenarios.names())
    ap.add_argument("--lambda", "--lambda_distance", dest="lambda_distance",
                    type=float, default=None,
                    help="Motion cost λ weight (larger values favour moving obstacles rather than detouring)")
    ap.add_argument("--time-importance", "-w", dest="time_importance",
                    type=float, default=None,
                    help="w in [0, 1] for C = (1-w)J + w*(time_value*T): 0 minimises "
                         "energy alone (default), 1 minimises time alone")
    ap.add_argument("--no-llm-order", action="store_true",
                    help="Disable LLM-based intelligent ordering of obstacle processing")
    frames = ap.add_mutually_exclusive_group()
    frames.add_argument("--frames", dest="save_frames", action="store_true",
                    default=None,
                    help="Save the per-step robot motion as an animated GIF: "
                         "img/frames_<map_name>.gif")
    frames.add_argument("--no-frames", dest="save_frames", action="store_false",
                        help="Do not save per-step frames or an animated GIF")
    ap.add_argument("--no-contact", action="store_true",
                    help="Drop the requirement that the robot stays in contact with "
                         "an obstacle while moving it (obstacles then move while the "
                         "robot waits on its node, and its escort travel is not charged)")
    ap.add_argument("--forward-penalty", type=float, default=None,
                    help="Soft bias towards dropping obstacles ahead of the robot "
                         "rather than behind it; 0 removes the bias entirely")
    args = ap.parse_args()

    s = scenarios.load(args.scenario)
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

    if args.no_llm_order:
        cfg.use_llm_ordering = False

    if args.no_contact:
        cfg.contact_required = False
    if args.forward_penalty is not None:
        cfg.manip_forward_penalty = max(0.0, args.forward_penalty)

    if args.save_frames is not None:
        cfg.save_frames = args.save_frames

    os.makedirs(cfg.out_dir, exist_ok=True)

    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal"], cfg, events=s.get("dynamics"))
    
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    print(f"Scenario: {s['name']}   {sim.roadmap}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
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
    print(f"{'Total time T (s)':<{W}} : {res.T:,}   (simulated clock: moving + thinking)")
    print(f"{'  of which moving':<{W}} : {res.move_time:,}   (driving and turning)")
    print(f"{'Total plan time (s)':<{W}} : {res.plan_time:,}")
    print(f"{'A* expansions':<{W}} : {res.total_expansions:,}")
    print(f"{'LLM calls':<{W}} : {res.llm_calls:,}  (mode={res.llm_mode})")
    print(f"{'Risk assessments':<{W}} : {len(sim.risk.level):,} seen, "
          f"{len(sim.risk.on_contact):,} revised on contact"
          f"  ({sim.risk.calls:,} calls, mode={sim.risk.mode})")
    if res.world_events:
        print(f"{'World events':<{W}} : {len(res.world_events):,}")
        for line in res.world_events:
            print(f"{'':<{W}}   {line}")

    out = os.path.join(cfg.out_dir, f"summary_{s['name']}.png")
    viz.visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")

    if cfg.save_frames:
        gif_path = os.path.join(cfg.out_dir, f"frames_{s['name']}.gif")
        n, step = viz.render_sequence(sim, res, original_poses, gif_path)
        print(f"Saved {n:,}-frame animation ({step:g}s of simulated time per frame, "
              f"{cfg.gif_fps:g} fps) -> {gif_path}")
    return res

if __name__ == "__main__":
    main()
