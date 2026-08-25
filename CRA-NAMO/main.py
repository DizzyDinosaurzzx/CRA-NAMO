"""Command-line entry point."""

from __future__ import annotations
import argparse
import os

import config
import scenarios
import viz
from executor import OnlineNAMO

# --- main function ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO,
                    choices=scenarios.names())
    ap.add_argument("--lambda", "--lambda_distance", dest="lambda_distance",
                    type=float, default=None,
                    help="Motion cost λ weight (larger values favour moving obstacles rather than detouring)")
    ap.add_argument("--no-llm-order", action="store_true",
                    help="Disable LLM-based intelligent ordering of obstacle processing")
    ap.add_argument("--frames", action="store_true",
                    help="Save the per-step robot motion as an animated GIF: "
                         "img/frames_<map_name>.gif")
    ap.add_argument("--strategy", default=None,
                    choices=["normal", "shortest", "easiest"],
                    help="Planning strategy: normal (optimal J=λD+W), "
                         "shortest (minimise path, ignore W), "
                         "easiest (detour around all obstacles)")
    ap.add_argument("--no-contact", action="store_true",
                    help="Drop the requirement that the robot stays in contact with "
                         "an obstacle while moving it (obstacles then move while the "
                         "robot waits on its node, and its escort travel is not charged)")
    ap.add_argument("--forward-penalty", type=float, default=None,
                    help="Soft bias towards dropping obstacles ahead of the robot "
                         "rather than behind it; 0 removes the bias entirely")
    args = ap.parse_args()

    # load scenario
    s = scenarios.load(args.scenario)
    cfg = s["cfg"]

    # command-line override of strategy
    if args.strategy is not None:
        cfg.strategy = args.strategy

    # command-line override of λ in config
    if args.lambda_distance is not None:
        try:
            cfg.lambda_distance = config.validate_lambda(args.lambda_distance)
        except ValueError as e:
            ap.error(str(e))

    # command-line override of LLM ordering switch
    if args.no_llm_order:
        cfg.use_llm_ordering = False

    # command-line override of the contact model
    if args.no_contact:
        cfg.contact_required = False
    if args.forward_penalty is not None:
        cfg.manip_forward_penalty = max(0.0, args.forward_penalty)

    # command-line enable per-frame saving
    if args.frames:
        cfg.save_frames = True

    os.makedirs(cfg.out_dir, exist_ok=True)

    # initialise CA-NAMO simulator
    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal"], cfg)
    
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    print(f"Scenario: {s['name']}   strategy={cfg.strategy}   {sim.roadmap}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
    print("-" * 60)

    res = sim.run()
    print("=" * 60)  # separate runtime [se2] logs from the stats below

    # print simulation statistics
    W = 22  # label field width – everything left-aligned
    print(f"{'Success':<{W}} : {res.success}   ({res.message})")
    print(f"{'Total cost J':<{W}} : {res.J}")
    print(f"{'motion lambda*D':<{W}} : {res.walk_cost}")
    print(f"{'  of which in contact':<{W}} : {res.manip_walk_cost}"
          f"   (robot travel while holding an obstacle)")
    print(f"{'obstacle work W':<{W}} : {res.work_cost}")
    print(f"{'Obstacles moved':<{W}} : {res.removed}")
    print(f"{'Replan cycles':<{W}} : {res.cycles}")
    print(f"{'Total plan time (s)':<{W}} : {res.plan_time}")
    print(f"{'A* expansions':<{W}} : {res.total_expansions}")
    print(f"{'LLM calls':<{W}} : {res.llm_calls}  (mode={res.llm_mode})")

    # render summary plot
    strategy_suffix = f"_{cfg.strategy}" if cfg.strategy != "normal" else ""
    out = os.path.join(cfg.out_dir, f"summary_{s['name']}{strategy_suffix}.png")
    viz.visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")

    # render every motion frame into one animation
    if cfg.save_frames:
        gif_path = os.path.join(cfg.out_dir, f"frames_{s['name']}{strategy_suffix}.gif")
        n = viz.render_sequence(sim, res, original_poses, gif_path)
        print(f"Saved {n}-frame animation ({cfg.gif_fps:g} fps) -> {gif_path}")
    return res

if __name__ == "__main__":
    main()


