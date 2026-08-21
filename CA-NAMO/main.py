"""Command-line entry point."""

from __future__ import annotations
import argparse
import os

import config
import cost
import scenarios
import viz
from executor import OnlineNAMO


def _hms(seconds: float) -> str:
    """Seconds, plus a mm:ss reading once the number stops being readable."""
    if seconds < 60.0:
        return f"{seconds:,.2f} s"
    m, s = divmod(seconds, 60.0)
    h, m = divmod(int(m), 60)
    clock = f"{h}h {m:02d}m {s:04.1f}s" if h else f"{m}m {s:04.1f}s"
    return f"{seconds:,.2f} s   ({clock})"


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
                    choices=["normal", "shortest"],
                    help="Planning strategy: normal (optimal J=λD+W), "
                         "shortest (minimise path, ignore W)")
    ap.add_argument("--time-importance", dest="time_importance",
                    type=float, default=None,
                    help="How much the route choice weighs mission time against "
                         "energy, 0..1. 0 = pure J (energy only), 1 = pure "
                         "minimum time. In between the search minimises "
                         "(1-w)*J + w*(λ*v_max)*T")
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

    # command-line override of the energy-vs-time weighting
    if args.time_importance is not None:
        try:
            cfg.time_importance = config.validate_time_importance(
                args.time_importance)
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
    w = cfg.time_importance
    objective = ("J (energy only)" if w <= 0.0 else
                 "T (time only)" if w >= 1.0 else
                 f"{1 - w:g}*J + {w:g}*{cost.time_price(cfg):,.0f}W*T")
    print(f"Objective: time_importance={w:g}  ->  minimise {objective}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
    print("-" * 60)

    res = sim.run()
    print("=" * 60)  # separate runtime [se2] logs from the stats below

    # print simulation statistics
    W = 22  # label field width – everything left-aligned
    print(f"{'Success':<{W}} : {res.success}   ({res.message})")
    print(f"{'Total cost J':<{W}} : {res.J:,.4f}")
    print(f"{'motion lambda*D':<{W}} : {res.walk_cost:,.4f}")
    print(f"{'  of which in contact':<{W}} : {res.manip_walk_cost:,.4f}"
          f"   (robot travel while holding an obstacle)")
    print(f"{'obstacle work W':<{W}} : {res.work_cost:,.4f}")
    print(f"{'Obstacles moved':<{W}} : {res.removed}")
    print(f"{'Replan cycles':<{W}} : {res.cycles}")
    print(f"{'A* expansions':<{W}} : {res.total_expansions:,}")
    print(f"{'LLM calls':<{W}} : {res.llm_calls:,}  (mode={res.llm_mode})")

    # mission time = how long the robot drives + how long it thinks
    print("-" * 60)
    print(f"{'Robot motion time':<{W}} : {_hms(res.motion_time)}"
          f"   (v={cfg.v_max:g}/{cfg.v_max_contact:g} m/s free/contact,"
          f" a={cfg.a_max:g}/{cfg.a_max_contact:g} m/s^2)")
    print(f"{'  of which handling':<{W}} : {_hms(res.manip_motion_time)}"
          f"   (gripping, escorting and backing out)")
    print(f"{'Decision compute time':<{W}} : {_hms(res.plan_time)}"
          f"   ({res.cycles} replans, measured wall clock)")
    print(f"{'Total mission time':<{W}} : {_hms(res.mission_time)}")

    # render summary plot. The suffix keeps runs that differ only in objective
    # from overwriting each other's output.
    strategy_suffix = f"_{cfg.strategy}" if cfg.strategy != "normal" else ""
    if cfg.time_importance > 0.0:
        strategy_suffix += f"_t{cfg.time_importance:g}"
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


