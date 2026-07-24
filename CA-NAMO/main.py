"""
入口：在指定场景上运行代价感知在线 NAMO 规划器并进行可视化。
Usage:
    python main.py                       # two_doors，使用启发式难度（无 API）
    python main.py --scenario two_doors  # 选择场景
    python main.py --lambda_w 5          # 提高操作代价 -> 更倾向绕行
    DEEPSEEK_API_KEY=sk-... python main.py   # 使用 DeepSeek 排列难度
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scenarios
from planner import OnlineNAMO


def _plot_poly(ax, poly, **kw):
    if poly.geom_type == "Polygon":
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kw)
    else:
        for g in poly.geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, **kw)


def visualize(sim: OnlineNAMO, res, original_poses, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_poly(ax, sim.workspace, color="whitesmoke", zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color="black", lw=1)

    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, color="dimgray", zorder=1)

    _plot_poly(ax, sim.goal_region, color="mediumseagreen", alpha=0.7, zorder=1)

    # 原始障碍物 footprint（虚线轮廓）
    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color="crimson", lw=1, ls="--", alpha=0.5, zorder=2)

    # 最终障碍物 footprint
    for w in sim.world:
        col = "orange" if w.removed else "crimson"
        _plot_poly(ax, w.polygon, color=col, alpha=0.6, zorder=3)
        ax.text(w.x, w.y, str(w.oid), ha="center", va="center", fontsize=8, zorder=4)

    # 机器人轨迹
    if res.robot_track:
        xs = [p[0] for p in res.robot_track]
        ys = [p[1] for p in res.robot_track]
        ax.plot(xs, ys, color="royalblue", lw=2, marker="o", ms=3, zorder=5)
        ax.plot(xs[0], ys[0], marker="*", color="blue", ms=16, zorder=6, label="start")

    ax.set_aspect("equal")
    ax.set_xlim(-1, sim.workspace.bounds[2] + 1)
    ax.set_ylim(-1, sim.workspace.bounds[3] + 1)
    title = (f"{res.message}  |  J={res.J} "
             f"(walk={res.walk_cost}, work={res.work_cost})  |  "
             f"moved={res.removed}  |  LLM={res.llm_mode}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="two_doors")
    ap.add_argument("--lambda_d", type=float, default=None)
    ap.add_argument("--lambda_w", type=float, default=None)
    ap.add_argument("--no-llm-order", action="store_true")
    args = ap.parse_args()

    s = scenarios.load(args.scenario)
    cfg = s["cfg"]
    if args.lambda_d is not None:
        cfg.lambda_d = args.lambda_d
    if args.lambda_w is not None:
        cfg.lambda_w = args.lambda_w
    if args.no_llm_order:
        cfg.use_llm_ordering = False

    os.makedirs(cfg.out_dir, exist_ok=True)

    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal_region"], cfg)
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    print(f"Scenario: {s['name']}   {sim.roadmap}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
    print("-" * 60)

    res = sim.run()

    print(f"Success           : {res.success}   ({res.message})")
    print(f"Total cost J       : {res.J}")
    print(f"  walk (lambda_d)  : {res.walk_cost}")
    print(f"  work (lambda_w)  : {res.work_cost}")
    print(f"Obstacles moved    : {res.removed}")
    print(f"Replan cycles      : {res.cycles}")
    print(f"First-plan time (s): {round(res.first_plan_time, 4)}")
    print(f"Total plan time (s): {res.plan_time}")
    print(f"A* expansions      : {res.total_expansions}")
    print(f"LLM calls          : {res.llm_calls}  (mode={res.llm_mode})")

    out = os.path.join(cfg.out_dir, f"summary_{s['name']}.png")
    visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")
    return res


if __name__ == "__main__":
    main()
