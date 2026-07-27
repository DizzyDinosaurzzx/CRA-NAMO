"""主程序+可视化"""

from __future__ import annotations
import argparse
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
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

def _draw_flag(ax, sim: OnlineNAMO, point, color: str, label: str):
    x, y = point
    s = 0.06 * max(sim.workspace.bounds[2], sim.workspace.bounds[3])
    top = y + s

    ax.plot([x, x], [y, top], color="black", lw=1.4,
            solid_capstyle="round", zorder=9)
    ax.fill([x, x + 0.62 * s, x],
            [top, top - 0.22 * s, top - 0.44 * s],
            facecolor=color, edgecolor="black", lw=0.8,
            zorder=9, label=label)
    ax.plot([x], [y], marker="o", color="black", ms=3.5, zorder=9)


def _draw_roadmap_bg(ax, sim: OnlineNAMO, cur_node: int | None = None):
    rm = sim.roadmap

    for u, v in rm.edge_len:
        x1, y1 = rm.nodes[u]
        x2, y2 = rm.nodes[v]
        ax.plot([x1, x2], [y1, y2], color="lightgray", lw=0.3, zorder=0.3)

    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color="silver", s=1, zorder=0.4)

    if cur_node is not None and cur_node in rm.adj:
        for v in rm.adj[cur_node]:
            x1, y1 = rm.nodes[cur_node]
            x2, y2 = rm.nodes[v]
            ax.plot([x1, x2], [y1, y2], color="dodgerblue", lw=1.2, alpha=0.7,
                    zorder=4.5)
        nb_xs = [rm.nodes[v][0] for v in rm.adj[cur_node]]
        nb_ys = [rm.nodes[v][1] for v in rm.adj[cur_node]]
        ax.scatter(nb_xs, nb_ys, color="dodgerblue", s=8, alpha=0.7, zorder=4.6)
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color="dodgerblue", s=20, marker="s", zorder=4.7)


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    _plot_poly(ax, sim.workspace, color="whitesmoke", zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color="black", lw=1)

    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, color="dimgray", zorder=1)

    _draw_flag(ax, sim, sim.start_point, "royalblue", "start")
    _draw_flag(ax, sim, sim.goal_point, "red", "goal")

    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color="crimson", lw=1, ls="--", alpha=0.5, zorder=2)


def _finish_ax(ax, sim: OnlineNAMO, title: str):
    ax.set_aspect("equal")
    ax.set_xlim(-1, sim.workspace.bounds[2] + 1)
    ax.set_ylim(-1, sim.workspace.bounds[3] + 1)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0, framealpha=0.9)


def visualize(sim: OnlineNAMO, res, original_poses, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)

    for w in sim.world:
        col = "orange" if w.removed else "crimson"
        _plot_poly(ax, w.polygon, color=col, alpha=0.6, zorder=3)
        ax.text(w.x, w.y, str(w.oid), ha="center", va="center", fontsize=8, zorder=4)

    if len(res.robot_track) >= 2:
        corridor = LineString(res.robot_track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, corridor, color="royalblue", alpha=0.3, zorder=5)
    elif res.robot_track:
        p = Point(res.robot_track[0]).buffer(sim.cfg.robot_radius)
        _plot_poly(ax, p, color="royalblue", alpha=0.3, zorder=5)

    title = (f"{res.message}  |  J={res.J} "
             f"(lambda*D={res.walk_cost}, W={res.work_cost})  |  "
             f"moved={res.removed}  |  LLM={res.llm_mode}")
    _finish_ax(ax, sim, title)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_frame(sim: OnlineNAMO, frame, original_poses, out_path: str,
                 idx: int, total: int, cur_node: int | None = None):
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim, cur_node=cur_node)

    perceived = frame["perceived"]
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            ax.plot(*poly.exterior.xy, color="gray", lw=1, ls=":", alpha=0.5, zorder=3)
            continue
        col = "orange" if removed else "crimson"
        _plot_poly(ax, poly, color=col, alpha=0.6, zorder=3)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, str(oid), ha="center", va="center", fontsize=8, zorder=4)

    track = frame["track"]
    if len(track) >= 2:
        buf = LineString(track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, buf, color="royalblue", alpha=0.25, zorder=5)

    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, color="red", alpha=0.7, zorder=7)
    ax.plot(rx, ry, marker="o", color="darkred", ms=4, zorder=8, label="robot")

    title = f"step {idx}/{total - 1}  |  {frame['label']}  |  J={frame['J']}"
    _finish_ax(ax, sim, title)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_sequence(sim: OnlineNAMO, res, original_poses, frames_dir: str):
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "step_*.png")):
        os.remove(old)
    total = len(res.frames)
    for i, frame in enumerate(res.frames):
        out = os.path.join(frames_dir, f"step_{i:03d}.png")
        render_frame(sim, frame, original_poses, out, i, total,
                     cur_node=frame.get("node"))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO,
                    choices=scenarios.names())
    ap.add_argument("--lambda", "--lambda_distance", dest="lambda_distance",
                    type=float, default=None)
    ap.add_argument("--no-llm-order", action="store_true")
    ap.add_argument("--frames", action="store_true",
                    help="逐步保存机器人每一步运动的帧图片到 img/frames_<地图名>/")
    args = ap.parse_args()

    s = scenarios.load(args.scenario)
    cfg = s["cfg"]
    if args.lambda_distance is not None:
        if args.lambda_distance < 0:
            ap.error("--lambda 必须为非负数")
        cfg.lambda_distance = args.lambda_distance
    if args.no_llm_order:
        cfg.use_llm_ordering = False
    if args.frames:
        cfg.save_frames = True

    os.makedirs(cfg.out_dir, exist_ok=True)

    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal"], cfg)
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    print(f"Scenario: {s['name']}   {sim.roadmap}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
    print("-" * 60)

    res = sim.run()

    print(f"Success           : {res.success}   ({res.message})")
    print(f"Total cost J       : {res.J}")
    print(f"  motion lambda*D  : {res.walk_cost}")
    print(f"  obstacle work W  : {res.work_cost}")
    print(f"Obstacles moved    : {res.removed}")
    print(f"Replan cycles      : {res.cycles}")
    print(f"First-plan time (s): {round(res.first_plan_time, 4)}")
    print(f"Total plan time (s): {res.plan_time}")
    print(f"A* expansions      : {res.total_expansions}")
    print(f"LLM calls          : {res.llm_calls}  (mode={res.llm_mode})")

    out = os.path.join(cfg.out_dir, f"summary_{s['name']}.png")
    visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")

    if cfg.save_frames:
        frames_dir = os.path.join(cfg.out_dir, f"frames_{s['name']}")
        n = render_sequence(sim, res, original_poses, frames_dir)
        print(f"Saved {n} step frames -> {frames_dir}/step_000.png ... "
              f"step_{n - 1:03d}.png")
    return res


if __name__ == "__main__":
    main()
