"""Rendering: the summary plot and the per-step animation.

Split out of `main.py`, which is now just the command-line entry point. Nothing
here is used by the simulation itself — it reads a finished `RunResult` and the
frame snapshots the executor recorded along the way.

Colour vocabulary:
    robot           yellow disc, dark goldenrod centre
    planned route   blue solid (where the robot intends to drive)
    obstacle route  blue dashed (where an obstacle is to be carried)
    planned contact dark goldenrod dotted (where the robot grips it)
    obstacle        crimson if untouched, orange once moved
    robot trail     translucent blue corridor, outlined; ground the robot drove
                    around rather than over stays empty
"""

from __future__ import annotations
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from PIL import Image
from shapely.geometry import LineString, Point

from executor import OnlineNAMO


def _ring(ring):
    """One closed ring as (vertices, path codes)."""
    verts = np.asarray(ring.coords)
    codes = np.full(len(verts), Path.LINETO, dtype=Path.code_type)
    codes[0] = Path.MOVETO
    return verts, codes


def _plot_poly(ax, poly, **kw):
    """Fill a (Multi)Polygon — holes included.

    Filling `exterior.xy` and stopping there, which is what this used to do,
    paints straight over every hole. It shows up the moment the robot's trail
    closes a loop: the buffered corridor is a ring, and the whole area the robot
    drove *around* was flooded with trail colour as if it had driven across it.
    Every ring goes into one compound path instead, which Agg fills by the
    even-odd rule, so interiors come out empty.
    """
    geoms = poly.geoms if poly.geom_type.startswith("Multi") else [poly]
    verts, codes = [], []
    for g in geoms:
        if g.is_empty:
            continue
        for ring in (g.exterior, *g.interiors):
            v, c = _ring(ring)
            verts.append(v)
            codes.append(c)
    if not verts:
        return
    ax.add_patch(PathPatch(Path(np.concatenate(verts), np.concatenate(codes)),
                           **kw))


# An estimate counts as correct only to floating-point noise. Difficulty is a
# derived quantity (mu*rho x volume x g), so when the estimator recognises the
# material it reproduces the scenario's value exactly; anything else is a miss,
# not a near miss. This is the comparison the label has always made.
_DIFF_MATCH_EPS = 1e-6


def _obstacle_label(oid: int, estimates, touched=None) -> str:
    """Obstacle ID, its estimated difficulty, and how that estimate held up.

    The second line is the estimate, the third is the verdict on it. The robot
    only learns the truth by physically touching the obstacle, so until then
    there is no third line at all; afterwards it reads `right` when the guess
    was correct and `real=<truth>` when it was not — the number is only worth
    the space when it says something the estimate did not.
    """
    est = estimates.get(oid) if estimates else None
    true = touched.get(oid) if touched else None
    parts = [str(oid)]
    if est is not None:
        parts.append(f"est={est:g}")
    if true is not None:
        # touched before ever being estimated: nothing to be right or wrong about
        if est is not None and abs(est - true) <= _DIFF_MATCH_EPS:
            parts.append("right")
        else:
            parts.append(f"real={true:g}")
    return "\n".join(parts)


# Outlined rather than a bare translucent wash: once holes render as holes, the
# boundary is carrying real information — a thin edge is what makes an enclosed
# pocket read as unvisited ground rather than as a lighter patch of trail.
_TRAIL_KW = dict(facecolor="royalblue", edgecolor="royalblue",
                 alpha=0.28, lw=0.6, zorder=5)


def _draw_trail(ax, track, radius: float):
    """Ground the robot's body has covered: the region swept by its disc.

    One buffered polygon over the whole track, not a stack of per-step circles,
    so ground crossed five times looks the same as ground crossed once — this
    answers *where it has been*, not how often. Ground it drove around but never
    over stays empty, which is the whole point of a loop having a hole.
    """
    if not track:
        return
    swept = (LineString(track).buffer(radius, cap_style=1) if len(track) >= 2
             else Point(track[0]).buffer(radius))
    _plot_poly(ax, swept, **_TRAIL_KW)


def _draw_flag(ax, sim: OnlineNAMO, point, color: str, label: str):  # draw flag marker
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
    segs = [(rm.nodes[u], rm.nodes[v]) for u, v in rm.edge_len]
    ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.2,
                                     zorder=0.3))
    # roadmap nodes: silver-grey scatter
    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color="silver", s=0.5, zorder=0.4)
    # highlight current node (blue square); reachable neighbours are not drawn separately
    if cur_node is not None and 0 <= cur_node < len(rm.nodes):
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color="dodgerblue", s=20, marker="s", zorder=4.7)


def _draw_plan_paths(ax, frame):  # draw the planned paths being executed at this frame (blue)
    labeled = set()
    for path in frame.get("plan_paths") or []:
        xs = [p[0] for p in path["pts"]]
        ys = [p[1] for p in path["pts"]]
        if path["kind"] == "route":
            # robot planned route: blue solid line + small dots at via-nodes
            label = None if "route" in labeled else "planned path"
            ax.plot(xs, ys, color="blue", lw=1.8, alpha=0.9, zorder=6,
                    solid_capstyle="round", label=label)
            ax.scatter(xs, ys, color="blue", s=6, zorder=6.1)
        elif path["kind"] == "contact":
            # where the robot holds the obstacle while it moves — the contact path
            label = None if "contact" in labeled else "planned contact"
            ax.plot(xs, ys, color="darkgreen", lw=1.1, ls=":", alpha=0.85, zorder=6,
                    label=label)
        else:
            # obstacle's own SE2 route: blue dashed line
            label = None if "obstacle" in labeled else "planned obstacle route"
            ax.plot(xs, ys, color="blue", lw=1.3, ls="--", alpha=0.75, zorder=6,
                    label=label)
        labeled.add(path["kind"])


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    # workspace background
    _plot_poly(ax, sim.workspace, color="whitesmoke", zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color="black", lw=1)
    # static obstacles (walls etc. that are never moved)
    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, color="dimgray", zorder=1)
    # start (blue) and goal (red) flags
    _draw_flag(ax, sim, sim.start_point, "royalblue", "start")
    _draw_flag(ax, sim, sim.goal_point, "red", "goal")
    # original positions of movable obstacles (red dashed), for comparison with final positions
    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color="crimson", lw=1, ls="--", alpha=0.5, zorder=2)


# --- Canvas layout ---
_PLOT_BOX = (8.0, 7.0)     # max plot area (width, height), inches
_MARGIN = 0.5              # left/right + bottom tick-label margin, inches
_TOP_PAD = 0.12            # whitespace above title, inches
_TITLE_LINE = 0.24         # height per title line, inches
_TITLE_LINES = 3           # always reserve height for the three-line caption
_LEGEND_H = 0.5            # bottom legend strip height, inches
_TITLE_FS = 9              # title font size
_LEGEND_NCOL = 5


# --- caption ---
# The summary plot and every animation frame carry the *same* three lines, so a
# still and a frame read the same way:
#     1. run configuration — fixed for the whole run
#     2. where the run is  — one step, or the final verdict
#     3. what it has cost  — identical fields either way, cumulative in a frame
# Structural sameness is the point. Line 2 is the one that differs by design, so
# anything belonging to only one of the two views belongs there.
def _mode_tag(sim: OnlineNAMO) -> str:
    cfg = sim.cfg
    return "[" + " | ".join((
        cfg.strategy,
        f"w={cfg.time_importance:g}",
        "SE2" if cfg.se2_use_planner else "teleport",
        sim.estimator.mode,
    )) + "]"


def _cost_line(J, walk, work, motion_s, plan_s) -> str:
    return (f"J {J:,.1f} = λD {walk:,.1f} + W {work:,.1f}"
            f"  |  move {motion_s:,.1f}s + plan {plan_s:,.1f}s")


def _wrap_title(lines, width_inch: float) -> str:
    """Fit the caption into the reserved block, one output line per input line.

    Long lines are ellipsised rather than reflowed: reflowing line 1 would push
    the cost line out of the block entirely, and the three lines have distinct
    jobs that wrapping would blur together.
    """
    ncols = max(20, int(width_inch / (0.55 * _TITLE_FS / 72.0)))
    out = []
    for line in list(lines)[:_TITLE_LINES]:
        if len(line) > ncols:
            line = line[:max(1, ncols - 1)] + "…"
        out.append(line)
    return "\n".join(out)

def _new_canvas(sim: OnlineNAMO):
    minx, miny, maxx, maxy = sim.workspace.bounds
    w, h = (maxx - minx) + 2.0, (maxy - miny) + 2.0   # consistent with xlim/ylim padding below
    s = min(_PLOT_BOX[0] / w, _PLOT_BOX[1] / h)
    pw, ph = w * s, h * s

    title_h = _TITLE_LINE * _TITLE_LINES
    fw = pw + 2 * _MARGIN
    fh = _LEGEND_H + _MARGIN + ph + title_h + _TOP_PAD

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes((_MARGIN / fw, (_LEGEND_H + _MARGIN) / fh, pw / fw, ph / fh))
    return fig, ax


def _finish_ax(ax, sim: OnlineNAMO, title_lines):
    minx, miny, maxx, maxy = sim.workspace.bounds
    ax.set_aspect("equal")
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_title(_wrap_title(title_lines, ax.figure.get_figwidth() - 2 * _MARGIN),
                 fontsize=_TITLE_FS)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.figure.legend(handles, labels, loc="lower center",
                         bbox_to_anchor=(0.5, 0.008),
                         ncol=min(len(handles), _LEGEND_NCOL), fontsize=8,
                         framealpha=0.9, borderaxespad=0.0)

def visualize(sim: OnlineNAMO, res, original_poses, out_path: str):
    """Generate the final summary plot of the simulation"""
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)
    # iterate over all movable obstacles, colour by whether they were moved
    for w in sim.world:
        col = "orange" if w.removed else "crimson"
        _plot_poly(ax, w.polygon, color=col, alpha=0.6, zorder=3)
        # label obstacle centroid with ID, estimate, and true difficulty if known
        ax.text(w.x, w.y, _obstacle_label(w.oid, sim.estimator.cache,
                                           sim.belief.touched_difficulty),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=4)

    _draw_trail(ax, res.robot_track, sim.cfg.robot_radius)
    # which obstacles moved is already on the map — they are the orange ones,
    # each drawn beside the dashed outline of where it started
    _finish_ax(ax, sim, (
        _mode_tag(sim),
        f"{res.message}  —  {res.cycles} replan cycles",
        _cost_line(res.J, res.walk_cost, res.work_cost,
                   res.motion_time, res.plan_time),
    ))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_frame(sim: OnlineNAMO, frame, original_poses,
                 idx: int, total: int, cur_node: int | None = None):
    """Render a single frame from the simulation; returns the figure (caller closes it)"""
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim, cur_node=cur_node)
    # draw obstacles classified by perception state
    perceived = frame["perceived"]
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            # unperceived: grey dotted outline only (robot doesn't know where this obstacle is)
            ax.plot(*poly.exterior.xy, color="gray", lw=1, ls=":", alpha=0.5, zorder=3)
            continue
        # perceived: colour by whether it has been moved
        col = "orange" if removed else "crimson"
        # the obstacle being moved sits above the robot, so a piece carried past
        # the robot is not hidden by it; on collisions the robot stays on top
        z = 8.5 if oid == frame.get("move_oid") else 3
        _plot_poly(ax, poly, color=col, alpha=0.6, zorder=z)
        cx, cy = poly.centroid.x, poly.centroid.y
        estimates = frame.get("estimated_difficulty", {})
        touched = frame.get("touched_difficulty", {})
        ax.text(cx, cy, _obstacle_label(oid, estimates, touched),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=z + 1)

    # trail up to this frame — same treatment as the summary plot
    _draw_trail(ax, frame["track"], sim.cfg.robot_radius)

    # draw all paths given by the planner at this frame (blue lines)
    _draw_plan_paths(ax, frame)

    # draw current robot position (filled green circle + dark green centre dot)
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, color="limegreen", alpha=0.85, zorder=7)
    ax.plot(rx, ry, marker="o", color="darkgreen", ms=4, zorder=8, label="robot")

    # same three lines as the summary; line 3 is cumulative up to this frame
    _finish_ax(ax, sim, (
        _mode_tag(sim),
        f"step {idx}/{total - 1}  —  {frame['label']}",
        _cost_line(frame["J"], frame.get("walk_cost", 0.0),
                   frame.get("work_cost", 0.0), frame.get("motion_time", 0.0),
                   frame.get("plan_time", 0.0)),
    ))
    return fig


def _shared_palette(buffers, colors: int = 255):
    """One palette for the whole animation, sampled from start / middle / end.

    Frames sharing a palette let Pillow store only the pixels that changed
    between them, which matters here: the roadmap background is redrawn
    identically every step and dominates the file size otherwise.
    """
    picks = sorted({0, len(buffers) // 2, len(buffers) - 1})
    tiles = [Image.open(buffers[i]).convert("RGB") for i in picks]
    w, h = tiles[0].size
    montage = Image.new("RGB", (w, h * len(tiles)))
    for k, tile in enumerate(tiles):
        montage.paste(tile, (0, h * k))
    return montage.quantize(colors=colors, method=Image.MEDIANCUT)


def render_sequence(sim: OnlineNAMO, res, original_poses, gif_path: str):
    """Render every motion frame into one animated GIF"""
    cfg = sim.cfg
    total = len(res.frames)
    if total == 0:
        return 0
    # keep the frames as encoded PNG bytes rather than decoded bitmaps: a long
    # run on a large map would otherwise hold hundreds of MB of pixels at once
    buffers = []
    for i, frame in enumerate(res.frames):
        fig = render_frame(sim, frame, original_poses, i, total,
                           cur_node=frame.get("node"))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=cfg.gif_dpi)
        plt.close(fig)
        buf.seek(0)
        buffers.append(buf)

    palette = _shared_palette(buffers)
    images = [Image.open(b).convert("RGB").quantize(palette=palette,
                                                    dither=Image.NONE)
              for b in buffers]
    step_ms = max(20, int(round(1000.0 / cfg.gif_fps)))
    durations = [step_ms] * len(images)
    durations[-1] = max(step_ms, int(cfg.gif_end_hold_s * 1000))  # pause on the result
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    return total


