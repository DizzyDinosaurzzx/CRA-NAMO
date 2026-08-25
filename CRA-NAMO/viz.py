"""Rendering: the summary plot and the per-step animation.

Split out of `main.py`, which is now just the command-line entry point. Nothing
here is used by the simulation itself — it reads a finished `RunResult` and the
frame snapshots the executor recorded along the way.

The animation is keyed to the simulated clock rather than to the frame list, so
a second of dragging an obstacle takes as much of the GIF as a second of driving
free, and a replan is a visible pause. See `_time_sampled`.

Colour vocabulary:
    robot           yellow disc, dark goldenrod centre
    planned route   blue solid (where the robot intends to drive)
    obstacle route  blue dashed (where an obstacle is to be carried)
    planned contact dark goldenrod dotted (where the robot grips it)
    obstacle        crimson if untouched, orange once moved
    robot trail     translucent blue corridor
"""

from __future__ import annotations
import io
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from shapely.geometry import LineString, Point

from executor import OnlineNAMO

def _plot_poly(ax, poly, **kw):  # draw basic polygon
    if poly.geom_type == "Polygon":
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kw)
    else:
        for g in poly.geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, **kw)


def _obstacle_label(oid: int, estimates, touched=None) -> str:
    """Obstacle ID, plus estimated / true difficulty when available."""
    est = estimates.get(oid) if estimates else None
    true = touched.get(oid) if touched else None
    parts = [str(oid)]
    if est is not None:
        parts.append(f"est={est:,g}")
    if true is not None:
        parts.append(f"true={true:,g}")
        if est is not None and abs(est - true) > 1e-6:
            parts[-1] = f"T={true:,g}"  # highlight mismatch with estimate
    return "\n".join(parts)


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
_TITLE_LINES = 3           # always reserve height for three title lines
_LEGEND_H = 0.5            # bottom legend strip height, inches
_TITLE_FS = 9              # title font size
_LEGEND_NCOL = 5


def _wrap_title(text: str, width_inch: float) -> str:
    ncols = max(20, int(width_inch / (0.55 * _TITLE_FS / 72.0)))
    lines = textwrap.wrap(text, width=ncols) or [""]
    if len(lines) > _TITLE_LINES:
        lines = lines[:_TITLE_LINES]
        lines[-1] = lines[-1][:max(1, ncols - 1)] + "…"
    return "\n".join(lines)

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


def _finish_ax(ax, sim: OnlineNAMO, title: str):
    minx, miny, maxx, maxy = sim.workspace.bounds
    ax.set_aspect("equal")
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_title(_wrap_title(title, ax.figure.get_figwidth() - 2 * _MARGIN),
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

    # draw robot motion trail corridor:
    if len(res.robot_track) >= 2:
        corridor = LineString(res.robot_track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, corridor, color="royalblue", alpha=0.3, zorder=5)
    elif res.robot_track:
        # if only a single point (robot did not move), draw a circle
        p = Point(res.robot_track[0]).buffer(sim.cfg.robot_radius)
        _plot_poly(ax, p, color="royalblue", alpha=0.3, zorder=5)
    # assemble title info. In maps like maze_doors, moved may have dozens of ids;
    # listing them all would span many lines, so only report the count when long.
    manip_mode = "SE2" if sim.cfg.se2_use_planner else "teleport"
    strategy = sim.cfg.strategy
    moved = (f"{len(res.removed)} obstacles" if len(res.removed) > 8
             else (str(res.removed) if res.removed else "none"))
    title = (f"[{strategy}] {res.message}  |  J={res.J:,} "
             f"(lambda*D={res.walk_cost:,}, W={res.work_cost:,})  |  "
             f"T={res.T:,g}s (moving {res.move_time:,g}s, planning {res.plan_time:,g}s)"
             f"  |  C={res.C:,} (w={sim.cfg.time_importance:g})  |  "
             f"moved={moved}  |  manip={manip_mode}  |  LLM={res.llm_mode}")
    _finish_ax(ax, sim, title)
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

    # draw robot motion trail up to this frame (semi-transparent blue corridor)
    track = frame["track"]
    if len(track) >= 2:
        buf = LineString(track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, buf, color="royalblue", alpha=0.25, zorder=5)

    # draw all paths given by the planner at this frame (blue lines)
    _draw_plan_paths(ax, frame)

    # draw current robot position (filled green circle + dark green centre dot)
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, color="limegreen", alpha=0.85, zorder=7)
    ax.plot(rx, ry, marker="o", color="darkgreen", ms=4, zorder=8, label="robot")

    # title: step N / total | action label | cumulative cost
    title = (f"[{sim.cfg.strategy}] step {idx}/{total - 1}  |  {frame['label']}"
             f"  |  J={frame['J']:,}  |  T={frame.get('t', 0.0):,.1f}s")
    _finish_ax(ax, sim, title)
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


def _time_sampled(frames, step: float, max_frames: int):
    """Indices of the frames to show, one per `step` seconds of simulated time.

    Frames are recorded at events — a node reached, a grip taken, a collision —
    and events are not evenly spaced in time: dragging an obstacle one metre
    takes several times longer than driving that metre free, and a replan takes
    seconds during which nothing moves at all. Playing the event list back at a
    fixed frame rate flattens all of that, which hides the very thing the time
    axis is for. So the animation is resampled onto the clock: at each tick it
    shows whichever frame was current then, repeating one that is still current
    and skipping past any that came and went inside a single tick.

    `step` is stretched if a run is long enough that honouring it would produce
    more frames than `max_frames` — a GIF too big to open shows nothing at all.
    Returns the indices and the step actually used, since a caller reporting the
    configured one would be quoting a number the animation does not run at.
    """
    times = [float(f.get("t", i)) for i, f in enumerate(frames)]
    span = times[-1] - times[0]
    if span <= 0.0:                       # no clock to speak of, show it as recorded
        return list(range(len(frames))), step
    step = max(step, span / max(1, max_frames))
    picks = []
    j = 0
    t = times[0]
    while t <= times[-1] + 1e-9:
        while j + 1 < len(frames) and times[j + 1] <= t + 1e-9:
            j += 1
        picks.append(j)
        t += step
    if picks[-1] != len(frames) - 1:      # always land on the finished world
        picks.append(len(frames) - 1)
    return picks, step


def render_sequence(sim: OnlineNAMO, res, original_poses, gif_path: str):
    """Render the run into one animated GIF, played on the simulated clock.

    Returns (frames written, simulated seconds each one stands for).
    """
    cfg = sim.cfg
    total = len(res.frames)
    if total == 0:
        return 0, cfg.gif_time_step
    picks, step = _time_sampled(res.frames, cfg.gif_time_step, cfg.gif_max_frames)
    # keep the frames as encoded PNG bytes rather than decoded bitmaps: a long
    # run on a large map would otherwise hold hundreds of MB of pixels at once.
    # A frame still current across several ticks is drawn once and shown again.
    rendered = {}
    for i in sorted(set(picks)):
        frame = res.frames[i]
        fig = render_frame(sim, frame, original_poses, i, total,
                           cur_node=frame.get("node"))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=cfg.gif_dpi)
        plt.close(fig)
        buf.seek(0)
        rendered[i] = buf
    buffers = [rendered[i] for i in picks]

    palette = _shared_palette(buffers)
    images = [Image.open(b).convert("RGB").quantize(palette=palette,
                                                    dither=Image.NONE)
              for b in buffers]
    step_ms = max(20, int(round(1000.0 / cfg.gif_fps)))
    durations = [step_ms] * len(images)
    durations[-1] = max(step_ms, int(cfg.gif_end_hold_s * 1000))  # pause on the result
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    return len(images), step


