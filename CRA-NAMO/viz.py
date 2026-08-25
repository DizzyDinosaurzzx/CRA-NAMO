"""Rendering: the summary plot and the per-step animation.

Split out of `main.py`, which is now just the command-line entry point. Nothing
here is used by the simulation itself — it reads a finished `RunResult` and the
frame snapshots the executor recorded along the way.

The animation is keyed to the simulated clock rather than to the frame list, so
a second of dragging an obstacle takes as much of the GIF as a second of driving
free, and a replan is a visible pause. See `_time_sampled`.

Colour vocabulary — three families that never borrow from each other, so no two
unrelated things can be confused at a glance:

    warm (obstacles)  fill shades an obstacle by how hard it is to move, light
                      cream through dark brown on a log scale (`difficulty_palette`)
    blue (intent)     everything the planner decided: route, obstacle route,
                      trail already driven
    green (the robot) the disc itself, the start flag, and the outline of any
                      obstacle it has actually moved

Grey is reserved for what the robot cannot act on — walls, the roadmap, and the
dotted outline of an obstacle it has not yet perceived. Red appears once, on the
goal flag.
"""

from __future__ import annotations
import io
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Times New Roman throughout — figures go into a paper, and the fallbacks are
# ordered so a machine without it still renders a serif rather than silently
# dropping to the sans-serif default.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral",
                                     "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Rectangle
from PIL import Image
from shapely.geometry import LineString, Point

from executor import OnlineNAMO

# --- palette ---
# Everything that has a colour has it named here, so a change is one edit and
# two things can never drift into the same shade by accident.
_BACKGROUND     = "#f6f6f3"      # workspace fill
_WALL           = "#585d63"      # static obstacles
_WALL_EDGE      = "#3a3e42"
_ROADMAP_EDGE   = "#e4e4e4"      # the graph is context, not content: keep it faint
_ROADMAP_NODE   = "#d2d2d2"
_CURRENT_NODE   = "#08519c"
_ROUTE          = "#08519c"      # where the robot intends to drive
_OBSTACLE_ROUTE = "#3182bd"      # where an obstacle is to be carried
_CONTACT        = "#6a51a3"      # where the robot grips it
_TRAIL          = "#9ecae1"      # ground already covered
_ROBOT          = "#31a354"
_ROBOT_CORE     = "#00441b"
_START          = "#31a354"
_GOAL           = "#d73027"
_GHOST          = "#b4b4b4"      # where an obstacle started
_OBSTACLE_EDGE  = "#8c510a"      # outline of a perceived obstacle
_MOVED_EDGE     = "#1a9850"      # ... once the robot has moved it
_UNPERCEIVED    = "#9a9a9a"      # outline of one it has not seen yet
_DIFFICULTY_CMAP = "YlOrBr"

# The gradient runs over the *true* difficulties, which is ground truth the robot
# does not have. That is deliberate and safe: it reaches the picture only, never
# the belief or the planner, exactly as the dashed "original pose" outlines
# already do. Reading the map, a human can see at a glance what the robot has to
# work out by touching things.
_DIFFICULTY_SHADE = (0.18, 0.92)   # crop the colormap: pure white reads as absent


def _difficulty_cmap():
    """The colormap, cropped to the shades actually used.

    Built once here and used for both the obstacle fills and the colourbar, so
    the key under the plot cannot drift out of step with what it is explaining.
    """
    base = matplotlib.colormaps[_DIFFICULTY_CMAP]
    lo, hi = _DIFFICULTY_SHADE
    return ListedColormap([base(lo + (hi - lo) * i / 255.0) for i in range(256)])


def difficulty_palette(world):
    """Map each obstacle id to a fill colour, shading by true difficulty.

    Log scale, because difficulty spans four orders of magnitude across the
    scenarios — on a linear ramp every obstacle in a map holding one heavy item
    collapses to the same pale shade. Returns (colours, low, high) so the key
    below the plot can be drawn over the same range.
    """
    values = {w.oid: max(float(w.difficulty), 1e-6) for w in world}
    if not values:
        return {}, 0.0, 0.0
    low, high = min(values.values()), max(values.values())
    cmap = _difficulty_cmap()
    if high - low < 1e-9:      # every obstacle equally hard: no gradient to show
        return {oid: cmap(0.5) for oid in values}, low, high
    norm = LogNorm(vmin=low, vmax=high)
    return {oid: cmap(norm(v)) for oid, v in values.items()}, low, high


def _label_colour(fill) -> str:
    """Black or white, whichever stays readable on `fill`.

    The gradient runs from near-white to dark brown, so a single fixed text
    colour is illegible at one end or the other. Relative luminance (Rec. 709)
    decides, with the threshold where the two contrast ratios cross.
    """
    r, g, b = fill[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 0.55 else "#ffffff"


def _draw_obstacle_key(ax, show_unperceived: bool):
    """Legend proxies for what an obstacle's *outline* means.

    Fill shades difficulty, so the state of an obstacle had to move to the
    outline — and an encoding nobody can read is not an encoding. Zero-size
    rectangles carry the styling into the legend without drawing on the plot.
    """
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="none",
                           edgecolor=_MOVED_EDGE, lw=2.0,
                           label="moved by the robot"))
    if show_unperceived:
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="none",
                               edgecolor=_UNPERCEIVED, lw=1.0, ls=":",
                               label="not yet perceived"))


def _draw_difficulty_key(fig, ax, colours, low, high):
    """Explain what an obstacle's shade means, below the plot.

    A colourbar rather than swatches: two squares can name the ends but cannot
    show that the ramp between them is continuous and logarithmic, and that is
    the part that makes a 66 N door and a 167 kN block legible on one map. Ticks
    are placed at the two ends and the geometric middle, which are meaningful for
    any range, rather than at whatever decades happen to fall inside it.

    A map whose obstacles are all equally hard has no ramp to draw, so it falls
    back to a single labelled swatch in the ordinary legend.
    """
    if not colours:
        return
    if high - low < 1e-9:
        ax.scatter([], [], marker="s", s=40, color=_difficulty_cmap()(0.5),
                   label=f"obstacle difficulty {low:,.0f} N")
        return
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    width = min(0.5 * fw, 3.4)
    cax = fig.add_axes(((fw - width) / 2.0 / fw, (_LEGEND_H + 0.17) / fh,
                        width / fw, 0.10 / fh))
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=LogNorm(vmin=low, vmax=high),
                              cmap=_difficulty_cmap()),
        cax=cax, orientation="horizontal")
    ticks = [low, math.sqrt(low * high), high]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{t:,.0f}" for t in ticks])
    # a log scale labels its minor ticks too, which over a range narrower than
    # two decades writes "3 x 10^1" on top of the three ticks actually wanted
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=6, length=2, pad=1)
    bar.outline.set_linewidth(0.5)
    # the title sits above the bar; a colourbar label would land under the ticks,
    # which is where the ordinary legend already is
    bar.ax.set_title("obstacle difficulty — push force [N], shown for the reader "
                     "only", fontsize=6.5, pad=3)


def _plot_poly(ax, poly, **kw):  # draw basic polygon
    if poly.geom_type == "Polygon":
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kw)
    else:
        for g in poly.geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, **kw)


def _obstacle_label(oid: int, estimates, touched=None, risk=None) -> str:
    """Obstacle ID, what moving it is thought to cost, and how dangerous it is.

    The difficulty line switches rather than accumulates: an estimate while the
    obstacle has only been seen, the measured value once it has been touched.
    Showing both would be showing a number nothing consults any more —
    `Belief.get_difficulty` stops asking the estimator the moment a true value
    exists, and so should the picture.

    The risk line is shown at every level including `low`, because a blank is
    ambiguous between "assessed as safe" and "not assessed yet". Obstacles that
    genuinely have no verdict — never perceived — get no line at all.
    """
    true = touched.get(oid) if touched else None
    est = estimates.get(oid) if estimates else None
    level = risk.get(oid) if risk else None
    parts = [str(oid)]
    if true is not None:
        parts.append(f"real={true:,g}")
    elif est is not None:
        parts.append(f"est={est:,g}")
    if level is not None:
        parts.append(f"risk={level}")
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
    ax.add_collection(LineCollection(segs, colors=_ROADMAP_EDGE, linewidths=0.2,
                                     zorder=0.3))
    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color=_ROADMAP_NODE, s=0.5, zorder=0.4)
    # highlight current node; reachable neighbours are not drawn separately
    if cur_node is not None and 0 <= cur_node < len(rm.nodes):
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color=_CURRENT_NODE, s=20, marker="s", zorder=4.7)


def _draw_plan_paths(ax, frame):  # draw the planned paths being executed at this frame (blue)
    labeled = set()
    for path in frame.get("plan_paths") or []:
        xs = [p[0] for p in path["pts"]]
        ys = [p[1] for p in path["pts"]]
        if path["kind"] == "route":
            # robot planned route: blue solid line + small dots at via-nodes
            label = None if "route" in labeled else "planned path"
            ax.plot(xs, ys, color=_ROUTE, lw=1.8, alpha=0.9, zorder=6,
                    solid_capstyle="round", label=label)
            ax.scatter(xs, ys, color=_ROUTE, s=6, zorder=6.1)
        elif path["kind"] == "contact":
            # where the robot holds the obstacle while it moves — the contact path
            label = None if "contact" in labeled else "planned contact"
            ax.plot(xs, ys, color=_CONTACT, lw=1.2, ls=":", alpha=0.9, zorder=6,
                    label=label)
        else:
            # obstacle's own SE2 route
            label = None if "obstacle" in labeled else "planned obstacle route"
            ax.plot(xs, ys, color=_OBSTACLE_ROUTE, lw=1.3, ls="--", alpha=0.85,
                    zorder=6, label=label)
        labeled.add(path["kind"])


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    # workspace background
    _plot_poly(ax, sim.workspace, facecolor=_BACKGROUND, zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color=_WALL_EDGE, lw=1)
    # static obstacles (walls etc. that are never moved)
    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, facecolor=_WALL, edgecolor=_WALL_EDGE,
                   lw=0.5, zorder=1)
    _draw_flag(ax, sim, sim.start_point, _START, "start")
    _draw_flag(ax, sim, sim.goal_point, _GOAL, "goal")
    # where each movable obstacle started, for comparison with where it ended up
    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color=_GHOST, lw=1, ls="--", alpha=0.8, zorder=2)


# --- Canvas layout ---
_PLOT_BOX = (8.0, 7.0)     # max plot area (width, height), inches
_MARGIN = 0.5              # left/right + bottom tick-label margin, inches
_TOP_PAD = 0.12            # whitespace above title, inches
_TITLE_LINE = 0.24         # height per title line, inches
_TITLE_LINES = 4           # height reserved above every animation frame
_LEGEND_H = 0.5            # bottom legend strip height, inches
_CBAR_H = 0.46             # difficulty colourbar strip, above the legend, inches
_TITLE_FS = 9              # title font size
_LEGEND_NCOL = 6


def _lay_out_title(groups, width_inch: float, max_lines: int = _TITLE_LINES) -> str:
    """Lay the title out one category per line: `groups` is [(label, segments)].

    Two rules, both about being read rather than being compact. Each category
    starts its own line, because someone scanning the figure is after one number
    and knowing which line it lives on is worth more than saving a line. And a
    line is only ever broken *between* segments, never inside one — `textwrap`
    breaks on spaces, which used to put a line end in the middle of
    "lambda*D=11,740.872" and orphan a bare number onto the next line.

    A category too wide for the plot continues on the following line instead of
    being squeezed or clipped.
    """
    ncols = max(20, int(width_inch / (0.55 * _TITLE_FS / 72.0)))
    lines = []
    for label, segments in groups:
        segments = [s for s in segments if s]
        if not segments:
            continue
        head = f"{label}:  " if label else ""
        current = ""
        for seg in segments:
            candidate = f"{current}  |  {seg}" if current else seg
            if len(head) + len(candidate) <= ncols or not current:
                current = candidate
            else:
                lines.append(head + current)
                head = ""          # the title is centred, so no indent to align to
                current = seg
        lines.append(head + current)
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(1, ncols - 1)] + "…"
    return "\n".join(lines)

def _new_canvas(sim: OnlineNAMO, title_lines: int = _TITLE_LINES):
    """Blank figure with room reserved above the axes for `title_lines` of title.

    Animation frames must all reserve the same height whatever their title says,
    or the GIF ends up with frames of differing sizes; a one-off summary can size
    itself to the title it actually has.
    """
    minx, miny, maxx, maxy = sim.workspace.bounds
    w, h = (maxx - minx) + 2.0, (maxy - miny) + 2.0   # consistent with xlim/ylim padding below
    s = min(_PLOT_BOX[0] / w, _PLOT_BOX[1] / h)
    pw, ph = w * s, h * s

    title_h = _TITLE_LINE * title_lines
    fw = pw + 2 * _MARGIN
    fh = _LEGEND_H + _CBAR_H + _MARGIN + ph + title_h + _TOP_PAD

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes((_MARGIN / fw, (_LEGEND_H + _CBAR_H + _MARGIN) / fh,
                       pw / fw, ph / fh))
    return fig, ax


def _finish_ax(ax, sim: OnlineNAMO, title: str):
    minx, miny, maxx, maxy = sim.workspace.bounds
    ax.set_aspect("equal")
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_title(title, fontsize=_TITLE_FS)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.figure.legend(handles, labels, loc="lower center",
                         bbox_to_anchor=(0.5, 0.008),
                         ncol=min(len(handles), _LEGEND_NCOL), fontsize=8,
                         framealpha=0.9, borderaxespad=0.0)

def _summary_title(sim: OnlineNAMO, res) -> list:
    """The facts above the summary plot, grouped into the categories they belong to.

    Deliberately absent: the combined objective C, the total elapsed time, which
    difficulty estimator ran, and the risk surcharge R. C and T are sums of
    things already on the line below; the estimator says nothing about the run
    itself; and R is carried by the obstacles — every one of them shows its own
    `risk=` verdict, which is the form in which it actually explains the route.
    """
    moved = (f"{len(res.removed)} obstacles" if len(res.removed) > 8
             else (str(res.removed) if res.removed else "none"))
    return [
        ("", [f"[{sim.cfg.strategy}] {res.message}"]),
        ("cost", [f"J={res.J:,}",
                  f"lambda*D={res.walk_cost:,}",
                  f"W={res.work_cost:,}"]),
        ("time", [f"plan {res.plan_time:,g}s",
                  f"move {res.move_time:,g}s",
                  f"time_importance={sim.cfg.time_importance:g}"]),
        ("moved", [moved]),
    ]


def visualize(sim: OnlineNAMO, res, original_poses, out_path: str):
    """Generate the final summary plot of the simulation"""
    title = _lay_out_title(_summary_title(sim, res),
                           _PLOT_BOX[0], max_lines=8)
    fig, ax = _new_canvas(sim, title_lines=title.count("\n") + 1)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)
    # fill shades true difficulty; the outline says whether the robot moved it
    colours, low, high = difficulty_palette(sim.world)
    for w in sim.world:
        _plot_poly(ax, w.polygon, facecolor=colours[w.oid], alpha=0.9,
                   edgecolor=_MOVED_EDGE if w.removed else _OBSTACLE_EDGE,
                   lw=2.0 if w.removed else 0.8, zorder=3)
        ax.text(w.x, w.y, _obstacle_label(w.oid, sim.estimator.cache,
                                           sim.belief.touched_difficulty,
                                           sim.risk.level),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=4,
                color=_label_colour(colours[w.oid]))
    _draw_difficulty_key(fig, ax, colours, low, high)
    _draw_obstacle_key(ax, show_unperceived=False)

    # draw robot motion trail corridor:
    if len(res.robot_track) >= 2:
        corridor = LineString(res.robot_track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, corridor, facecolor=_TRAIL, alpha=0.45, zorder=5)
    elif res.robot_track:
        # if only a single point (robot did not move), draw a circle
        p = Point(res.robot_track[0]).buffer(sim.cfg.robot_radius)
        _plot_poly(ax, p, facecolor=_TRAIL, alpha=0.45, zorder=5)
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
    colours, low, high = difficulty_palette(sim.world)
    perceived = frame["perceived"]
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            # unperceived: grey dotted outline only (robot doesn't know where this obstacle is)
            ax.plot(*poly.exterior.xy, color=_UNPERCEIVED, lw=1, ls=":",
                    alpha=0.6, zorder=3)
            continue
        # the obstacle being moved sits above the robot, so a piece carried past
        # the robot is not hidden by it; on collisions the robot stays on top
        z = 8.5 if oid == frame.get("move_oid") else 3
        _plot_poly(ax, poly, facecolor=colours.get(oid, _OBSTACLE_EDGE), alpha=0.9,
                   edgecolor=_MOVED_EDGE if removed else _OBSTACLE_EDGE,
                   lw=2.0 if removed else 0.8, zorder=z)
        cx, cy = poly.centroid.x, poly.centroid.y
        estimates = frame.get("estimated_difficulty", {})
        touched = frame.get("touched_difficulty", {})
        ax.text(cx, cy, _obstacle_label(oid, estimates, touched,
                                        frame.get("risk")),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=z + 1,
                color=_label_colour(colours.get(oid, (1.0, 1.0, 1.0))))

    # draw robot motion trail up to this frame (semi-transparent blue corridor)
    track = frame["track"]
    if len(track) >= 2:
        buf = LineString(track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, buf, facecolor=_TRAIL, alpha=0.4, zorder=5)

    # draw all paths given by the planner at this frame (blue lines)
    _draw_plan_paths(ax, frame)

    # draw current robot position (filled green circle + dark green centre dot)
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, facecolor=_ROBOT, alpha=0.9, zorder=7)
    ax.plot(rx, ry, marker="o", color=_ROBOT_CORE, ms=4, zorder=8, label="robot")
    _draw_difficulty_key(fig, ax, colours, low, high)
    _draw_obstacle_key(ax, show_unperceived=True)

    # every frame reserves the same title height, or the GIF frames differ in size
    title = _lay_out_title([
        ("", [f"[{sim.cfg.strategy}] step {idx}/{total - 1}", frame["label"]]),
        ("cost", [f"J={frame['J']:,}"]),
        ("time", [f"plan {frame.get('plan_t', 0.0):,g}s",
                  f"move {frame.get('move_t', 0.0):,g}s",
                  f"time_importance={sim.cfg.time_importance:g}"]),
    ], _PLOT_BOX[0])
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


