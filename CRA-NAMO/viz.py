"""绘制汇总图和按时间采样的动画。"""

from __future__ import annotations
import io
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral",
                                     "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.patheffects as patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, to_rgb
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from PIL import Image
from shapely.geometry import LineString, Point

from executor import OnlineNAMO

_BACKGROUND     = "#f6f6f3"      # 工作空间填充色
_WALL           = "#585d63"      # 静态障碍物
_WALL_EDGE      = "#3a3e42"
_ROADMAP_EDGE   = "#e4e4e4"      # 路线图只作背景，使用浅色
_ROADMAP_NODE   = "#d2d2d2"
_CURRENT_NODE   = "#08519c"
_ROUTE          = "#08519c"      # 机器人计划行驶的位置
_OBSTACLE_ROUTE = "#3182bd"      # 障碍物计划被搬移的位置
_CONTACT        = "#6a51a3"      # 机器人握持障碍物的位置
_TRAIL          = "#9ecae1"      # 已经走过的路径
_ROBOT          = "#31a354"
_ROBOT_CORE     = "#00441b"
_START          = "#31a354"
_GOAL           = "#d73027"
_GHOST          = "#b4b4b4"      # 障碍物的起始位置
_OBSTACLE_EDGE  = "#8c510a"      # 已感知障碍物的轮廓
_MOVED_EDGE     = "#1a9850"      # 机器人搬移后的轮廓
_WORLD_EDGE     = "#7a0177"      # 世界自主移动后的轮廓
_UNPERCEIVED    = "#9a9a9a"      # 未感知障碍物的轮廓
_DIFFICULTY_CMAP = "YlOrBr"

_DIFFICULTY_SHADE = (0.18, 0.92)   # 截取颜色图范围，避免纯白看起来像缺失


_OBSTACLE_ALPHA = 0.9
_TEXT_DARK      = "#111111"
_TEXT_LIGHT     = "#ffffff"
_TRAIL_Z        = 5                # 机器人的轨迹，标签放在轨迹上方


def _difficulty_cmap():
    base = matplotlib.colormaps[_DIFFICULTY_CMAP]
    lo, hi = _DIFFICULTY_SHADE
    return ListedColormap([base(lo + (hi - lo) * i / 255.0) for i in range(256)])


def difficulty_palette(world):
    values = {w.oid: max(float(w.difficulty), 1e-6) for w in world}
    if not values:
        return {}, 0.0, 0.0
    low, high = min(values.values()), max(values.values())
    cmap = _difficulty_cmap()
    if high - low < 1e-9:      # 所有障碍物难度相同，不显示渐变
        return {oid: cmap(0.5) for oid in values}, low, high
    norm = LogNorm(vmin=low, vmax=high)
    return {oid: cmap(norm(v)) for oid, v in values.items()}, low, high


def _luminance(colour) -> float:
    r, g, b = colour[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _over(fg, bg, alpha: float) -> tuple:
    return tuple(alpha * f + (1.0 - alpha) * b for f, b in zip(fg[:3], to_rgb(bg)))


def _contrast(a, b) -> float:
    la, lb = _luminance(to_rgb(a)) + 0.05, _luminance(to_rgb(b)) + 0.05
    return max(la, lb) / min(la, lb)


def _label_colour(fill, alpha: float = _OBSTACLE_ALPHA) -> str:
    seen = _over(fill, _BACKGROUND, alpha)
    return max((_TEXT_DARK, _TEXT_LIGHT), key=lambda c: _contrast(c, seen))


def _label_effects(colour: str):
    edge = _TEXT_LIGHT if colour == _TEXT_DARK else _TEXT_DARK
    return [patheffects.withStroke(linewidth=1.6, foreground=edge, alpha=0.75)]


def _obstacle_edge(oid, removed: bool, world_moved) -> tuple:
    if removed:
        return _MOVED_EDGE, 2.0
    if oid in world_moved:
        return _WORLD_EDGE, 2.0
    return _OBSTACLE_EDGE, 0.8


def _draw_obstacle_key(ax, show_unperceived: bool, show_world_moved: bool = False):
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="none",
                           edgecolor=_MOVED_EDGE, lw=2.0,
                           label="moved by the robot"))
    if show_world_moved:
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="none",
                               edgecolor=_WORLD_EDGE, lw=2.0,
                               label="moved on its own"))
    if show_unperceived:
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="none",
                               edgecolor=_UNPERCEIVED, lw=1.0, ls=":",
                               label="not yet perceived"))


def _draw_difficulty_key(fig, ax, colours, low, high):
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
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=_COLORBAR_TICK_FS, length=2, pad=1)
    bar.outline.set_linewidth(0.5)
    bar.ax.set_title("obstacle difficulty — push force [N], shown for the reader "
                     "only", fontsize=_COLORBAR_TITLE_FS, pad=3)


def _poly_path(poly) -> Path:
    """返回包含多边形及其孔洞的 Matplotlib 路径。"""
    verts, codes = [], []
    for ring in (poly.exterior, *poly.interiors):
        xy = list(ring.coords)
        if len(xy) < 4:            # 三个不同顶点，再重复首点
            continue
        verts.extend(xy)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2)
                     + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _plot_poly(ax, poly, **kw):  # 绘制多边形，保留孔洞为空
    """填充多边形外壳，同时保留孔洞。"""
    kw.setdefault("edgecolor", "none")   # 仅提供面颜色时 ax.fill 的默认行为
    for part in getattr(poly, "geoms", (poly,)):
        if part.is_empty or part.geom_type != "Polygon":
            continue
        ax.add_patch(PathPatch(_poly_path(part), **kw))


def _obstacle_label(oid: int, estimates, touched=None, risk=None) -> str:
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


def _draw_flag(ax, sim: OnlineNAMO, point, color: str, label: str):  # 绘制旗标
    x, y = point
    s = _FLAG_SCALE * max(sim.workspace.bounds[2], sim.workspace.bounds[3])
    top = y + s
    ax.plot([x, x], [y, top], color="black", lw=1.4,
            solid_capstyle="round", zorder=9)
    ax.fill([x, x + 0.62 * s, x],
            [top, top - 0.22 * s, top - 0.44 * s],
            facecolor=color, edgecolor="black", lw=0.8,
            zorder=9, label=label)
    ax.plot([x], [y], marker="o", color="black", ms=_FLAG_MARKER_MS, zorder=9)


def _draw_roadmap_bg(ax, sim: OnlineNAMO, cur_node: int | None = None):
    rm = sim.roadmap
    segs = [(rm.nodes[u], rm.nodes[v]) for u, v in rm.edge_len]
    ax.add_collection(LineCollection(segs, colors=_ROADMAP_EDGE, linewidths=0.2,
                                     zorder=0.3))
    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color=_ROADMAP_NODE, s=0.5, zorder=0.4)
    # 高亮当前路线图节点。
    if cur_node is not None and 0 <= cur_node < len(rm.nodes):
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color=_CURRENT_NODE, s=20, marker="s", zorder=4.7)


def _draw_plan_paths(ax, frame):  # 绘制当前帧的计划路径。
    labeled = set()
    for path in frame.get("plan_paths") or []:
        xs = [p[0] for p in path["pts"]]
        ys = [p[1] for p in path["pts"]]
        if path["kind"] == "route":
            # 绘制计划的路线图路径。
            label = None if "route" in labeled else "planned path"
            ax.plot(xs, ys, color=_ROUTE, lw=1.8, alpha=0.9, zorder=6,
                    solid_capstyle="round", label=label)
            ax.scatter(xs, ys, color=_ROUTE, s=6, zorder=6.1)
        elif path["kind"] == "contact":
            # 绘制计划的接触路径。
            label = None if "contact" in labeled else "planned contact"
            ax.plot(xs, ys, color=_CONTACT, lw=1.2, ls=":", alpha=0.9, zorder=6,
                    label=label)
        else:
            # 绘制障碍物计划的 SE(2) 路径。
            label = None if "obstacle" in labeled else "planned obstacle route"
            ax.plot(xs, ys, color=_OBSTACLE_ROUTE, lw=1.3, ls="--", alpha=0.85,
                    zorder=6, label=label)
        labeled.add(path["kind"])


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    # 绘制工作空间背景。
    _plot_poly(ax, sim.workspace, facecolor=_BACKGROUND, zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color=_WALL_EDGE, lw=1)
    # 绘制静态障碍物。
    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, facecolor=_WALL, edgecolor=_WALL_EDGE,
                   lw=0.5, zorder=1)
    _draw_flag(ax, sim, sim.start_point, _START, "start")
    _draw_flag(ax, sim, sim.goal_point, _GOAL, "goal")
    # 绘制障碍物初始轮廓以便对比。
    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color=_GHOST, lw=1, ls="--", alpha=0.8, zorder=2)


_PLOT_BOX = (8.0, 7.0)     # 图像区域最大尺寸（宽、高），单位为英寸
_MARGIN = 0.5              # 左右及底部刻度标签边距，单位为英寸
_TOP_PAD = 0.12            # 标题上方留白，单位为英寸
_TITLE_LINE = 0.30         # 每行标题高度，单位为英寸
_TITLE_LINES = 5           # 每个动画帧上方预留的高度
_LEGEND_H = 0.58           # 底部图例条高度，单位为英寸
_CBAR_H = 0.52             # 图例上方难度色条高度，单位为英寸
_TITLE_FS = 11             # 标题字号
_LEGEND_FS = 9              # 图例字号
_OBSTACLE_LABEL_FS = 8.5    # 障碍物标签字号
_COLORBAR_TICK_FS = 7       # 难度色条刻度字号
_COLORBAR_TITLE_FS = 7.5    # 难度色条标题字号
_FLAG_SCALE = 0.045         # 起点和终点旗标的相对大小
_FLAG_MARKER_MS = 3.0       # 旗杆底部圆点大小
_LEGEND_NCOL = 6


def _lay_out_title(groups, width_inch: float, max_lines: int = _TITLE_LINES) -> str:
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
                head = ""          # 标题居中，因此无需对齐缩进
                current = seg
        lines.append(head + current)
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(1, ncols - 1)] + "…"
    return "\n".join(lines)

def _new_canvas(sim: OnlineNAMO, title_lines: int = _TITLE_LINES):
    minx, miny, maxx, maxy = sim.workspace.bounds
    w, h = (maxx - minx) + 2.0, (maxy - miny) + 2.0   # 与下方 xlim/ylim 的边距一致
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
                         ncol=min(len(handles), _LEGEND_NCOL), fontsize=_LEGEND_FS,
                         framealpha=0.9, borderaxespad=0.0)

def _strategy_line(sim: OnlineNAMO) -> tuple:
    """返回标注本次运行策略的标题行，用于区分各 arm 的输出。"""
    if sim.cfg.shortest_path_mode:
        detail = ["baseline: shortest path, obstacle cost ignored"]
    else:
        detail = [f"cost={sim.estimator.mode}", f"risk={sim.risk.mode}"]
    return ("strategy", [sim.cfg.strategy] + detail)


def _summary_title(sim: OnlineNAMO, res) -> list:
    moved = (f"{len(res.removed)} obstacles" if len(res.removed) > 8
             else (str(res.removed) if res.removed else "none"))
    return [
        _strategy_line(sim),
        ("", [res.message]),
        ("cost", [f"J={res.J:,}",
                  f"lambda*D={res.walk_cost:,}",
                  f"W={res.work_cost:,}"]),
        ("time", [f"plan {res.plan_time:,g}s",
                  f"move {res.move_time:,g}s",
                  f"time_importance={sim.cfg.time_importance:g}"]),
        ("moved", [moved]),
    ]


def visualize(sim: OnlineNAMO, res, original_poses, out_path: str):
    title = _lay_out_title(_summary_title(sim, res),
                           _PLOT_BOX[0], max_lines=8)
    fig, ax = _new_canvas(sim, title_lines=title.count("\n") + 1)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)
    # 填充色表示真实难度，轮廓表示运动状态。
    colours, low, high = difficulty_palette(sim.world)
    world_moved = set(sim.dynamics.moved_on_own)
    for w in sim.world:
        edge, lw = _obstacle_edge(w.oid, w.removed, world_moved)
        _plot_poly(ax, w.polygon, facecolor=colours[w.oid], alpha=_OBSTACLE_ALPHA,
                   edgecolor=edge, lw=lw, zorder=3)
        colour = _label_colour(colours[w.oid])
        ax.text(w.x, w.y, _obstacle_label(w.oid, sim.estimator.cache,
                                           sim.belief.touched_difficulty,
                                           sim.risk.level),
                ha="center", va="center", fontsize=_OBSTACLE_LABEL_FS,
                linespacing=0.9,
                zorder=_TRAIL_Z + 1, color=colour,
                path_effects=_label_effects(colour))
    _draw_difficulty_key(fig, ax, colours, low, high)
    _draw_obstacle_key(ax, show_unperceived=False,
                       show_world_moved=bool(world_moved))

    # 绘制机器人运动轨迹。
    if len(res.robot_track) >= 2:
        corridor = LineString(res.robot_track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, corridor, facecolor=_TRAIL, alpha=0.45, zorder=_TRAIL_Z)
    elif res.robot_track:
        # 机器人未移动时绘制圆点。
        p = Point(res.robot_track[0]).buffer(sim.cfg.robot_radius)
        _plot_poly(ax, p, facecolor=_TRAIL, alpha=0.45, zorder=_TRAIL_Z)
    _finish_ax(ax, sim, title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_frame(sim: OnlineNAMO, frame, original_poses,
                 idx: int, total: int, cur_node: int | None = None):
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim, cur_node=cur_node)
    # 按感知状态绘制障碍物。
    colours, low, high = difficulty_palette(sim.world)
    perceived = frame["perceived"]
    world_moved = frame.get("world_moved", ())
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            # 未感知障碍物使用灰色点状轮廓。
            ax.plot(*poly.exterior.xy, color=_UNPERCEIVED, lw=1, ls=":",
                    alpha=0.6, zorder=3)
            continue
        # 将握持的障碍物绘制在机器人上方。
        z = 8.5 if oid == frame.get("move_oid") else 3
        edge, lw = _obstacle_edge(oid, removed, world_moved)
        fill = colours.get(oid, _OBSTACLE_EDGE)
        _plot_poly(ax, poly, facecolor=fill, alpha=_OBSTACLE_ALPHA,
                   edgecolor=edge, lw=lw, zorder=z)
        cx, cy = poly.centroid.x, poly.centroid.y
        estimates = frame.get("estimated_difficulty", {})
        touched = frame.get("touched_difficulty", {})
        colour = _label_colour(colours.get(oid, (1.0, 1.0, 1.0)))
        ax.text(cx, cy, _obstacle_label(oid, estimates, touched,
                                        frame.get("risk")),
                ha="center", va="center", fontsize=_OBSTACLE_LABEL_FS,
                linespacing=0.9,
                zorder=max(z + 1, _TRAIL_Z + 1), color=colour,
                path_effects=_label_effects(colour))

    # 绘制截至当前帧的轨迹。
    track = frame["track"]
    if len(track) >= 2:
        buf = LineString(track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, buf, facecolor=_TRAIL, alpha=0.4, zorder=_TRAIL_Z)

    # 绘制规划器提供的路径。
    _draw_plan_paths(ax, frame)

    # 绘制当前机器人位置。
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, facecolor=_ROBOT, alpha=0.9, zorder=7)
    ax.plot(rx, ry, marker="o", color=_ROBOT_CORE, ms=4, zorder=8, label="robot")
    _draw_difficulty_key(fig, ax, colours, low, high)
    _draw_obstacle_key(ax, show_unperceived=True,
                       show_world_moved=bool(world_moved))

    # 为每个帧预留相同的标题高度。
    title = _lay_out_title([
        _strategy_line(sim),
        ("", [f"step {idx}/{total - 1}", frame["label"]]),
        ("cost", [f"J={frame['J']:,}"]),
        ("time", [f"plan {frame.get('plan_t', 0.0):,g}s",
                  f"move {frame.get('move_t', 0.0):,g}s",
                  f"time_importance={sim.cfg.time_importance:g}"]),
    ], _PLOT_BOX[0])
    _finish_ax(ax, sim, title)
    return fig


def _shared_palette(buffers, colors: int = 255):
    picks = sorted({0, len(buffers) // 2, len(buffers) - 1})
    tiles = [Image.open(buffers[i]).convert("RGB") for i in picks]
    w, h = tiles[0].size
    montage = Image.new("RGB", (w, h * len(tiles)))
    for k, tile in enumerate(tiles):
        montage.paste(tile, (0, h * k))
    return montage.quantize(colors=colors, method=Image.MEDIANCUT)


def _time_sampled(frames, step: float, max_frames: int):
    times = [float(f.get("t", i)) for i, f in enumerate(frames)]
    span = times[-1] - times[0]
    if span <= 0.0:                       # 时钟未推进，保留已记录的帧。
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
    if picks[-1] != len(frames) - 1:      # 始终落在最终世界状态
        picks.append(len(frames) - 1)
    return picks, step


def render_sequence(sim: OnlineNAMO, res, original_poses, gif_path: str):
    cfg = sim.cfg
    total = len(res.frames)
    if total == 0:
        return 0, cfg.gif_time_step
    picks, step = _time_sampled(res.frames, cfg.gif_time_step, cfg.gif_max_frames)
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
    durations[-1] = max(step_ms, int(cfg.gif_end_hold_s * 1000))  # 在结果处暂停
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    return len(images), step
