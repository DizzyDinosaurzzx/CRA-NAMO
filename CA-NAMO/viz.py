"""渲染汇总图与逐帧动画：只读取已完成的 RunResult 与执行器记录的帧快照，不参与仿真本身。"""

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
    """把单个闭合环转为 (顶点, 路径码)。"""
    verts = np.asarray(ring.coords)
    codes = np.full(len(verts), Path.LINETO, dtype=Path.code_type)
    codes[0] = Path.MOVETO
    return verts, codes


def _plot_poly(ax, poly, **kw):
    """填充 (Multi)Polygon（含洞）：所有环合成一条复合路径，由 Agg 按奇偶规则填充。
    只填 exterior 会把洞盖住——轨迹闭环时绕行的整片空地会被误涂成走过的区域。
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


# 估计仅在浮点误差内才算正确：难度是推导量(mu*rho×V×g)，认出材料即精确复现场景值，其余一律算错
_DIFF_MATCH_EPS = 1e-6


def _obstacle_label(oid: int, estimates, touched=None) -> str:
    """障碍物 ID、难度估计及其对错的标注：第二行为估计值，触碰获知真相后才有第三行。"""
    est = estimates.get(oid) if estimates else None
    true = touched.get(oid) if touched else None
    parts = [str(oid)]
    if est is not None:
        parts.append(f"est={est:g}")
    if true is not None:
        # 触碰早于估计：无对错可言
        if est is not None and abs(est - true) <= _DIFF_MATCH_EPS:
            parts.append("right")
        else:
            parts.append(f"real={true:g}")
    return "\n".join(parts)


# 加描边而非纯半透明填充：洞真正渲染成洞后边界携带信息，细边使围合空地读作未经过区域
_TRAIL_KW = dict(facecolor="royalblue", edgecolor="royalblue",
                 alpha=0.28, lw=0.6, zorder=5)


def _draw_trail(ax, track, radius: float):
    """机器人圆盘扫过的地面：整条轨迹一次 buffer 而非逐帧叠圆。

    走过五次与一次显示相同——回答"去过哪"而非"去过几次"；绕行未压过的地面留空。
    """
    if not track:
        return
    swept = (LineString(track).buffer(radius, cap_style=1) if len(track) >= 2
             else Point(track[0]).buffer(radius))
    _plot_poly(ax, swept, **_TRAIL_KW)


def _draw_flag(ax, sim: OnlineNAMO, point, color: str, label: str):  # 绘制旗帜标记
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
    # 路网节点：银灰色散点
    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color="silver", s=0.5, zorder=0.4)
    # 高亮当前节点（蓝色方块）；可达邻节点不单独绘制
    if cur_node is not None and 0 <= cur_node < len(rm.nodes):
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color="dodgerblue", s=20, marker="s", zorder=4.7)


def _draw_plan_paths(ax, frame):  # 绘制本帧正在执行的规划路径（蓝色）
    labeled = set()
    for path in frame.get("plan_paths") or []:
        xs = [p[0] for p in path["pts"]]
        ys = [p[1] for p in path["pts"]]
        if path["kind"] == "route":
            # 机器人规划路线：蓝色实线 + 途经节点小圆点
            label = None if "route" in labeled else "planned path"
            ax.plot(xs, ys, color="blue", lw=1.8, alpha=0.9, zorder=6,
                    solid_capstyle="round", label=label)
            ax.scatter(xs, ys, color="blue", s=6, zorder=6.1)
        elif path["kind"] == "contact":
            # 机器人搬运障碍物时的贴身抓持路径
            label = None if "contact" in labeled else "planned contact"
            ax.plot(xs, ys, color="darkgreen", lw=1.1, ls=":", alpha=0.85, zorder=6,
                    label=label)
        else:
            # 障碍物自身 SE2 路线：蓝色虚线
            label = None if "obstacle" in labeled else "planned obstacle route"
            ax.plot(xs, ys, color="blue", lw=1.3, ls="--", alpha=0.75, zorder=6,
                    label=label)
        labeled.add(path["kind"])


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    # 工作区背景
    _plot_poly(ax, sim.workspace, color="whitesmoke", zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color="black", lw=1)
    # 静态障碍物（墙等，永不移动）
    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, color="dimgray", zorder=1)
    # 起点(蓝)与终点(红)旗帜
    _draw_flag(ax, sim, sim.start_point, "royalblue", "start")
    _draw_flag(ax, sim, sim.goal_point, "red", "goal")
    # 可移动障碍物初始位置（红虚线），与最终位置对比
    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color="crimson", lw=1, ls="--", alpha=0.5, zorder=2)


# --- 画布布局 ---
_PLOT_BOX = (8.0, 7.0)     # 最大绘图区（宽, 高），英寸
_MARGIN = 0.5              # 左右及底部刻度标签边距，英寸
_TOP_PAD = 0.12            # 标题上方留白，英寸
_TITLE_LINE = 0.24         # 每行标题高度，英寸
_TITLE_LINES = 3           # 始终为三行标题预留高度
_LEGEND_H = 0.5            # 底部图例条高度，英寸
_TITLE_FS = 9              # 标题字号
_LEGEND_NCOL = 5


# --- 标题栏 ---
# 汇总图与每帧动画共用同样三行结构：1) 运行配置（全程不变） 2) 当前进度或最终结果
# 3) 代价（字段一致，帧内为累计值）。仅属于其中一个视图的内容只放在第 2 行。
def _mode_tag(sim: OnlineNAMO) -> str:
    cfg = sim.cfg
    return "[" + " | ".join((
        cfg.strategy,
        f"w={cfg.time_importance:g}",
        sim.estimator.mode,
    )) + "]"


def _cost_line(J, walk, work, motion_s, plan_s) -> str:
    return (f"J {J:,.1f} = λD {walk:,.1f} + W {work:,.1f}"
            f"  |  move {motion_s:,.1f}s & plan {plan_s:,.1f}s")


def _wrap_title(lines, width_inch: float) -> str:
    """把标题压进预留块，一行输入对应一行输出；超长截断省略而非换行，以免代价行被挤出块外。"""
    ncols = max(20, int(width_inch / (0.55 * _TITLE_FS / 72.0)))
    out = []
    for line in list(lines)[:_TITLE_LINES]:
        if len(line) > ncols:
            line = line[:max(1, ncols - 1)] + "…"
        out.append(line)
    return "\n".join(out)

def _new_canvas(sim: OnlineNAMO):
    minx, miny, maxx, maxy = sim.workspace.bounds
    w, h = (maxx - minx) + 2.0, (maxy - miny) + 2.0   # 与下方 xlim/ylim 留白一致
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
    """生成仿真的最终汇总图。"""
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)
    # 遍历所有可移动障碍物，按是否被移动着色
    for w in sim.world:
        col = "orange" if w.removed else "crimson"
        _plot_poly(ax, w.polygon, color=col, alpha=0.6, zorder=3)
        # 在障碍物质心标注 ID、估计难度及已知真实难度
        ax.text(w.x, w.y, _obstacle_label(w.oid, sim.estimator.cache,
                                           sim.belief.touched_difficulty),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=4)

    _draw_trail(ax, res.robot_track, sim.cfg.robot_radius)
    # 哪些障碍物被移动图上已可见——即橙色者，旁有其初始位置虚线轮廓
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
    """渲染单帧仿真画面；返回 figure，由调用方关闭。"""
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim, cur_node=cur_node)
    # 按感知状态分类绘制障碍物
    perceived = frame["perceived"]
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            # 未感知：仅灰色点线轮廓（机器人不知其位置）
            ax.plot(*poly.exterior.xy, color="gray", lw=1, ls=":", alpha=0.5, zorder=3)
            continue
        # 已感知：按是否被移动着色
        col = "orange" if removed else "crimson"
        # 被搬运障碍物画在机器人之上，经过机器人身侧的部件不被遮挡；碰撞时机器人在上层
        z = 8.5 if oid == frame.get("move_oid") else 3
        _plot_poly(ax, poly, color=col, alpha=0.6, zorder=z)
        cx, cy = poly.centroid.x, poly.centroid.y
        estimates = frame.get("estimated_difficulty", {})
        touched = frame.get("touched_difficulty", {})
        ax.text(cx, cy, _obstacle_label(oid, estimates, touched),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=z + 1)

    # 截至本帧的轨迹——与汇总图同法绘制
    _draw_trail(ax, frame["track"], sim.cfg.robot_radius)

    # 绘制本帧规划器给出的所有路径（蓝线）
    _draw_plan_paths(ax, frame)

    # 绘制机器人当前位置（绿色实心圆 + 深绿圆心点）
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, color="limegreen", alpha=0.85, zorder=7)
    ax.plot(rx, ry, marker="o", color="darkgreen", ms=4, zorder=8, label="robot")

    # 与汇总图相同的三行标题；第 3 行为本帧累计值
    _finish_ax(ax, sim, (
        _mode_tag(sim),
        f"step {idx}/{total - 1}  —  {frame['label']}",
        _cost_line(frame["J"], frame.get("walk_cost", 0.0),
                   frame.get("work_cost", 0.0), frame.get("motion_time", 0.0),
                   frame.get("plan_time", 0.0)),
    ))
    return fig


def _shared_palette(buffers, colors: int = 255):
    """全动画共用一张调色板（取首/中/末帧采样）：Pillow 只存帧间变化像素，否则逐帧重绘的路网背景会撑大文件。"""
    picks = sorted({0, len(buffers) // 2, len(buffers) - 1})
    tiles = [Image.open(buffers[i]).convert("RGB") for i in picks]
    w, h = tiles[0].size
    montage = Image.new("RGB", (w, h * len(tiles)))
    for k, tile in enumerate(tiles):
        montage.paste(tile, (0, h * k))
    return montage.quantize(colors=colors, method=Image.MEDIANCUT)


def _frame_durations(frames: list, cfg) -> list:
    """按相邻帧 motion_time(仿真物理耗时，不含 plan_time——思考时间只衡量算法，
    不代表世界在动，见 executor._tick) 差值折算每帧停留的毫秒数；gif_speed 是相对
    仿真时间的播放倍率，gif_fps/gif_max_frame_s 分别给单帧时长设下限与上限，
    避免几毫秒的子步一闪而过、或一次缓慢搬运把动画卡住半天。"""
    n = len(frames)
    min_ms = max(20, int(round(1000.0 / cfg.gif_fps)))
    max_ms = max(min_ms, int(round(cfg.gif_max_frame_s * 1000)))
    speed = max(cfg.gif_speed, 1e-6)
    out = []
    for i in range(n):
        dt = (max(0.0, frames[i + 1]["motion_time"] - frames[i]["motion_time"])
              if i + 1 < n else 0.0)
        ms = int(round(dt / speed * 1000.0))
        out.append(min(max(ms, min_ms), max_ms))
    out[-1] = max(out[-1], int(cfg.gif_end_hold_s * 1000))
    return out


def render_sequence(sim: OnlineNAMO, res, original_poses, gif_path: str):
    """将全部运动帧渲染为一个 GIF 动画。"""
    cfg = sim.cfg
    total = len(res.frames)
    if total == 0:
        return 0
    # 帧保存为 PNG 编码字节而非解码位图，否则大地图长跑会同时驻留数百 MB 像素
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
    durations = _frame_durations(res.frames, cfg)
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    return total


