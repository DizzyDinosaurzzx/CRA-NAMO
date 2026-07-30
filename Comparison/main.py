"""主程序入口 & 可视化模块"""

from __future__ import annotations
import argparse
import glob
import os
import textwrap
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from shapely.geometry import LineString, Point
import config
import scenarios
from planner import OnlineNAMO

def _plot_poly(ax, poly, **kw): #绘制基础图
    if poly.geom_type == "Polygon":
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kw)
    else:
        for g in poly.geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, **kw)


def _obstacle_label(oid: int, estimates) -> str:
    """障碍物 ID，以及规划器实际使用过的难度估计值。"""
    if oid not in estimates:
        return str(oid)
    return f"{oid}\nest={estimates[oid]:g}"


def _draw_flag(ax, sim: OnlineNAMO, point, color: str, label: str): #绘制旗帜
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


def _draw_roadmap_bg(ax, sim: OnlineNAMO, cur_node: int | None = None): #
    rm = sim.roadmap
    segs = [(rm.nodes[u], rm.nodes[v]) for u, v in rm.edge_len]
    ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.2,
                                     zorder=0.3))
    # 路网节点：银灰色散点图
    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color="silver", s=0.5, zorder=0.4)
    # 高亮当前所在节点（蓝色方块）；相邻可达节点不再单独绘制
    if cur_node is not None and 0 <= cur_node < len(rm.nodes):
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color="dodgerblue", s=20, marker="s", zorder=4.7)


def _draw_plan_paths(ax, frame): #绘制该帧时刻正在执行的规划路径（蓝色）
    labeled = set()
    for path in frame.get("plan_paths") or []:
        xs = [p[0] for p in path["pts"]]
        ys = [p[1] for p in path["pts"]]
        if path["kind"] == "move":
            # 机器人计划走的路线：蓝色实线 + 途经节点小圆点
            label = None if "move" in labeled else "planned path"
            ax.plot(xs, ys, color="blue", lw=1.8, alpha=0.9, zorder=6,
                    solid_capstyle="round", label=label)
            ax.scatter(xs, ys, color="blue", s=6, zorder=6.1)
        else:
            # 障碍物计划被推动的 SE2 路径：蓝色虚线
            label = None if "push" in labeled else "planned push"
            ax.plot(xs, ys, color="blue", lw=1.3, ls="--", alpha=0.75, zorder=6,
                    label=label)
        labeled.add(path["kind"])


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    # 工作空间背景
    _plot_poly(ax, sim.workspace, color="whitesmoke", zorder=0)
    ax.plot(*sim.workspace.exterior.xy, color="black", lw=1)
    # 静态障碍物（不会被移动的墙体等）
    for so in sim.static_obstacles:
        _plot_poly(ax, so.polygon, color="dimgray", zorder=1)
    # 起点（蓝色）和终点（红色）旗帜
    _draw_flag(ax, sim, sim.start_point, "royalblue", "start")
    _draw_flag(ax, sim, sim.goal_point, "red", "goal")
    # 可移动障碍物的原始位置（红色虚线），便于对比其最终位置
    for oid, poly in original_poses.items():
        ax.plot(*poly.exterior.xy, color="crimson", lw=1, ls="--", alpha=0.5, zorder=2)


# ---------------- 画布布局 ---------------- #
_PLOT_BOX = (8.0, 7.0)     # 绘图区最大 (宽, 高)，英寸
_MARGIN = 0.5              # 左右两侧 + 底部刻度标签的边距，英寸
_TOP_PAD = 0.12            # 标题上方的留白，英寸
_TITLE_LINE = 0.24         # 每行标题的高度，英寸
_TITLE_LINES = 2           # 标题固定按两行预留高度
_LEGEND_H = 0.5            # 底部图例条带高度，英寸
_TITLE_FS = 9              # 标题字号
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
    w, h = (maxx - minx) + 2.0, (maxy - miny) + 2.0   # 与下面 xlim/ylim 的外扩一致
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
    """生成仿真最终汇总图"""
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)
    # 遍历所有可移动障碍物，根据是否被移走使用不同颜色
    for w in sim.world:
        col = "orange" if w.removed else "crimson"
        _plot_poly(ax, w.polygon, color=col, alpha=0.6, zorder=3)
        # 在障碍物质心标注 ID 和规划器实际使用过的难度估计值
        ax.text(w.x, w.y, _obstacle_label(w.oid, sim.estimator.cache),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=4)

    # 绘制机器人运动轨迹走廊：
    if len(res.robot_track) >= 2:
        corridor = LineString(res.robot_track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, corridor, color="royalblue", alpha=0.3, zorder=5)
    elif res.robot_track:
        # 若只有单个点（机器人未移动），画一个圆形
        p = Point(res.robot_track[0]).buffer(sim.cfg.robot_radius)
        _plot_poly(ax, p, color="royalblue", alpha=0.3, zorder=5)
    # 组装标题信息。moved 在 maze_doors 这类图里可能有几十个 id，全列出来会把标题
    # 撑到几行开外，超长时只报数量。
    push_mode = "SE2" if sim.cfg.push_use_planner else "teleport"
    moved = (f"{len(res.removed)} obstacles" if len(res.removed) > 8
             else (str(res.removed) if res.removed else "none"))
    title = (f"{res.message}  |  J={res.J} "
             f"(lambda*D={res.walk_cost}, W={res.work_cost})  |  "
             f"moved={moved}  |  push={push_mode}  |  LLM={res.llm_mode}")
    _finish_ax(ax, sim, title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_frame(sim: OnlineNAMO, frame, original_poses, out_path: str,
                 idx: int, total: int, cur_node: int | None = None):
    """渲染仿真过程中的单帧画面"""
    fig, ax = _new_canvas(sim)
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim, cur_node=cur_node)
    # 按感知状态分类绘制障碍物
    perceived = frame["perceived"]
    estimates = frame.get("estimated_difficulty", {})
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            # 未被感知：灰色虚线，仅显示轮廓（机器人"不知道"这个障碍物在哪）
            ax.plot(*poly.exterior.xy, color="gray", lw=1, ls=":", alpha=0.5, zorder=3)
            continue
        # 已被感知：根据是否被移走着色
        col = "orange" if removed else "crimson"
        _plot_poly(ax, poly, color=col, alpha=0.6, zorder=3)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, _obstacle_label(oid, estimates),
                ha="center", va="center", fontsize=7, linespacing=0.9, zorder=4)

    # 绘制当前帧为止的机器人运动轨迹（半透明蓝色走廊）
    track = frame["track"]
    if len(track) >= 2:
        buf = LineString(track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, buf, color="royalblue", alpha=0.25, zorder=5)

    # 绘制该帧时刻规划器给出的全部路径（蓝线）
    _draw_plan_paths(ax, frame)

    # 绘制机器人当前位置（红色填充圆 + 深红色圆心点）
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, color="red", alpha=0.7, zorder=7)
    ax.plot(rx, ry, marker="o", color="darkred", ms=4, zorder=8, label="robot")

    # 标题：步数 / 总步数 | 操作标签 | 累计代价
    title = f"step {idx}/{total - 1}  |  {frame['label']}  |  J={frame['J']}"
    _finish_ax(ax, sim, title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_sequence(sim: OnlineNAMO, res, original_poses, frames_dir: str):
    os.makedirs(frames_dir, exist_ok=True) 
    for old in glob.glob(os.path.join(frames_dir, "step_*.png")):
        os.remove(old) # 清空旧图片
    total = len(res.frames)
    for i, frame in enumerate(res.frames):
        out = os.path.join(frames_dir, f"step_{i:03d}.png")
        render_frame(sim, frame, original_poses, out, i, total,
                     cur_node=frame.get("node"))
    return total


def create_gif(frames_dir: str, out_path: str, duration_ms: int = 150):
    """将按编号保存的过程帧合成为循环播放的 GIF。"""
    paths = sorted(glob.glob(os.path.join(frames_dir, "step_*.png")))
    if not paths:
        raise ValueError(f"没有可用于生成 GIF 的过程帧：{frames_dir}")

    frames = [Image.open(path).convert("RGB") for path in paths]
    try:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    finally:
        for frame in frames:
            frame.close()
    return len(paths)


# ---------------- main函数 ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO,
                    choices=scenarios.names())
    ap.add_argument("--lambda", "--lambda_distance", dest="lambda_distance",
                    type=float, default=None,
                    help="运动代价 λ 权重（越大越倾向推开障碍物而非绕路）")
    ap.add_argument("--no-llm-order", action="store_true",
                    help="禁用 LLM 对障碍物处理顺序的智能排序")
    ap.add_argument("--frames", action="store_true",
                    help="逐步保存机器人每一步运动的帧图片到 img/frames_<地图名>/")
    args = ap.parse_args()

    # 加载场景
    s = scenarios.load(args.scenario)
    cfg = s["cfg"]

    # 命令行覆盖配置文件中的 λ 值
    if args.lambda_distance is not None:
        try:
            cfg.lambda_distance = config.validate_lambda(args.lambda_distance)
        except ValueError as e:
            ap.error(str(e))

    # 命令行覆盖 LLM 排序开关
    if args.no_llm_order:
        cfg.use_llm_ordering = False

    # 命令行启用逐帧保存
    if args.frames:
        cfg.save_frames = True

    os.makedirs(cfg.out_dir, exist_ok=True)

    # 初始化CA-NAMO仿真器
    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal"], cfg)
    
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    print(f"Scenario: {s['name']}   {sim.roadmap}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
    print("-" * 60)

    res = sim.run()
    print("-" * 60) # 把运行期的 [push] 日志与下面的统计结果隔开

    # 打印仿真统计结果
    print(f"Success           : {res.success}   ({res.message})")
    print(f"Total cost J       : {res.J}")
    print(f"motion lambda*D  : {res.walk_cost}")   # λ × 运动距离
    print(f"obstacle work W  : {res.work_cost}")   # 推开障碍物的操作代价
    print(f"Obstacles moved    : {res.removed}")      # 被移动的障碍物总数
    print(f"Replan cycles      : {res.cycles}")       # 重规划次数
    print(f"Total plan time (s): {res.plan_time}")    # 总规划耗时
    print(f"A* expansions      : {res.total_expansions}")  # A* 搜索节点扩展总数
    print(f"LLM calls          : {res.llm_calls}  (mode={res.llm_mode})")  # LLM API 调用次数
    print(f"Difficulty cache   : oid_hits={sim.estimator.object_cache_hits}, "
          f"material_hits={sim.estimator.material_cache_hits}")
    print(f"Estimated difficulty ({res.llm_mode}):")
    if sim.estimator.cache:
        for oid, value in sorted(sim.estimator.cache.items()):
            obs = sim.belief.perceived.get(oid)
            material = obs.material if obs is not None else "unknown"
            density = sim.estimator.density_cache.get(oid)
            density_text = f"{density:g}" if density is not None else "n/a"
            source = sim.estimator.source_cache.get(oid, "unknown")
            print(f"  oid={oid:<3} material={material:<20} "
                  f"density={density_text:<9} estimate={value:g} "
                  f"source={source}")
    else:
        print("  (no difficulty estimate was requested by the planner)")

    # 渲染总体图
    out = os.path.join(cfg.out_dir, f"summary_{s['name']}.png")
    visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")

    # 渲染每一个运动帧
    if cfg.save_frames:
        frames_dir = os.path.join(cfg.out_dir, f"frames_{s['name']}")
        n = render_sequence(sim, res, original_poses, frames_dir)
        print(f"Saved {n} step frames -> {frames_dir}/step_000.png ... "
              f"step_{n - 1:03d}.png")
        if cfg.save_gif:
            gif_out = os.path.join(cfg.out_dir, f"{s['name']}.gif")
            create_gif(frames_dir, gif_out)
            print(f"Saved animated GIF -> {gif_out}")
    return res

if __name__ == "__main__":
    main()
