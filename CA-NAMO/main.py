"""
main.py – 主程序入口 & 可视化模块
=====================================

本文件包含两个部分：
1. **可视化函数**：将仿真结果渲染为 matplotlib 图片，包括：
   - 最终汇总图（完整路径 + 障碍物最终状态 + 路网 + 机器人扫过区域）
   - 逐帧动画帧（机器人分步移动过程，含感知状态变化）
2. **main() 入口**：解析命令行参数、加载场景、运行 OnlineNAMO 仿真、输出统计结果。

可视化图例：
-----------
- 蓝色三角旗 → 起点 (start)
- 红色三角旗 → 终点 (goal)
- 深灰色实心多边形 → 静态障碍物（墙体等不可移动物体）
- 红色实心多边形 → 可移动障碍物，未被移开
- 橙色实心多边形 → 可移动障碍物，已被机器人推开/移走
- 红色虚线轮廓 → 可移动障碍物的原始位置（被移开后仍保留虚线标记）
- 浅灰色细线 & 银灰色点 → 概率路网 (PRM roadmap) 的边和节点
- 蓝色高亮边/点 → 当前时刻机器人所在路网节点及其邻居
- 半透明蓝色条带 → 机器人走过的轨迹（以机器人半径为宽度的走廊）
- 红色圆 + 深红圆心 → 机器人当前位置
- 灰色虚线轮廓 → 尚未被机器人感知到的障碍物（仅逐帧动画中出现）
"""

from __future__ import annotations
import argparse
import glob
import os
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免在没有 GUI 的服务器上报错
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Point
import config
import scenarios
from planner import OnlineNAMO


# ═══════════════════════════════════════════════════════════════════════════════
# 底层绘图辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_poly(ax, poly, **kw):
    """在指定 Axes 上绘制 Shapely 多边形。

    同时支持单多边形 (Polygon) 和多边形集合 (MultiPolygon)，
    均使用 ax.fill 填充颜色。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标子图。
    poly : shapely.geometry.Polygon | shapely.geometry.MultiPolygon
        要绘制的多边形对象。
    **kw : dict
        传递给 ax.fill 的样式参数（如 color, alpha, zorder 等）。
    """
    if poly.geom_type == "Polygon":
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kw)
    else:
        for g in poly.geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, **kw)


def _draw_flag(ax, sim: OnlineNAMO, point, color: str, label: str):
    """绘制起点或终点的三角旗标志。

    旗帜由三部分组成：
    - 一根竖直旗杆（黑色线段）
    - 一面三角形旗帜（填充颜色）
    - 底部一个圆点标记精确位置

    旗帜大小随工作空间自动缩放（取 workspace 长宽中较大者的 6%）。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标子图。
    sim : OnlineNAMO
        仿真对象，用于获取 workspace 尺寸。
    point : (float, float)
        旗杆底部的坐标 (x, y)。
    color : str
        旗帜填充颜色。起点用 "royalblue"，终点用 "red"。
    label : str
        matplotlib 图例标签。起点为 "start"，终点为 "goal"。
    """
    x, y = point
    # 旗帜高度 = workspace 最大边长的 6%，保证在不同地图中视觉比例一致
    s = 0.06 * max(sim.workspace.bounds[2], sim.workspace.bounds[3])
    top = y + s

    # 旗杆：从 (x, y) 到 (x, top) 的黑色竖线
    ax.plot([x, x], [y, top], color="black", lw=1.4,
            solid_capstyle="round", zorder=9)
    # 三角旗面：向右展开的三角形
    ax.fill([x, x + 0.62 * s, x],
            [top, top - 0.22 * s, top - 0.44 * s],
            facecolor=color, edgecolor="black", lw=0.8,
            zorder=9, label=label)
    # 底部圆点：标记精确的起点/终点位置
    ax.plot([x], [y], marker="o", color="black", ms=3.5, zorder=9)


def _draw_roadmap_bg(ax, sim: OnlineNAMO, cur_node: int | None = None):
    """绘制概率路网 (PRM roadmap) 作为背景。

    路网是规划器在自由空间中采样的可达图，用于路径搜索。
    节点 = 无碰撞采样点，边 = 两节点间的直线可行路径。

    绘制内容：
    - 所有边：浅灰色细线（用 LineCollection 批量绘制，避免逐条 ax.plot 导致渲染极慢）
    - 所有节点：银灰色小散点
    - 若指定 cur_node：高亮该节点（蓝色方块）及其邻居节点和连接边（蓝色）

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标子图。
    sim : OnlineNAMO
        仿真对象，包含 roadmap 属性。
    cur_node : int | None
        当前机器人所在的路网节点编号。为 None 时不绘制高亮。
    """
    rm = sim.roadmap

    # 细化后的路网可能包含几万条边，使用 LineCollection 批量绘制
    # 比逐条调用 ax.plot 快约两个数量级，对逐帧渲染至关重要
    segs = [(rm.nodes[u], rm.nodes[v]) for u, v in rm.edge_len]
    ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.2,
                                     zorder=0.3))

    # 路网节点：银灰色散点图
    xs = [p[0] for p in rm.nodes]
    ys = [p[1] for p in rm.nodes]
    ax.scatter(xs, ys, color="silver", s=0.5, zorder=0.4)

    # 如果指定了当前所在节点，高亮显示该节点及其邻居
    if cur_node is not None and cur_node in rm.adj:
        # 高亮当前节点到各邻居的边（蓝色，略粗）
        nbr_segs = [(rm.nodes[cur_node], rm.nodes[v]) for v in rm.adj[cur_node]]
        ax.add_collection(LineCollection(nbr_segs, colors="dodgerblue",
                                         linewidths=1.2, alpha=0.7, zorder=4.5))
        # 高亮邻居节点（蓝色散点）
        nb_xs = [rm.nodes[v][0] for v in rm.adj[cur_node]]
        nb_ys = [rm.nodes[v][1] for v in rm.adj[cur_node]]
        ax.scatter(nb_xs, nb_ys, color="dodgerblue", s=8, alpha=0.7, zorder=4.6)
        # 高亮当前节点（蓝色方块，更大）
        hx, hy = rm.nodes[cur_node]
        ax.scatter([hx], [hy], color="dodgerblue", s=20, marker="s", zorder=4.7)


def _draw_static(ax, sim: OnlineNAMO, original_poses):
    """绘制场景中不随仿真步骤变化的静态元素。

    包括：
    - 工作空间边界（白色填充 + 黑色边框）
    - 静态障碍物（深灰色填充，如墙体）
    - 起点和终点旗帜
    - 可移动障碍物的原始位置（红色虚线轮廓，用于对比被推走后的位置）

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标子图。
    sim : OnlineNAMO
        仿真对象。
    original_poses : dict[int, Polygon]
        {障碍物ID: 初始位置多边形}，用于绘制虚线原始轮廓。
    """
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


def _finish_ax(ax, sim: OnlineNAMO, title: str):
    """统一设置坐标轴属性：等比例、边界、标题和图例。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标子图。
    sim : OnlineNAMO
        仿真对象，用于获取 workspace 边界。
    title : str
        图片标题（包含仿真统计信息）。
    """
    ax.set_aspect("equal")
    ax.set_xlim(-1, sim.workspace.bounds[2] + 1)
    ax.set_ylim(-1, sim.workspace.bounds[3] + 1)
    ax.set_title(title, fontsize=10)
    # 图例放在图外右侧，避免遮挡地图内容
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0, framealpha=0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# 高层可视化函数
# ═══════════════════════════════════════════════════════════════════════════════

def visualize(sim: OnlineNAMO, res, original_poses, out_path: str):
    """生成仿真最终汇总图。

    汇总图展示的是仿真结束后的全局状态，一张图即可概览整个导航过程：
    - 静态背景（工作空间、静态障碍物、起点/终点旗帜）
    - 路网结构（所有节点和边）
    - 所有可移动障碍物的最终状态：
      * 红色 = 仍在原位（未被移动）
      * 橙色 = 已被机器人推开/移走
      * 障碍物上标注数字 ID
    - 机器人完整运动轨迹（以 robot_radius 为半径的蓝色半透明走廊）
    - 标题栏显示：结果状态、总代价 J、运动代价 λ·D、操作代价 W、
      被移动障碍物数、推动模式（SE2物理推动 vs teleport瞬移）、LLM 模式

    Parameters
    ----------
    sim : OnlineNAMO
        已运行完的仿真对象。
    res : Result
        sim.run() 返回的结果对象，包含轨迹、帧序列、代价等。
    original_poses : dict[int, Polygon]
        障碍物初始位置映射。
    out_path : str
        输出图片的保存路径（如 img/summary_maze.png）。
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim)

    # 遍历所有可移动障碍物，根据是否被移走使用不同颜色
    for w in sim.world:
        col = "orange" if w.removed else "crimson"
        _plot_poly(ax, w.polygon, color=col, alpha=0.6, zorder=3)
        # 在障碍物质心标注其 ID 编号
        ax.text(w.x, w.y, str(w.oid), ha="center", va="center", fontsize=8, zorder=4)

    # 绘制机器人运动轨迹走廊：
    # 将轨迹点连成折线，以 robot_radius 做 buffer 膨胀，
    # 得到机器人实际扫过的区域（含体积）
    if len(res.robot_track) >= 2:
        corridor = LineString(res.robot_track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, corridor, color="royalblue", alpha=0.3, zorder=5)
    elif res.robot_track:
        # 若只有单个点（机器人未移动），画一个圆形
        p = Point(res.robot_track[0]).buffer(sim.cfg.robot_radius)
        _plot_poly(ax, p, color="royalblue", alpha=0.3, zorder=5)

    # 组装标题信息
    push_mode = "SE2" if sim.cfg.push_use_planner else "teleport"
    title = (f"{res.message}  |  J={res.J} "
             f"(lambda*D={res.walk_cost}, W={res.work_cost})  |  "
             f"moved={res.removed}  |  push={push_mode}  |  LLM={res.llm_mode}")
    _finish_ax(ax, sim, title)
    # rect 参数为右侧图例留出 15% 的空白
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_frame(sim: OnlineNAMO, frame, original_poses, out_path: str,
                 idx: int, total: int, cur_node: int | None = None):
    """渲染仿真过程中的单帧画面。

    与 visualize() 的全局汇总不同，本函数渲染的是某一时刻的瞬时状态，
    用于生成逐帧动画。主要区别：
    - 只显示到当前帧为止的轨迹（而非完整轨迹）
    - 障碍物按感知状态分类显示：
      * 灰色虚线轮廓 = 尚未被机器人感知到（传感器范围外）
      * 红色/橙色 = 已被感知（颜色含义同 visualize）
    - 绘制机器人当前位置（红色圆）
    - 标题显示步数、当前操作标签、累计代价

    Parameters
    ----------
    sim : OnlineNAMO
        仿真对象。
    frame : dict
        单帧数据，包含 robot（位置）、track（轨迹点列表）、
        obstacles（障碍物列表）、perceived（已感知 ID 集合）、
        label（操作描述）、J（累计代价）。
    original_poses : dict[int, Polygon]
        障碍物初始位置映射。
    out_path : str
        输出图片路径。
    idx : int
        当前帧序号（从 0 开始）。
    total : int
        总帧数。
    cur_node : int | None
        当前路网节点编号，用于高亮。
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_static(ax, sim, original_poses)
    _draw_roadmap_bg(ax, sim, cur_node=cur_node)

    # 按感知状态分类绘制障碍物
    perceived = frame["perceived"]
    for oid, poly, removed in frame["obstacles"]:
        if oid not in perceived:
            # 未被感知：灰色虚线，仅显示轮廓（机器人"不知道"这个障碍物在哪）
            ax.plot(*poly.exterior.xy, color="gray", lw=1, ls=":", alpha=0.5, zorder=3)
            continue
        # 已被感知：根据是否被移走着色
        col = "orange" if removed else "crimson"
        _plot_poly(ax, poly, color=col, alpha=0.6, zorder=3)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, str(oid), ha="center", va="center", fontsize=8, zorder=4)

    # 绘制当前帧为止的机器人运动轨迹（半透明蓝色走廊）
    track = frame["track"]
    if len(track) >= 2:
        buf = LineString(track).buffer(sim.cfg.robot_radius, cap_style=1)
        _plot_poly(ax, buf, color="royalblue", alpha=0.25, zorder=5)

    # 绘制机器人当前位置（红色填充圆 + 深红色圆心点）
    rx, ry = frame["robot"]
    robot_circle = Point(rx, ry).buffer(sim.cfg.robot_radius)
    _plot_poly(ax, robot_circle, color="red", alpha=0.7, zorder=7)
    ax.plot(rx, ry, marker="o", color="darkred", ms=4, zorder=8, label="robot")

    # 标题：步数 / 总步数 | 操作标签 | 累计代价
    title = f"step {idx}/{total - 1}  |  {frame['label']}  |  J={frame['J']}"
    _finish_ax(ax, sim, title)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_sequence(sim: OnlineNAMO, res, original_poses, frames_dir: str):
    """批量渲染仿真过程中所有帧，生成逐帧动画图片序列。

    先清空目标目录中的旧帧图片，然后遍历 res.frames 逐帧调用 render_frame。
    输出的图片序列可以用 ffmpeg 等工具合成为视频。

    Parameters
    ----------
    sim : OnlineNAMO
        仿真对象。
    res : Result
        仿真结果，包含 frames 列表。
    original_poses : dict[int, Polygon]
        障碍物初始位置映射。
    frames_dir : str
        帧图片输出目录（如 img/frames_maze/）。

    Returns
    -------
    total : int
        渲染的总帧数。
    """
    os.makedirs(frames_dir, exist_ok=True)
    # 清空旧的帧图片，避免残留上轮仿真结果
    for old in glob.glob(os.path.join(frames_dir, "step_*.png")):
        os.remove(old)
    total = len(res.frames)
    for i, frame in enumerate(res.frames):
        out = os.path.join(frames_dir, f"step_{i:03d}.png")
        render_frame(sim, frame, original_poses, out, i, total,
                     cur_node=frame.get("node"))
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """程序主入口。

    执行流程：
    1. 解析命令行参数（场景名、λ 权重、是否禁用 LLM 排序、是否保存逐帧图片）
    2. 加载场景配置（工作空间、障碍物布局、起点/终点、算法参数）
    3. 初始化 OnlineNAMO 仿真器
    4. 运行仿真，在线规划 + 执行
    5. 打印统计结果（成功与否、总代价、规划时间、A* 扩展次数等）
    6. 保存最终汇总图（summary_*.png）
    7. 若启用 --frames，保存逐帧动画图片序列

    命令行参数
    ----------
    --scenario : str
        场景名称，对应 scenarios 模块中定义的场景。
    --lambda / --lambda_distance : float
        运动代价 λ 权重。λ 越大，机器人越倾向于推开障碍物而非绕远路。
    --no-llm-order : flag
        禁用 LLM 对障碍物处理顺序的智能排序，改用启发式规则。
    --frames : flag
        启用后会在 img/frames_<地图名>/ 目录下保存每一步的帧图片。
    """
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

    # 加载场景：包含 workspace、静态/可移动障碍物、起点终点、配置
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

    # 确保输出目录存在（如 img/）
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 初始化在线 NAMO 仿真器
    # OnlineNAMO 整合了：路网构建、路径搜索、障碍物感知与推动、重规划
    sim = OnlineNAMO(s["workspace"], s["static"], s["movable"],
                     s["start"], s["goal"], cfg)
    # 记录障碍物的原始位置（用于可视化中画虚线轮廓）
    original_poses = {w.oid: w.polygon for w in s["movable"]}

    # 打印场景基本信息
    print(f"Scenario: {s['name']}   {sim.roadmap}")
    print(f"Difficulty estimator: {sim.estimator.mode}"
          + ("" if sim.estimator.mode == "heuristic" else " (DeepSeek)"))
    print("-" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 核心：运行在线 NAMO 仿真
    # 仿真循环：规划路径 → 沿路径移动 → 遇阻则感知障碍物 →
    #           决定推开或绕路 → 重规划 → 重复直到到达目标或失败
    # ═══════════════════════════════════════════════════════════════
    res = sim.run()

    # 打印仿真统计结果
    print(f"Success           : {res.success}   ({res.message})")
    print(f"Total cost J       : {res.J}")
    print(f"  motion lambda*D  : {res.walk_cost}")   # λ × 运动距离
    print(f"  obstacle work W  : {res.work_cost}")   # 推开障碍物的操作代价
    print(f"Obstacles moved    : {res.removed}")      # 被移动的障碍物总数
    print(f"Replan cycles      : {res.cycles}")       # 重规划次数
    print(f"First-plan time (s): {round(res.first_plan_time, 4)}")  # 首次规划耗时
    print(f"Total plan time (s): {res.plan_time}")    # 总规划耗时
    print(f"A* expansions      : {res.total_expansions}")  # A* 搜索节点扩展总数
    print(f"LLM calls          : {res.llm_calls}  (mode={res.llm_mode})")  # LLM API 调用次数

    # 保存最终汇总图：完整路径 + 障碍物最终状态 + 路网 + 代价信息
    out = os.path.join(cfg.out_dir, f"summary_{s['name']}.png")
    visualize(sim, res, original_poses, out)
    print(f"\nSaved visualisation -> {out}")

    # 如果启用逐帧保存，渲染每一步的动画帧
    if cfg.save_frames:
        frames_dir = os.path.join(cfg.out_dir, f"frames_{s['name']}")
        n = render_sequence(sim, res, original_poses, frames_dir)
        print(f"Saved {n} step frames -> {frames_dir}/step_000.png ... "
              f"step_{n - 1:03d}.png")
    return res


if __name__ == "__main__":
    main()
