"""Generate the CRA-NAMO core planning pipeline diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")

BG = "#F7F9FC"
INK = "#172033"
MUTED = "#536078"
LINE = "#AAB8CC"
BLUE = "#2F6FED"
BLUE_BG = "#EAF2FF"
PURPLE = "#7357D9"
PURPLE_BG = "#F1EDFF"
ORANGE = "#D86D1F"
ORANGE_BG = "#FFF0E4"
GREEN = "#16865C"
GREEN_BG = "#E7F7EF"
RED = "#C44949"
RED_BG = "#FDEDEC"
WHITE = "#FFFFFF"


if FONT_PATH.exists():
    mpl.font_manager.fontManager.addfont(str(FONT_PATH))
    FONT = mpl.font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT = "sans-serif"

mpl.rcParams.update({
    "font.family": FONT,
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
})


def rounded_box(ax, x, y, w, h, face, edge, title, lines=(), *,
                title_color=INK, tag=None, title_size=14.5, body_size=11.2,
                radius=1.3, linewidth=1.35, zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.32,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + 1.25, y + h - 1.55, title, ha="left", va="top",
            fontsize=title_size, fontweight="bold", color=title_color,
            zorder=zorder + 1)
    if tag:
        ax.text(x + w - 1.1, y + 0.72, tag, ha="right", va="bottom",
                fontsize=8.1, color=MUTED, zorder=zorder + 1)
    if lines:
        top = y + h - 4.55
        gap = 2.08 if len(lines) <= 3 else 1.78
        for i, line in enumerate(lines):
            ax.text(x + 1.3, top - i * gap, line, ha="left", va="top",
                    fontsize=body_size, color=INK, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, *, color=LINE, width=1.8, style="-|>",
          mutation=13, connection="arc3", zorder=2):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=width, color=color, connectionstyle=connection,
        shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def pill(ax, x, y, w, h, text, face, edge, color=INK, size=10.5):
    rounded_box(ax, x, y, w, h, face, edge, "", (), radius=h / 2.2,
                linewidth=1.1, zorder=4)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, fontweight="bold", color=color, zorder=5)


def diamond(ax, cx, cy, w, h, text):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=RED_BG,
                         edgecolor=RED, linewidth=1.35, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.2,
            fontweight="bold", color=RED, zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(BG)

    ax.text(3, 96.5, "CRA-NAMO 路径规划核心 Pipeline", ha="left", va="top",
            fontsize=25, fontweight="bold", color=INK)
    ax.text(3, 92.4,
            "Belief 驱动的代价/风险感知、接触可行 Online NAMO；执行中持续感知并触发响应式重规划",
            ha="left", va="top", fontsize=12.5, color=MUTED)

    # Main online pipeline.
    boxes = {
        "input": (3, 75, 15.5, 13),
        "roadmap": (21.5, 75, 16.5, 13),
        "belief": (41, 75, 17, 13),
        "search": (61, 75, 16.5, 13),
        "execute": (80.5, 75, 16.5, 13),
    }
    for a, b in zip(("input", "roadmap", "belief", "search"),
                    ("roadmap", "belief", "search", "execute")):
        x, y, w, h = boxes[a]
        nx, ny, nw, nh = boxes[b]
        arrow(ax, (x + w + 0.5, y + h / 2), (nx - 0.5, ny + nh / 2),
              color=BLUE if a in ("input", "roadmap", "belief") else PURPLE)

    rounded_box(ax, *boxes["input"], BLUE_BG, BLUE, "1  场景输入", (
        "workspace + 静态障碍物",
        "真实可移动障碍物",
        "起点 start → 目标 goal",
    ), tag="main.py", title_size=13.8, body_size=10.4)
    rounded_box(ax, *boxes["roadmap"], BLUE_BG, BLUE, "2  静态路线图", (
        "自由空间按机器人半径腐蚀",
        "均匀采样 + 邻域连边",
        "每条边扩成通行 corridor",
    ), tag="roadmap.py", title_size=13.8, body_size=10.25)
    rounded_box(ax, *boxes["belief"], BLUE_BG, BLUE, "3  局部感知与 Belief", (
        "视距 / 遮挡 / 接触观测",
        "障碍物 → 被阻挡的边",
        "难度估计 + 语义风险",
    ), tag="perception.py · risk.py", title_size=13.6, body_size=10.25)
    rounded_box(ax, *boxes["search"], PURPLE_BG, PURPLE, "4  联合路线规划", (
        "Roadmap 上 A* / 最佳优先",
        "逐边比较 MOVE / WAIT",
        "与 REMOVE → 有序动作序列",
    ), tag="search.py", title_color=PURPLE, title_size=13.6,
                body_size=10.15)
    rounded_box(ax, *boxes["execute"], GREEN_BG, GREEN, "5  分段执行", (
        "MOVE：沿路线图行驶",
        "REMOVE：接近、伴随搬移、释放",
        "WAIT：推进动态世界",
    ), tag="executor.py", title_color=GREEN, title_size=13.8,
                body_size=10.05)

    # Detailed planning panel.
    panel = FancyBboxPatch(
        (3, 18.5), 72.5, 50.5,
        boxstyle="round,pad=0.4,rounding_size=1.6",
        facecolor=WHITE, edgecolor="#D8E0EC", linewidth=1.2, zorder=0,
    )
    ax.add_patch(panel)
    ax.text(5, 66.7, "阻挡边的 REMOVE 候选如何生成", ha="left", va="top",
            fontsize=16.5, fontweight="bold", color=INK)
    ax.text(73, 66.7, "候选代价回填为路线图边权", ha="right", va="top",
            fontsize=10.2, color=PURPLE)

    # Link global search to the detail panel and the scored candidate back up.
    arrow(ax, (69.25, 74.4), (69.25, 69.7), color=PURPLE, width=1.7)
    arrow(ax, (72.6, 62.2), (74.5, 74.3), color=PURPLE, width=1.5,
          connection="arc3,rad=-0.16")

    detail_boxes = {
        "edge": (5, 49, 14.5, 12.5),
        "filter": (22.5, 49, 14.5, 12.5),
        "se2": (40, 49, 15.5, 12.5),
        "contact": (58.5, 49, 14.5, 12.5),
    }
    for a, b in zip(("edge", "filter", "se2"), ("filter", "se2", "contact")):
        x, y, w, h = detail_boxes[a]
        nx, ny, nw, nh = detail_boxes[b]
        arrow(ax, (x + w + 0.35, y + h / 2), (nx - 0.35, ny + nh / 2),
              color=ORANGE, width=1.55, mutation=11)

    rounded_box(ax, *detail_boxes["edge"], PURPLE_BG, PURPLE, "A  边状态分解", (
        "畅通 → MOVE",
        "已观测运动 → WAIT",
        "可移动阻挡 → REMOVE",
    ), tag="_edge_cost", title_size=12.7, body_size=10.1,
                title_color=PURPLE)
    rounded_box(ax, *detail_boxes["filter"], ORANGE_BG, ORANGE, "B  可搬移性过滤", (
        "高风险禁搬策略",
        "最大推力 / 倾覆约束",
        "已失败姿态与边缓存",
    ), tag="_off_limits", title_size=12.7, body_size=10.1,
                title_color=ORANGE)
    rounded_box(ax, *detail_boxes["se2"], ORANGE_BG, ORANGE, "C  障碍物 SE(2) 搜索", (
        "离散配置空间 (x, y, θ)",
        "分桶 Dijkstra：平移 + 旋转",
        "清空 corridor 且扫掠体无碰撞",
    ), tag="se2_planner.py", title_size=12.3, body_size=9.8,
                title_color=ORANGE)
    rounded_box(ax, *detail_boxes["contact"], ORANGE_BG, ORANGE, "D  接触轨迹 DP", (
        "接近抓握点并持续接触",
        "滑移 / 力臂 / 机器人碰撞",
        "释放后可离开并回到路线图",
    ), tag="contact.py", title_size=12.4, body_size=9.65,
                title_color=ORANGE)

    # Drop-pose lookahead and objective.
    arrow(ax, (65.75, 48.4), (65.75, 44.5), color=ORANGE, width=1.55,
          mutation=11)
    rounded_box(ax, 48, 35.2, 25, 8.6, ORANGE_BG, ORANGE,
                "E  放置姿态前瞻", (
                    "拒绝切断后续去路的落点；惩罚额外阻挡的路线图边",
                ), tag="_placement_extra", title_size=12.6, body_size=9.8,
                title_color=ORANGE)

    rounded_box(ax, 7, 25, 66, 7.4, "#F6F3FF", PURPLE,
                "统一目标函数", (), title_size=12.3, title_color=PURPLE)
    ax.text(40, 27.35,
            "C = (1-w)·( λD_robot + F_est·D_SE2,eq )  +  w·v_T·T  +  R_sem",
            ha="center", va="center", fontsize=15.1, fontweight="bold",
            color=INK, zorder=5)
    ax.text(40, 21.3,
            "机器人行驶  ·  障碍物搬移工作量代理  ·  转向/接近/伴随/释放/等待时间  ·  语义后果风险附加代价",
            ha="center", va="center", fontsize=9.6, color=MUTED)
    arrow(ax, (60.5, 35), (60.5, 32.8), color=PURPLE, width=1.45,
          mutation=10)

    # Online feedback panel.
    loop_panel = FancyBboxPatch(
        (78.5, 18.5), 18.5, 50.5,
        boxstyle="round,pad=0.4,rounding_size=1.6",
        facecolor=WHITE, edgecolor="#D8E0EC", linewidth=1.2, zorder=0,
    )
    ax.add_patch(loop_panel)
    ax.text(80.5, 66.7, "在线执行与响应式重规划", ha="left", va="top",
            fontsize=14.7, fontweight="bold", color=INK)

    rounded_box(ax, 80.5, 54.4, 14.5, 8.2, GREEN_BG, GREEN,
                "执行一小段", (
                    "每步后 perceive / touch",
                ), tag="executor.py", title_size=12.2, body_size=9.8,
                title_color=GREEN)
    arrow(ax, (87.75, 54), (87.75, 50.2), color=RED, width=1.5,
          mutation=11)
    diamond(ax, 87.75, 46.2, 14.2, 7.2, "信息或世界\n发生变化？")

    ax.text(80.8, 40.3, "触发源", ha="left", va="top", fontsize=10.3,
            fontweight="bold", color=RED)
    triggers = (
        "• 新障碍物 / 姿态或属性变化",
        "• 未知碰撞 / 动态物体穿越",
        "• 接触揭示真实难度或风险",
        "• 达到分段执行上限",
    )
    for i, line in enumerate(triggers):
        ax.text(80.8, 37.8 - i * 2.55, line, ha="left", va="top",
                fontsize=9.25, color=INK)

    pill(ax, 80.7, 22.2, 14.1, 4.3, "目标节点 → 成功结束", GREEN_BG, GREEN,
         color=GREEN, size=9.4)
    ax.text(90.1, 48.8, "否：继续", ha="left", va="center",
            fontsize=8.8, color=GREEN)
    ax.text(79.7, 46.2, "是", ha="right", va="center",
            fontsize=9.2, fontweight="bold", color=RED)

    # Feedback loop: changed belief returns to perception/planning.
    arrow(ax, (80.3, 46.2), (76.8, 46.2), color=RED, width=1.7,
          mutation=11)
    arrow(ax, (76.8, 46.2), (76.8, 71.3), color=RED, width=1.7,
          style="-", mutation=11)
    arrow(ax, (76.8, 71.3), (49.5, 71.3), color=RED, width=1.7,
          mutation=12)
    ax.text(62.5, 72.2, "更新 belief.version → 重新规划",
            ha="center", va="bottom", fontsize=9.5, color=RED,
            fontweight="bold")

    # Continue and goal paths in the loop panel.
    arrow(ax, (94.9, 46.2), (96.1, 46.2), color=GREEN, width=1.4,
          style="-", mutation=10)
    arrow(ax, (96.1, 46.2), (96.1, 58.5), color=GREEN, width=1.4,
          style="-", mutation=10)
    arrow(ax, (96.1, 58.5), (95.2, 58.5), color=GREEN, width=1.4,
          mutation=10)

    # Output summary.
    ax.text(3, 13.8, "输出", ha="left", va="center", fontsize=10.2,
            fontweight="bold", color=MUTED)
    pill(ax, 8, 11.6, 16, 4.5, "MOVE  机器人行驶", BLUE_BG, BLUE,
         color=BLUE, size=10.2)
    pill(ax, 25.5, 11.6, 22.5, 4.5, "REMOVE  搬移障碍物并释放", ORANGE_BG, ORANGE,
         color=ORANGE, size=10.2)
    pill(ax, 49.5, 11.6, 17.5, 4.5, "WAIT  等待世界变化", PURPLE_BG, PURPLE,
         color=PURPLE, size=10.2)
    pill(ax, 68.5, 11.6, 28.5, 4.5, "核心特征：分层规划 + 在线闭环", GREEN_BG, GREEN,
         color=GREEN, size=10.2)

    ax.text(3, 6.8,
            "实现定位：cost-guided, risk-aware, contact-feasible online NAMO with reactive replanning",
            ha="left", va="center", fontsize=10.8, color=MUTED)
    ax.text(97, 6.8, "基于当前代码调用链整理", ha="right", va="center",
            fontsize=9.4, color=MUTED)

    png = OUT_DIR / "cra-namo-core-planning-pipeline.png"
    svg = OUT_DIR / "cra-namo-core-planning-pipeline.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.18,
                facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.18,
                facecolor=BG)
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
