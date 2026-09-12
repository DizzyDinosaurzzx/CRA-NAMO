"""Generate the paper schematic for roadmap construction and locked routes."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")

BG = "#F6F8FB"
WHITE = "#FFFFFF"
INK = "#182235"
MUTED = "#5D6A7E"
FRAME = "#D5DEEA"
GRID = "#AAB7C9"
BLUE = "#3274D9"
BLUE_SOFT = "#EAF2FF"
PURPLE = "#7357D9"
PURPLE_SOFT = "#F1EDFF"
ORANGE = "#DF7628"
ORANGE_SOFT = "#FFF0E4"
RED = "#C84A4A"
RED_SOFT = "#FDECEC"
GREEN = "#18845E"
GREEN_SOFT = "#E7F6EF"
WALL = "#697588"
WALL_LIGHT = "#DDE3EB"


if FONT_PATH.exists():
    mpl.font_manager.fontManager.addfont(str(FONT_PATH))
    FONT = mpl.font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT = "sans-serif"

mpl.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
})


def rounded(ax, x, y, w, h, face=WHITE, edge=FRAME, radius=1.2,
            linewidth=1.1, zorder=0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=GRID, width=1.5, *, connection="arc3",
          style="-|>", mutation=12, zorder=3):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=width, color=color, connectionstyle=connection,
        shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def pill(ax, x, y, w, h, text, face, edge, color, size=9.5):
    rounded(ax, x, y, w, h, face, edge, radius=h / 2.0,
            linewidth=1.0, zorder=8)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, fontweight="bold", color=color, zorder=9)


def panel_title(ax, x, number, title, subtitle):
    ax.text(x, 74.2, number, ha="left", va="center", fontsize=20,
            fontweight="bold", color=BLUE)
    ax.text(x + 4.1, 74.2, title, ha="left", va="center", fontsize=15.2,
            fontweight="bold", color=INK)
    ax.text(x + 4.1, 70.9, subtitle, ha="left", va="center", fontsize=9.4,
            color=MUTED)


def draw_static_map(ax, x, y, w, h):
    """Draw the shared workspace, walls, nodes, and base roadmap."""
    rounded(ax, x, y, w, h, WHITE, FRAME, radius=0.8, linewidth=1.0, zorder=0)

    # Eroded free-space boundary.
    ax.add_patch(Rectangle((x + 1.3, y + 1.3), w - 2.6, h - 2.6,
                           facecolor=BLUE_SOFT, edgecolor=BLUE,
                           linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.45,
                           zorder=0.5))

    # Static obstacles used to construct the roadmap.
    walls = [
        (x + 18.0, y + 14.8, 3.7, 5.0),
        (x + 31.0, y + 5.6, 3.7, 3.6),
        (x + 36.5, y + 15.2, 5.8, 3.6),
    ]
    for wx, wy, ww, wh in walls:
        ax.add_patch(Rectangle((wx, wy), ww, wh, facecolor=WALL_LIGHT,
                               edgecolor=WALL, linewidth=1.0, hatch="////",
                               zorder=5))

    nodes = {
        "S": (x + 4.0, y + 12.0),
        "a": (x + 11.5, y + 12.0),
        "b": (x + 18.0, y + 8.0),
        "c": (x + 25.5, y + 8.0),
        "d": (x + 34.5, y + 11.8),
        "G": (x + 45.5, y + 12.0),
        "u1": (x + 11.5, y + 21.8),
        "u2": (x + 25.5, y + 23.8),
        "u3": (x + 34.5, y + 23.8),
        "l1": (x + 11.5, y + 4.3),
        "l2": (x + 25.5, y + 3.8),
        "l3": (x + 40.0, y + 4.2),
    }
    edges = [
        ("S", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "G"),
        ("a", "u1"), ("u1", "u2"), ("u2", "u3"), ("u3", "G"),
        ("a", "l1"), ("l1", "l2"), ("l2", "l3"), ("l3", "G"),
        ("b", "l1"), ("c", "l2"), ("d", "l3"),
    ]
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], color=GRID, linewidth=1.15,
                zorder=2, solid_capstyle="round")
    for name, (nx, ny) in nodes.items():
        special = name in ("S", "G")
        ax.add_patch(Circle((nx, ny), 0.66 if special else 0.42,
                            facecolor=INK if special else WHITE,
                            edgecolor=INK if special else BLUE,
                            linewidth=1.1, zorder=6))
        if special:
            ax.text(nx, ny, name, ha="center", va="center", fontsize=7.4,
                    fontweight="bold", color=WHITE, zorder=7)
    return nodes, edges


def highlight_path(ax, nodes, path, color, width=3.0, *, dashed=False,
                   zorder=4, alpha=1.0):
    for a, b in zip(path, path[1:]):
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=width,
                linestyle=(0, (3, 2)) if dashed else "-",
                solid_capstyle="round", dash_capstyle="round",
                zorder=zorder, alpha=alpha)


def obstacle(ax, cx, cy, *, ghost=False, label="o7"):
    face = "none" if ghost else ORANGE_SOFT
    alpha = 0.52 if ghost else 1.0
    ax.add_patch(Rectangle((cx - 2.45, cy - 1.75), 4.9, 3.5,
                           angle=0, rotation_point="center",
                           facecolor=face, edgecolor=ORANGE,
                           linewidth=1.5, linestyle=(0, (3, 2)) if ghost else "-",
                           alpha=alpha, zorder=7))
    if not ghost:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=ORANGE, zorder=8)


def main():
    fig, ax = plt.subplots(figsize=(18, 8.6), facecolor=BG)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 86)
    ax.axis("off")
    ax.set_facecolor(BG)

    ax.text(4, 83.2, "Roadmap 构建、Locked Route 与代价解锁",
            ha="left", va="top", fontsize=24, fontweight="bold", color=INK)
    ax.text(4, 79.3,
            "静态拓扑一次构建；在线阶段仅根据当前 Belief 更新边的 blocker 标注，并在绕行与付费解锁之间联合决策",
            ha="left", va="top", fontsize=11.2, color=MUTED)

    panels = [(3, 16.5, 55, 60), (62.5, 16.5, 55, 60), (122, 16.5, 55, 60)]
    for x, y, w, h in panels:
        rounded(ax, x, y, w, h, WHITE, FRAME, radius=1.35,
                linewidth=1.1, zorder=-1)
    arrow(ax, (58.8, 47), (61.6, 47), color=BLUE, width=1.8, mutation=13)
    arrow(ax, (118.3, 47), (121.1, 47), color=PURPLE, width=1.8, mutation=13)

    # Panel 1: construction.
    panel_title(ax, 5, "1", "构建静态 Roadmap",
                "roadmap.py · workspace + static obstacles only")
    nodes1, _ = draw_static_map(ax, 6, 39.2, 49, 28)
    # Highlight one edge corridor.
    p, q = nodes1["b"], nodes1["c"]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, linewidth=8.2,
            alpha=0.16, solid_capstyle="round", zorder=1)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, linewidth=2.2,
            solid_capstyle="round", zorder=4)
    ax.text((p[0] + q[0]) / 2, p[1] + 2.5, r"edge corridor  $\Gamma_e$",
            ha="center", va="center", fontsize=9.4, color=BLUE,
            fontfamily="DejaVu Sans", zorder=8)
    pill(ax, 7.5, 35.0, 14.8, 3.2, "① 腐蚀静态自由空间", BLUE_SOFT, BLUE, BLUE, 8.7)
    pill(ax, 23.3, 35.0, 13.8, 3.2, "② 均匀采样节点", BLUE_SOFT, BLUE, BLUE, 8.7)
    pill(ax, 38.1, 35.0, 15.4, 3.2, "③ 邻域无碰撞连边", BLUE_SOFT, BLUE, BLUE, 8.7)
    ax.text(7, 31.2,
            r"$\mathcal{F}_{robot}=(workspace\setminus static)\ominus r_{robot}$",
            ha="left", va="center", fontsize=10.3, color=INK,
            fontfamily="DejaVu Sans")
    ax.text(7, 27.7,
            r"$\Gamma_e=Buffer(Line(u,v),\ r_{robot})$",
            ha="left", va="center", fontsize=10.3, color=INK,
            fontfamily="DejaVu Sans")
    ax.text(7, 23.2, "可移动障碍物不删除图结构；它们只会在 Belief 中改变边状态。",
            ha="left", va="center", fontsize=9.2, color=MUTED)

    # Panel 2: belief update and locked route.
    panel_title(ax, 64.5, "2", "Belief 更新并标记 Locked Route",
                "perception.py · obstacle polygon intersects edge corridor")
    nodes2, _ = draw_static_map(ax, 65.5, 39.2, 49, 28)
    direct = ["S", "a", "b", "c", "d", "G"]
    highlight_path(ax, nodes2, direct, PURPLE, width=2.8, zorder=4)
    p, q = nodes2["b"], nodes2["c"]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=RED, linewidth=9.0,
            alpha=0.16, solid_capstyle="round", zorder=4)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=RED, linewidth=3.2,
            linestyle=(0, (3, 2)), dash_capstyle="round", zorder=6)
    obstacle(ax, (p[0] + q[0]) / 2, p[1])
    pill(ax, 85.5, 61.8, 12.0, 3.2, "LOCKED EDGE", RED_SOFT, RED, RED, 8.8)
    ax.text(66.8, 35.8,
            r"$\mathcal{B}_t(e)=\{o\in Belief_t\mid P_o\cap\Gamma_e\neq\varnothing\}$",
            ha="left", va="center", fontsize=10.5, color=INK,
            fontfamily="DejaVu Sans")
    ax.text(66.8, 31.8,
            r"$locked(e,t)\Longleftrightarrow\mathcal{B}_t(e)\neq\varnothing$",
            ha="left", va="center", fontsize=10.5, color=RED,
            fontfamily="DejaVu Sans", fontweight="bold")
    ax.text(66.8, 27.8,
            r"$locked(\pi,t)\Longleftrightarrow\exists e\in\pi:\ locked(e,t)$",
            ha="left", va="center", fontsize=10.5, color=RED,
            fontfamily="DejaVu Sans", fontweight="bold")
    pill(ax, 66.8, 21.1, 46.2, 3.6,
         "perceive / relocate → forget old edges → re-intersect corridors → version++",
         PURPLE_SOFT, PURPLE, PURPLE, 8.25)
    ax.text(90, 18.7, "定义基于当前 Belief：尚未感知的障碍物不会预先锁边。",
            ha="center", va="center", fontsize=8.6, color=MUTED)

    # Panel 3: pay unified cost or detour.
    panel_title(ax, 124, "3", "比较绕行与付费解锁",
                "search.py · blocked edge receives an unlock surcharge")
    nodes3, _ = draw_static_map(ax, 125, 39.2, 49, 28)
    detour = ["S", "a", "u1", "u2", "u3", "G"]
    highlight_path(ax, nodes3, detour, BLUE, width=2.9, zorder=4)
    direct3 = ["S", "a", "b", "c", "d", "G"]
    highlight_path(ax, nodes3, direct3, GREEN, width=3.0, zorder=5)
    p3, q3 = nodes3["b"], nodes3["c"]
    old = ((p3[0] + q3[0]) / 2, p3[1])
    new = (old[0] + 6.5, old[1] + 8.0)
    obstacle(ax, *old, ghost=True)
    obstacle(ax, *new, label="o7'")
    arrow(ax, (old[0], old[1] + 2.1), (new[0], new[1] - 2.1),
          color=ORANGE, width=1.8, connection="arc3,rad=-0.16",
          mutation=12, zorder=8)
    pill(ax, 126.5, 63.0, 16.8, 3.1, "DETOUR  C_detour", BLUE_SOFT, BLUE, BLUE, 8.4)
    pill(ax, 151.8, 50.2, 19.7, 3.2, "PAY C_unlock → UNLOCK", GREEN_SOFT, GREEN, GREEN, 8.25)

    ax.text(126.5, 35.8,
            r"$C_{unlock}(o,e)=(1-w)[\hat{F}_oD^{eq}_{SE(2)}+\lambda D_{contact}]$",
            ha="left", va="center", fontsize=9.7, color=INK,
            fontfamily="DejaVu Sans")
    ax.text(126.5, 32.4,
            r"$\qquad\qquad\quad +\ wv_TT_{remove}+R_{sem}(o)$",
            ha="left", va="center", fontsize=9.7, color=INK,
            fontfamily="DejaVu Sans")
    ax.text(126.5, 28.3,
            r"$C_{edge}=C_{drive}(e)+\sum_{o\in\mathcal{B}_t(e)}C_{unlock}(o,e)$",
            ha="left", va="center", fontsize=9.8, color=PURPLE,
            fontfamily="DejaVu Sans", fontweight="bold")
    ax.text(126.5, 24.2,
            r"$\pi^*=\mathrm{argmin}_{\pi\in\Pi_{feasible}}\ \sum_{e\in\pi} C_{edge}$",
            ha="left", va="center", fontsize=10.2, color=PURPLE,
            fontfamily="DejaVu Sans", fontweight="bold")
    ax.text(126.5, 19.6,
            "只有 SE(2) 搬移、扫掠体碰撞、接触保持和释放均可行时，才允许“付费解锁”。",
            ha="left", va="center", fontsize=8.6, color=MUTED)

    # Bottom legend and update note.
    ax.text(4, 11.9, "图例", ha="left", va="center", fontsize=9.5,
            fontweight="bold", color=MUTED)
    ax.plot([10, 14], [11.9, 11.9], color=GRID, linewidth=1.4)
    ax.text(15, 11.9, "available edge", ha="left", va="center",
            fontsize=8.8, color=MUTED)
    ax.plot([30, 34], [11.9, 11.9], color=RED, linewidth=3,
            linestyle=(0, (3, 2)))
    ax.text(35, 11.9, "locked edge", ha="left", va="center",
            fontsize=8.8, color=MUTED)
    ax.plot([48, 52], [11.9, 11.9], color=GREEN, linewidth=3)
    ax.text(53, 11.9, "unlocked edge", ha="left", va="center",
            fontsize=8.8, color=MUTED)
    ax.add_patch(Rectangle((69, 10.7), 3.5, 2.4, facecolor=ORANGE_SOFT,
                           edgecolor=ORANGE, linewidth=1.2))
    ax.text(74, 11.9, "perceived movable obstacle", ha="left", va="center",
            fontsize=8.8, color=MUTED)
    ax.add_patch(Rectangle((105, 10.7), 3.5, 2.4, facecolor=WALL_LIGHT,
                           edgecolor=WALL, linewidth=1.0, hatch="////"))
    ax.text(110, 11.9, "static obstacle", ha="left", va="center",
            fontsize=8.8, color=MUTED)
    pill(ax, 136.5, 9.9, 39.5, 4.0,
         "核心：拓扑不重建，Belief 边状态随观测和搬移在线更新",
         PURPLE_SOFT, PURPLE, PURPLE, 8.6)

    ax.text(4, 6.0,
            "Locked route is a belief-dependent route containing at least one blocked roadmap edge.",
            ha="left", va="center", fontsize=9.2, color=MUTED,
            fontfamily="DejaVu Sans")
    ax.text(176, 6.0, "CRA-NAMO methodology schematic",
            ha="right", va="center", fontsize=8.8, color=MUTED,
            fontfamily="DejaVu Sans")

    png = OUT_DIR / "cra-namo-roadmap-locked-route.png"
    svg = OUT_DIR / "cra-namo-roadmap-locked-route.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.16,
                facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.16,
                facecolor=BG)
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
