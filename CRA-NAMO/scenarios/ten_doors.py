"""用于测量估计误差对路线影响的静态十门走廊。"""

from __future__ import annotations

from shapely.geometry import box

import risk as risk_model
from config import Config
from obstacle import MovableObstacle, StaticObstacle

# 几何常量。
WALL_T = 0.4                 # 墙和外壳厚度 [米]
CORRIDOR_H = 18.0            # 工作空间高度 [米]
GATE_SPACING = 5.0           # 墙间距 [米]
FIRST_GATE_X = 5.0           # 第一面墙的 x 坐标 [米]
N_GATES = 10

DOOR_H = 1.8                 # 沿 y 方向的开口尺寸 [米]
DOOR_A_Y = 7.6               # 门 A 占据 [7.6, 9.4]
DOOR_B_Y = 9.7               # 门 B 占据 [9.7, 11.5]
TRAVEL_Y = 9.55              # 走廊轴线，位于 A、B 中间
BLOCKER_CLEARANCE = 0.2      # 阻塞物比门洞窄的尺寸

WORKSPACE_W = FIRST_GATE_X * 2 + GATE_SPACING * (N_GATES - 1)   # 55.0 米

A_OID_BASE = 100             # 第 i 个门的 A 门 ID 为 100 + 2i，B 门 ID 加 1

# 门组数据：绕行位置、材料和真实难度。
#
# 需要校准。2026-09-12 用 exact belief 跑了一次，十道门里九道绕行，只搬走了
# oid 102（gate 1 的 A 门，400 N），W 只占 C 的 0.9%。决策这样一边倒时，
# 估计误差没有第二个选项可换，Gap 扫描量到的会接近平坦。
#
# 把难度整体压到 300..2400 N 试过一次，结果不变，仍然只搬一扇。再看两个
# 数据点就知道为什么猜不出来：gate 1 在 9.9 m 绕行下搬走了 400 N，而
# gate 8 在 14.9 m 绕行下连 2600 N 都不搬 —— 「搬移固定开销 + 难度 x 距离」
# 这种常数模型解释不了，接近障碍物要走的路和绕行之间的串联都跟每道门的
# 几何有关。所以这里不再猜，先跑一次
# `LLM_benchmark/llm_accuracy.py doors-calibrate`：它把每道门的三个选项
# 分别单独计价，直接给出每扇门正好与绕行持平所需的难度，照着那一列把下面
# 的十个数字铺在平衡点两侧即可。
GATES = (
    (4.0,  "wooden_crate",    1600.0, "wooden_crate",    2600.0),  # 接近平衡
    (13.6, "wooden_crate",     400.0, "wooden_crate",    3000.0),  # 低价对照
    (3.0,  "wooden_crate",    2200.0, "wooden_crate",    1800.0),  # 接近平衡
    (14.6, "wooden_crate",    3400.0, "wooden_crate",    2900.0),  # 接近平衡
    (4.4,  "glassware_crate",  800.0, "wooden_crate",    2400.0),  # 风险敏感
    (15.6, "wooden_crate",    1200.0, "glassware_crate",  900.0),  # 风险敏感
    (2.2,  "wooden_crate",    2600.0, "wooden_crate",    2000.0),  # 接近平衡
    (13.0, "wooden_crate",    5000.0, "wooden_crate",    4500.0),  # 高价对照
    (1.2,  "glassware_crate", 1000.0, "wooden_crate",    2600.0),  # 风险敏感
    (15.8, "wooden_crate",    2000.0, "glassware_crate", 1500.0),  # 风险敏感
)

# 每个门组的绕行距离，单位为米。
DETOUR_M = tuple(round(2.0 * abs(y0 + DOOR_H / 2.0 - TRAVEL_Y), 2)
                 for y0, *_ in GATES)


def gate_x(i: int) -> float:
    """返回第 i 个门墙近侧面的 x 坐标。"""
    return FIRST_GATE_X + GATE_SPACING * i


def _blocker(oid: int, x: float, door_y: float, material: str,
             difficulty: float) -> MovableObstacle:
    """创建使剩余净空小于机器人宽度的门洞阻塞物。"""
    return MovableObstacle(
        x=x + WALL_T / 2.0,
        y=door_y + DOOR_H / 2.0,
        l=WALL_T,
        d=DOOR_H - BLOCKER_CLEARANCE,
        h=1.0,
        theta=0.0,
        material=material,
        difficulty=float(difficulty),
        oid=oid,
    )


def gate_table() -> list:
    """返回估计误差研究所需的门洞真实记录。"""
    rows = []
    for i, (bypass_y0, mat_a, diff_a, mat_b, diff_b) in enumerate(GATES):
        risk_a = risk_model.keyword_level(mat_a)
        risk_b = risk_model.keyword_level(mat_b)
        # 一道门的两扇之间的难度比：`mixed` 方向下决策翻转所需的相对误差下界。
        ratio = max(diff_a, diff_b) / max(min(diff_a, diff_b), 1e-9)
        # 一道门只要有一扇不是 low，决策就由风险而不是代价主导。
        kind = ("risk" if risk_model.LOW not in (risk_a, risk_b)
                or risk_a != risk_b else "cost")
        for side, material, difficulty, level in (("A", mat_a, diff_a, risk_a),
                                                  ("B", mat_b, diff_b, risk_b)):
            rows.append({
                "gate": i,
                "side": side,
                "oid": A_OID_BASE + 2 * i + (0 if side == "A" else 1),
                "material": material,
                "difficulty": float(difficulty),
                "risk": level,
                "detour_m": DETOUR_M[i],
                "bypass_y": bypass_y0,
                # 以下三项对同一道门的两行相同，便于按门切片。
                "kind": kind,
                "ab_ratio": round(ratio, 3),
                "cheaper_side": "A" if diff_a <= diff_b else "B",
            })
    return rows


def gate_summary() -> list:
    """每道门一行，供报告按决策裕度而不是按障碍物切片。"""
    rows = gate_table()
    out = []
    for i in range(N_GATES):
        a, b = [r for r in rows if r["gate"] == i]
        out.append({
            "gate": i, "kind": a["kind"], "detour_m": a["detour_m"],
            "ab_ratio": a["ab_ratio"], "cheaper_side": a["cheaper_side"],
            "A": {"oid": a["oid"], "material": a["material"],
                  "difficulty": a["difficulty"], "risk": a["risk"]},
            "B": {"oid": b["oid"], "material": b["material"],
                  "difficulty": b["difficulty"], "risk": b["risk"]},
        })
    return out


def _wall_segments(bypass_y0: float):
    """从墙上切出三个开口后剩余的墙段。"""
    openings = sorted([(DOOR_A_Y, DOOR_A_Y + DOOR_H),
                       (DOOR_B_Y, DOOR_B_Y + DOOR_H),
                       (bypass_y0, bypass_y0 + DOOR_H)])
    edges = [0.0]
    for lo, hi in openings:
        edges += [lo, hi]
    edges.append(CORRIDOR_H)
    return [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]


def _wall_segments_no_bypass():
    """只留下门 A 和门 B 的墙段；校准阶段用它砌死绕行开口。"""
    edges = [0.0, DOOR_A_Y, DOOR_A_Y + DOOR_H, DOOR_B_Y, DOOR_B_Y + DOOR_H,
             CORRIDOR_H]
    return [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]


def create(closed_bypasses=()):
    """构建十门走廊。

    `closed_bypasses` 列出要砌死第三个开口的门号。校准阶段用它把一道门的
    选项逐个关掉，从而单独测量「搬 A」「搬 B」「绕行」各自的真实代价。
    """
    closed = {int(i) for i in closed_bypasses}
    unknown = sorted(i for i in closed if not 0 <= i < N_GATES)
    if unknown:
        raise ValueError(f"ten_doors has gates 0..{N_GATES - 1}; "
                         f"closed_bypasses names {unknown}")
    workspace = box(0.0, 0.0, WORKSPACE_W, CORRIDOR_H)

    walls = [
        StaticObstacle(box(0.0, 0.0, WORKSPACE_W, WALL_T), "shell_bottom"),
        StaticObstacle(box(0.0, CORRIDOR_H - WALL_T, WORKSPACE_W, CORRIDOR_H),
                       "shell_top"),
        StaticObstacle(box(0.0, 0.0, WALL_T, CORRIDOR_H), "shell_left"),
        StaticObstacle(box(WORKSPACE_W - WALL_T, 0.0, WORKSPACE_W, CORRIDOR_H),
                       "shell_right"),
    ]

    movable = []
    for i, (bypass_y0, mat_a, diff_a, mat_b, diff_b) in enumerate(GATES):
        x = gate_x(i)
        segments = (_wall_segments_no_bypass() if i in closed
                    else _wall_segments(bypass_y0))
        for k, (y0, y1) in enumerate(segments):
            if y1 - y0 > 1e-9:
                walls.append(StaticObstacle(box(x, y0, x + WALL_T, y1),
                                            f"gate{i}_seg{k}"))
        movable.append(_blocker(A_OID_BASE + 2 * i, x, DOOR_A_Y, mat_a, diff_a))
        movable.append(_blocker(A_OID_BASE + 2 * i + 1, x, DOOR_B_Y, mat_b, diff_b))

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": (2.2, TRAVEL_Y),
        "goal": (WORKSPACE_W - 2.2, TRAVEL_Y),
        "cfg": Config(),
        # 估计误差研究使用的真实数据。
        "gates": gate_table(),
        "gate_summary": gate_summary(),
        "closed_bypasses": tuple(sorted(closed)),
    }
