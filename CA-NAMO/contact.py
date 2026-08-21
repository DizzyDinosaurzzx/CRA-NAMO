"""贴身搬运的机器人轨迹规划:障碍物移动期间机器人须保持物理接触,抓持点可沿表面滑动,按子步用动态规划选取。"""


from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import LineString

import geometry

Pose = Tuple[float, float, float]
XY = Tuple[float, float]

_INF = float("inf")
_MIN_STATIONS = 8
_MAX_STATIONS = 64


@dataclass
class ContactPlan:
    """一次搬运中机器人每一步的位置。

    robot_path 为一条连续折线:[起点] + [每个子步的抓持点] + [终点];
    障碍物在 poses[i] 时机器人在 robot_path[move_offset + i],
    前段是接近与绕行到抓持侧,后段是对称的撤离绕行。
    """
    feasible: bool
    reason: str = ""
    robot_path: List[XY] = field(default_factory=list)
    move_offset: int = 0
    travel: float = 0.0            # 整次搬运中机器人总行程 [m]

    def at(self, i: int) -> XY:
        return self.robot_path[self.move_offset + i]

    def leg_length(self, a: int, b: int) -> float:
        """robot_path[a:b+1] 的折线长度(索引基于 robot_path,a <= b)。"""
        return sum(math.dist(self.robot_path[t], self.robot_path[t + 1])
                   for t in range(a, b))


def idle_plan(robot_pos: XY, n_poses: int) -> ContactPlan:
    """无需贴身接触时的退化方案:机器人原地不动,结构上与真实方案一致以便执行器统一处理,且不产生行程。"""
    return ContactPlan(True, "", [tuple(robot_pos)] * (max(1, n_poses) + 2), 1, 0.0)


# --- 抓持站 ---
def contact_stations(l: float, d: float, r: float, spacing: float) -> np.ndarray:
    """与 l x d 矩形接触的机器人圆心位置。

    返回障碍物局部系下的 (K, 2) 点列:按弧长等距、逆时针排列,
    每点距矩形边界恰好为 r。
    """
    hl, hd = l / 2.0, d / 2.0
    quarter = 0.5 * math.pi * r
    # 逆时针排列:右边、圆角、上边、圆角……
    segments = [
        ("line", (hl + r, -hd), (0.0, 1.0), d),
        ("arc", (hl, hd), 0.0, quarter),
        ("line", (hl, hd + r), (-1.0, 0.0), l),
        ("arc", (-hl, hd), 0.5 * math.pi, quarter),
        ("line", (-hl - r, hd), (0.0, -1.0), d),
        ("arc", (-hl, -hd), math.pi, quarter),
        ("line", (-hl, -hd - r), (1.0, 0.0), l),
        ("arc", (hl, -hd), 1.5 * math.pi, quarter),
    ]
    perimeter = 2.0 * (l + d) + 2.0 * math.pi * r
    k = int(math.ceil(perimeter / max(spacing, 1e-3)))
    k = max(_MIN_STATIONS, min(_MAX_STATIONS, k))

    pts = np.empty((k, 2), dtype=float)
    step = perimeter / k
    seg_i, seg_s = 0, 0.0          # 当前段索引及段内弧长位置
    for n in range(k):
        target = n * step
        # 前进到包含 target 的段
        acc = 0.0
        for si, seg in enumerate(segments):
            length = seg[3]
            if acc + length > target or si == len(segments) - 1:
                seg_i, seg_s = si, target - acc
                break
            acc += length
        seg = segments[seg_i]
        if seg[0] == "line":
            (px, py), (ux, uy) = seg[1], seg[2]
            pts[n] = (px + ux * seg_s, py + uy * seg_s)
        else:
            (cx, cy), a0 = seg[1], seg[2]
            a = a0 + (seg_s / max(quarter, 1e-12)) * (0.5 * math.pi)
            pts[n] = (cx + r * math.cos(a), cy + r * math.sin(a))
    return pts


def _unwrapped_angles(poses: Sequence[Pose]) -> np.ndarray:
    """沿路径连续展开的朝向角。

    SE(2) 规划器按模 pi 存朝向(矩形在 theta 与 theta+pi 的 footprint 相同),
    但抓持点不满足这一等价:若直接跟踪存储角,索引自 pi 绕回 0 时机器人会被
    甩到对面。逐段累加 wrap_dtheta 短转角即可保持在实际抓持的那一面。
    """
    th = [float(poses[0][2])]
    for a, b in zip(poses, poses[1:]):
        th.append(th[-1] + geometry.wrap_dtheta(a[2], b[2]))
    return np.array(th, dtype=float)


def _world_positions(stations: np.ndarray, poses: Sequence[Pose]) -> np.ndarray:
    """每个位姿下所有抓持站的世界坐标,形状 (T, K, 2)。"""
    th = _unwrapped_angles(poses)
    c, s = np.cos(th), np.sin(th)
    sx = stations[:, 0][None, :]
    sy = stations[:, 1][None, :]
    x = c[:, None] * sx - s[:, None] * sy + np.array([p[0] for p in poses])[:, None]
    y = s[:, None] * sx + c[:, None] * sy + np.array([p[1] for p in poses])[:, None]
    return np.stack([x, y], axis=2)


# --- 直线通行测试 ---
def _clear_line(a: XY, b: XY, free_geom, blocked_geom, trim: float) -> bool:
    """机器人能否从 a 直线驶向 b。

    墙体沿整段检查(free_geom 已按机器人半径收缩);可移动障碍物只检查
    两端各去掉一个机器人半径的区段——端点是已知可达的位置,
    接近末端的那个半径正是贴上障碍物的对接动作。
    """
    seg = LineString([a, b])
    if not shapely.contains(free_geom, seg):
        return False
    if blocked_geom is None:
        return True
    length = seg.length
    if length <= 2.0 * trim:
        return True
    inner = LineString([seg.interpolate(trim), seg.interpolate(length - trim)])
    return not shapely.intersects(blocked_geom, inner)


# --- 规划器 ---
def plan_contact(obs,
                 poses: Sequence[Pose],
                 robot_start: XY,
                 robot_end: XY,
                 free_geom,
                 others_inflated,
                 cfg) -> ContactPlan:
    """为一次搬运规划机器人轨迹。

    free_geom 为机器人圆心可达的静态自由区(静态自由空间收缩一个机器人半径);
    others_inflated 为其他可移动障碍物禁止的圆心区(并集膨胀一个机器人半径),可为 None。
    """
    if not poses:
        return ContactPlan(False, "empty move path")

    r = float(cfg.robot_radius)
    tol = float(cfg.contact_clearance)
    stations = contact_stations(obs.l, obs.d, r, cfg.contact_station_spacing)
    k = len(stations)

    # 单个子步内抓持点可沿表面滑过的弧长
    step_arc = (2.0 * (obs.l + obs.d) + 2.0 * math.pi * r) / k
    # 向下取整而非四舍五入:取大会让每子步滑动超过 contact_max_slide;
    # 至少允许移一站,否则粗糙采样会把抓持点彻底冻死
    max_shift = max(1, int(cfg.contact_max_slide / max(step_arc, 1e-6)))
    # 搬运前后预留绕行站数:直线被挡时仍能绕到背面
    pad = int(math.ceil(k / (2.0 * max_shift)))

    ext: List[Pose] = ([poses[0]] * pad) + list(poses) + ([poses[-1]] * pad)
    world = _world_positions(stations, ext)
    t_total = len(ext)

    # --- 抓持可行性(向量化) ---
    flat_x = world[:, :, 0].ravel()
    flat_y = world[:, :, 1].ravel()
    ok = shapely.contains_xy(free_geom, flat_x, flat_y)
    if others_inflated is not None:
        ok &= ~shapely.intersects_xy(others_inflated, flat_x, flat_y)
    feas = ok.reshape(t_total, k)
    if not feas.any():
        return ContactPlan(False, "no reachable grip point on this obstacle")

    # 障碍物自身阻挡接近与撤离绕行,但不阻挡抓持点(后者本就贴着它)
    def _with_body(pose: Pose):
        body = obs.polygon_at(pose[0], pose[1], pose[2]).buffer(max(r - tol, 0.0))
        merged = body if others_inflated is None else others_inflated.union(body)
        shapely.prepare(merged)
        return merged

    approach_blockers = _with_body(poses[0])
    exit_blockers = approach_blockers if len(poses) == 1 else _with_body(poses[-1])

    # 参考点是路网点,而路网忽略可移动障碍物,可能正落在被搬障碍物底下:
    # 从一个本就不可能站的位置无从校验驶入,该段的直线测试跳过,距离照常计费。
    start_under = bool(shapely.intersects_xy(approach_blockers, *robot_start))
    end_under = bool(shapely.intersects_xy(exit_blockers, *robot_end))

    # --- 进入代价 ---
    cost = np.full(k, _INF)
    for s in np.flatnonzero(feas[0]):
        p = (float(world[0, s, 0]), float(world[0, s, 1]))
        if start_under or _clear_line(robot_start, p, free_geom,
                                      approach_blockers, r):
            cost[s] = math.dist(robot_start, p)
    if not np.isfinite(cost).any():
        return ContactPlan(False, "cannot reach any grip point on this obstacle")

    # --- 动态规划 ---
    parent = np.full((t_total, k), -1, dtype=np.int64)
    idx = np.arange(k)
    shifts = range(-max_shift, max_shift + 1)
    for t in range(t_total - 1):
        feas_next = feas[t + 1]
        best = np.full(k, _INF)
        best_src = np.full(k, -1, dtype=np.int64)
        for shift in shifts:
            # 滑过的每个站都必须在下一姿态可行
            slide_ok = np.ones(k, dtype=bool)
            span = range(0, shift + 1) if shift >= 0 else range(shift, 1)
            for e in span:
                slide_ok &= np.roll(feas_next, -e)
            step = np.linalg.norm(np.roll(world[t + 1], -shift, axis=0) - world[t],
                                  axis=1)
            cand = np.where(slide_ok, cost + step, _INF)
            cand = np.roll(cand, shift)          # 把源站 s 的代价散布到目标站 s+shift
            upd = cand < best
            best[upd] = cand[upd]
            best_src[upd] = (idx[upd] - shift) % k
        cost, parent[t + 1] = best, best_src
        if not np.isfinite(cost).any():
            return ContactPlan(
                False, "robot cannot stay in contact for the whole manipulation")

    # --- 撤离代价 ---
    # 候选按总代价升序,第一个能顺利驶出的即胜出
    exit_dist = np.linalg.norm(world[-1] - np.asarray(robot_end), axis=1)
    total = cost + exit_dist
    chosen = -1
    for s in np.argsort(total):
        if not np.isfinite(total[s]):
            break
        p = (float(world[-1, s, 0]), float(world[-1, s, 1]))
        if end_under or _clear_line(p, robot_end, free_geom, exit_blockers, r):
            chosen = int(s)
            break
    if chosen < 0:
        return ContactPlan(False, "robot cannot leave the obstacle after moving it")

    # --- 回溯 ---
    seq = [chosen]
    for t in range(t_total - 1, 0, -1):
        seq.append(int(parent[t, seq[-1]]))
    seq.reverse()
    grip = [(float(world[t, s, 0]), float(world[t, s, 1]))
            for t, s in enumerate(seq)]
    path = [tuple(robot_start)] + grip + [tuple(robot_end)]
    travel = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    return ContactPlan(True, "", path, 1 + pad, travel)

