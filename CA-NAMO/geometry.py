"""
方法部分中的两个几何计算器
1. push_plan()        

计算清除障碍物所阻挡边所需的实际推动距离、推动是否可行，以及障碍物最终位置
（“放置位姿”）。默认 direct 模式的做功 W 由障碍物直接给出，推动距离不参与 W；
保留的 estimated 模式仍可使用 difficulty * push_distance。

2. walking_distance() 

计算两点经静态自由空间的实际行驶距离（用于路网边长度和可采纳启发式函数的合理性检查）。

两者仅在静态地图（工作空间减去不可移动障碍物）上运行，这也是机器人预先知道的地图。
"""


from __future__ import annotations

import heapq
import math
from typing import Iterable, List, Optional, Tuple

from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union

from obstacle import MovableObstacle
from config import Config


# --------------------------------------------------------------------------- #
#  计算器 1：推动规划
# --------------------------------------------------------------------------- #
def _swept_region(obs: MovableObstacle, nx: float, ny: float) -> Polygon:

    """
    计算矩形从当前位姿平移到 (nx, ny) 时扫过的区域。
    由于矩形是凸形且推动过程中保持方向不变，精确的扫掠区域就是起始和终止footprint的凸包。
    """
    
    start = obs.polygon
    end = obs.polygon_at(nx, ny)
    return unary_union([start, end]).convex_hull


def push_plan(
    obs: MovableObstacle,
    static_free,                       # shapely (Multi)Polygon：仅由墙壁界定的自由空间
    must_clear: Optional[Polygon],     # 放置位姿【必须】腾空的通道（硬约束）
    avoid: Optional[Polygon],          # 应避免重新阻挡的其他畅通通道（软约束）
    cfg: Config,
    others: Optional[Polygon] = None,  # 其他可移动障碍物当前占据的区域（硬约束：不得碰撞）
) -> Tuple[bool, float, Optional[Tuple[float, float, float]]]:

    """
    为障碍物找到代价最小的可行重新放置方案

    一个候选放置位姿为【可行】当且仅当：
      * 它完全落在静态自由空间内（不撞墙）
      * 到达该位姿的直线推动过程也始终处于静态自由空间内
      * 它完全腾空 `must_clear`——即该障碍物当前挡住的通道（从而让“付费可解锁”的边真正被打通）
      * 终点位姿与推动扫掠路径都不与其他可移动障碍物 `others` 相交（物体不能互相穿透/叠放）

    在可行位姿中，我们将避免反效果的偏好作为软目标
    优先选择不会重新阻挡其他当前畅通通道（`avoid`）的位姿，并以最短推动距离打破平局。
    由于密集路网覆盖自由空间，将避免反效果作为硬约束会禁止所有放置；任何残余的新阻挡都会通过重新感知发现，并由重规划处理。
    返回 (feasible, push_distance, drop_pose)。push_distance 是障碍物中心的欧氏移动
    距离；direct 模式仅把它用于几何放置选择，不用于计算 W。
    """

    best = None                      # (penalty, distance, pose)
    cx, cy = obs.x, obs.y
    for k in range(1, cfg.drop_radius_steps + 1):
        r = cfg.R_push * k / cfg.drop_radius_steps
        for a in range(cfg.drop_ring_samples):
            ang = 2.0 * math.pi * a / cfg.drop_ring_samples
            nx = round(cx + r * math.cos(ang), 3)
            ny = round(cy + r * math.sin(ang), 3)
            end_poly = obs.polygon_at(nx, ny)

            if not static_free.contains(end_poly):
                continue
            if must_clear is not None and end_poly.intersects(must_clear):
                continue
            if others is not None and end_poly.intersects(others):
                continue                              # 终点压到了别的障碍物
            swept = _swept_region(obs, nx, ny)
            if not static_free.contains(swept):
                continue
            if others is not None and swept.intersects(others):
                continue                              # 推动路径穿过了别的障碍物

            penalty = 1 if (avoid is not None and end_poly.intersects(avoid)) else 0
            dist = math.hypot(nx - cx, ny - cy)
            cand = (penalty, round(dist, 3), (nx, ny, obs.theta))
            if best is None or cand[:2] < best[:2]:
                best = cand
    if best is None:
        return (False, math.inf, None)
    return (True, best[1], best[2])


def push_work(obs: MovableObstacle, push_distance: float) -> float:
    """保留的 estimated 模式：difficulty x push_distance。"""
    return obs.difficulty * push_distance


# --------------------------------------------------------------------------- #
#  计算器 2：经静态自由空间的行驶距离
# --------------------------------------------------------------------------- #


def line_of_sight(free_space, a: Tuple[float, float], b: Tuple[float, float],
                  inflate: float = 0.0) -> bool:
    
    """如果线段 a-b（可按 `inflate` 膨胀）始终位于自由空间内，则返回 True。"""

    seg = LineString([a, b])
    if inflate > 0:
        seg = seg.buffer(inflate)
    return free_space.contains(seg)


def walking_distance(
    free_space,
    a: Tuple[float, float],
    b: Tuple[float, float],
    cfg: Config,
    grid: float = 1.0,
) -> float:
    
    """计算从 a 到 b 经 `free_space` 的实际行驶距离。
    通道畅通时返回直线距离；否则在自由空间上使用 8 连通网格 A*。
    如果 b 不可达，则返回 math.inf。
    """
    
    if line_of_sight(free_space, a, b, inflate=cfg.robot_radius):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    minx, miny, maxx, maxy = free_space.bounds

    def snap(p):
        return (round((p[0] - minx) / grid), round((p[1] - miny) / grid))

    def to_xy(cell):
        return (minx + cell[0] * grid, miny + cell[1] * grid)

    def walkable(cell):
        x, y = to_xy(cell)
        return free_space.contains(Point(x, y).buffer(cfg.robot_radius * 0.5))

    start, goal = snap(a), snap(b)
    open_heap = [(0.0, start)]
    g = {start: 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            return g[cur]
        for dx, dy in nbrs:
            nxt = (cur[0] + dx, cur[1] + dy)
            step = grid * math.hypot(dx, dy)
            ng = g[cur] + step
            if ng < g.get(nxt, math.inf) and walkable(nxt):
                g[nxt] = ng
                gx, gy = to_xy(goal)
                nxy = to_xy(nxt)
                h = math.hypot(nxy[0] - gx, nxy[1] - gy)
                heapq.heappush(open_heap, (ng + h, nxt))
    return math.inf
