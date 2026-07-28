"""
几何计算：
1. push_plan(): 计算清除障碍物所阻挡边所需的实际推动距离、推动是否可行，以及障碍物最终位置（“放置位姿”）
2. walking_distance() ：计算两点经静态自由空间的实际行驶距离（用于路网边长度和可采纳启发式函数的合理性检查）
"""

from __future__ import annotations
import heapq
import math
from typing import Iterable, List, Optional, Tuple
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
from obstacle import MovableObstacle
from config import Config
import push_planner


# -------------计算器 1：推动规划--------------- #

# 放置位姿的软偏好权重，单位是"等效推动米数"，直接加到推动距离上参与比较。
# 障碍物可在可达范围内任意平移，只受墙体与其他障碍物约束

W_ROUTE = 5.0     # 落点压在"障碍物->目标"直线走廊上：机器人下一段路会再被它挡住
W_AVOID = 0.5     # 落点重新挡住了其他当前畅通的通道
# 落点仍压在该障碍物原本阻挡的其他通道上：机器人稍后还得再推一次。
# 八张图实测：0 -> 总J 1032/重复推 24；0.5 与 1.0 同为最优 841.7/2；2.0 起回升到
# 1009.5。取平台区上沿。
W_RESIDUAL = 1.0

def _swept_region(obs: MovableObstacle, nx: float, ny: float) -> Polygon: #计算矩形平移时扫过的区域
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
    goal_xy: Optional[Tuple[float, float]] = None,   # 目标点：用于避开剩余路线
    residual: Optional[Polygon] = None,  # 该障碍物当前阻挡的全部通道（软约束：尽量一并让开）
    sample_thetas: Optional[list] = None,  # 额外采样的 theta 值；None=仅采样当前朝向
) -> Tuple[bool, float, Optional[Tuple[float, float, float]]]:

    """
    为障碍物找到代价最小的可行重新放置方案

    一个候选放置位姿为【可行】当且仅当：
      * 它完全落在静态自由空间内（不撞墙）
      * 到达该位姿的直线推动过程也始终处于静态自由空间内
      * 它完全腾空 `must_clear`——即该障碍物当前挡住的通道（从而让“付费可解锁”的边真正被打通）
      * 终点位姿与推动扫掠路径都不与其他可移动障碍物 `others` 相交（物体不能互相穿透/叠放）

    障碍物在可达范围内自由平移：方向不受限（不区分推/拉），只要终点位姿与平移
    扫掠区都不撞墙、不撞其他障碍物即可。移动过程中与机器人自身的碰撞不予考虑。

    可行位姿按加权得分取最小：

        score = 推动距离 + W_ROUTE * 挡住剩余路线 + W_AVOID * 挡住其他畅通通道

    `route`（障碍物 -> 目标的直线走廊）这一项是关键：只用 `avoid` 时，密集路网让
    走廊铺满自由空间，几乎每个候选都判为“挡住了别的路”，该项退化成常数，选择塌缩
    成“最短距离 + 枚举顺序”，于是障碍物被顺着机器人的行进方向一路顶着走，每个
    重规划周期再撞上、再推。用加权和而不是字典序，是因为 route 只是直线代理，在
    迷宫里并不可靠；让它无条件压倒距离会把障碍物推到很远的糟糕位置。

    返回 (feasible, push_distance, drop_pose)。push_distance 是障碍物中心的欧氏移动距离。
    """

    best = None                      # (score, distance, pose)
    cx, cy = obs.x, obs.y

    # 额外采样的 theta 值
    thetas_to_try = [obs.theta]
    if sample_thetas:
        for th in sample_thetas:
            th_wrapped = th % math.pi
            if not any(abs(t - th_wrapped) < 1e-9 for t in thetas_to_try):
                thetas_to_try.append(th_wrapped)

    # 障碍物到目标的直线走廊：落在其中意味着机器人下一段路会再次被它挡住。
    route = None
    if goal_xy is not None:
        route = LineString([(cx, cy), goal_xy]).buffer(cfg.robot_radius)

    # 注意：曾尝试把枚举起点改成"背离机器人"的方向再向两侧展开。它只影响同分候选
    # 之间的取舍，但实测会让 two_doors_hidden_c 与 maze_to_house 从成功变为失败
    # （见提交说明的消融数据），因此保持固定的 0..2pi 顺序。
    n = cfg.drop_ring_samples

    for k in range(1, cfg.drop_radius_steps + 1):
        r = cfg.R_push * k / cfg.drop_radius_steps
        for a in range(n):
            ang = 2.0 * math.pi * a / n
            nx = round(cx + r * math.cos(ang), 3)
            ny = round(cy + r * math.sin(ang), 3)

            for th in thetas_to_try:
                end_poly = obs.polygon_at(nx, ny, th)

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

                blocks_route = 1 if (route is not None
                                     and end_poly.intersects(route)) else 0
                penalty = 1 if (avoid is not None and end_poly.intersects(avoid)) else 0
                # 仍压在该障碍物原本阻挡的通道上的比例（0~1）。硬约束只保证腾空了当前
                # 这一条边，这一项负责把"顺带让开其他通道"变成偏好，抑制反复推同一个
                # 物体。用面积占比而非计数：一次相交运算即可，且天然归一化。
                residual_frac = 0.0
                if residual is not None:
                    inter = end_poly.intersection(residual)
                    if not inter.is_empty:
                        residual_frac = inter.area / end_poly.area
                dist = math.hypot(nx - cx, ny - cy)
                # 加权求和而非字典序：各偏好都折算成"等效米数"叠加到推动距离上。
                # 字典序会让任意一项无条件压倒距离，从而选出很远、很糟的落点——迷宫里
                # route 只是直线代理、并不可靠，一旦让它独裁就会把障碍物推进真正的通道。
                score = (dist + W_ROUTE * blocks_route + W_AVOID * penalty
                         + W_RESIDUAL * residual_frac)
                cand = (round(score, 3), round(dist, 3), (nx, ny, th))
                if best is None or cand[:2] < best[:2]:
                    best = cand
    if best is None:
        return (False, math.inf, None)
    return (True, best[1], best[2])


# ---------- 计算器 1b：SE2 推动路径规划 ---------- #

def push_plan_se2(
    obs: MovableObstacle,
    must_clear_polys,                   # corridor polygons that must be cleared
    static_obstacles,                   # list of StaticObstacle
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys = None,                # additional polygons to avoid (other movable obstacles)
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    """Use PushPlanner to search full SE2 space for the cheapest pose that clears the route.

    Unlike the two-step push_plan()+push_path_plan(), this lets rotated and unrotated
    goals compete on equal footing in a single Dijkstra search.
    Returns (feasible, path, push_cost, goal_pose).
    """
    if not cfg.push_use_planner:
        return (False, None, math.inf, None)

    # cache key
    if not hasattr(push_plan_se2, '_cache'):
        push_plan_se2._cache = {}

    try:
        start_pose = (obs.x, obs.y, obs.theta)

        xmin, xmax, ymin, ymax = bounds
        bw, bh = xmax - xmin, ymax - ymin
        cell = cfg.push_cell
        nx_c = int(bw / cell)
        ny_c = int(bh / cell)
        max_states = 200000
        if nx_c * ny_c * cfg.push_n_theta > max_states:
            cell = max(cell, math.sqrt(bw * bh * cfg.push_n_theta / max_states))

        # Include others in cache key (different obstacles => different C-space)
        others_key = ("none" if others_polys is None
                      else (round(others_polys.area, 1), round(others_polys.bounds[0], 1)))
        cache_key = (obs.l, obs.d, round(obs.x, 1), round(obs.y, 1),
                     bounds, robot_pos, cfg.R_push, cell, others_key)
        if cache_key in push_plan_se2._cache:
            planner = push_plan_se2._cache[cache_key]
        else:
            # 工作圆半径 = 机器人到障碍物距离 + R_push（障碍物始终在圆内且有推送空间）
            dist_to_obs = math.hypot(robot_pos[0] - obs.x, robot_pos[1] - obs.y)
            # 合并静态障碍物和 other 可移动障碍物作为不可穿越区域
            all_walls = [so.polygon for so in static_obstacles]
            if others_polys is not None:
                if hasattr(others_polys, 'geoms'):
                    all_walls.extend(others_polys.geoms)
                elif not others_polys.is_empty:
                    all_walls.append(others_polys)

            wr = dist_to_obs + cfg.R_push + 1.0
            planner = push_planner.build_push_planner(
                static_free=None,
                wall_polys=all_walls,
                obstacle_w=obs.l, obstacle_h=obs.d,
                bounds=bounds, robot_pos=robot_pos,
                work_radius=wr if cfg.push_containment != "none" else float("inf"),
                cell=cell, n_theta=cfg.push_n_theta,
                connectivity=cfg.push_connectivity,
                rot_weight=cfg.push_rot_weight,
                containment=cfg.push_containment,
                verbose=cfg.verbose,
            )
            push_plan_se2._cache[cache_key] = planner

        # Set the corridor to clear (re-set each call as blocked edges may differ)
        corridor_verts = [push_planner.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        result = planner.plan_anywhere(start_pose)

        if result.success:
            return (True, result.path, result.cost, result.goal)
        else:
            return (False, None, math.inf, None)
    except Exception as e:
        if cfg.verbose:
            print(f"[push_plan_se2] error: {e}")
        return (False, None, math.inf, None)


def push_path_plan(
    obs: MovableObstacle,
    drop_pose: Tuple[float, float, float],
    static_obstacles,
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
) -> Tuple[bool, Optional[list], float]:
    """使用 SE2 推动规划器，规划从障碍物当前位置到目标位姿的连续推动路径。"""
    if not cfg.push_use_planner:
        return (False, None, math.inf)

    cache_key = (obs.l, obs.d, bounds, robot_pos, cfg.R_push)
    if not hasattr(push_path_plan, '_planner_cache'):
        push_path_plan._planner_cache = {}

    try:
        start_pose = (obs.x, obs.y, obs.theta)
        if (abs(start_pose[0] - drop_pose[0]) < 1e-6
                and abs(start_pose[1] - drop_pose[1]) < 1e-6
                and abs(push_planner.wrap_dtheta(start_pose[2], drop_pose[2])) < 1e-6):
            return (True, [start_pose, drop_pose], 0.0)

        xmin, xmax, ymin, ymax = bounds
        bw, bh = xmax - xmin, ymax - ymin
        cell = cfg.push_cell
        nx_c = int(bw / cell)
        ny_c = int(bh / cell)
        max_states = 200000
        if nx_c * ny_c * cfg.push_n_theta > max_states:
            cell = max(cell, math.sqrt(bw * bh * cfg.push_n_theta / max_states))

        actual_key = (obs.l, obs.d, round(obs.x, 1), round(obs.y, 1), bounds, robot_pos, cfg.R_push, cell)
        if actual_key in push_path_plan._planner_cache:
            planner = push_path_plan._planner_cache[actual_key]
        else:
            dist_to_obs = math.hypot(robot_pos[0] - obs.x, robot_pos[1] - obs.y)
            wr = dist_to_obs + cfg.R_push + 1.0
            planner = push_planner.build_push_planner(
                static_free=None,
                wall_polys=[so.polygon for so in static_obstacles],
                obstacle_w=obs.l, obstacle_h=obs.d,
                bounds=bounds, robot_pos=robot_pos,
                work_radius=wr if cfg.push_containment != "none" else float("inf"),
                cell=cell, n_theta=cfg.push_n_theta,
                connectivity=cfg.push_connectivity,
                rot_weight=cfg.push_rot_weight,
                containment=cfg.push_containment,
                verbose=cfg.verbose,
            )
            push_path_plan._planner_cache[actual_key] = planner

        result = planner.plan_path(start_pose, drop_pose)
        if result.success:
            return (True, result.path, result.cost)
        else:
            return (False, None, math.inf)
    except Exception as e:
        if cfg.verbose:
            print(f"[push_path_plan] SE2 planner error: {e}")
        return (False, None, math.inf)


# ---------- 计算器 1c：推动做功 ---------- #

def push_work(obs: MovableObstacle, push_distance: float) -> float:
    return obs.difficulty * push_distance   #推开障碍物的操作功


# -------------计算器 2：自由空间距离计算器--------------- #

def line_of_sight(free_space, a: Tuple[float, float], b: Tuple[float, float],
                  inflate: float = 0.0) -> bool:
    seg = LineString([a, b])
    if inflate > 0:
        seg = seg.buffer(inflate)
    return free_space.contains(seg)

def walking_distance(
    free_space,
    a: Tuple[float, float],
    b: Tuple[float, float],
    cfg: Config,
    grid: float = 1,
) -> float:
    
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
