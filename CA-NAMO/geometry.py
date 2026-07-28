"""
几何计算：
1. push_plan()     : 环形采样放置位姿——推动距离、可行性与最终“放置位姿”
2. push_plan_se2() : 在 SE2 全空间搜索能腾空走廊的最优位姿（首选路径）
3. push_path_plan(): 给定放置位姿，规划到达它的连续推动路径（回退路径）
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
from shapely.geometry import Polygon, LineString
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
    static_free,                       # 仅由墙壁界定的自由空间（prepared 几何，只用 contains）
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

_MAX_PUSH_STATES = 200_000
# 构型空间的构建是全流程最贵的一步，故按"决定该 C-space 的全部参数"缓存规划器实例。
# 每个实例约 4 MB（若干 200k 元素的布尔网格 + Dijkstra 结果），而缓存键现在是精确的，
# 命中率低于从前的模糊键 —— 必须限容，否则长时间运行会把内存吃光。按 LRU 淘汰。
_PLANNER_CACHE: Dict[tuple, push_planner.PushPlanner] = {}
_PLANNER_CACHE_MAX = 32


def _polygon_parts(geom):
    """把任意 shapely 几何摊平成多边形列表（丢弃线/点等退化部件）。

    `others` 里混有 `swept.intersection(wall)` 产生的接触区，退化情况下会是
    LineString 或 GeometryCollection；它们没有 `.exterior`，直接喂给规划器会抛异常
    并被吞掉，表现为"这个障碍物忽然不可推动"。
    """
    if geom is None or geom.is_empty:
        return []
    parts = getattr(geom, "geoms", None)
    parts = list(parts) if parts is not None else [geom]
    return [p for p in parts if p.geom_type == "Polygon" and not p.is_empty]


def _geometry_signature(geom) -> tuple:
    """几何体的精确缓存标识。

    曾用 `(面积, bounds[0])` 概括 `others`，但障碍物做竖直平移时这两个量都不变，
    不同布局会命中同一条缓存 —— 于是复用了按【旧位置】构建的构型空间，规划出的
    推动路径直接穿过实际存在的障碍物。这里改用精确的 WKB 字节：命中即等价，
    不命中最多是重建一次（安全），绝不会张冠李戴。
    """
    if geom is None or geom.is_empty:
        return ("none",)
    return ("wkb", geom.wkb)


def _resolve_cell(bounds: Tuple[float, float, float, float], cfg: Config) -> float:
    """在状态总数上限内自适应放大网格单元。"""
    xmin, xmax, ymin, ymax = bounds
    bw, bh = xmax - xmin, ymax - ymin
    cell = cfg.push_cell
    if int(bw / cell) * int(bh / cell) * cfg.push_n_theta > _MAX_PUSH_STATES:
        cell = max(cell, math.sqrt(bw * bh * cfg.push_n_theta / _MAX_PUSH_STATES))
    return cell


def _get_push_planner(obs: MovableObstacle, static_obstacles,
                      bounds: Tuple[float, float, float, float],
                      robot_pos: Tuple[float, float], cfg: Config,
                      others_polys=None) -> push_planner.PushPlanner:
    """取得（或构建）与这组参数对应的 PushPlanner，两条推动路径共用。"""
    cell = _resolve_cell(bounds, cfg)
    # 工作圆半径 = 机器人到障碍物距离 + R_push（障碍物始终在圆内且有推送空间）
    dist_to_obs = math.hypot(robot_pos[0] - obs.x, robot_pos[1] - obs.y)
    work_radius = (float("inf") if cfg.push_containment == "none"
                   else dist_to_obs + cfg.R_push + 1.0)

    key = (obs.l, obs.d, bounds, robot_pos, round(work_radius, 6), cell,
           cfg.push_n_theta, cfg.push_connectivity, cfg.push_rot_weight,
           cfg.push_containment, _geometry_signature(others_polys))
    planner = _PLANNER_CACHE.get(key)
    if planner is not None:
        _PLANNER_CACHE[key] = _PLANNER_CACHE.pop(key)   # 标记为最近使用
        return planner

    # 静态墙体 + 其他可移动障碍物，一起作为不可穿越区域
    walls = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    planner = push_planner.build_push_planner(
        wall_polys=walls,
        obstacle_w=obs.l, obstacle_h=obs.d,
        bounds=bounds, robot_pos=robot_pos,
        work_radius=work_radius,
        cell=cell, n_theta=cfg.push_n_theta,
        connectivity=cfg.push_connectivity,
        rot_weight=cfg.push_rot_weight,
        containment=cfg.push_containment,
        verbose=cfg.verbose,
    )
    _PLANNER_CACHE[key] = planner
    while len(_PLANNER_CACHE) > _PLANNER_CACHE_MAX:
        _PLANNER_CACHE.pop(next(iter(_PLANNER_CACHE)))   # 淘汰最久未用的
    return planner


def push_plan_se2(
    obs: MovableObstacle,
    must_clear_polys,                   # 必须被腾空的走廊多边形
    static_obstacles,                   # StaticObstacle 列表
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,                  # 需要避开的其他可移动障碍物
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    """在 SE2 全空间搜索能腾空走廊的最省代价位姿。

    与 push_plan() + push_path_plan() 的两步法不同，这里让旋转与不旋转的候选目标
    在同一次 Dijkstra 里平等竞争。返回 (feasible, path, push_cost, goal_pose)。
    """
    if not cfg.push_use_planner:
        return (False, None, math.inf, None)
    try:
        planner = _get_push_planner(obs, static_obstacles, bounds, robot_pos,
                                    cfg, others_polys)
        # 每次都要重设走廊：被阻挡的边逐周期不同
        corridor_verts = [push_planner.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        result = planner.plan_anywhere((obs.x, obs.y, obs.theta))
        if result.success:
            return (True, result.path, result.cost, result.goal)
        return (False, None, math.inf, None)
    except Exception as e:
        cfg.log(f"[push_plan_se2] error: {e}")
        return (False, None, math.inf, None)


def push_path_plan(
    obs: MovableObstacle,
    drop_pose: Tuple[float, float, float],
    static_obstacles,
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,
) -> Tuple[bool, Optional[list], float]:
    """规划从障碍物当前位置到指定放置位姿的连续推动路径。

    `others_polys` 必须与 push_plan() 用的是同一组：漏传会让这条回退路径规划出
    穿过其他可移动障碍物的推动轨迹。
    """
    if not cfg.push_use_planner:
        return (False, None, math.inf)
    try:
        start_pose = (obs.x, obs.y, obs.theta)
        if (abs(start_pose[0] - drop_pose[0]) < 1e-6
                and abs(start_pose[1] - drop_pose[1]) < 1e-6
                and abs(push_planner.wrap_dtheta(start_pose[2], drop_pose[2])) < 1e-6):
            return (True, [start_pose, drop_pose], 0.0)

        planner = _get_push_planner(obs, static_obstacles, bounds, robot_pos,
                                    cfg, others_polys)
        result = planner.plan_path(start_pose, drop_pose)
        if result.success:
            return (True, result.path, result.cost)
        return (False, None, math.inf)
    except Exception as e:
        cfg.log(f"[push_path_plan] SE2 planner error: {e}")
        return (False, None, math.inf)


# ---------- 计算器 1c：推动做功 ---------- #

def push_work(difficulty: float, push_distance: float) -> float:
    """推开障碍物的操作功。difficulty 即物理模型中的 μmg。

    刚体在均匀压力下被推动，做功分为平移与绕形心转动两部分：

        J1 = μmg · d                 (平移距离 d)
        J2 = μmg · θ · r̄             (转动 θ，r̄ = 摩擦力矩臂)

        W  = J1 + J2 = difficulty · (d + θ·r̄)

    `push_distance` 传入的正是括号里那个 `d + θ·r̄`：SE2 规划器以
    rot_weight = r̄ 累加代价（见 push_planner.mean_rotation_radius），
    所以它返回的 cost 天然就是这个和，无需在此再拆分。

    显式传入 difficulty 而不是从障碍物对象里读：规划期用的是估计值（来自信念），
    结算期用的是 ground truth（来自世界），两者绝不能混。
    """
    return difficulty * push_distance
