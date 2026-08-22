"""仿真世界与 SE(2) 规划器之间的适配层:扫掠体计算与 plan_move_se2 封装。"""


from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
from shapely.geometry import Polygon
from shapely.ops import unary_union

import geometry
import se2_planner
from config import Config
from obstacle import MovableObstacle

# --- 扫掠体 ---

_SWEPT_MAX_DTHETA = math.pi / 12.0  # 基准角步长,对应约 1 m² 障碍物(半对角线约 0.5 m)
_SWEPT_REF_HALF_DIAG = 0.5

def swept_between(obs: MovableObstacle, a, b) -> Polygon:
    """计算 *obs* 从位姿 *a* 运动到 *b* 的扫掠区域。

    凸包以弦代弧会漏掉弓高 R(1 - cos(dtheta/2)),它随物体尺寸增大,
    故角步长按半对角线长度反比缩放(小物体保持 15 度,长物体细分)。
    """
    ax, ay, ath = a
    bx, by, bth = b
    dtheta = geometry.wrap_dtheta(ath, bth)
    if abs(dtheta) < 1e-9:
        return unary_union([obs.polygon_at(ax, ay, ath),
                            obs.polygon_at(bx, by, ath)]).convex_hull
    half_diag = 0.5 * math.hypot(obs.l, obs.d)
    max_dtheta = _SWEPT_MAX_DTHETA * _SWEPT_REF_HALF_DIAG / max(half_diag, _SWEPT_REF_HALF_DIAG)
    steps = max(2, int(math.ceil(abs(dtheta) / max_dtheta)))
    poses = [obs.polygon_at(ax + (bx - ax) * i / steps,
                            ay + (by - ay) * i / steps,
                            ath + dtheta * i / steps)
             for i in range(steps + 1)]
    return unary_union([unary_union([p, q]).convex_hull
                        for p, q in zip(poses, poses[1:])])

def swept_region(obs: MovableObstacle, nx: float, ny: float,
                 theta: Optional[float] = None) -> Polygon:
    """计算 *obs* 从当前位姿移动到指定位姿的扫掠区域。"""
    return swept_between(obs, (obs.x, obs.y, obs.theta),
                         (nx, ny, obs.theta if theta is None else theta))


# --- 规划器缓存 ---

_MAX_SE2_STATES = 200_000
_PLANNER_CACHE: Dict[tuple, se2_planner.SE2Planner] = {}
_PLANNER_CACHE_MAX = 32

def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    parts = getattr(geom, "geoms", None)
    parts = list(parts) if parts is not None else [geom]
    return [p for p in parts if p.geom_type == "Polygon" and not p.is_empty]

def _geometry_signature(geom) -> tuple:
    if geom is None or geom.is_empty:
        return ("none",)
    return ("wkb", geom.wkb)

def _resolve_cell(bounds: Tuple[float, float, float, float], cfg: Config) -> float:
    xmin, xmax, ymin, ymax = bounds
    bw, bh = xmax - xmin, ymax - ymin
    cell = cfg.se2_cell
    if int(bw / cell) * int(bh / cell) * cfg.se2_n_theta > _MAX_SE2_STATES:
        cell = max(cell, math.sqrt(bw * bh * cfg.se2_n_theta / _MAX_SE2_STATES))
    return cell


def _get_planner(obs: MovableObstacle, static_obstacles,
                 bounds: Tuple[float, float, float, float],
                 robot_pos: Tuple[float, float], cfg: Config,
                 others_polys=None) -> se2_planner.SE2Planner:
    cell = _resolve_cell(bounds, cfg)
    dist_to_obs = math.hypot(robot_pos[0] - obs.x, robot_pos[1] - obs.y)
    work_radius = (float("inf") if cfg.se2_containment == "none"
                   else dist_to_obs + cfg.R_manip + 1.0)

    key = (obs.l, obs.d, bounds, robot_pos, round(work_radius, 6), cell,
           cfg.se2_n_theta, cfg.se2_connectivity, cfg.se2_rot_weight,
           cfg.se2_containment, cfg.manip_forward_penalty,
           _geometry_signature(others_polys))
    planner = _PLANNER_CACHE.get(key)
    if planner is not None:
        _PLANNER_CACHE[key] = _PLANNER_CACHE.pop(key)   # 标记为最近使用
        return planner

    # 静态墙体与其他可移动障碍物,合并为不可通行区域
    walls = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    planner = se2_planner.build_se2_planner(
        wall_polys=walls,
        obstacle_w=obs.l, obstacle_h=obs.d,
        bounds=bounds, robot_pos=robot_pos,
        work_radius=work_radius,
        cell=cell, n_theta=cfg.se2_n_theta,
        connectivity=cfg.se2_connectivity,
        rot_weight=cfg.se2_rot_weight,
        containment=cfg.se2_containment,
        forward_penalty=cfg.manip_forward_penalty,
        oid=obs.oid,
        verbose=cfg.verbose,
    )
    _PLANNER_CACHE[key] = planner
    while len(_PLANNER_CACHE) > _PLANNER_CACHE_MAX:
        _PLANNER_CACHE.pop(next(iter(_PLANNER_CACHE)))   # 淘汰最久未使用的
    return planner


# --- 路径校验 ---

def path_is_clear_against(obs: MovableObstacle, path, blockers) -> bool:
    """整条路径上物体是否与所有阻挡物保持净空。

    使用与执行器相同的 geometry.CONTACT_AREA_EPS 容差:规划若比执行宽松,
    产生的路线会被执行器拒绝。
    """
    if not path or len(path) < 2 or not blockers:
        return True
    for a, b in zip(path, path[1:]):
        swept = swept_between(obs, a, b)
        sminx, sminy, smaxx, smaxy = swept.bounds
        for poly, (pminx, pminy, pmaxx, pmaxy) in blockers:
            if pmaxx < sminx or pminx > smaxx or pmaxy < sminy or pminy > smaxy:
                continue
            if not swept.intersects(poly):
                continue
            if swept.intersection(poly).area > geometry.CONTACT_AREA_EPS:
                return False
    return True

def blocker_index(static_obstacles, others_polys):
    """多边形及其包围盒,用于廉价的 AABB 预筛。"""
    polys = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    return [(p, p.bounds) for p in polys]


def plan_move_se2(
    obs: MovableObstacle,
    must_clear_polys,                   # 必须清空的走廊多边形
    static_obstacles,                   # StaticObstacle 列表
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,                  # 需避让的其他可移动障碍物
    goal_accept=None,                   # (goal_pose) -> bool, 过滤候选放置位姿
    path_accept=None,                   # (poses) -> bool, 对整条路径的额外硬约束
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    """规划把 *obs* 挪到哪里才能不再挡路,以及怎么挪过去。"""
    try:
        planner = _get_planner(obs, static_obstacles, bounds, robot_pos,
                               cfg, others_polys)
        # 每次重设走廊:每个周期被堵的边不同
        corridor_verts = [geometry.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        blockers = blocker_index(static_obstacles, others_polys)

        def _validate(poses):
            if not path_is_clear_against(obs, poses, blockers):
                return False
            # 障碍物能过去还不够,还须机器人全程贴身护送才算解
            return path_accept is None or path_accept(poses)

        result = planner.plan_anywhere((obs.x, obs.y, obs.theta),
                                       validate=_validate, goal_accept=goal_accept)
        if not result.success:
            cfg.log(f"[plan_move_se2] oid={obs.oid} {result.reason}")
            return (False, None, math.inf, None)
        end_poly = obs.polygon_at(*result.goal)
        if any(end_poly.intersects(p) for p in (must_clear_polys or [])):
            cfg.log(f"[plan_move_se2] oid={obs.oid} target pose does not actually clear the corridor – no solution")
            return (False, None, math.inf, None)
        return (True, result.path, result.cost, result.goal)
    except Exception as e:
        cfg.log(f"[plan_move_se2] error: {e}")
        return (False, None, math.inf, None)

