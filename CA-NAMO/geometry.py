"""Adaptation layer between the simulated world and the SE2 push planner
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
from shapely.geometry import Polygon
from shapely.ops import unary_union
from obstacle import MovableObstacle
from config import Config
import push_planner

# ------------- Calculator 1: push planning --------------- #

_SWEPT_MAX_DTHETA = math.pi / 12.0  # baseline for a ~1 m² obstacle (half-diag ≈ 0.5 m)
_SWEPT_REF_HALF_DIAG = 0.5

def _swept_between(obs: MovableObstacle, a, b) -> Polygon:
    ax, ay, ath = a
    bx, by, bth = b
    dtheta = push_planner.wrap_dtheta(ath, bth)
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

def _swept_region(obs: MovableObstacle, nx: float, ny: float,
                  theta: Optional[float] = None) -> Polygon:
    return _swept_between(obs, (obs.x, obs.y, obs.theta),
                          (nx, ny, obs.theta if theta is None else theta))


# ---------- Calculator 1b: SE2 push path planning ---------- #

_MAX_PUSH_STATES = 200_000
_PLANNER_CACHE: Dict[tuple, push_planner.PushPlanner] = {}
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
    cell = cfg.push_cell
    if int(bw / cell) * int(bh / cell) * cfg.push_n_theta > _MAX_PUSH_STATES:
        cell = max(cell, math.sqrt(bw * bh * cfg.push_n_theta / _MAX_PUSH_STATES))
    return cell


def _get_push_planner(obs: MovableObstacle, static_obstacles,
                      bounds: Tuple[float, float, float, float],
                      robot_pos: Tuple[float, float], cfg: Config,
                      others_polys=None) -> push_planner.PushPlanner:
    cell = _resolve_cell(bounds, cfg)
    dist_to_obs = math.hypot(robot_pos[0] - obs.x, robot_pos[1] - obs.y)
    work_radius = (float("inf") if cfg.push_containment == "none"
                   else dist_to_obs + cfg.R_push + 1.0)

    key = (obs.l, obs.d, bounds, robot_pos, round(work_radius, 6), cell,
           cfg.push_n_theta, cfg.push_connectivity, cfg.push_rot_weight,
           cfg.push_containment, cfg.push_forward_penalty,
           _geometry_signature(others_polys))
    planner = _PLANNER_CACHE.get(key)
    if planner is not None:
        _PLANNER_CACHE[key] = _PLANNER_CACHE.pop(key)   # mark as most recently used
        return planner

    # static walls + other movable obstacles, together as impassable region
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
        forward_penalty=cfg.push_forward_penalty,
        oid=obs.oid,
        verbose=cfg.verbose,
    )
    _PLANNER_CACHE[key] = planner
    while len(_PLANNER_CACHE) > _PLANNER_CACHE_MAX:
        _PLANNER_CACHE.pop(next(iter(_PLANNER_CACHE)))   # evict least recently used
    return planner

_PATH_CONTACT_AREA_EPS = 1e-6

def _path_is_clear_against(obs: MovableObstacle, path, blockers) -> bool:
    if not path or len(path) < 2 or not blockers:
        return True
    for a, b in zip(path, path[1:]):
        swept = _swept_between(obs, a, b)
        sminx, sminy, smaxx, smaxy = swept.bounds
        for poly, (pminx, pminy, pmaxx, pmaxy) in blockers:
            if pmaxx < sminx or pminx > smaxx or pmaxy < sminy or pminy > smaxy:
                continue
            if not swept.intersects(poly):
                continue
            if swept.intersection(poly).area > _PATH_CONTACT_AREA_EPS:
                return False
    return True

def _blocker_index(static_obstacles, others_polys):
    polys = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    return [(p, p.bounds) for p in polys]

def push_plan_se2(
    obs: MovableObstacle,
    must_clear_polys,                   # corridor polygons that must be cleared
    static_obstacles,                   # list of StaticObstacle
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,                  # other movable obstacles to avoid
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    if not cfg.push_use_planner:
        return (False, None, math.inf, None)
    try:
        planner = _get_push_planner(obs, static_obstacles, bounds, robot_pos,
                                    cfg, others_polys)
        # reset corridor every time: the blocked edge differs per cycle
        corridor_verts = [push_planner.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        blockers = _blocker_index(static_obstacles, others_polys)

        def _validate(poses):
            return _path_is_clear_against(obs, poses, blockers)

        result = planner.plan_anywhere((obs.x, obs.y, obs.theta),
                                       validate=_validate)
        if not result.success:
            cfg.log(f"[push_plan_se2] oid={obs.oid} {result.reason}")
            return (False, None, math.inf, None)
        end_poly = obs.polygon_at(*result.goal)
        if any(end_poly.intersects(p) for p in (must_clear_polys or [])):
            cfg.log(f"[push_plan_se2] oid={obs.oid} target pose does not actually clear the corridor – no solution")
            return (False, None, math.inf, None)
        return (True, result.path, result.cost, result.goal)
    except Exception as e:
        cfg.log(f"[push_plan_se2] error: {e}")
        return (False, None, math.inf, None)
    
# ---------- Calculator 1c: push work ---------- #

def se2_path_cost(obs: MovableObstacle, poses, cfg: Config) -> float:
    if poses is None or len(poses) < 2:
        return 0.0
    rot_weight = (push_planner.mean_rotation_radius(obs.l, obs.d)
                  if cfg.push_rot_weight is None else float(cfg.push_rot_weight))
    total = 0.0
    for a, b in zip(poses, poses[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        total += rot_weight * abs(push_planner.wrap_dtheta(a[2], b[2]))
    return total

def push_work(difficulty: float, push_distance: float) -> float:
    return difficulty * push_distance
