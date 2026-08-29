"""Connect world geometry to the obstacle SE(2) planner."""

from __future__ import annotations
import hashlib
import math
from typing import Dict, Optional, Tuple
from shapely.geometry import Polygon
from shapely.ops import unary_union

import geometry
import se2_planner
from config import Config
from obstacle import MovableObstacle


_SWEPT_MAX_DTHETA = math.pi / 12.0  # baseline for a ~1 m² obstacle (half-diag ≈ 0.5 m)
_SWEPT_REF_HALF_DIAG = 0.5

def swept_between(obs: MovableObstacle, a, b) -> Polygon:
    """Return an interpolated swept region between two obstacle poses."""
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
    """Region swept moving *obs* from where it is now to the given pose."""
    return swept_between(obs, (obs.x, obs.y, obs.theta),
                         (nx, ny, obs.theta if theta is None else theta))



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


def _walls_signature(static_obstacles) -> bytes:
    """Return a geometry digest for caching C-space planners."""
    digest = hashlib.blake2b(digest_size=16)
    for so in static_obstacles:
        digest.update(so.polygon.wkb)
    return digest.digest()

def _resolve_cell(bounds: Tuple[float, float, float, float], cfg: Config) -> float:
    xmin, xmax, ymin, ymax = bounds
    bw, bh = xmax - xmin, ymax - ymin
    cell = cfg.se2_cell
    if int(bw / cell) * int(bh / cell) * cfg.se2_n_theta > _MAX_SE2_STATES:
        cell = max(cell, math.sqrt(bw * bh * cfg.se2_n_theta / _MAX_SE2_STATES))
    return cell


def _get_planner(obs: MovableObstacle, static_obstacles,
                 bounds: Tuple[float, float, float, float],
                 cfg: Config, others_polys=None,
                 work_radius: Optional[float] = None,
                 forward_penalty: Optional[float] = None,
                 transition_safe: bool = False) -> se2_planner.SE2Planner:
    """Return a cached SE(2) planner for one obstacle and world arrangement."""
    forward_penalty = (cfg.manip_forward_penalty if forward_penalty is None
                       else forward_penalty)
    cell = _resolve_cell(bounds, cfg)
    if work_radius is None:
        work_radius = (float("inf") if cfg.se2_containment == "none"
                       else cfg.R_manip + 1.0)
    centre = (round(obs.x, 6), round(obs.y, 6))

    key = (obs.l, obs.d, bounds, centre, round(work_radius, 6), cell,
           cfg.se2_n_theta, cfg.se2_connectivity, cfg.se2_rot_weight,
           cfg.se2_containment, forward_penalty, transition_safe,
           _walls_signature(static_obstacles),
           _geometry_signature(others_polys))
    planner = _PLANNER_CACHE.get(key)
    if planner is not None:
        _PLANNER_CACHE[key] = _PLANNER_CACHE.pop(key)   # Mark as most recently used.
        return planner

    # Static and movable exclusions share one C-space.
    walls = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    planner = se2_planner.build_se2_planner(
        wall_polys=walls,
        obstacle_w=obs.l, obstacle_h=obs.d,
        bounds=bounds, robot_pos=centre,
        work_radius=work_radius,
        cell=cell, n_theta=cfg.se2_n_theta,
        connectivity=cfg.se2_connectivity,
        rot_weight=cfg.se2_rot_weight,
        containment=cfg.se2_containment,
        forward_penalty=forward_penalty,
        transition_safe=transition_safe,
        oid=obs.oid,
        verbose=cfg.verbose,
        logger=cfg.log,
    )
    _PLANNER_CACHE[key] = planner
    while len(_PLANNER_CACHE) > _PLANNER_CACHE_MAX:
        _PLANNER_CACHE.pop(next(iter(_PLANNER_CACHE)))   # Evict the least-recently-used entry.
    return planner



def path_is_clear_against(obs: MovableObstacle, path, blockers) -> bool:
    """Return whether the swept body path clears every blocker."""
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
    """Polygons plus their bounding boxes, for cheap AABB pre-filtering."""
    polys = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    return [(p, p.bounds) for p in polys]


def plan_move_se2(
    obs: MovableObstacle,
    must_clear_polys,
    static_obstacles,
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,
    goal_accept=None,
    goal_rank=None,
    path_accept=None,
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    """The cheapest place to put *obs* by push cost, for callers wanting just one."""
    for path, cost_, goal in move_se2_options(
            obs, must_clear_polys, static_obstacles, bounds, robot_pos, cfg,
            others_polys, goal_accept, goal_rank, path_accept):
        return (True, path, cost_, goal)
    return (False, None, math.inf, None)


def move_se2_options(
    obs: MovableObstacle,
    must_clear_polys,                   # corridor polygons that must be cleared
    static_obstacles,                   # list of StaticObstacle
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,                  # other movable obstacles to avoid
    goal_accept=None,                   # (goal_pose) -> bool, filters candidate drop poses
    goal_rank=None,                     # (goal_pose) -> float, extra metres of regret, for ordering
    path_accept=None,                   # (poses) -> bool, extra hard constraint on the whole path
):
    """Yield candidate obstacle relocations ordered by push cost."""
    try:
        # Use the same swept-volume-safe C-space as obstacle routing.
        planner = _get_planner(obs, static_obstacles, bounds, cfg, others_polys,
                               transition_safe=True)
        # The blocked corridor may change between planning cycles.
        corridor_verts = [geometry.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        blockers = blocker_index(static_obstacles, others_polys)

        def _validate(poses):
            if not path_is_clear_against(obs, poses, blockers):
                return False
            # A candidate is valid only if the robot can escort it in contact.
            return path_accept is None or path_accept(poses)

        for result in planner.acceptable_goals(
                (obs.x, obs.y, obs.theta),
                validate=_validate, goal_accept=goal_accept, goal_rank=goal_rank,
                n_candidates=cfg.se2_goal_candidates, ref_pos=robot_pos,
                widen=cfg.se2_goal_widen):
            end_poly = obs.polygon_at(*result.goal)
            if any(end_poly.intersects(p) for p in (must_clear_polys or [])):
                continue
            yield (result.path, result.cost, result.goal)
        reason = planner._last_refusal.reason
        if reason:
            cfg.log(f"[plan_move_se2] oid={obs.oid} {reason}")
    except Exception as e:
        cfg.log(f"[plan_move_se2] error: {e}")


def _split_mixed_leg(obs: MovableObstacle, a, b, blockers) -> Optional[list]:
    """Split a mixed translation-and-rotation leg into two validated legs."""
    for middle in ((b[0], b[1], a[2]), (a[0], a[1], b[2])):
        if path_is_clear_against(obs, [a, middle, b], blockers):
            return [a, middle, b]
    return None


def _verified_prefix(obs: MovableObstacle, path, blockers, cfg: Config) -> list:
    """Return the longest prefix whose swept legs pass collision validation."""
    if not path or len(path) < 2:
        return []
    out = [path[0]]
    for a, b in zip(path, path[1:]):
        if path_is_clear_against(obs, [a, b], blockers):
            out.append(b)
            continue
        repaired = _split_mixed_leg(obs, a, b, blockers)
        if repaired is None:
            cfg.log(f"[dynamics] oid={obs.oid} route cut short at "
                    f"({a[0]:,.2f}, {a[1]:,.2f}): next leg does not clear")
            break
        out += repaired[1:]
    return out if len(out) >= 2 else []


def plan_route_se2(obs: MovableObstacle,
                   goal_pose: Tuple[float, float, float],
                   static_obstacles,
                   bounds: Tuple[float, float, float, float],
                   cfg: Config,
                   others_polys=None) -> Optional[list]:
    """Plan and swept-validate a route for an independently moving obstacle."""
    try:
        planner = _get_planner(obs, static_obstacles, bounds, cfg, others_polys,
                               work_radius=float("inf"), forward_penalty=0.0,
                               transition_safe=True)
        planner.set_corridor([])
        result = planner.plan_path((obs.x, obs.y, obs.theta), tuple(goal_pose))
        if not result.success:
            cfg.log(f"[dynamics] oid={obs.oid} no route: {result.reason}")
            return None
        path = _verified_prefix(obs, result.path,
                                blocker_index(static_obstacles, others_polys),
                                cfg)
        if not path:
            cfg.log(f"[dynamics] oid={obs.oid} no route: "
                    "nothing it can reach without clipping something")
            return None
        return path
    except Exception as e:
        cfg.log(f"[dynamics] oid={obs.oid} route error: {type(e).__name__}: {e}")
        return None
