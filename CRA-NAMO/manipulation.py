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
    """Region swept as *obs* moves from pose *a* to pose *b*.

    Interpolates the motion and unions the convex hull of each consecutive pair.
    A hull approximates the arc of a rotation by its chord, so the bulge between
    chord and arc is missed — a false negative, exactly the kind that makes a
    path look safe at planning time and graze something at execution time. The
    bulge is `R(1 - cos(dtheta/2))` and scales with body size, so the angular
    step is scaled inversely: small parts keep a coarse 15 degrees, long ones get
    subdivided until their missed bulge is comparable.
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
    """What the walls are, for a cache key.

    The C-space is built around the walls, so two scenarios sharing a bounds and
    an obstacle size would otherwise share a planner built for the wrong
    building — one run of a process would answer the next one's questions. The
    digest is over the geometry itself rather than the identity of the list,
    because a list is not a thing that stays put.
    """
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
    """Planner for one body in one arrangement of the world.

    The reachable disc is centred on the body itself, which is what `R_manip`
    has always been documented to mean — how far an obstacle may be relocated
    from where it stands. Anchoring it there rather than to the robot is what
    lets one planner serve every edge of a cycle and keep the Dijkstra result it
    has already paid for; where the robot is standing enters only as the
    drop-pose bias, which `plan_anywhere` takes per call.

    An obstacle moving under its own steam overrides both: the whole map is in
    reach and there is nobody to be nudged away from.
    """
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
        _PLANNER_CACHE[key] = _PLANNER_CACHE.pop(key)   # mark as most recently used
        return planner

    # static walls + other movable obstacles, together as impassable region
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
        _PLANNER_CACHE.pop(next(iter(_PLANNER_CACHE)))   # evict least recently used
    return planner



def path_is_clear_against(obs: MovableObstacle, path, blockers) -> bool:
    """Does the body clear every blocker along the whole path?

    Uses the shared `geometry.CONTACT_AREA_EPS`, the same tolerance the executor
    applies — a planner that is more permissive than the executor produces routes
    the executor then refuses.
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
    """Polygons plus their bounding boxes, for cheap AABB pre-filtering."""
    polys = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    return [(p, p.bounds) for p in polys]


def plan_move_se2(
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
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    """Plan where to put *obs* so it stops blocking, and how to get it there."""
    try:
        # Same C-space the body's own routes are planned in, and for the same
        # reason: a cell whose centre is clear says nothing about the step into
        # it, and `_validate` below tests the steps. Planning in cells that are
        # only clear at their centres means generating routes for the validator
        # to throw away — and when it throws away all of them, reporting that a
        # body has nowhere to go when it has somewhere the robot could push it.
        planner = _get_planner(obs, static_obstacles, bounds, cfg, others_polys,
                               transition_safe=True)
        # reset corridor every time: the blocked edge differs per cycle
        corridor_verts = [geometry.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        blockers = blocker_index(static_obstacles, others_polys)

        def _validate(poses):
            if not path_is_clear_against(obs, poses, blockers):
                return False
            # the obstacle can go there, but only counts as a solution if the
            # robot can escort it the whole way while staying in contact
            return path_accept is None or path_accept(poses)

        result = planner.plan_anywhere((obs.x, obs.y, obs.theta),
                                       validate=_validate, goal_accept=goal_accept,
                                       goal_rank=goal_rank,
                                       n_candidates=cfg.se2_goal_candidates,
                                       ref_pos=robot_pos,
                                       widen=cfg.se2_goal_widen)
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


def _split_mixed_leg(obs: MovableObstacle, a, b, blockers) -> Optional[list]:
    """Turn one leg that both slides and turns into two that do one thing each.

    The SE(2) move set never combines the two — every planned leg is a pure
    slide or a pure pivot, which is what the C-space was built to answer for.
    The one leg that mixes them is the one joining the body's true pose to the
    grid pose the search started from, and it is the one that clips door jambs.
    Doing the two halves in turn is a motion of the kind the planner validated;
    which half goes first is whichever of the two is clear.
    """
    for middle in ((b[0], b[1], a[2]), (a[0], a[1], b[2])):
        if path_is_clear_against(obs, [a, middle, b], blockers):
            return [a, middle, b]
    return None


def _verified_prefix(obs: MovableObstacle, path, blockers, cfg: Config) -> list:
    """The longest part of *path* the body can be shown to travel safely.

    The search answers in cell centres, so what it hands back is a sequence of
    poses that are each clear rather than a motion that is clear throughout: a
    grid pose either side of a door jamb says nothing about the corner between
    them. Every leg is therefore swept and tested against the walls and the
    other bodies before the route is believed, and the route is cut at the first
    leg that cannot be shown to be clear.
    """
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
    """Plan an obstacle's own route from where it stands to *goal_pose*.

    The counterpart of `plan_move_se2` for a body that moves under its own
    steam: no corridor to clear, no drop pose to choose, and no robot to stay
    within reach of — only "get from here to there without hitting anything".
    Returns the pose list, or None when there is no route.

    The route comes back swept-volume verified, the same way `plan_move_se2`
    verifies the robot's manipulations. Nothing downstream re-tests it against
    the walls, so this is the only place a body under its own steam is stopped
    from cutting a corner off a doorway. It is planned in a C-space that keeps
    enough clearance for the step between two cells to be as clear as the cells
    themselves, so the verification is a check on the answer rather than the
    thing standing between the body and the wall.
    """
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
