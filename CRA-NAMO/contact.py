"""Robot–obstacle contact trajectory planning.

The manipulation model is: the robot is a disc that must stay *physically in
contact* with the obstacle for the whole time the obstacle is moving.  It may
grip anywhere on the perimeter, so it can push (robot behind the motion) or
pull (robot ahead of it) with no restriction on direction, and it may slide its
grip point along the surface while the obstacle moves.

Given the obstacle's SE(2) path, this module picks where the robot holds it at
every sub-step so that

  * the robot disc never overlaps a wall or another obstacle,
  * the grip point moves continuously along the surface (no teleporting from
    one side to the other),
  * a grip that has to *turn* the body has the leverage to do it (see below),
  * the total robot travel distance is minimal.

Travel is charged at the same rate as ordinary driving (lambda [N] per metre),
so the joules the robot spends carting itself around the obstacle land in the
same cost function as the joules it spends overcoming the obstacle's friction.

Leverage
--------
A grip transmits a force, and a force turns a body only through its lever arm.
The friction moment resisting rotation is ``f * rho`` — the obstacle's friction
force times its mean rotation radius, the same ``rho`` that turns an angle into
a distance everywhere else in the project — so a grip ``a`` metres from the
centre has to push with ``f * rho / a`` to keep the body turning. Halfway along
a long crate that is an unbounded force, which is exactly where a planner that
counts nothing but its own footsteps parks itself: the middle of the long face
is the point that moves least when the body pivots. Capping the ratio to the
friction force the robot is overcoming anyway (``cfg.contact_max_force_ratio``)
turns that into ``a >= rho / ratio`` and puts the grip back out towards an end,
where a person would take hold of it. It costs no new units and it binds only on
elongated bodies: ``rho`` is a *mean* radius, so it is always shorter than the
half-diagonal, and a corner grip always qualifies.

One way out
-----------
A manipulation is not a round trip. The robot is given a list of places it may
let go — the endpoints of the edge it is clearing, wherever it is standing now —
each with what it would still cost to be there rather than on its route, and it
releases at whichever comes out cheapest. Forcing it back to the exact spot it
gripped from is what used to make it walk its grip back along the surface it had
just come down, and what made a manipulation that was perfectly possible get
rejected as "cannot leave the obstacle" whenever the way back was through the
obstacle it had just moved.

Discretisation
--------------
Candidate grip points ("stations") are sampled uniformly by arc length along
the *offset curve* of the rectangle at distance ``robot_radius`` — four straight
runs joined by four quarter arcs.  A station is a fixed point in the obstacle's
local frame, so it rides along with both translation and rotation.  Choosing one
station per manipulation sub-step is then a shortest-path problem over a
(sub-step x station) lattice, solved exactly by dynamic programming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import geometry

Pose = Tuple[float, float, float]
XY = Tuple[float, float]

_INF = float("inf")
_MIN_STATIONS = 8
_MAX_STATIONS = 64
_INFLATED_CACHE_MAX = 64
_INFLATED_CACHE: Dict[Tuple[bytes, float], object] = {}


def inflate_others(others, cfg):
    """Robot-centre positions ruled out by the obstacles that are *not* being moved.

    Content-addressed by WKB: the union is rebuilt from scratch for every removal
    candidate, but the same set of obstacles recurs constantly across edges and
    cycles, and inflating it is the expensive half.
    """
    if others is None or others.is_empty:
        return None
    pad = max(cfg.robot_radius - cfg.contact_clearance, 1e-6)
    key = (others.wkb, round(pad, 9))     # a different robot inflates it differently
    cached = _INFLATED_CACHE.get(key)
    if cached is not None:
        return cached
    geom = others.buffer(pad)
    shapely.prepare(geom)
    _INFLATED_CACHE[key] = geom
    while len(_INFLATED_CACHE) > _INFLATED_CACHE_MAX:
        _INFLATED_CACHE.pop(next(iter(_INFLATED_CACHE)))
    return geom


@dataclass
class ContactPlan:
    """Where the robot is at every step of a manipulation.

    ``robot_path`` is one continuous polyline:
        [start] + [grip point per extended pose] + [end]
    ``move_offset`` maps move-path indices onto it: the robot is at
    ``robot_path[move_offset + i]`` while the obstacle is at ``poses[i]``.
    The entries before ``move_offset`` are the approach plus any walk around the
    stationary obstacle to reach the chosen grip side; the entries after the
    push are the walk out to wherever it lets go.

    ``exit_index`` says which of the caller's candidate release points that was,
    so the caller can tell where the robot now stands — -1 when it never left
    the roadmap position it set out from.
    """
    feasible: bool
    reason: str = ""
    robot_path: List[XY] = field(default_factory=list)
    move_offset: int = 0
    travel: float = 0.0            # total robot travel over the whole manipulation [m]
    exit_index: int = -1           # which of the candidate release points was used

    def at(self, i: int) -> XY:
        return self.robot_path[self.move_offset + i]

    def leg_length(self, a: int, b: int) -> float:
        """Length of robot_path[a:b+1] (indices into robot_path, a <= b)."""
        return sum(math.dist(self.robot_path[t], self.robot_path[t + 1])
                   for t in range(a, b))


def idle_plan(robot_pos: XY, n_poses: int) -> ContactPlan:
    """Degenerate plan used when contact is not required: the obstacle moves on
    its own and the robot stays put. Shaped like a real plan — start, one entry
    per pose, end — so the executor needs no special case, but every entry is the
    same point, so it accrues no travel."""
    return ContactPlan(True, "", [tuple(robot_pos)] * (max(1, n_poses) + 2), 1, 0.0)


# --- grip stations ---
def contact_stations(l: float, d: float, r: float, spacing: float) -> np.ndarray:
    """Robot centre positions in contact with an ``l`` x ``d`` rectangle.

    Returns a (K, 2) array of points in the obstacle's local frame, ordered
    counter-clockwise and equally spaced by arc length, each exactly ``r`` away
    from the rectangle's boundary.
    """
    hl, hd = l / 2.0, d / 2.0
    quarter = 0.5 * math.pi * r
    # counter-clockwise: right edge, corner, top edge, corner, ...
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
    seg_i, seg_s = 0, 0.0          # current segment and how far into it we are
    for n in range(k):
        target = n * step
        # advance to the segment containing `target`
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


def lever_arms(stations: np.ndarray, l: float, d: float) -> np.ndarray:
    """Distance from the body centre to the contact point behind each station.

    The force acts where the robot touches the rectangle, not at its own centre,
    so the station is walked back onto the surface first — for a rectangle that
    is just a clamp of the local coordinates to the faces.
    """
    cx = np.clip(stations[:, 0], -l / 2.0, l / 2.0)
    cy = np.clip(stations[:, 1], -d / 2.0, d / 2.0)
    return np.hypot(cx, cy)


def min_lever_arm(l: float, d: float, cfg) -> float:
    """Shortest lever arm that can still turn an ``l`` x ``d`` body.

    ``f * rho / a <= ratio * f``, so ``a >= rho / ratio``. A ratio of 0 (or less)
    switches the rule off and every grip may turn anything.
    """
    ratio = float(getattr(cfg, "contact_max_force_ratio", 0.0))
    if ratio <= 0.0:
        return 0.0
    return geometry.mean_rotation_radius(l, d) / ratio


def _unwrapped_angles(poses: Sequence[Pose]) -> np.ndarray:
    """Continuous heading along the path.

    The SE(2) planner stores orientation modulo pi, because a rectangle at theta
    and at theta+pi has the same footprint. A grip point does not: the body
    really did turn, and tracking the stored angle would slide the robot to the
    opposite face the moment the index wraps from pi back to 0. Accumulating the
    short-way delta — the same `wrap_dtheta` the swept volume and the rotation
    cost use — keeps the grip on the face it is actually holding.
    """
    th = [float(poses[0][2])]
    for a, b in zip(poses, poses[1:]):
        th.append(th[-1] + geometry.wrap_dtheta(a[2], b[2]))
    return np.array(th, dtype=float)


def _world_positions(stations: np.ndarray, poses: Sequence[Pose]) -> np.ndarray:
    """(T, K, 2) world positions of every station at every pose."""
    th = _unwrapped_angles(poses)
    c, s = np.cos(th), np.sin(th)
    sx = stations[:, 0][None, :]
    sy = stations[:, 1][None, :]
    x = c[:, None] * sx - s[:, None] * sy + np.array([p[0] for p in poses])[:, None]
    y = s[:, None] * sx + c[:, None] * sy + np.array([p[1] for p in poses])[:, None]
    return np.stack([x, y], axis=2)


# --- line-of-travel test ---
def _clear_line(a: XY, b: XY, free_geom, blocked_geom, trim: float) -> bool:
    """Can the robot drive straight from *a* to *b*?

    Walls are checked over the whole segment (``free_geom`` is the static free
    space already eroded by the robot radius).  Movable obstacles are checked
    over the segment trimmed by one robot radius at each end: both endpoints are
    positions the robot is known to be able to occupy, and the last radius of an
    approach is the docking move that ends flush against the obstacle.
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


def _standable(p: XY, blockers, free_geom) -> Optional[XY]:
    """The nearest spot to *p* the robot could actually be standing on.

    The reference point a manipulation is measured from is a roadmap position,
    and roadmap positions ignore movable obstacles — so it routinely lies within
    a robot radius of the very obstacle being moved, which is nowhere the robot
    could be. Testing a drive-in from there is meaningless, but *skipping* the
    test makes every grip point on the obstacle look directly reachable,
    including the ones on the far face: that is how the robot ends up crossing
    the obstacle instead of walking round it. Shifting the test out to the
    nearest place it could stand keeps the test.

    Returns *p* unchanged when it already is such a place, and None when there is
    nowhere to shift it to — the caller then genuinely has no line to test.
    """
    if not shapely.intersects_xy(blockers, *p):
        return p
    boundary = blockers.boundary
    if boundary.is_empty:
        return None
    q = nearest_points(boundary, Point(p))[0]
    if not shapely.contains_xy(free_geom, q.x, q.y):
        return None
    return (q.x, q.y)


# --- planner ---
def plan_contact(obs,
                 poses: Sequence[Pose],
                 robot_start: XY,
                 exits: Sequence[Tuple[XY, float]],
                 free_geom,
                 others_inflated,
                 cfg) -> ContactPlan:
    """Plan the robot's trajectory for one manipulation.

    ``exits``           candidate release points as ``(point, detour)`` pairs, at
                        least one. ``detour`` is what it would still cost the
                        robot to be *there* rather than on its route — 0 for the
                        far end of the edge it is clearing, the edge's length for
                        the end it set out from. It steers the choice only; the
                        travel reported is the travel actually driven.
    ``free_geom``       prepared polygon of robot-centre positions that clear the
                        static walls (static free space eroded by the robot radius).
    ``others_inflated`` prepared polygon of robot-centre positions forbidden by the
                        *other* movable obstacles (their union inflated by the robot
                        radius), or None.
    """
    if not poses:
        return ContactPlan(False, "empty move path")
    if not exits:
        return ContactPlan(False, "nowhere to let go of this obstacle")

    r = float(cfg.robot_radius)
    tol = float(cfg.contact_clearance)
    stations = contact_stations(obs.l, obs.d, r, cfg.contact_station_spacing)
    k = len(stations)
    # a grip may only be on the body while it turns if it can supply the moment
    has_lever = (lever_arms(stations, obs.l, obs.d)
                 >= min_lever_arm(obs.l, obs.d, cfg) - 1e-9)

    # how far the grip may slide along the surface within one manipulation sub-step
    step_arc = (2.0 * (obs.l + obs.d) + 2.0 * math.pi * r) / k
    # floor, not round: rounding up would let the grip slide further per sub-step
    # than contact_max_slide allows. One station is always permitted, otherwise a
    # coarsely sampled perimeter would freeze the grip in place entirely.
    max_shift = max(1, int(cfg.contact_max_slide / max(step_arc, 1e-6)))
    # room to walk around the stationary obstacle before and after the move, so
    # the far side stays reachable even when the direct line to it is blocked
    pad = int(math.ceil(k / (2.0 * max_shift)))

    ext: List[Pose] = ([poses[0]] * pad) + list(poses) + ([poses[-1]] * pad)
    world = _world_positions(stations, ext)
    t_total = len(ext)
    # the padding poses are the obstacle standing still, so only the real
    # sub-steps in the middle can ever be turns
    turns = [abs(geometry.wrap_dtheta(a[2], b[2])) > 1e-9
             for a, b in zip(ext, ext[1:])]

    # --- grip feasibility (vectorised) ---
    flat_x = world[:, :, 0].ravel()
    flat_y = world[:, :, 1].ravel()
    ok = shapely.contains_xy(free_geom, flat_x, flat_y)
    if others_inflated is not None:
        ok &= ~shapely.intersects_xy(others_inflated, flat_x, flat_y)
    feas = ok.reshape(t_total, k)
    if not feas.any():
        return ContactPlan(False, "no reachable grip point on this obstacle")

    # the obstacle itself blocks the approach and the exit walk, but not
    # the grip points (those are flush against it by construction)
    def _with_body(pose: Pose):
        body = obs.polygon_at(pose[0], pose[1], pose[2]).buffer(max(r - tol, 0.0))
        merged = body if others_inflated is None else others_inflated.union(body)
        shapely.prepare(merged)
        return merged

    approach_blockers = _with_body(poses[0])
    exit_blockers = approach_blockers if len(poses) == 1 else _with_body(poses[-1])

    # Where the drive-in and drive-out get tested from. Every reference point is a
    # roadmap position, so any of them may lie under the obstacle itself; the test
    # then runs from the nearest spot the robot could stand on instead. Distances
    # are still charged from the original points — only the line test moves.
    start_ref = _standable(robot_start, approach_blockers, free_geom)
    exit_pts = np.asarray([e[0] for e in exits], dtype=float)
    exit_detour = np.asarray([e[1] for e in exits], dtype=float)
    exit_refs = [_standable((float(p[0]), float(p[1])), exit_blockers, free_geom)
                 for p in exit_pts]

    # --- entry costs ---
    cost = np.full(k, _INF)
    for s in np.flatnonzero(feas[0]):
        p = (float(world[0, s, 0]), float(world[0, s, 1]))
        if start_ref is None or _clear_line(start_ref, p, free_geom,
                                            approach_blockers, r):
            cost[s] = math.dist(robot_start, p)
    if not np.isfinite(cost).any():
        return ContactPlan(False, "cannot reach any grip point on this obstacle")

    # --- dynamic programming ---
    parent = np.full((t_total, k), -1, dtype=np.int64)
    idx = np.arange(k)
    shifts = range(-max_shift, max_shift + 1)
    for t in range(t_total - 1):
        feas_next = feas[t + 1]
        # On a sub-step that turns the body, both ends of the slide have to be
        # grips that could have turned it. The stations passed over in between
        # are transient and are only asked to be clear.
        turning = turns[t]
        src_cost = np.where(has_lever, cost, _INF) if turning else cost
        best = np.full(k, _INF)
        best_src = np.full(k, -1, dtype=np.int64)
        for shift in shifts:
            # every station the grip slides across must be free at the next pose
            slide_ok = np.ones(k, dtype=bool)
            span = range(0, shift + 1) if shift >= 0 else range(shift, 1)
            for e in span:
                slide_ok &= np.roll(feas_next, -e)
            if turning:
                slide_ok &= np.roll(has_lever, -shift)
            step = np.linalg.norm(np.roll(world[t + 1], -shift, axis=0) - world[t],
                                  axis=1)
            cand = np.where(slide_ok, src_cost + step, _INF)
            cand = np.roll(cand, shift)          # scatter source s onto target s+shift
            upd = cand < best
            best[upd] = cand[upd]
            best_src[upd] = (idx[upd] - shift) % k
        cost, parent[t + 1] = best, best_src
        if not np.isfinite(cost).any():
            return ContactPlan(
                False, "robot cannot stay in contact for the whole manipulation")

    # --- exit costs ---
    # every (grip, release point) pair in ascending total cost, `detour` included
    # so that letting go on the far side of a cleared edge beats letting go on the
    # near side by exactly the length of the edge it saves. The first pair with a
    # clear drive-out wins.
    exit_dist = np.linalg.norm(world[-1][:, None, :] - exit_pts[None, :, :], axis=2)
    total = cost[:, None] + exit_dist + exit_detour[None, :]
    chosen, chosen_exit = -1, -1
    for flat in np.argsort(total, axis=None):
        st, ex = divmod(int(flat), len(exits))
        if not np.isfinite(total[st, ex]):
            break
        p = (float(world[-1, st, 0]), float(world[-1, st, 1]))
        ref = exit_refs[ex]
        if ref is None or _clear_line(p, ref, free_geom, exit_blockers, r):
            chosen, chosen_exit = st, ex
            break
    if chosen < 0:
        return ContactPlan(False, "robot cannot leave the obstacle after moving it")

    # --- backtrack ---
    seq = [chosen]
    for t in range(t_total - 1, 0, -1):
        seq.append(int(parent[t, seq[-1]]))
    seq.reverse()
    grip = [(float(world[t, s, 0]), float(world[t, s, 1]))
            for t, s in enumerate(seq)]
    end = (float(exit_pts[chosen_exit][0]), float(exit_pts[chosen_exit][1]))
    path = [tuple(robot_start)] + grip + [end]
    travel = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    return ContactPlan(True, "", path, 1 + pad, travel, chosen_exit)

