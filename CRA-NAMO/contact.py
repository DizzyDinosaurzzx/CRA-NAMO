"""Plan robot contact trajectories during obstacle manipulation."""

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
    """Return cached robot-centre exclusions for stationary obstacles."""
    if others is None or others.is_empty:
        return None
    pad = max(cfg.robot_radius - cfg.contact_clearance, 1e-6)
    key = (others.wkb, round(pad, 9))
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
    """Store the robot path, alignment and travel for one manipulation."""
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
    """Return a zero-travel plan for an obstacle that moves without contact."""
    return ContactPlan(True, "", [tuple(robot_pos)] * (max(1, n_poses) + 2), 1, 0.0)


def contact_stations(l: float, d: float, r: float, spacing: float) -> np.ndarray:
    """Return equally spaced contact centres around a rectangle."""
    hl, hd = l / 2.0, d / 2.0
    quarter = 0.5 * math.pi * r
    # Segments follow the rectangle boundary counter-clockwise.
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
    seg_i, seg_s = 0, 0.0
    for n in range(k):
        target = n * step
        # Locate the segment containing this perimeter distance.
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
    """Return the lever arm from the body centre to each contact station."""
    cx = np.clip(stations[:, 0], -l / 2.0, l / 2.0)
    cy = np.clip(stations[:, 1], -d / 2.0, d / 2.0)
    return np.hypot(cx, cy)


def min_lever_arm(l: float, d: float, cfg) -> float:
    """Return the minimum lever arm allowed by the configured force ratio."""
    ratio = float(getattr(cfg, "contact_max_force_ratio", 0.0))
    if ratio <= 0.0:
        return 0.0
    return geometry.mean_rotation_radius(l, d) / ratio


def _unwrapped_angles(poses: Sequence[Pose]) -> np.ndarray:
    """Unwrap headings so a tracked grip stays on the same rectangle face."""
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


def _clear_line(a: XY, b: XY, free_geom, blocked_geom, trim: float) -> bool:
    """Check a straight robot segment against static and movable exclusions."""
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
    """Return the nearest feasible robot position to a roadmap reference point."""
    if not shapely.intersects_xy(blockers, *p):
        return p
    boundary = blockers.boundary
    if boundary.is_empty:
        return None
    q = nearest_points(boundary, Point(p))[0]
    if shapely.contains_xy(free_geom, q.x, q.y):
        return (q.x, q.y)
    standable = boundary.intersection(free_geom)
    if standable.is_empty:
        return None
    q = nearest_points(standable, Point(p))[0]
    return (q.x, q.y)


def plan_contact(obs,
                 poses: Sequence[Pose],
                 robot_start: XY,
                 exits: Sequence[Tuple[XY, float]],
                 free_geom,
                 others_inflated,
                 cfg) -> ContactPlan:
    """Plan a collision-free robot trajectory for one obstacle manipulation."""
    if not poses:
        return ContactPlan(False, "empty move path")
    if not exits:
        return ContactPlan(False, "nowhere to let go of this obstacle")

    r = float(cfg.robot_radius)
    tol = float(cfg.contact_clearance)
    stations = contact_stations(obs.l, obs.d, r, cfg.contact_station_spacing)
    k = len(stations)
    # Turning requires a contact station with enough lever arm.
    has_lever = (lever_arms(stations, obs.l, obs.d)
                 >= min_lever_arm(obs.l, obs.d, cfg) - 1e-9)

    # Limit contact-index movement to the configured slide distance.
    step_arc = (2.0 * (obs.l + obs.d) + 2.0 * math.pi * r) / k
    max_shift = max(1, int(cfg.contact_max_slide / max(step_arc, 1e-6)))
    # Padding gives the robot room to walk around the stationary obstacle.
    pad = int(math.ceil(k / (2.0 * max_shift)))

    ext: List[Pose] = ([poses[0]] * pad) + list(poses) + ([poses[-1]] * pad)
    world = _world_positions(stations, ext)
    t_total = len(ext)
    turns = [abs(geometry.wrap_dtheta(a[2], b[2])) > 1e-9
             for a, b in zip(ext, ext[1:])]

    flat_x = world[:, :, 0].ravel()
    flat_y = world[:, :, 1].ravel()
    ok = shapely.contains_xy(free_geom, flat_x, flat_y)
    if others_inflated is not None:
        ok &= ~shapely.intersects_xy(others_inflated, flat_x, flat_y)
    feas = ok.reshape(t_total, k)
    if not feas.any():
        return ContactPlan(False, "no reachable grip point on this obstacle")

    # The body blocks approach and exit paths, but not its contact stations.
    def _with_body(pose: Pose):
        body = obs.polygon_at(pose[0], pose[1], pose[2]).buffer(max(r - tol, 0.0))
        merged = body if others_inflated is None else others_inflated.union(body)
        shapely.prepare(merged)
        return merged

    approach_blockers = _with_body(poses[0])
    exit_blockers = approach_blockers if len(poses) == 1 else _with_body(poses[-1])

    # Roadmap references may lie inside the body, so line tests use standable points.
    start_ref = _standable(robot_start, approach_blockers, free_geom)
    exit_pts = np.asarray([e[0] for e in exits], dtype=float)
    exit_detour = np.asarray([e[1] for e in exits], dtype=float)
    exit_refs = [_standable((float(p[0]), float(p[1])), exit_blockers, free_geom)
                 for p in exit_pts]

    cost = np.full(k, _INF)
    for s in np.flatnonzero(feas[0]):
        p = (float(world[0, s, 0]), float(world[0, s, 1]))
        if start_ref is not None and _clear_line(start_ref, p, free_geom,
                                                 approach_blockers, r):
            cost[s] = math.dist(robot_start, p)
    if not np.isfinite(cost).any():
        return ContactPlan(
            False, "the robot has nowhere to stand to reach this obstacle"
            if start_ref is None else "cannot reach any grip point on this obstacle")

    parent = np.full((t_total, k), -1, dtype=np.int64)
    idx = np.arange(k)
    shifts = range(-max_shift, max_shift + 1)
    for t in range(t_total - 1):
        feas_next = feas[t + 1]
        # A turning step requires both endpoint grips to supply the needed moment.
        turning = turns[t]
        src_cost = np.where(has_lever, cost, _INF) if turning else cost
        best = np.full(k, _INF)
        best_src = np.full(k, -1, dtype=np.int64)
        for shift in shifts:
            # Every station crossed during a slide must be free at the next pose.
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

    # Evaluate release points by total travel plus the caller's detour estimate.
    exit_dist = np.linalg.norm(world[-1][:, None, :] - exit_pts[None, :, :], axis=2)
    total = cost[:, None] + exit_dist + exit_detour[None, :]
    chosen, chosen_exit = -1, -1
    for flat in np.argsort(total, axis=None):
        st, ex = divmod(int(flat), len(exits))
        if not np.isfinite(total[st, ex]):
            break
        p = (float(world[-1, st, 0]), float(world[-1, st, 1]))
        ref = exit_refs[ex]
        if ref is not None and _clear_line(p, ref, free_geom, exit_blockers, r):
            chosen, chosen_exit = st, ex
            break
    if chosen < 0:
        return ContactPlan(False, "robot cannot leave the obstacle after moving it")

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
