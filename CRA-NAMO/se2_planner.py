"""SE(2) manipulation path planner.

Searches a discretised (x, y, theta) grid with a Dial-bucket Dijkstra for a
continuous path that carries a rectangular body from its current pose to one
that no longer blocks a corridor. Knows nothing about robots, costs, or
contact - it answers only "can this body get there, and by what route".
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

import log
from geometry import (
    c_obstacle,
    convex_hull,
    inside_convex,
    mean_rotation_radius,
    offset_bbox,
    polygon_exterior_coords,
    rect_corners,
    sat_rect_intersect,
    wrap_dtheta,
)

# --- SE2 grid + Dial-bucket Dijkstra ---

@dataclass
class SE2PlanResult:
    success: bool
    reason: str = ""
    cost: float = float("nan")
    trans_length: float = float("nan")
    rot_total: float = float("nan")
    goal: Optional[Tuple[float, float, float]] = None
    path: List[Tuple[float, float, float]] = field(default_factory=list)


_UNSTICK_RELAX_STEPS = 4
_START_COLLISION_EPS = -1e-6
_CORRIDOR_MASK_CACHE: Dict[tuple, tuple] = {}
_CORRIDOR_MASK_CACHE_MAX = 4096


class SE2Planner:
    _W_AXIS = 990
    _W_DIAG = 1400
    _W_KNIGHT = 2214

    def __init__(self,
                 wall_polys: List[np.ndarray],
                 obstacle_w: float,
                 obstacle_h: float,
                 bounds: Tuple[float, float, float, float],
                 robot_pos: Tuple[float, float] = (0.0, 0.0),
                 work_radius: float = float("inf"),
                 cell: float = 0.08,
                 n_theta: int = 24,
                 connectivity: int = 8,
                 rot_weight: Optional[float] = None,
                 containment: str = "centroid",
                 forward_penalty: float = 0.0,
                 oid: int = -1,
                 verbose: bool = False):
        assert connectivity in (8, 16), "connectivity supports only 8 or 16"
        assert containment in ("body", "centroid")

        self.oid = oid          # only used for log identification, not in any computation
        self.forward_penalty = float(forward_penalty)
        self.wall_polys = wall_polys
        self.obstacle_w = obstacle_w
        self.obstacle_h = obstacle_h
        self.robot_pos = robot_pos
        self.work_radius = work_radius
        self.cell = cell
        self.n_theta = n_theta
        self.connectivity = connectivity
        self.containment = containment
        self.verbose = verbose

        xmin, xmax, ymin, ymax = bounds
        self.xs = np.arange(xmin + cell / 2, xmax, cell)
        self.ys = np.arange(ymin + cell / 2, ymax, cell)
        self.nx, self.ny = len(self.xs), len(self.ys)
        self.thetas = np.arange(n_theta) * (math.pi / n_theta)
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")

        self.r_half_diag = 0.5 * math.hypot(obstacle_w, obstacle_h)
        self.dtheta = math.pi / n_theta
        self.rot_weight = (mean_rotation_radius(obstacle_w, obstacle_h)
                           if rot_weight is None else float(rot_weight))
        self.rot_step_cost = self.rot_weight * self.dtheta

        self.bulge = self.r_half_diag * (1.0 - math.cos(self.dtheta / 2.0))
        self.pose_margin = 0.1 * cell + self.bulge
        self.margin = self.pose_margin
        self.snap_margin = 0.5 * math.hypot(cell, cell) + self.bulge
        self.unit = cell / self._W_AXIS
        self.W_rot = int(round(self.rot_step_cost / self.unit))

        self._build_moves()
        self._build_cspace()
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, int]] = None

        if verbose:
            log.emit(f"[se2] oid={self.oid} | "
                     f"free {self.free.mean() * 100:.0f}% "
                     f"-> reachable {self.allowed.mean() * 100:.0f}%")

    # --- move definitions ---
    def _build_moves(self) -> None:
        """(di, dj, dk, integer weight). Translation and rotation are mutually exclusive."""
        mv: List[Tuple[int, int, int, int]] = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            mv.append((di, dj, 0, self._W_AXIS))
        for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            mv.append((di, dj, 0, self._W_DIAG))
        if self.connectivity == 16:
            for di, dj in ((1, 2), (2, 1), (-1, 2), (-2, 1),
                           (1, -2), (2, -1), (-1, -2), (-2, -1)):
                mv.append((di, dj, 0, self._W_KNIGHT))
        mv.append((0, 0, 1, self.W_rot))
        mv.append((0, 0, -1, self.W_rot))
        self.moves = mv

        groups: Dict[int, List[Tuple[int, int]]] = {}
        for di, dj, dk, w in mv:
            if dk == 0:
                groups.setdefault(w, []).append((di, dj))
        self._move_groups = [
            (w, np.array([m[0] for m in g], dtype=np.int64)[:, None],
                np.array([m[1] for m in g], dtype=np.int64)[:, None])
            for w, g in groups.items()]

    # --- configuration space build ---
    def _build_cspace(self) -> None:
        rx, ry = self.robot_pos
        R = self.work_radius
        w, h = self.obstacle_w, self.obstacle_h

        shape = (self.nx, self.ny, self.n_theta)
        blocked = np.zeros(shape, dtype=bool)
        indisk = np.zeros(shape, dtype=bool)
        rot_blocked = np.zeros(shape, dtype=bool)

        corners = [rect_corners(0.0, 0.0, w, h, th) for th in self.thetas]

        for k, th in enumerate(self.thetas):
            O = corners[k]

            # collision check against walls
            layer_b = np.zeros((self.nx, self.ny), dtype=bool)
            for wp in self.wall_polys:
                layer_b |= inside_convex(c_obstacle(wp, O), self.X, self.Y,
                                         margin=self.margin)
            blocked[:, :, k] = layer_b
            H = convex_hull(np.vstack([O, corners[(k + 1) % self.n_theta]]))
            layer_r = np.zeros((self.nx, self.ny), dtype=bool)
            for wp in self.wall_polys:
                layer_r |= inside_convex(c_obstacle(wp, H), self.X, self.Y,
                                         margin=self.pose_margin)
            rot_blocked[:, :, k] = layer_r

            # workspace circle constraint
            if not np.isfinite(R):
                indisk[:, :, k] = True
            elif self.containment == "centroid":
                indisk[:, :, k] = ((self.X - rx) ** 2 + (self.Y - ry) ** 2) <= R * R
            else:  # body mode
                ok = np.ones((self.nx, self.ny), dtype=bool)
                for ox, oy in O:  # O = corner offsets relative to centroid
                    ok &= ((self.X + ox - rx) ** 2 + (self.Y + oy - ry) ** 2) <= R * R
                indisk[:, :, k] = ok

        self.free = ~blocked
        self.in_disk = indisk
        self.allowed = self.free & self.in_disk
        self.rot_ok = ~rot_blocked
        self._rot_ok_base = self.rot_ok.copy()
        self._unstuck = False
        self._unstuck_for: Optional[Tuple[int, int, int]] = None
        self._route_window: Optional[Tuple[int, int, int, int]] = None
        self._route_mask: Optional[np.ndarray] = None

    # --- start un-stuck ---
    def _true_collision(self, pose: Tuple[float, float, float],
                        clearance: float = 1e-9) -> bool:
        O = rect_corners(0.0, 0.0, self.obstacle_w, self.obstacle_h, pose[2])
        corners = O + np.array([pose[0], pose[1]])
        for wp in self.wall_polys:
            if sat_rect_intersect(corners, wp, eps=clearance):
                return True
        return False

    def _unstick_clearance(self, pose: Tuple[float, float, float]) -> Optional[float]:
        c = self.margin
        for _ in range(_UNSTICK_RELAX_STEPS):
            if not self._true_collision(pose, c):
                return c
            c *= 0.5
        return None if self._true_collision(pose) else 0.0

    def _unstick_start(self, start_idx: Tuple[int, int, int]) -> int:
        if self._unstuck:
            if self._unstuck_for == start_idx:
                return 0
            np.copyto(self.allowed, self._allowed_base)   # start changed, restore old exemptions
            np.copyto(self.rot_ok, self._rot_ok_base)
            self._unstuck = False
            self._unstuck_for = None
        if self.allowed[start_idx]:
            return 0
        if not self.in_disk[start_idx]:
            return 0                      # outside workspace circle – not a margin artifact, let caller error
        clearance = self._unstick_clearance(self._pose(*start_idx))
        if clearance is None:
            return 0                      # genuinely embedded in wall, cannot exempt
        max_cells = max(3, int(math.ceil(self.r_half_diag / self.cell)) + 2)
        si, sj, sk = start_idx
        stack = [start_idx]
        freed = 0
        seen = {start_idx}
        while stack:
            i, j, k = stack.pop()
            if not self.allowed[i, j, k]:
                self.allowed[i, j, k] = True
                self.rot_ok[i, j, k] = True
                self.rot_ok[i, j, (k - 1) % self.n_theta] = True
                freed += 1
            for di, dj, dk, _w in self.moves:
                ni, nj = i + di, j + dj
                nk = (k + dk) % self.n_theta
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                if abs(ni - si) > max_cells or abs(nj - sj) > max_cells:
                    continue
                nxt = (ni, nj, nk)
                if nxt in seen or self.allowed[nxt]:
                    continue              # already unblocked or originally free -> reached normal space, stop
                seen.add(nxt)
                if not self.in_disk[nxt]:
                    continue
                if self._true_collision(self._pose(*nxt), clearance):
                    continue              # true collision, keep as occupied
                stack.append(nxt)

        if freed:
            self._unstuck = True
            self._unstuck_for = start_idx
            if self.verbose:
                x, y, _ = self._pose(*start_idx)
                log.emit(f"[se2] oid={self.oid} unstuck {freed:,} states at "
                         f"({x:.1f}, {y:.1f}) clearance={clearance:.3f}")
        return freed

    # --- index helpers ---
    def _snap(self, x: float, y: float, theta: float) -> Tuple[int, int, int]:
        i = int(np.clip(round((x - self.xs[0]) / self.cell), 0, self.nx - 1))
        j = int(np.clip(round((y - self.ys[0]) / self.cell), 0, self.ny - 1))
        k = int(round((theta % math.pi) / self.dtheta)) % self.n_theta
        return i, j, k

    def _pose(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        return float(self.xs[i]), float(self.ys[j]), float(self.thetas[k])

    def _flat(self, i, j, k):
        return (np.int64(i) * self.ny + np.int64(j)) * self.n_theta + np.int64(k)

    def _unflat(self, idx: np.ndarray):
        nT = self.n_theta
        nyT = self.ny * nT
        i = idx // nyT
        rem = idx - i * nyT
        j = rem // nT
        return i, j, rem - j * nT

    # --- Dial-bucket Dijkstra ---
    def _search(self, start_idx: Tuple[int, int, int], max_bucket: int | None = None):
        """Dial-bucket Dijkstra. If *max_bucket* is given, stop expanding once the
        current bucket exceeds that value — states beyond that distance are unreachable
        in practice and skipping them saves the majority of the search time."""
        N = self.nx * self.ny * self.n_theta
        allowed = self.allowed.reshape(-1)
        rot_ok = self.rot_ok
        INF = np.int64(1) << 62
        dist = np.full(N, INF, dtype=np.int64)
        parent = np.full(N, -1, dtype=np.int64)

        s = int(self._flat(*start_idx))
        dist[s] = 0
        buckets: Dict[int, List[np.ndarray]] = {0: [np.array([s], dtype=np.int64)]}
        max_b = 0
        nx, ny, nT = self.nx, self.ny, self.n_theta

        b = 0
        while b <= max_b:
            while True:  # inner loop: compatible with zero-weight edges (when rotation is free)
                arrs = buckets.pop(b, None)
                if not arrs:
                    break
                idx = arrs[0] if len(arrs) == 1 else np.concatenate(arrs)
                idx = np.unique(idx[dist[idx] == b])  # discard stale entries
                if idx.size == 0:
                    continue
                i, j, k = self._unflat(idx)
                # --- translation ---
                for w, DI, DJ in self._move_groups:
                    ni, nj = i + DI, j + DJ            # (num directions, frontier size)
                    ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
                    if not ok.any():
                        continue
                    nidx = ((ni * ny + nj) * nT + np.broadcast_to(k, ni.shape))[ok]
                    src = np.broadcast_to(idx, ok.shape)[ok]
                    nidx, first = np.unique(nidx, return_index=True)
                    src = src[first]
                    nd = b + w
                    better = allowed[nidx] & (dist[nidx] > nd)
                    tgt = nidx[better]
                    if tgt.size == 0:
                        continue
                    dist[tgt] = nd
                    parent[tgt] = src[better]
                    buckets.setdefault(nd, []).append(tgt)
                    if nd > max_b:
                        max_b = nd

                # --- rotation ---
                nd = b + self.W_rot
                for dk in (1, -1):
                    kk = k if dk == 1 else (k - 1) % nT
                    nidx = (i * ny + j) * nT + (k + dk) % nT
                    # (i, j, k) pairwise distinct -> (i, j, k+dk) also pairwise distinct, no dedup needed
                    better = (rot_ok[i, j, kk] & allowed[nidx]
                              & (dist[nidx] > nd))
                    tgt = nidx[better]
                    if tgt.size == 0:
                        continue
                    dist[tgt] = nd
                    parent[tgt] = idx[better]
                    buckets.setdefault(nd, []).append(tgt)
                    if nd > max_b:
                        max_b = nd
            b += 1
            if max_bucket is not None and b > max_bucket:
                break

        return dist, parent

    def _trace(self, parent: np.ndarray, goal_flat: int,
               start_pose: Optional[Tuple[float, float, float]] = None
               ) -> List[Tuple[float, float, float]]:
        chain = [goal_flat]
        cur = goal_flat
        while parent[cur] >= 0:
            cur = int(parent[cur])
            chain.append(cur)
        chain.reverse()
        i, j, k = self._unflat(np.array(chain, dtype=np.int64))
        path = [self._pose(int(a), int(b_), int(c)) for a, b_, c in zip(i, j, k)]
        if start_pose is not None and self._pose_gap(start_pose, path[0]) > 1e-9:
            path.insert(0, start_pose)
        return path

    @staticmethod
    def _pose_gap(a, b) -> float:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(wrap_dtheta(a[2], b[2])))

    # --- planning ---
    def _build_corridor_mask(self, corridor_polys: List[np.ndarray]):
        x0, y0, cell = float(self.xs[0]), float(self.ys[0]), self.cell
        layers = []                       # (k, i0, i1, j0, j1, layer)
        wi0, wi1, wj0, wj1 = self.nx, 0, self.ny, 0
        for k, th in enumerate(self.thetas):
            O = rect_corners(0.0, 0.0, self.obstacle_w, self.obstacle_h, th)
            for cp in corridor_polys:
                C = c_obstacle(cp, O)
                bb = offset_bbox(C, self.snap_margin)
                if bb is None:                       # unreliable solution -> fall back to full grid layer
                    i0, i1, j0, j1 = 0, self.nx, 0, self.ny
                else:
                    bxmin, bxmax, bymin, bymax = bb
                    i0 = max(0, int(math.ceil((bxmin - x0) / cell)))
                    i1 = min(self.nx, int(math.floor((bxmax - x0) / cell)) + 1)
                    j0 = max(0, int(math.ceil((bymin - y0) / cell)))
                    j1 = min(self.ny, int(math.floor((bymax - y0) / cell)) + 1)
                    if i0 >= i1 or j0 >= j1:         # window entirely outside the grid
                        continue
                layer = inside_convex(C, self.xs[i0:i1, None],
                                      self.ys[None, j0:j1], margin=self.snap_margin)
                if not layer.any():
                    continue
                layers.append((k, i0, i1, j0, j1, layer))
                wi0, wi1 = min(wi0, i0), max(wi1, i1)
                wj0, wj1 = min(wj0, j0), max(wj1, j1)

        if wi0 >= wi1:
            return None, None
        mask = np.zeros((wi1 - wi0, wj1 - wj0, self.n_theta), dtype=bool)
        for k, i0, i1, j0, j1, layer in layers:
            mask[i0 - wi0:i1 - wi0, j0 - wj0:j1 - wj0, k] |= layer
        return (wi0, wi1, wj0, wj1), mask

    def set_corridor(self, corridor_polys: List[np.ndarray]):
        key = (self.obstacle_w, self.obstacle_h, self.n_theta, self.cell,
               self.nx, self.ny, float(self.xs[0]), float(self.ys[0]),
               round(self.snap_margin, 12),
               tuple(np.asarray(cp, dtype=float).tobytes() for cp in corridor_polys))
        hit = _CORRIDOR_MASK_CACHE.get(key)
        if hit is not None:
            _CORRIDOR_MASK_CACHE[key] = _CORRIDOR_MASK_CACHE.pop(key)  # mark as most recently used
            win, packed, shape = hit
            self._route_window = win
            self._route_mask = (None if win is None else
                                np.unpackbits(packed, count=shape[0] * shape[1] * shape[2])
                                .astype(bool).reshape(shape))
            return

        win, mask = self._build_corridor_mask(corridor_polys)
        self._route_window, self._route_mask = win, mask
        _CORRIDOR_MASK_CACHE[key] = (
            win,
            None if mask is None else np.packbits(mask),
            None if mask is None else mask.shape)
        while len(_CORRIDOR_MASK_CACHE) > _CORRIDOR_MASK_CACHE_MAX:
            _CORRIDOR_MASK_CACHE.pop(next(iter(_CORRIDOR_MASK_CACHE)))

    # --- goal preference ---
    def _forward_bias(self, start_pose: Tuple[float, float, float]):
        if self.forward_penalty <= 0.0:
            return None
        fx = start_pose[0] - self.robot_pos[0]
        fy = start_pose[1] - self.robot_pos[1]
        n = math.hypot(fx, fy)
        if n < 1e-9:            # robot and obstacle coincide, no meaningful direction
            return None
        fx, fy = fx / n, fy / n
        along = (self.X - start_pose[0]) * fx + (self.Y - start_pose[1]) * fy
        # dist uses self.unit metres per integer unit; penalty must use same scale
        return np.maximum(along, 0.0) * (self.forward_penalty / self.unit)

    def _select_goal(self, dist: np.ndarray,
                     start_pose: Tuple[float, float, float],
                     n_best: int = 1) -> Tuple[list, bool]:
        INF = np.int64(1) << 62
        nx, ny, nT = self.nx, self.ny, self.n_theta
        d3 = dist.reshape(nx, ny, nT)
        win = self._route_window
        sub = saved = None
        if win is not None:
            i0, i1, j0, j1 = win
            sub = d3[i0:i1, j0:j1, :]
            saved = sub[self._route_mask]    # cached dist is shared, must be restored exactly
            sub[self._route_mask] = INF
        try:
            bias = self._forward_bias(start_pose)
            if bias is None:
                score = np.where(dist >= INF, np.inf, dist.astype(float))
                order = self._smallest(score, n_best)
                return order, bool(order and score[order[0]] < np.inf)
            kbest = d3.argmin(axis=2)
            dmin = np.take_along_axis(d3, kbest[:, :, None], axis=2)[:, :, 0]
            score = np.where(dmin >= INF, np.inf, dmin.astype(float))
            score2 = score + bias
            flat_ij = self._smallest(score2.reshape(-1), n_best)
            order = []
            for f in flat_ij:
                i, j = divmod(int(f), ny)
                order.append(int((i * ny + j) * nT + int(kbest[i, j])))
            return order, bool(flat_ij and score.reshape(-1)[flat_ij[0]] < np.inf)
        finally:
            if sub is not None:
                sub[self._route_mask] = saved

    @staticmethod
    def _smallest(score: np.ndarray, n: int) -> list:
        """Indices of the n smallest finite elements in score, sorted by value ascending."""
        n = max(1, n)
        if n == 1:
            b = int(np.argmin(score))
            return [b] if np.isfinite(score[b]) else [b]
        idx = np.argpartition(score, min(n, score.size - 1))[:n]
        idx = idx[np.argsort(score[idx])]
        return [int(v) for v in idx if np.isfinite(score[v])] or [int(np.argmin(score))]

    def plan_anywhere(self,
                      start_pose: Tuple[float, float, float],
                      validate=None,
                      n_candidates: int = 10,
                      goal_accept=None) -> SE2PlanResult:
        start_idx = self._snap(*start_pose)

        if not self.in_disk[start_idx]:
            return SE2PlanResult(False, "start pose is outside the workspace circle")
        O_start = rect_corners(0, 0, self.obstacle_w, self.obstacle_h, start_pose[2])
        for wp in self.wall_polys:
            if sat_rect_intersect(
                O_start + np.array([start_pose[0], start_pose[1]]),
                wp, eps=_START_COLLISION_EPS):
                return SE2PlanResult(False, "start pose collides with wall")

        if self._unstick_start(start_idx):
            self._cache = None            # C-space changed, invalidate previous Dijkstra result

        # Cap the Dijkstra search radius to the corridor extent + obstacle size.
        # An obstacle never needs to be moved farther than this to clear the corridor;
        # exploring beyond it wastes time on unreachable / irrelevant states.
        max_bucket = None
        if self._route_window is not None:
            i0, i1, j0, j1 = self._route_window
            corridor_diag = math.hypot((i1 - i0) * self.cell, (j1 - j0) * self.cell)
            max_reach = (corridor_diag + self.r_half_diag * 2.0) * 1.5
            max_bucket = int(max_reach / self.unit)

        start_flat = int(self._flat(*start_idx))
        if self._cache is None or self._cache[2] != start_flat:
            self._cache = self._search(start_idx, max_bucket=max_bucket) + (start_flat,)
        dist, parent, _cached_start = self._cache

        one_shot = validate is None and goal_accept is None
        candidates, reachable = self._select_goal(
            dist, start_pose, n_best=1 if one_shot else n_candidates)
        if not reachable:
            return SE2PlanResult(False, "no reachable pose can clear the path")

        nyT = self.ny * self.n_theta
        fallback = None
        for best in candidates:
            goal_idx = (best // nyT, (best % nyT) // self.n_theta,
                        best % self.n_theta)
            goal_pose = self._pose(*goal_idx)
            wanted = goal_accept is None or goal_accept(goal_pose)
            # `validate` is the expensive one — for CA-NAMO it plans the robot's
            # whole escort — and `goal_accept` is a couple of predicates, so ask
            # the cheap question first. A drop pose the caller does not want is
            # only ever validated to fill the one fallback slot; once that is
            # taken the rest cost nothing but a trace.
            if not wanted and fallback is not None:
                continue
            poses = self._trace(parent, best, start_pose)
            if validate is not None and not validate(poses):
                continue                  # this path fails swept-volume validation, try next candidate
            trans = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                        for a, b in zip(poses, poses[1:]))
            rot = sum(abs(wrap_dtheta(a[2], b[2])) for a, b in zip(poses, poses[1:]))
            cost = trans + self.rot_weight * rot
            result = SE2PlanResult(True, "", cost, trans, rot, goal_pose, poses)
            # candidates come in cost order, so the first accepted one is the
            # cheapest acceptable drop pose; keep the cheapest overall as fallback
            if wanted:
                return result
            fallback = result
        if fallback is not None:
            return fallback

        return SE2PlanResult(False, "all candidate routes failed swept-volume validation")

    def plan_path(self,
                  start_pose: Tuple[float, float, float],
                  goal_pose: Tuple[float, float, float]) -> SE2PlanResult:
        start_idx = self._snap(*start_pose)
        goal_idx = self._snap(*goal_pose)

        if not self.in_disk[start_idx]:
            return SE2PlanResult(False, "start pose is outside the robot workspace circle")
        # check start against walls (no margin). tangent contact allowed, see _START_COLLISION_EPS.
        O_start = rect_corners(0, 0, self.obstacle_w, self.obstacle_h, start_pose[2])
        for wp in self.wall_polys:
            if sat_rect_intersect(
                O_start + np.array([start_pose[0], start_pose[1]]),
                wp, eps=_START_COLLISION_EPS):
                return SE2PlanResult(False, "start pose collides with wall")

        if not self.free[goal_idx]:
            return SE2PlanResult(False, "goal pose collides with wall (or is too close)")

        if not self.in_disk[goal_idx]:
            return SE2PlanResult(False, "goal pose is outside the robot workspace circle")

        if self._unstick_start(start_idx):
            self._cache = None

        start_flat = int(self._flat(*start_idx))
        if self._cache is None or self._cache[2] != start_flat:
            self._cache = self._search(start_idx) + (start_flat,)
        dist, parent, _cs = self._cache

        goal_flat = int(self._flat(*goal_idx))
        INF = np.int64(1) << 62

        if dist[goal_flat] >= INF:
            return SE2PlanResult(False, "no feasible path from start to goal in the discrete search space")

        poses = self._trace(parent, goal_flat, start_pose)
        trans = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses, poses[1:]))
        rot = sum(abs(wrap_dtheta(a[2], b[2])) for a, b in zip(poses, poses[1:]))
        cost = trans + self.rot_weight * rot     # same as plan_anywhere: path length is the basis
        return SE2PlanResult(True, "", cost, trans, rot, self._pose(*goal_idx), poses)

# --- Factory function ---
def build_se2_planner(wall_polys,             # list of shapely Polygons — impassable region outlines
                       obstacle_w: float,
                       obstacle_h: float,
                       bounds: Tuple[float, float, float, float],
                       robot_pos: Tuple[float, float] = (0.0, 0.0),
                       work_radius: float = float("inf"),
                       cell: float = 0.08,
                       n_theta: int = 24,
                       connectivity: int = 8,
                       rot_weight: Optional[float] = None,
                       containment: str = "centroid",
                       forward_penalty: float = 0.0,
                       oid: int = -1,
                       verbose: bool = False) -> SE2Planner:
    """Build a SE2Planner from CA-NAMO-style data."""
    wall_verts = [polygon_exterior_coords(p) for p in wall_polys]

    return SE2Planner(
        wall_polys=wall_verts,
        obstacle_w=obstacle_w,
        obstacle_h=obstacle_h,
        bounds=bounds,
        robot_pos=robot_pos,
        work_radius=work_radius,
        cell=cell,
        n_theta=n_theta,
        connectivity=connectivity,
        rot_weight=rot_weight,
        containment=containment,
        forward_penalty=forward_penalty,
        oid=oid,
        verbose=verbose,
    )

