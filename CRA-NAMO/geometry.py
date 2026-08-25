"""Provide shared geometric and collision primitives."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# How much overlap counts as "still clear". About 1 mm^2: purely there to absorb
# convex-hull approximation and floating-point residue, not a licence for real
# penetration. Planning and execution MUST share it — when planning is the more
# permissive of the two, it hands the executor paths the executor then rejects,
# and the manipulation gets blacklisted for a rounding error.
CONTACT_AREA_EPS = 1e-6


def rect_corners(cx: float, cy: float, w: float, h: float,
                 theta: float) -> np.ndarray:
    """Corners of a w x h rectangle centred at (cx, cy), rotated by theta.

    Counter-clockwise, which is what `inside_convex` and `convex_hull` assume.
    Called with (0, 0, ...) it yields corner offsets in the body frame.
    """
    dx, dy = w / 2.0, h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=float)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cy])


def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(np.asarray(pts, dtype=float), 9), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def minkowski_sum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    pairwise = (A[:, None, :] + B[None, :, :]).reshape(-1, 2)
    return convex_hull(pairwise)


def c_obstacle(shape_poly: np.ndarray, obs_corners_local: np.ndarray) -> np.ndarray:
    """Configuration-space obstacle: where the body centre may not go."""
    return minkowski_sum(shape_poly, -obs_corners_local)


def inside_convex(poly: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  margin: float = 0.0) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    shape = np.broadcast(X, Y).shape
    if n == 0:
        return np.zeros(shape, dtype=bool)
    if n < 3:  # degenerate case: point or line segment
        a, b = poly[0], poly[-1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-18:
            d2 = (X - a[0]) ** 2 + (Y - a[1]) ** 2
        else:
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0.0, 1.0)
            d2 = (X - (a[0] + t * ab[0])) ** 2 + (Y - (a[1] + t * ab[1])) ** 2
        return d2 <= margin ** 2

    inside = np.ones(shape, dtype=bool)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        e = q - p
        L = math.hypot(e[0], e[1])
        if L < 1e-12:
            continue
        nx, ny = e[1] / L, -e[0] / L  # outward normal of counter-clockwise polygon
        inside &= ((X - p[0]) * nx + (Y - p[1]) * ny) <= margin
    return inside


def offset_bbox(poly: np.ndarray, margin: float):
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    if n == 0:
        return None
    if n < 3:      # degenerate point/segment: criterion is "distance <= margin", just expand bbox
        return (poly[:, 0].min() - margin, poly[:, 0].max() + margin,
                poly[:, 1].min() - margin, poly[:, 1].max() + margin)

    normals, offsets = [], []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        e = q - p
        L = math.hypot(e[0], e[1])
        if L < 1e-12:
            continue
        nx, ny = e[1] / L, -e[0] / L
        normals.append((nx, ny))
        offsets.append(p[0] * nx + p[1] * ny + margin)
    m = len(normals)
    if m < 3:
        return None

    xs, ys = [], []
    for i in range(m):
        (ax, ay), ca = normals[i - 1], offsets[i - 1]
        (bx, by), cb = normals[i], offsets[i]
        det = ax * by - ay * bx
        if abs(det) < 1e-12:      # adjacent edges collinear: vertex does not exist, skip
            continue
        xs.append((ca * by - cb * ay) / det)
        ys.append((ax * cb - bx * ca) / det)
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def mean_rotation_radius(w: float, h: float) -> float:
    """Mean distance from the centroid to a point of a w x h rectangle.

    Converts an angle into an equivalent translation distance, so rotating and
    sliding can be summed into one path length.
    """
    if w <= 0.0 or h <= 0.0:
        return 0.0
    s = math.hypot(w, h)
    return (s / 6.0
            + (w * w / (12.0 * h)) * math.log((s + h) / w)
            + (h * h / (12.0 * w)) * math.log((s + w) / h))


def wrap_dtheta(a: float, b: float) -> float:
    """Shortest turn from a to b, modulo pi.

    A rectangle at theta and theta+pi has the same footprint, so orientation is
    only meaningful mod pi. Everything that follows a body's rotation — swept
    volumes, rotation cost, contact grip points — must go through this, and
    anything tracking a body-fixed point must *accumulate* it rather than read
    the stored angle, or it will jump half a turn when the index wraps.
    """
    return (b - a + math.pi / 2) % math.pi - math.pi / 2


def sat_rect_intersect(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> bool:
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            axis = np.array([-e[1], e[0]])
            L = math.hypot(axis[0], axis[1])
            if L < 1e-12:
                continue
            axis /= L
            pa, pb = A @ axis, B @ axis
            if pa.max() < pb.min() - eps or pb.max() < pa.min() - eps:
                return False
    return True


def polygon_exterior_coords(polygon) -> np.ndarray:
    """Vertices of a shapely Polygon as an (N, 2) array, without the repeated last point."""
    coords = list(polygon.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return np.array(coords, dtype=float)

