"""Construct and query a uniform augmented roadmap."""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import shapely
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from scipy.spatial import KDTree
from config import Config
EdgeKey = Tuple[int, int]

class Roadmap:
    """Uniform roadmap over free space defined by static obstacles."""

    def __init__(self, workspace: Polygon, static_obstacles, cfg: Config):
        self.cfg = cfg
        self.workspace = workspace
        polys = [so.polygon for so in static_obstacles]
        self.static_obstacles = static_obstacles  # for use by the SE(2) planner
        # Nested free-space sets use progressively stricter robot clearance:
        #   static_free      — anywhere inside the workspace that is not a wall
        #   free_eroded      — where the robot's *centre* may sit (eroded by r)
        #   free_eroded_tol  — same, minus a contact_clearance hair of slack
        self.static_free = workspace.difference(unary_union(polys)) if polys else workspace
        self.static_free_prep = prep(self.static_free)
        self.free_eroded = self.static_free.buffer(-cfg.robot_radius, quad_segs=16)
        self.free_eroded_prep = prep(self.free_eroded)
        self.nodes: List[Tuple[float, float]] = []
        self.adj: Dict[int, List[int]] = {}
        self.edge_len: Dict[EdgeKey, float] = {}
        self.edge_corridor: Dict[EdgeKey, Polygon] = {}
        self._kdtree: KDTree | None = None
        self._corridor_tree: STRtree | None = None
        self._corridor_keys: List[EdgeKey] = []
        self._free_eroded_tol: Polygon | None = None
        self._build()
        self._rebuild_kdtree()

    @property
    def free_eroded_tol(self):
        if self._free_eroded_tol is None:
            eps = max(1e-6, self.cfg.robot_radius - self.cfg.contact_clearance)
            geom = self.static_free.buffer(-eps, quad_segs=16)
            shapely.prepare(geom)
            self._free_eroded_tol = geom
        return self._free_eroded_tol

    def _build(self):
        cfg = self.cfg
        assert cfg.grid_step > 0 and cfg.conn_radius > 0
        minx, miny, maxx, maxy = self.workspace.bounds
        step = cfg.grid_step
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for iy in range(int((maxy - miny) / step) + 1):
            y = miny + iy * step
            for ix in range(int((maxx - minx) / step) + 1):
                x = minx + ix * step
                if not self.free_eroded_prep.contains(Point(x, y)):
                    continue
                nid = len(self.nodes)
                self.nodes.append((round(x, 3), round(y, 3)))
                self.adj[nid] = []
                b = (int(x // cfg.conn_radius), int(y // cfg.conn_radius))
                buckets.setdefault(b, []).append(nid)
        # A 3x3 bucket neighborhood covers every edge within conn_radius.
        for nid, (x, y) in enumerate(self.nodes):
            bx, by = int(x // cfg.conn_radius), int(y // cfg.conn_radius)
            for dbx in (-1, 0, 1):
                for dby in (-1, 0, 1):
                    for mid in buckets.get((bx + dbx, by + dby), ()):
                        if mid <= nid:
                            continue
                        self._try_edge(nid, mid)

    def _try_edge(self, u: int, v: int):
        a, b = self.nodes[u], self.nodes[v]
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        if dist > self.cfg.conn_radius + 1e-9:
            return
        seg = LineString([a, b])
        if not self.free_eroded_prep.contains(seg):
            return
        key = (u, v) if u < v else (v, u)
        self.edge_len[key] = round(dist, 4)
        self.edge_corridor[key] = seg.buffer(self.cfg.robot_radius, cap_style=2)
        self.adj[u].append(v)
        self.adj[v].append(u)

    def neighbors(self, u: int):
        for v in self.adj[u]:
            key = (u, v) if u < v else (v, u)
            yield v, key, self.edge_len[key]

    def count_blocked_edges(self, poly: Polygon) -> int:
        return len(self.corridors_intersecting(poly))

    def corridors_intersecting(self, poly: Polygon) -> List[EdgeKey]:
        """Return only roadmap edges whose swept corridor intersects *poly*."""
        if self._corridor_tree is None:
            self._corridor_keys = list(self.edge_corridor)
            self._corridor_tree = STRtree(
                [self.edge_corridor[key] for key in self._corridor_keys])
        indices = self._corridor_tree.query(poly, predicate="intersects")
        return [self._corridor_keys[int(i)] for i in indices]

    def _rebuild_kdtree(self):
        if self.nodes:
            self._kdtree = KDTree(self.nodes)
        else:
            self._kdtree = None

    def nearest_node(self, p: Tuple[float, float]) -> int:
        if self._kdtree is None:
            return 0
        _, idx = self._kdtree.query(p)
        return int(idx)

    def can_drive(self, a: Tuple[float, float], b: Tuple[float, float],
                  blocked=None) -> bool:
        if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9:
            # degenerate segment: shapely predicates on a zero-length LineString
            # are unreliable, and standing still is trivially possible anyway
            return shapely.contains(self.free_eroded_tol, Point(a))
        seg = LineString([a, b])
        if not shapely.contains(self.free_eroded_tol, seg):
            return False
        return blocked is None or not shapely.intersects(blocked, seg)

    def nearest_reachable_node(self, p: Tuple[float, float], blocked=None,
                               k: int = 24) -> int | None:
        """Nearest node the robot can drive to from *p* in a straight line.

        Used to put the robot back on the roadmap after a manipulation ended
        somewhere off-graph.
        """
        if self._kdtree is None:
            return None
        k = min(k, len(self.nodes))
        _, idx = self._kdtree.query(p, k=k)
        for nid in (idx if hasattr(idx, "__iter__") else [idx]):
            nid = int(nid)
            if self.can_drive(p, self.nodes[nid], blocked):
                return nid
        return None

    def add_terminal(self, p: Tuple[float, float]) -> int:
        cfg = self.cfg
        nid = len(self.nodes)
        self.nodes.append((round(p[0], 3), round(p[1], 3)))
        self.adj[nid] = []
        for other, (x, y) in enumerate(self.nodes[:-1]):
            dist = math.hypot(p[0] - x, p[1] - y)
            if dist > cfg.conn_radius * 2 + 1e-9:
                continue
            seg = LineString([p, (x, y)])
            if self.free_eroded_prep.contains(seg):
                key = (nid, other) if nid < other else (other, nid)
                self.edge_len[key] = round(dist, 4)
                self.edge_corridor[key] = seg.buffer(cfg.robot_radius, cap_style=2)
                self.adj[nid].append(other)
                self.adj[other].append(nid)
        self._corridor_tree = None      # new corridors, index is stale
        self._corridor_keys = []
        self._rebuild_kdtree()
        return nid

    def __repr__(self):
        return (f"Roadmap(nodes={len(self.nodes):,}, edges={len(self.edge_len):,}, "
                f"step={self.cfg.grid_step:g}m)")
