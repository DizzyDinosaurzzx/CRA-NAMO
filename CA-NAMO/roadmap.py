"""Augmented roadmap construction (uniform grid across the entire map)"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from scipy.spatial import KDTree
from config import Config
EdgeKey = Tuple[int, int]

class Roadmap:
    def __init__(self, workspace: Polygon, static_obstacles, cfg: Config):
        self.cfg = cfg
        self.workspace = workspace
        polys = [so.polygon for so in static_obstacles]
        self.static_obstacles = static_obstacles  # for use by the push planner
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
        self._build()
        self._rebuild_kdtree()

    # ------------------ Construction ---------------- #
    def _build(self):
        """Uniform grid across the entire map"""
        cfg = self.cfg
        assert cfg.grid_step > 0 and cfg.conn_radius > 0
        minx, miny, maxx, maxy = self.workspace.bounds
        step = cfg.grid_step
        # sample nodes on a grid, keeping only positions where the robot disc fits
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
        # connect nearby nodes whose inflated segment does not intersect walls. bucket size = conn_radius
        #    so a 3×3 neighbourhood still covers every connectable pair.
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

    # ------------------ Query ---------------- #
    def neighbors(self, u: int):
        for v in self.adj[u]:
            key = (u, v) if u < v else (v, u)
            yield v, key, self.edge_len[key]

    def count_blocked_edges(self, poly: Polygon) -> int:
        """How many edge corridors this footprint would block"""
        if self._corridor_tree is None:
            self._corridor_tree = STRtree(list(self.edge_corridor.values()))
        return len(self._corridor_tree.query(poly, predicate="intersects"))

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
        self._rebuild_kdtree()
        return nid

    def __repr__(self):
        return (f"Roadmap(nodes={len(self.nodes)}, edges={len(self.edge_len)}, "
                f"step={self.cfg.grid_step:g}m)")
