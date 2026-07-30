"""增广路网的构建（全图统一网格）"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
from config import Config
EdgeKey = Tuple[int, int]

class Roadmap:
    def __init__(self, workspace: Polygon, static_obstacles, cfg: Config):
        self.cfg = cfg
        self.workspace = workspace
        polys = [so.polygon for so in static_obstacles]
        self.static_obstacles = static_obstacles  # 供推动规划器使用
        self.static_free = workspace.difference(unary_union(polys)) if polys else workspace
        self.static_free_prep = prep(self.static_free)
        self.free_eroded = self.static_free.buffer(-cfg.robot_radius, quad_segs=16)
        self.free_eroded_prep = prep(self.free_eroded)
        self.nodes: List[Tuple[float, float]] = []
        self.adj: Dict[int, List[int]] = {}
        self.edge_len: Dict[EdgeKey, float] = {}
        self.edge_corridor: Dict[EdgeKey, Polygon] = {}
        self._build()

    # ------------------ 构建 ---------------- #
    def _build(self):
        """全图统一间距的网格"""
        cfg = self.cfg
        assert cfg.grid_step > 0 and cfg.conn_radius > 0
        minx, miny, maxx, maxy = self.workspace.bounds
        step = cfg.grid_step
        # 在网格上采样节点，仅保留机器人圆盘能够放入的位置
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
        # 连接相近且膨胀后的线段不与墙体相交的节点。桶边长取连接半径
        #    cfg.conn_radius，因此 3x3 邻域仍能覆盖任意一对可连边的节点。
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

    # ------------------ 查询 ---------------- #
    def neighbors(self, u: int):
        for v in self.adj[u]:
            key = (u, v) if u < v else (v, u)
            yield v, key, self.edge_len[key]

    def nearest_node(self, p: Tuple[float, float]) -> int:
        best, bd = -1, math.inf
        for nid, (x, y) in enumerate(self.nodes):
            d = (x - p[0]) ** 2 + (y - p[1]) ** 2
            if d < bd:
                bd, best = d, nid
        return best

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
        return nid

    def __repr__(self):
        return (f"Roadmap(nodes={len(self.nodes)}, edges={len(self.edge_len)}, "
                f"step={self.cfg.grid_step:g}m)")
