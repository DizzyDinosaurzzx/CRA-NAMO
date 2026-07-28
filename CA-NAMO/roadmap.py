"""
增广路网的构建
"""

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
        # 预处理版本：static_free 全程只读，而 contains() 在建图/可见性/放置判定里被
        # 调用成千上万次。prep() 预先建好边索引，谓词判定快一个数量级，语义完全等价。
        self.static_free_prep = prep(self.static_free)
        self.nodes: List[Tuple[float, float]] = []
        self.adj: Dict[int, List[int]] = {}
        self.edge_len: Dict[EdgeKey, float] = {}
        self.edge_corridor: Dict[EdgeKey, Polygon] = {}
        self._build()
    # ------------------ 构建 ---------------- #
    def _build(self):
        cfg = self.cfg
        minx, miny, maxx, maxy = self.workspace.bounds
        # 1) 在网格上采样节点，仅保留机器人圆盘能够放入的位置
        buckets: Dict[Tuple[int, int], List[int]] = {}
        step = cfg.grid_step
        y = miny
        while y <= maxy + 1e-9:
            x = minx
            while x <= maxx + 1e-9:
                if self.static_free_prep.contains(Point(x, y).buffer(cfg.robot_radius)):
                    nid = len(self.nodes)
                    self.nodes.append((round(x, 3), round(y, 3)))
                    self.adj[nid] = []
                    b = (int(x // cfg.conn_radius), int(y // cfg.conn_radius))
                    buckets.setdefault(b, []).append(nid)
                x += step
            y += step
        # 2) 连接相近且膨胀后的线段不与墙体相交的节点
        for nid, (x, y) in enumerate(self.nodes):
            bx, by = int(x // cfg.conn_radius), int(y // cfg.conn_radius)
            for dbx in (-1, 0, 1):
                for dby in (-1, 0, 1):
                    for mid in buckets.get((bx + dbx, by + dby), []):
                        if mid <= nid:
                            continue
                        self._try_edge(nid, mid)

    def _try_edge(self, u: int, v: int):
        cfg = self.cfg
        a, b = self.nodes[u], self.nodes[v]
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        if dist > cfg.conn_radius + 1e-9:
            return
        corridor = LineString([a, b]).buffer(cfg.robot_radius, cap_style=2)
        if not self.static_free_prep.contains(corridor):
            return
        key = (u, v) if u < v else (v, u)
        self.edge_len[key] = round(dist, 4)
        self.edge_corridor[key] = corridor
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
            corridor = LineString([p, (x, y)]).buffer(cfg.robot_radius, cap_style=2)
            if self.static_free_prep.contains(corridor):
                key = (nid, other) if nid < other else (other, nid)
                self.edge_len[key] = round(dist, 4)
                self.edge_corridor[key] = corridor
                self.adj[nid].append(other)
                self.adj[other].append(nid)
        return nid

    def __repr__(self):
        return f"Roadmap(nodes={len(self.nodes)}, edges={len(self.edge_len)})"
