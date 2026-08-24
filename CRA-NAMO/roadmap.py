"""增强路网构建（整图均匀网格采样）"""

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
    def __init__(self, workspace: Polygon, static_obstacles, cfg: Config):
        self.cfg = cfg
        self.workspace = workspace
        polys = [so.polygon for so in static_obstacles]
        self.static_obstacles = static_obstacles  # 供 SE(2) 规划器使用
        # 三个嵌套的自由空间集合（从大到小），按"被谁腐蚀"命名——三者仅差在此，
        # 选错不会有任何报错，只能靠命名区分：
        #   static_free      — 工作区内除墙外的全部区域
        #   free_eroded      — 机器人中心可处位置（按 r 腐蚀）
        #   free_eroded_tol  — 同上，再减去 contact_clearance 的微小余量
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
        self._free_eroded_tol: Polygon | None = None
        self._build()
        self._rebuild_kdtree()

    @property
    def free_eroded_tol(self):
        """free_eroded 放宽 contact_clearance 的版本；建图刻意用严格版（节点是停靠点，应有真实间隙），而抓取点恰好压在边界上会因舍入误差被拒，故另留松弛"""
        if self._free_eroded_tol is None:
            eps = max(1e-6, self.cfg.robot_radius - self.cfg.contact_clearance)
            geom = self.static_free.buffer(-eps, quad_segs=16)
            shapely.prepare(geom)
            self._free_eroded_tol = geom
        return self._free_eroded_tol

    # --- 构建 ---
    def _build(self):
        """在全图上生成均匀网格路网"""
        cfg = self.cfg
        assert cfg.grid_step > 0 and cfg.conn_radius > 0
        minx, miny, maxx, maxy = self.workspace.bounds
        step = cfg.grid_step
        # 网格采样节点，只保留机器人圆盘能容纳的位置
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
        # 连接邻近节点，要求膨胀后的线段不与墙相交；桶大小取 conn_radius，
        # 这样 3×3 邻域即可覆盖所有可连接的点对
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

    # --- 查询 ---
    def neighbors(self, u: int):
        for v in self.adj[u]:
            key = (u, v) if u < v else (v, u)
            yield v, key, self.edge_len[key]

    def count_blocked_edges(self, poly: Polygon) -> int:
        """该形状会堵住多少条边走廊"""
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

    def can_drive(self, a: Tuple[float, float], b: Tuple[float, float],
                  blocked=None) -> bool:
        """机器人能否从 a 直线驶达 b；blocked 为已知可动障碍对应的中心禁区（可选）"""
        if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9:
            # 零长线段上 shapely 谓词不可靠，且原地不动显然可行，只判点位即可
            return shapely.contains(self.free_eroded_tol, Point(a))
        seg = LineString([a, b])
        if not shapely.contains(self.free_eroded_tol, seg):
            return False
        return blocked is None or not shapely.intersects(blocked, seg)

    def nearest_reachable_node(self, p: Tuple[float, float], blocked=None,
                               k: int = 24) -> int | None:
        """从 p 出发可直线驶达的最近节点；用于操作结束后把机器人放回路网"""
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
        self._corridor_tree = None      # 新增了走廊，索引已失效
        self._rebuild_kdtree()
        return nid

    def __repr__(self):
        return (f"Roadmap(nodes={len(self.nodes)}, edges={len(self.edge_len)}, "
                f"step={self.cfg.grid_step:g}m)")

