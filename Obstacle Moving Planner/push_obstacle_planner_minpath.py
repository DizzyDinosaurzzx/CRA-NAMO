"""
push_obstacle_planner_minpath.py
================================
在 SE(2) 中把一个矩形障碍物「推」到不遮挡给定折线路线的位置。
—— v2:目标函数改为 **最小推动路程**,并新增 **机器人工作圆** 约束。

与 v1 (push_obstacle_planner.py) 的差别
---------------------------------------
v1: minimize  ||t* - t0||          (首尾直线位移)   s.t. ||t*-t0|| <= D
    => 代价只看端点,路径只用来证明可达 => 连通性问题 => BFS 足够。
    => 但最优终点可能要绕到 D 外面再绕回来,所以要迭代加深探索半径。

v2: minimize  L(path) = 平移弧长 + w_rot * 总转角      <- 边可加代价
    s.t.  推动全程 O(q(s)) 完全落在圆 B(robot, work_radius) 内
    => 代价沿边累加 => 真正的最短路问题 => 需要 Dijkstra(本实现用 Dial 桶队列)。
    => 工作圆天然把搜索空间封死,迭代加深半径的机制**不再需要**,
       一次搜索就能展开整个可达连通分量,结果在离散化意义下即全局最优。

三个必须显式定下来的建模决定
----------------------------
1) 旋转怎么计入「路程」?
   纯平移路程对原地旋转是 0,那样规划器会疯狂免费旋转。这里用
       每档旋转代价 = w_rot * dtheta,   默认 w_rot = 矩形半对角线 r
   即「最远角点扫过的弧长」,量纲与平移一致,物理上对应推手要走的距离。
   想要纯平移路程就把 w_rot 设成 0(代码里对 0 权边做了闭包处理,仍然正确)。

2) 圆约束约束的是什么?
   containment="centroid" 只要形心在圆内(**默认**)
   containment="body"     整个矩形都要在圆内(对应机械臂可达域 / 安全罩)

   centroid 模式在离散化意义下是**精确**的,不含任何保守成分:
     * 圆是凸集,约束只作用于形心。相邻两个栅格中心都在圆内
       => 二者连线整段也在圆内。平移边天然安全。
     * 原地旋转形心不动,旋转边天然安全。
   body 模式则逐 theta 层用四个角点判定,单个构型是精确的,但**旋转边**
   只查了两个端档:转动过程中某个角点可能瞬时探出圆外再缩回来。
   (量级是 r*(1-cos(dtheta/2)),和 self.bulge 同阶,通常可忽略;
    要严格的话把 bulge 也从 R 里扣掉即可。)

3) 圆心是谁?
   scene.robot —— 机器人推动开始时的位置,视为固定。如果机器人会跟着障碍物
   一起走,这个约束就该变成移动的圆,那是另一个问题(见文末 TODO)。

离散化与正确性
--------------
* C-space = (x, y, theta) 三维栅格;矩形 180 度对称 => theta in [0, pi) 且环绕。
* 每个 theta 层内是纯平移:C_obs(theta) = U_i ( Wall_i (+) (-O_theta) ),
  凸多边形 => 半平面取交 => 整层向量化。
* 因为现在要**最小化路程**,4 邻接的栅格距离退化成曼哈顿距离,误差最大 41%。
  所以改用 8 邻接(对角权 sqrt2,误差 <= 7.6%),可选 16 邻接(加 +-(1,2)/+-(2,1),
  权 sqrt5,误差 <= 2.9%)。
* 边的整段无碰撞仍用「保守膨胀」保证,膨胀量随最长一步自动调整:
      margin = 0.5 * (最长一步长度) + r * (1 - cos(dtheta/2))
  因为线段 a->b 上任一点到某端点的距离 <= |ab|/2,而障碍物是刚体平移,
  所以「两端点相对膨胀 |ab|/2 后的 C_obs 自由」=> 整段真实自由。
  (v1 的 0.5*cell*sqrt2 正好是 8 邻接的特例。)

依赖: numpy, matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.animation import FuncAnimation, PillowWriter


# =============================================================================
# 1. 基础几何(与 v1 相同)
# =============================================================================

def rect_corners(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """矩形的四个角点,逆时针顺序,shape (4, 2)。"""
    dx, dy = w / 2.0, h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=float)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cy])


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew monotone chain,返回 CCW 顶点序列。"""
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
    """两个凸多边形的 Minkowski 和(顶点数很小,直接两两求和取凸包)。"""
    pairwise = (A[:, None, :] + B[None, :, :]).reshape(-1, 2)
    return convex_hull(pairwise)


def c_obstacle(shape_poly: np.ndarray, obs_corners_local: np.ndarray) -> np.ndarray:
    """构型空间障碍 = shape_poly (+) (-O_theta)。"""
    return minkowski_sum(shape_poly, -obs_corners_local)


def inside_convex(poly: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  margin: float = 0.0) -> np.ndarray:
    """点是否落在【向外膨胀 margin 后的】CCW 凸多边形内。向量化,保守(外接超集)。"""
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    if n == 0:
        return np.zeros(X.shape, dtype=bool)
    if n < 3:                                   # 退化成点或线段
        a, b = poly[0], poly[-1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-18:
            d2 = (X - a[0]) ** 2 + (Y - a[1]) ** 2
        else:
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0.0, 1.0)
            d2 = (X - (a[0] + t * ab[0])) ** 2 + (Y - (a[1] + t * ab[1])) ** 2
        return d2 <= margin ** 2

    inside = np.ones(X.shape, dtype=bool)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        e = q - p
        L = math.hypot(e[0], e[1])
        if L < 1e-12:
            continue
        nx, ny = e[1] / L, -e[0] / L            # CCW 多边形的外法线
        inside &= ((X - p[0]) * nx + (Y - p[1]) * ny) <= margin
    return inside


def regular_polygon(cx: float, cy: float, r: float, k: int = 8) -> np.ndarray:
    """半径 r 的正 k 边形(取外接,保证保守),近似折线拐点的圆头。"""
    r_out = r / math.cos(math.pi / k)
    ang = np.arange(k) * 2 * math.pi / k + math.pi / k
    return np.stack([cx + r_out * np.cos(ang), cy + r_out * np.sin(ang)], axis=1)


def corridor_polygons(route: np.ndarray, width: float) -> List[np.ndarray]:
    """「折线 + 宽度」-> 一组凸多边形(每段一个矩形 + 每个顶点一个正八边形)。"""
    route = np.asarray(route, dtype=float)
    polys: List[np.ndarray] = []
    half = width / 2.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        e = b - a
        L = math.hypot(e[0], e[1])
        if L < 1e-12:
            continue
        if half <= 1e-12:
            polys.append(np.array([a, b]))
            continue
        u = e / L
        nvec = np.array([-u[1], u[0]])
        polys.append(np.array([a + half * nvec, a - half * nvec,
                               b - half * nvec, b + half * nvec]))
    if half > 1e-12:
        for p in route:
            polys.append(regular_polygon(p[0], p[1], half, k=8))
    return polys


def wrap_dtheta(a: float, b: float) -> float:
    """theta 在 [0,pi) 上环绕,返回 a->b 的最短有向增量,落在 [-pi/2, pi/2)。"""
    return (b - a + math.pi / 2) % math.pi - math.pi / 2


# =============================================================================
# 2. 场景描述
# =============================================================================

@dataclass
class Rect:
    cx: float
    cy: float
    w: float
    h: float
    theta: float = 0.0

    def corners(self) -> np.ndarray:
        return rect_corners(self.cx, self.cy, self.w, self.h, self.theta)


@dataclass
class Scene:
    walls: List[Rect]
    route: np.ndarray                             # (n, 2) 折线顶点
    corridor_width: float
    obstacle_w: float
    obstacle_h: float
    start: Tuple[float, float, float]             # 障碍物初始位姿 (x, y, theta)
    bounds: Tuple[float, float, float, float]     # xmin, xmax, ymin, ymax
    robot: Tuple[float, float] = (0.0, 0.0)       # 工作圆圆心(机器人位置)
    work_radius: float = float("inf")             # 工作圆半径
    name: str = "scene"


@dataclass
class PlanResult:
    success: bool
    reason: str = ""                    # 中文说明(控制台)
    reason_en: str = ""                 # 英文说明(图上标题,避免 CJK 字体缺失)
    objective: str = "path"
    goal: Optional[Tuple[float, float, float]] = None
    cost: float = float("nan")          # 被最小化的量
    trans_length: float = float("nan")  # 形心平移弧长
    rot_total: float = float("nan")     # 总转角(弧度)
    displacement: float = float("nan")  # 首尾直线位移
    path: List[Tuple[float, float, float]] = field(default_factory=list)


# =============================================================================
# 3. 规划器:SE(2) 栅格 + Dial 桶队列 Dijkstra
# =============================================================================

class PushPlanner:
    """最小推动路程版本。

    参数
    ----
    cell          : 平移栅格边长
    n_theta       : theta 离散档数([0, pi) 上均分)
    connectivity  : 8 或 16。16 更接近欧氏距离,但膨胀量更大、更保守。
    rot_weight    : 每弧度旋转折算成多少平移距离。None => 用矩形半对角线 r
                    (最远角点扫过的弧长)。设 0 => 旋转免费(纯平移路程)。
    containment   : "centroid" 只要形心在圆内(默认,离散化下精确)
                    "body"     整个矩形都要在工作圆内
    """

    # 整数化权重:1 格轴向 = 990 单位。
    #   sqrt2 ~ 1400/990 = 1.4141414 (相对误差 5.2e-5)
    #   sqrt5 ~ 2214/990 = 2.2363636 (相对误差 1.3e-4)
    # 取这么细是因为 W_rot 的舍入误差是**系统性**的(每步同号),会随旋转步数累积;
    # unit 越小,累积上界 0.5*unit*步数 越小。桶数量随之变多,但空桶几乎不耗时。
    _W_AXIS = 990
    _W_DIAG = 1400
    _W_KNIGHT = 2214

    def __init__(self, scene: Scene,
                 cell: float = 0.08, n_theta: int = 24,
                 connectivity: int = 8,
                 rot_weight: Optional[float] = None,
                 containment: str = "centroid",
                 verbose: bool = True):
        assert connectivity in (8, 16), "connectivity 只支持 8 或 16"
        assert containment in ("body", "centroid")
        self.scene = scene
        self.cell = cell
        self.n_theta = n_theta
        self.connectivity = connectivity
        self.containment = containment
        self.verbose = verbose

        xmin, xmax, ymin, ymax = scene.bounds
        self.xs = np.arange(xmin + cell / 2, xmax, cell)
        self.ys = np.arange(ymin + cell / 2, ymax, cell)
        self.nx, self.ny = len(self.xs), len(self.ys)
        self.thetas = np.arange(n_theta) * (math.pi / n_theta)
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")

        # ---- 代价参数 ----
        self.r_half_diag = 0.5 * math.hypot(scene.obstacle_w, scene.obstacle_h)
        self.dtheta = math.pi / n_theta
        self.rot_weight = self.r_half_diag if rot_weight is None else float(rot_weight)
        self.rot_step_cost = self.rot_weight * self.dtheta

        # ---- 保守膨胀量:随最长一步自动调整 ----
        max_step = cell * (math.sqrt(5.0) if connectivity == 16 else math.sqrt(2.0))
        self.bulge = self.r_half_diag * (1 - math.cos(self.dtheta / 2))
        self.margin = 0.5 * max_step + self.bulge

        # ---- 整数量化单位 ----
        self.unit = cell / self._W_AXIS
        self.W_rot = int(round(self.rot_step_cost / self.unit))

        if verbose:
            print(f"[grid]   {self.nx} x {self.ny} x {n_theta} = "
                  f"{self.nx*self.ny*n_theta:,} 构型, cell={cell}, "
                  f"dtheta={math.degrees(self.dtheta):.1f}deg, {connectivity}-邻接")
            print(f"[margin] 扫掠 {0.5*max_step:.4f} + 旋转外凸 {self.bulge:.5f} "
                  f"= {self.margin:.4f}")
            print(f"[cost]   平移 1 格 = {cell:.4f} | 旋转 1 档 = "
                  f"{self.rot_step_cost:.4f} (w_rot={self.rot_weight:.3f})")

        self._build_moves()
        self._build_cspace()
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------ 邻接定义
    def _build_moves(self) -> None:
        """(di, dj, dk, 整数权重)。平移与旋转互斥:要么纯平移,要么原地转一档。"""
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

    # -------------------------------------------------------------- C-space
    def _build_cspace(self) -> None:
        """构造三张布尔栅格:
             self.free        —— 不与墙碰撞(障碍物按 margin 膨胀,保守)
             self.in_disk     —— 满足机器人工作圆约束(精确)
             self.route_block —— 与走廊相交(真实尺寸,不膨胀;只对终点生效)
        """
        sc = self.scene
        wall_polys = [w.corners() for w in sc.walls]
        corr_polys = corridor_polygons(sc.route, sc.corridor_width)
        rx, ry = sc.robot
        R = sc.work_radius

        shape = (self.nx, self.ny, self.n_theta)
        blocked = np.zeros(shape, dtype=bool)
        rblock = np.zeros(shape, dtype=bool)
        indisk = np.zeros(shape, dtype=bool)

        for k, th in enumerate(self.thetas):
            O = rect_corners(0.0, 0.0, sc.obstacle_w, sc.obstacle_h, th)

            layer_b = np.zeros((self.nx, self.ny), dtype=bool)
            for wp in wall_polys:
                layer_b |= inside_convex(c_obstacle(wp, O), self.X, self.Y,
                                         margin=self.margin)
            blocked[:, :, k] = layer_b

            layer_r = np.zeros((self.nx, self.ny), dtype=bool)
            for cp in corr_polys:
                layer_r |= inside_convex(c_obstacle(cp, O), self.X, self.Y, margin=0.0)
            rblock[:, :, k] = layer_r

            # 工作圆:body 模式对四个角点取最大距离(精确,非保守近似)
            if not np.isfinite(R):
                indisk[:, :, k] = True
            elif self.containment == "centroid":
                indisk[:, :, k] = ((self.X - rx) ** 2 + (self.Y - ry) ** 2) <= R * R
            else:
                ok = np.ones((self.nx, self.ny), dtype=bool)
                for ox, oy in O:                 # O 已是相对形心的角点偏移
                    ok &= ((self.X + ox - rx) ** 2 + (self.Y + oy - ry) ** 2) <= R * R
                indisk[:, :, k] = ok

        self.free = ~blocked
        self.in_disk = indisk
        self.route_block = rblock
        self.allowed = self.free & self.in_disk

        if self.verbose:
            print(f"[cspace] 无碰撞 {self.free.mean()*100:.1f}% | "
                  f"在工作圆内 {self.in_disk.mean()*100:.1f}% | "
                  f"两者都满足 {self.allowed.mean()*100:.1f}%")

    # ------------------------------------------------------------- 索引工具
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

    # -------------------------------------------------- Dial 桶队列 Dijkstra
    def _search(self, start_idx: Tuple[int, int, int]):
        """所有边权为非负整数 => 用桶队列 Dijkstra,整批向量化松弛。

        返回 (dist_int, parent);dist_int == INF 表示不可达。
        真实代价 = dist_int * self.unit。
        """
        N = self.nx * self.ny * self.n_theta
        allowed = self.allowed.reshape(-1)
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
            while True:                    # 内层循环:兼容 0 权边(旋转免费时)
                arrs = buckets.pop(b, None)
                if not arrs:
                    break
                idx = arrs[0] if len(arrs) == 1 else np.concatenate(arrs)
                idx = np.unique(idx[dist[idx] == b])     # 丢掉过期条目
                if idx.size == 0:
                    continue
                i, j, k = self._unflat(idx)
                for di, dj, dk, w in self.moves:
                    ni, nj = i + di, j + dj
                    ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
                    if not ok.any():
                        continue
                    src = idx[ok]
                    nidx = (ni[ok] * ny + nj[ok]) * nT + ((k[ok] + dk) % nT)
                    nd = b + w
                    better = allowed[nidx] & (dist[nidx] > nd)
                    tgt = nidx[better]
                    if tgt.size == 0:
                        continue
                    dist[tgt] = nd         # 同一 move 下 src->tgt 是单射,无写冲突
                    parent[tgt] = src[better]
                    buckets.setdefault(nd, []).append(tgt)
                    if nd > max_b:
                        max_b = nd
            b += 1

        return dist, parent

    def _trace(self, parent: np.ndarray, goal_flat: int) -> List[Tuple[float, float, float]]:
        chain = [goal_flat]
        cur = goal_flat
        while parent[cur] >= 0:
            cur = int(parent[cur])
            chain.append(cur)
        chain.reverse()
        i, j, k = self._unflat(np.array(chain, dtype=np.int64))
        return [self._pose(int(a), int(b_), int(c)) for a, b_, c in zip(i, j, k)]

    # ----------------------------------------------------------------- plan
    def plan(self, objective: str = "path", D: Optional[float] = None) -> PlanResult:
        """objective:
             "path"          最小化推动路程(平移弧长 + w_rot * 总转角)  <- v2 主目标
             "displacement"  最小化首尾直线位移(v1 目标;复用同一次搜索,便于对比)
           D: 可选的额外硬约束,要求首尾直线位移 <= D。None 表示不限制。
        """
        assert objective in ("path", "displacement")
        sc = self.scene
        start_idx = self._snap(*sc.start)

        if not self.free[start_idx]:
            return PlanResult(False, "初始位姿本身就与墙碰撞(或离墙太近被保守膨胀判死)",
                              "start pose already collides with a wall", objective)
        if not self.in_disk[start_idx]:
            return PlanResult(False, "初始位姿就不在机器人工作圆内,约束自相矛盾",
                              "start pose is already outside the robot work circle",
                              objective)

        if self._cache is None:
            self._cache = self._search(start_idx)
        dist, parent = self._cache

        INF = np.int64(1) << 62
        dist3 = dist.reshape(self.nx, self.ny, self.n_theta)
        reachable = dist3 < INF
        goals = reachable & ~self.route_block

        x0, y0 = self.xs[start_idx[0]], self.ys[start_idx[1]]
        disp2 = ((self.X - x0) ** 2 + (self.Y - y0) ** 2)[:, :, None]
        if D is not None:
            goals = goals & (disp2 <= D * D)

        if self.verbose:
            print(f"[dijkstra] 可达构型 {int(reachable.sum()):,} | "
                  f"候选终点 {int(goals.sum()):,}")

        if not goals.any():
            why = "工作圆内的可达自由空间中,不存在任何让开路线的位姿"
            why_en = "no reachable pose inside the work circle clears the route"
            if D is not None:
                why += f"(且满足位移 <= {D})"
            return PlanResult(False, why, why_en, objective)

        if objective == "path":
            score = np.where(goals, dist3.astype(np.float64), np.inf)
        else:
            score = np.where(goals, np.broadcast_to(disp2, goals.shape), np.inf)
        gi, gj, gk = np.unravel_index(int(np.argmin(score)), score.shape)
        idx = (int(gi), int(gj), int(gk))

        poses = self._trace(parent, int(self._flat(*idx)))
        trans = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses, poses[1:]))
        rot = sum(abs(wrap_dtheta(a[2], b[2])) for a, b in zip(poses, poses[1:]))
        disp = math.hypot(poses[-1][0] - poses[0][0], poses[-1][1] - poses[0][1])
        cost = float(dist3[idx]) * self.unit

        return PlanResult(True, "", "", objective, self._pose(*idx),
                          cost, trans, rot, disp, poses)

    # 供可视化使用
    def reachable_mask(self) -> np.ndarray:
        if self._cache is None:
            return np.zeros(self.allowed.shape, dtype=bool)
        INF = np.int64(1) << 62
        return (self._cache[0] < INF).reshape(self.nx, self.ny, self.n_theta)


# =============================================================================
# 4. 独立校验(不复用规划器的栅格,防止自证自洽)
# =============================================================================

def sat_rect_intersect(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> bool:
    """分离轴定理:两个凸多边形是否相交。"""
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


def verify(scene: Scene, res: PlanResult, planner: "PushPlanner",
           D: Optional[float] = None) -> List[str]:
    """对返回的解做一遍独立的连续几何校验。"""
    if not res.success:
        return ["(无解,跳过校验)"]
    msgs = []
    G = rect_corners(res.goal[0], res.goal[1], scene.obstacle_w,
                     scene.obstacle_h, res.goal[2])

    hit_wall = [i for i, w in enumerate(scene.walls) if sat_rect_intersect(G, w.corners())]
    msgs.append(f"终点 vs 墙体        : {'OK 无碰撞' if not hit_wall else f'!! 碰到墙 {hit_wall}'}")

    corr = corridor_polygons(scene.route, scene.corridor_width)
    hit_c = [i for i, c in enumerate(corr) if len(c) >= 3 and sat_rect_intersect(G, c)]
    msgs.append(f"终点 vs 走廊        : {'OK 已让开' if not hit_c else f'!! 仍压住走廊 {hit_c}'}")

    # 沿路径细分采样:全程不撞墙 + 全程在工作圆内
    rx, ry = scene.robot
    R = scene.work_radius
    bad_wall = None
    worst_reach = 0.0
    for a, b in zip(res.path, res.path[1:]):
        for t in np.linspace(0, 1, 8):
            th = a[2] + t * wrap_dtheta(a[2], b[2])
            cx, cy = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            P = rect_corners(cx, cy, scene.obstacle_w, scene.obstacle_h, th)
            if bad_wall is None:
                for i, w in enumerate(scene.walls):
                    if sat_rect_intersect(P, w.corners()):
                        bad_wall = (a, b, i)
                        break
            if planner.containment == "centroid":
                d = math.hypot(cx - rx, cy - ry)
            else:
                d = float(np.max(np.hypot(P[:, 0] - rx, P[:, 1] - ry)))
            worst_reach = max(worst_reach, d)
    msgs.append(f"推动全程 vs 墙体    : "
                f"{'OK 无碰撞' if bad_wall is None else f'!! 在 {bad_wall[0]}->{bad_wall[1]} 撞墙 {bad_wall[2]}'}")
    msgs.append(f"推动全程 vs 工作圆  : 最远 {worst_reach:.3f} <= R={R:.3f}  "
                f"{'OK' if worst_reach <= R + 1e-6 else '!! 出圈'}  "
                f"({planner.containment} 模式)")

    # 报告的 cost 来自整数桶队列,与连续几何重算值之间只应差一个量化误差。
    # 严格上界:每步舍入 <= unit/2(旋转)或 5.2e-5 * 步长(对角),这里取前者放大。
    recomputed = res.trans_length + planner.rot_weight * res.rot_total
    tol = 0.5 * planner.unit * len(res.path) + 1e-4 * res.trans_length + 1e-9
    msgs.append(f"代价一致性          : cost={res.cost:.5f} vs 重算 {recomputed:.5f} "
                f"(量化上界 {tol:.5f})  "
                f"{'OK' if abs(res.cost - recomputed) <= tol else '!! 超出量化误差'}")

    if D is not None:
        msgs.append(f"位移额外预算        : {res.displacement:.3f} <= {D:.3f} "
                    f"{'OK' if res.displacement <= D + 1e-9 else '!! 超预算'}")

    start_rect = rect_corners(*scene.start[:2], scene.obstacle_w,
                              scene.obstacle_h, scene.start[2])
    blocked_now = any(len(c) >= 3 and sat_rect_intersect(start_rect, c) for c in corr)
    msgs.append(f"初始确实遮挡路线    : {'OK 是' if blocked_now else '?? 初始就没挡住,题目无意义'}")
    return msgs


# =============================================================================
# 5. 可视化
# =============================================================================

C_WALL = "#4a4a52"
C_CORR = "#4c8bf5"
C_START = "#e2483d"
C_GOAL = "#2ca05a"
C_ROBOT = "#f0a03c"


def _draw_map(ax, scene: Scene, show_circle: bool = True):
    xmin, xmax, ymin, ymax = scene.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    for w in scene.walls:
        ax.add_patch(MplPolygon(w.corners(), closed=True, fc=C_WALL, ec="none", zorder=2))
    for poly in corridor_polygons(scene.route, scene.corridor_width):
        if len(poly) >= 3:
            ax.add_patch(MplPolygon(poly, closed=True, fc=C_CORR, ec="none",
                                    alpha=0.20, zorder=1))
    ax.plot(scene.route[:, 0], scene.route[:, 1], color=C_CORR, lw=2.0, ls="--",
            zorder=3, label="route (polyline)")
    ax.plot(scene.route[:, 0], scene.route[:, 1], "o", color=C_CORR, ms=4, zorder=3)
    if show_circle and np.isfinite(scene.work_radius):
        ax.add_patch(plt.Circle(scene.robot, scene.work_radius, fill=False,
                                ec=C_ROBOT, ls="--", lw=2.0, zorder=7))
        ax.plot(*scene.robot, "P", color=C_ROBOT, ms=13, mec="k", mew=0.8, zorder=8)


def _add_rect(ax, scene, pose, fc, ec, alpha=1.0, lw=1.6, z=4):
    ax.add_patch(MplPolygon(rect_corners(pose[0], pose[1], scene.obstacle_w,
                                         scene.obstacle_h, pose[2]),
                            closed=True, fc=fc, ec=ec, alpha=alpha, lw=lw, zorder=z))


def plot_solution(scene: Scene, res: PlanResult, planner: "PushPlanner",
                  out_png: str, D: Optional[float] = None):
    fig = plt.figure(figsize=(16.0, 10.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.30, wspace=0.20)

    # ---------------- (a) 地图总览 ----------------
    ax = fig.add_subplot(gs[0, 0:2])
    _draw_map(ax, scene)
    ax.plot([], [], color=C_ROBOT, lw=2, ls="--",
            label=f"robot work circle  R={scene.work_radius:g}  ({planner.containment})")

    if res.success:
        cmap = plt.get_cmap("plasma")
        step = max(1, len(res.path) // 16)
        for ii in range(0, len(res.path), step):
            _add_rect(ax, scene, res.path[ii], "none",
                      cmap(0.05 + 0.85 * ii / max(1, len(res.path) - 1)), lw=1.1, z=3)
        ax.plot([p[0] for p in res.path], [p[1] for p in res.path], "-",
                color="#8a63d2", lw=2.0, zorder=5,
                label=f"push path  (cost = {res.cost:.2f})")
        _add_rect(ax, scene, res.goal, C_GOAL, "#14522f", alpha=0.6, z=6)
        ax.plot([scene.start[0], res.goal[0]], [scene.start[1], res.goal[1]],
                color=C_GOAL, lw=1.6, ls="-.", zorder=6,
                label=f"net displacement = {res.displacement:.2f}")

    _add_rect(ax, scene, scene.start, C_START, "#7d1c14", alpha=0.6, z=6)
    ax.plot([], [], color=C_START, lw=6, alpha=0.6, label="obstacle @ start (blocking)")
    if res.success:
        ax.plot([], [], color=C_GOAL, lw=6, alpha=0.6, label="obstacle @ goal (clear)")
    ax.plot([], [], color=C_WALL, lw=6, label="walls (hard: never touch)")
    obj_en = "min PUSH PATH LENGTH" if res.objective == "path" else "min NET DISPLACEMENT"
    ax.set_title(f"{'SUCCESS' if res.success else 'FAILURE'}   [{obj_en}]   -   {scene.name}",
                 fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # ---------------- 信息面板 ----------------
    axi = fig.add_subplot(gs[0, 2])
    axi.axis("off")
    h, l = ax.get_legend_handles_labels()
    axi.legend(h, l, loc="upper left", fontsize=8.5, framealpha=1.0,
               bbox_to_anchor=(0.0, 1.02))
    if res.success:
        info = (
            f"objective      : {res.objective}\n"
            f"goal pose      : ({res.goal[0]:.2f}, {res.goal[1]:.2f}, "
            f"{math.degrees(res.goal[2]):.0f} deg)\n"
            f"COST           : {res.cost:.3f}\n"
            f"  translation  : {res.trans_length:.3f}\n"
            f"  rotation     : {math.degrees(res.rot_total):.0f} deg x w_rot="
            f"{planner.rot_weight:.3f}\n"
            f"                 -> {planner.rot_weight*res.rot_total:.3f}\n"
            f"net displace.  : {res.displacement:.3f}\n"
            f"work circle    : R={scene.work_radius:g} @ "
            f"({scene.robot[0]:.1f}, {scene.robot[1]:.1f})\n"
            f"connectivity   : {planner.connectivity}\n"
            f"optimality     : globally optimal on the grid.\n"
            "                 The work circle bounds the search,\n"
            "                 so no iterative deepening is needed\n"
            "                 (unlike the min-displacement version)."
        )
    else:
        info = f"FAILURE\n\n{res.reason_en}"
    axi.text(0.0, 0.46, info, fontsize=8.5, family="monospace",
             va="top", ha="left", transform=axi.transAxes)

    # ---------------- (b)(c)(d) C-space 切片 ----------------
    reach = planner.reachable_mask()
    k_start = planner._snap(*scene.start)[2]
    k_goal = planner._snap(*res.goal)[2] if res.success else k_start
    ks = sorted({k_start, k_goal, (k_start + planner.n_theta // 2) % planner.n_theta})
    while len(ks) < 3:
        ks.append((ks[-1] + 3) % planner.n_theta)
    ks = ks[:3]

    xmin, xmax, ymin, ymax = scene.bounds
    for col, k in enumerate(ks):
        axs = fig.add_subplot(gs[1, col])
        img = np.ones((planner.ny, planner.nx, 3), dtype=float)
        blocked = (~planner.free[:, :, k]).T
        outside = (~planner.in_disk[:, :, k]).T
        rc = (reach[:, :, k]).T
        goal_ok = rc & (~planner.route_block[:, :, k]).T
        img[outside] = np.array([0.93, 0.93, 0.93])         # 圆外
        img[blocked] = np.array([0.29, 0.29, 0.32])         # 墙的 C-obstacle
        img[rc & ~blocked] = np.array([0.80, 0.90, 1.00])   # 可达但压走廊
        img[goal_ok] = np.array([0.55, 0.88, 0.66])         # 可达且让开
        axs.imshow(img, origin="lower", extent=[xmin, xmax, ymin, ymax],
                   interpolation="nearest")
        axs.plot(scene.start[0], scene.start[1], "*", color=C_START, ms=13, mec="k", mew=0.6)
        if res.success and k == k_goal:
            axs.plot(res.goal[0], res.goal[1], "*", color=C_GOAL, ms=13, mec="k", mew=0.6)
        if np.isfinite(scene.work_radius):
            axs.add_patch(plt.Circle(scene.robot, scene.work_radius, fill=False,
                                     ec=C_ROBOT, ls="--", lw=1.2))
            axs.plot(*scene.robot, "P", color=C_ROBOT, ms=7, mec="k", mew=0.5)
        # 圆可能比地图大:先 imshow 再加 patch 会触发 autoscale 把地图挤小,
        # 所以在所有 artist 加完之后再把范围钉回地图 bounds(圆会被裁掉一部分)。
        axs.set_xlim(xmin, xmax)
        axs.set_ylim(ymin, ymax)
        axs.set_aspect("equal")
        tag = (["start th"] if k == k_start else []) + \
              (["goal th"] if res.success and k == k_goal else [])
        axs.set_title(f"C-space slice theta={math.degrees(planner.thetas[k]):.0f} deg"
                      + (f"  [{', '.join(tag)}]" if tag else ""), fontsize=9)
        axs.tick_params(labelsize=7)

    fig.text(0.012, 0.012,
             "C-space slices:  grey = outside the robot work circle   |   "
             "dark = collides with walls   |   light blue = reachable but still covers "
             "the corridor   |   green = reachable AND corridor is clear (candidate goals)",
             fontsize=8.5, color="#444")
    fig.suptitle("SE(2) push planning v2:  minimum PUSH PATH LENGTH, "
                 "constrained to the robot's work circle", fontsize=13, y=0.965)
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {out_png}")


def animate_push(scene: Scene, res: PlanResult, out_gif: str, fps: int = 20):
    if not res.success:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    _draw_map(ax, scene)
    _add_rect(ax, scene, scene.start, "none", C_START, lw=1.2, z=3)
    _add_rect(ax, scene, res.goal, "none", C_GOAL, lw=1.2, z=3)
    ax.plot([p[0] for p in res.path], [p[1] for p in res.path],
            color="#8a63d2", lw=1.2, alpha=0.6, zorder=3)
    moving = MplPolygon(rect_corners(*scene.start[:2], scene.obstacle_w,
                                     scene.obstacle_h, scene.start[2]),
                        closed=True, fc="#f0a03c", ec="#8a4f00", lw=1.6, zorder=9)
    ax.add_patch(moving)
    ax.set_title(f"min-path push  (cost {res.cost:.2f}) - {scene.name}", fontsize=11)
    idx = np.linspace(0, len(res.path) - 1, min(len(res.path), 160)).astype(int)

    def upd(f):
        p = res.path[idx[f]]
        moving.set_xy(rect_corners(p[0], p[1], scene.obstacle_w, scene.obstacle_h, p[2]))
        return (moving,)

    FuncAnimation(fig, upd, frames=len(idx), interval=1000 / fps,
                  blit=True).save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[out] {out_gif}")


# =============================================================================
# 6. Demo
# =============================================================================

def demo_scene(robot=(4.5, 4.0), work_radius=8.0,
               seal_left_pocket: bool = False) -> Scene:
    """「窄门」场景,额外给出机器人位置和工作圆半径。

        y 5.9 ~ 9.5   上层房间
        y 5.5 ~ 5.9   房间地板 —— 只铺到 x=9.3,右边留一道 2.2 宽的门
        y 2.6 ~ 5.5   走廊所在的通道带(路线 y=4,走廊宽 1.3 => 占 3.35~4.65)
        y 0.5 ~ 2.6   下方实心块

    障碍物 2.4 x 0.9 横躺在走廊正中 (6, 4);机器人在它左边 (4.5, 4) 等着通过。

    seal_left_pocket
    ----------------
    v1 的注释声称「唯一出路是上层房间」,这其实是**错的**:下方实心块只铺到
    x >= 4.0,左下角 (x 0.5..4.0, y 0.5..2.6) 是一个空口袋,把箱子往左下方一塞
    就让开了路线,既不用旋转也不用穿门。v1 之所以没选它,只是因为它的位移
    (3.49) 比上层房间 (2.40) 大 —— 换成【最小路程】目标后,这个口袋立刻胜出
    (路程 3.92 vs 13.04)。这恰好说明目标函数换了,最优解可以完全换一个地方。

    seal_left_pocket=True 会把下方实心块向左延伸到 x=0.5 封死这个口袋,
    此时「必须旋转 + 穿窄门」才是真的,两种目标函数就能在同一条走廊上对比。
    """
    T = 0.5
    if seal_left_pocket:
        lower = Rect(6.0, 1.55, 11.0, 2.1)     # x 0.5..11.5, y 0.5..2.6
    else:
        lower = Rect(7.75, 1.55, 7.5, 2.1)     # x 4.0..11.5, y 0.5..2.6(左下留口袋)
    walls = [
        Rect(6.0, 0.25, 12.0, T),      # 外墙 下
        Rect(6.0, 9.75, 12.0, T),      # 外墙 上
        Rect(0.25, 5.0, T, 10.0),      # 外墙 左
        Rect(11.75, 5.0, T, 10.0),     # 外墙 右
        lower,
        Rect(4.90, 5.70, 8.8, 0.4),    # 房间地板 x 0.5..9.3, y 5.5..5.9(门在 x 9.3..11.5)
    ]
    return Scene(
        walls=walls,
        route=np.array([[0.9, 4.0], [11.1, 4.0]]),
        corridor_width=1.3,
        obstacle_w=2.4,
        obstacle_h=0.9,
        start=(6.0, 4.0, 0.0),
        bounds=(0.0, 12.0, 0.0, 10.0),
        robot=robot,
        work_radius=work_radius,
        name=("narrow door only (left pocket sealed)" if seal_left_pocket
              else "narrow door + open left pocket"),
    )


def _report(scene: Scene, res: PlanResult, planner: "PushPlanner",
            D: Optional[float] = None):
    print()
    if res.success:
        print(f"  SUCCESS [{res.objective}]  goal = ({res.goal[0]:.3f}, "
              f"{res.goal[1]:.3f}, {math.degrees(res.goal[2]):.1f}deg)")
        print(f"     代价 cost    = {res.cost:.3f}   "
              f"(平移 {res.trans_length:.3f} + 旋转 {math.degrees(res.rot_total):.0f}deg "
              f"x {planner.rot_weight:.3f} = {planner.rot_weight*res.rot_total:.3f})")
        print(f"     首尾直线位移 = {res.displacement:.3f}")
    else:
        print(f"  FAILURE  {res.reason}")
    print("\n  --- 独立几何校验 ---")
    for m in verify(scene, res, planner, D):
        print("   ", m)
    print()


def run(scene: Scene, tag: str, objective: str = "path",
        cell: float = 0.08, n_theta: int = 24, connectivity: int = 8,
        rot_weight: Optional[float] = None, containment: str = "centroid",
        D: Optional[float] = None, make_gif: bool = False,
        planner: Optional["PushPlanner"] = None):
    print("=" * 78)
    print(f"### {tag}   objective={objective}   R={scene.work_radius}")
    print("=" * 78)
    if planner is None:
        planner = PushPlanner(scene, cell=cell, n_theta=n_theta,
                              connectivity=connectivity, rot_weight=rot_weight,
                              containment=containment)
    res = planner.plan(objective=objective, D=D)
    _report(scene, res, planner, D)
    plot_solution(scene, res, planner, f"{tag}_solution.png", D)
    if make_gif and res.success:
        animate_push(scene, res, f"{tag}_animation.gif")
    return res, planner


def compare(tag: str, res_path: PlanResult, res_disp: PlanResult) -> None:
    print("=" * 78)
    print(f"### 两种目标函数的对比 —— {tag}(同一场景、同一次 Dijkstra)")
    print("=" * 78)
    if not (res_path.success and res_disp.success):
        print("  至少一侧无解,跳过对比")
        print()
        return
    for name, r in (("min-path        ", res_path), ("min-displacement", res_disp)):
        print(f"  {name}: cost {r.cost:6.3f}   平移 {r.trans_length:6.3f}   "
              f"转角 {math.degrees(r.rot_total):5.0f}deg   位移 {r.displacement:5.3f}   "
              f"goal ({r.goal[0]:.2f}, {r.goal[1]:.2f}, {math.degrees(r.goal[2]):.0f}deg)")
    print(f"  => 推动代价省 {(1 - res_path.cost / res_disp.cost) * 100:.1f}%,"
          f"  终点多离开原位 {res_path.displacement - res_disp.displacement:+.2f}")
    print()


if __name__ == "__main__":
    # ---- A. 左下口袋敞开:换目标函数后,最优解跑到了地图的另一头 ----
    scA = demo_scene(robot=(4.5, 4.0), work_radius=8.0)
    rA_path, pA = run(scA, "v2_case1_minpath", objective="path", make_gif=True)
    rA_disp, _ = run(scA, "v2_case2_mindisp", objective="displacement", planner=pA)
    compare("左下口袋敞开", rA_path, rA_disp)

    # ---- B. 口袋封死:两种目标都必须旋转穿窄门,差别只在停在门的哪一侧 ----
    scB = demo_scene(robot=(4.5, 4.0), work_radius=8.0, seal_left_pocket=True)
    rB_path, pB = run(scB, "v2_case3_sealed_minpath", objective="path", make_gif=True)
    rB_disp, _ = run(scB, "v2_case4_sealed_mindisp", objective="displacement", planner=pB)
    compare("左下口袋封死(必须穿窄门)", rB_path, rB_disp)

    # ---- C. 工作圆太小:能让开路线的位姿全都够不着 -> FAILURE ----
    scC = demo_scene(robot=(4.5, 4.0), work_radius=1.8)
    run(scC, "v2_case5_circle_too_small", objective="path")

    # ---- D. 旋转免费(w_rot=0,纯平移路程);顺便验证 0 权边的桶队列闭包处理 ----
    scD = demo_scene(robot=(4.5, 4.0), work_radius=8.0, seal_left_pocket=True)
    run(scD, "v2_case6_free_rotation", objective="path", rot_weight=0.0)
