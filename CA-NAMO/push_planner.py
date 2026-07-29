"""
SE2推动路径规划器
在 SE2 构型空间 (x, y, theta) 中为矩形障碍物规划从起始位姿到目标位姿的连续推动路径。
路径满足：
1. 静态障碍物碰撞约束（采用保守膨胀以保证离散安全性）
2. 可选的工作圆约束（障碍物必须在机器人可达范围内）
3. 最小推动路径长度（平移弧长 + 加权旋转）
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# ============ 基础几何 ============= #
def rect_corners(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """矩形的四个角点，逆时针顺序，形状 (4, 2)。"""
    dx, dy = w / 2.0, h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=float)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cy])

def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew 单调链算法，返回逆时针顶点序列。"""
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
    """两个凸多边形的 Minkowski 和，通过逐对点求和 + 凸包计算。"""
    pairwise = (A[:, None, :] + B[None, :, :]).reshape(-1, 2)
    return convex_hull(pairwise)


def c_obstacle(shape_poly: np.ndarray, obs_corners_local: np.ndarray) -> np.ndarray:
    """构型空间障碍物 = shape_poly (+) (-O_theta)。"""
    return minkowski_sum(shape_poly, -obs_corners_local)


def inside_convex(poly: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  margin: float = 0.0) -> np.ndarray:
    """判断点是否在逆时针凸多边形内部（向外膨胀 margin） """
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    shape = np.broadcast(X, Y).shape
    if n == 0:
        return np.zeros(shape, dtype=bool)
    if n < 3:  # 退化情况：点或线段
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
        nx, ny = e[1] / L, -e[0] / L  # 逆时针多边形的外法向量
        inside &= ((X - p[0]) * nx + (Y - p[1]) * ny) <= margin
    return inside


def offset_bbox(poly: np.ndarray, margin: float):
    """`inside_convex(poly, ., ., margin)` 为真的那片凸区域的轴对齐包围盒。

    该区域是各条边【外推 margin】之后的半平面交集，**不等于** poly 的包围盒外扩
    margin：锐角顶点处外推后会伸得更远（边长为 1 的正菱形外推 m，顶点跑到距原顶点
    √2·m 处，落在 bbox+m 之外）。按包围盒+margin 开窗会把本该判为"挡路"的格子漏掉。
    这里直接解相邻两条外推直线的交点得到精确顶点，再取包围盒。

    返回 (xmin, xmax, ymin, ymax)；无法可靠求解时返回 None（调用方应退回全网格）。
    """
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    if n == 0:
        return None
    if n < 3:      # 退化成点/线段时判据是"到它的距离 <= margin"，包围盒外扩即可
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
        if abs(det) < 1e-12:      # 相邻边共线：该顶点不存在，跳过即可
            continue
        xs.append((ca * by - cb * ay) / det)
        ys.append((ax * cb - bx * ca) / det)
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def mean_rotation_radius(w: float, h: float) -> float:
    """矩形绕形心转动时的等效力矩臂r̄"""
    if w <= 0.0 or h <= 0.0:
        return 0.0
    s = math.hypot(w, h)
    return (s / 6.0
            + (w * w / (12.0 * h)) * math.log((s + h) / w)
            + (h * h / (12.0 * w)) * math.log((s + w) / h))


def wrap_dtheta(a: float, b: float) -> float:
    """角度在 [0, pi) 环上的最短有向旋转增量，范围 [-pi/2, pi/2)。"""
    return (b - a + math.pi / 2) % math.pi - math.pi / 2

# ============ SAT 碰撞检测 ============= #

def sat_rect_intersect(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> bool:
    """分离轴定理：判断两个凸多边形是否相交。"""
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
# ============ SE2网络+ Dial桶队列 Dijkstra ============= #

@dataclass
class PushPlanResult:
    """单次推动路径规划的结果。"""
    success: bool
    reason: str = ""
    cost: float = float("nan")
    trans_length: float = float("nan")
    rot_total: float = float("nan")
    goal: Optional[Tuple[float, float, float]] = None
    path: List[Tuple[float, float, float]] = field(default_factory=list)


# 起点解卡时，间隙从完整 margin 起最多减半这么多次；再不行才退到"精确不相交"。
_UNSTICK_RELAX_STEPS = 4

# 走廊掩码的缓存（见 PushPlanner.set_corridor）。值是 packbits 压缩的窗口掩码，
# 一条走廊约几 KB，故上限可以开得很大；仍按 LRU 淘汰以防长时间运行时无界增长。
_CORRIDOR_MASK_CACHE: Dict[tuple, tuple] = {}
_CORRIDOR_MASK_CACHE_MAX = 4096


class PushPlanner:
    """矩形障碍物的 SE2 最小推动路径规划器。"""

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
                 verbose: bool = False):
        assert connectivity in (8, 16), "connectivity supports only 8 or 16"
        assert containment in ("body", "centroid")

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

        # 代价参数。r_half_diag 只用于【几何】外延（膨胀量、解卡半径）；
        # 旋转【代价】用的是力矩臂 r̄，两者相差约 1.9 倍，不可混用。
        self.r_half_diag = 0.5 * math.hypot(obstacle_w, obstacle_h)
        self.dtheta = math.pi / n_theta
        self.rot_weight = (mean_rotation_radius(obstacle_w, obstacle_h)
                           if rot_weight is None else float(rot_weight))
        self.rot_step_cost = self.rot_weight * self.dtheta

        # 保守膨胀：0.1*cell 可减少靠墙障碍物的误判阻塞，
        # 同时仍能防止大多数离散化伪影。
        self.bulge = self.r_half_diag * (1.0 - math.cos(self.dtheta / 2.0))
        self.margin = 0.1 * cell + self.bulge
        # 起点被 _snap() 吸附到格心时的最大位姿误差：平移至多半个格对角线，
        # 转角至多半个 theta 步（后者的径向位移就是 bulge）。走廊腾空判定必须按这个
        # 量放大，否则会出现"真实位姿压着走廊、吸附后的格心不压"的错位——那会让
        # plan_anywhere 直接把起点当成零代价目标返回，谎称"这个障碍物不用动"。
        self.snap_margin = 0.5 * math.hypot(cell, cell) + self.bulge

        # 整数量化单位
        self.unit = cell / self._W_AXIS
        self.W_rot = int(round(self.rot_step_cost / self.unit))

        self._build_moves()
        self._build_cspace()
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, int]] = None

        if verbose:
            print(f"[push] {obstacle_w:g}x{obstacle_h:g} obstacle | "
                  f"grid {self.nx}x{self.ny}x{n_theta} @ cell {cell:.2f} | "
                  f"free {self.free.mean() * 100:.0f}% "
                  f"-> reachable {self.allowed.mean() * 100:.0f}%")

    # -------------------------------------------------------- 移动定义
    def _build_moves(self) -> None:
        """(di, dj, dk, 整数权重)。平移和旋转互斥。"""
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

        # 按权重分组，组内保持 self.moves 的原始先后。_search 对一组里的所有方向
        # 一次性广播展开：桶队列的前沿平均只有几个状态，逐个 move 调用 numpy 时
        # 几乎全部时间都花在调用开销上，而不是真正的计算。
        # 只有权重相同（= 落进同一个桶）的方向之间才可能出现"谁先定下 parent"的
        # 竞争，所以按权重分组不会改变任何结果：跨组的先后由严格的 dist > nd 决定。
        groups: Dict[int, List[Tuple[int, int, int]]] = {}
        for di, dj, dk, w in mv:
            groups.setdefault(w, []).append((di, dj, dk))
        self._move_groups = [
            (w, np.array([m[0] for m in g], dtype=np.int64)[:, None],
                np.array([m[1] for m in g], dtype=np.int64)[:, None],
                np.array([m[2] for m in g], dtype=np.int64)[:, None])
            for w, g in groups.items()]

    # ------------------------------------------------------------ 构型空间构建
    def _build_cspace(self) -> None:
        """构建两个布尔网格：
             free      — 不与墙壁碰撞（带 margin 膨胀，保守近似）
             in_disk   — 满足工作圆约束（精确判定）
        """
        rx, ry = self.robot_pos
        R = self.work_radius

        shape = (self.nx, self.ny, self.n_theta)
        blocked = np.zeros(shape, dtype=bool)
        indisk = np.zeros(shape, dtype=bool)

        for k, th in enumerate(self.thetas):
            O = rect_corners(0.0, 0.0, self.obstacle_w, self.obstacle_h, th)

            # 与墙壁的碰撞检测
            layer_b = np.zeros((self.nx, self.ny), dtype=bool)
            for wp in self.wall_polys:
                layer_b |= inside_convex(c_obstacle(wp, O), self.X, self.Y,
                                         margin=self.margin)
            blocked[:, :, k] = layer_b

            # 工作圆约束
            if not np.isfinite(R):
                indisk[:, :, k] = True
            elif self.containment == "centroid":
                indisk[:, :, k] = ((self.X - rx) ** 2 + (self.Y - ry) ** 2) <= R * R
            else:  # body 模式
                ok = np.ones((self.nx, self.ny), dtype=bool)
                for ox, oy in O:  # O 是角点相对于形心的偏移
                    ok &= ((self.X + ox - rx) ** 2 + (self.Y + oy - ry) ** 2) <= R * R
                indisk[:, :, k] = ok

        self.free = ~blocked
        self.in_disk = indisk
        self.allowed = self.free & self.in_disk
        # 未经 _unstick_start 豁免的原始版本。规划器实例是跨调用缓存的，而
        # _unstick_start() 会就地放开一小片格子——那是只对【某个起点】成立的豁免，
        # 必须在下次规划前还原，否则会永久累积成虚假的可通行区域。
        self._allowed_base = self.allowed.copy()
        self._unstuck = False

        # 走廊掩码只存被挡格子的 (i, j) 外包窗口那一小块，不再物化整格布尔数组：
        # 窗口通常只占全网格的百分之几，而 set_corridor 每轮规划要跑上百次。
        # _route_window 为 None 表示没有任何格子被挡。
        self._route_window: Optional[Tuple[int, int, int, int]] = None
        self._route_mask: Optional[np.ndarray] = None

    # -------------------------------------------------- 起点解卡（离散化伪影修复）
    def _true_collision(self, pose: Tuple[float, float, float],
                        clearance: float = 1e-9) -> bool:
        """用精确 SAT 判断该位姿是否与墙壁相交，且要求至少留出 `clearance` 的间隙。

        `sat_rect_intersect` 的 eps 语义就是"必须分离超过 eps 才算不相交"，因此把
        clearance 传给它等价于把矩形各向外膨胀 clearance 后再判交。
        """
        O = rect_corners(0.0, 0.0, self.obstacle_w, self.obstacle_h, pose[2])
        corners = O + np.array([pose[0], pose[1]])
        for wp in self.wall_polys:
            if sat_rect_intersect(corners, wp, eps=clearance):
                return True
        return False

    def _unstick_clearance(self, pose: Tuple[float, float, float]) -> Optional[float]:
        """能让该位姿脱离碰撞的最大间隙；从完整 margin 起逐次减半，全都不行返回 None"""
        c = self.margin
        for _ in range(_UNSTICK_RELAX_STEPS):
            if not self._true_collision(pose, c):
                return c
            c *= 0.5
        # 每一档都不行：退回"精确不相交"这一原有行为，绝不比改动前更保守
        return None if self._true_collision(pose) else 0.0

    def _unstick_start(self, start_idx: Tuple[int, int, int]) -> int:
        """优化器起点解卡：如果起点被膨胀障碍物阻塞，则在其附近搜索一小片可通行区域，豁免这些格子。"""
        if self._unstuck:                 # 还原上一个起点留下的豁免
            np.copyto(self.allowed, self._allowed_base)
            self._unstuck = False
        if self.allowed[start_idx]:
            return 0
        if not self.in_disk[start_idx]:
            return 0                      # 在工作圆外，不是膨胀造成的，交给上层报错
        clearance = self._unstick_clearance(self._pose(*start_idx))
        if clearance is None:
            return 0                      # 真的埋在墙里，不能豁免
        # 只在起点附近搜索：脱困所需的位移是亚米级的，限制半径可避免
        # 在真正被墙围死时退化成全图 BFS。
        max_cells = max(3, int(math.ceil(self.r_half_diag / self.cell)) + 2)
        si, sj, sk = start_idx
        stack = [start_idx]
        freed = 0
        seen = {start_idx}
        while stack:
            i, j, k = stack.pop()
            if not self.allowed[i, j, k]:
                self.allowed[i, j, k] = True
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
                    continue              # 已解锁或本来就通畅 -> 到达正常空间，停止
                seen.add(nxt)
                if not self.in_disk[nxt]:
                    continue
                # 整条脱困通道统一用起点那一档间隙，而不是各自"不相交即可"：
                # 否则通道中段会退化成零余量，规划器沿着它擦墙而过。
                if self._true_collision(self._pose(*nxt), clearance):
                    continue              # 真实碰撞，保持占据
                stack.append(nxt)

        if freed:
            self._unstuck = True
            if self.verbose:
                x, y, _ = self._pose(*start_idx)
                print(f"[push] unstuck {freed} states at ({x:.1f}, {y:.1f}) "
                      f"clearance={clearance:.3f}")
        return freed

    # ----------------------------------------------------------- 索引辅助函数
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

    # ------------------------------------------------- Dial 桶队列 Dijkstra
    def _search(self, start_idx: Tuple[int, int, int]):
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
            while True:  # 内层循环：兼容零权重边（旋转免费时）
                arrs = buckets.pop(b, None)
                if not arrs:
                    break
                idx = arrs[0] if len(arrs) == 1 else np.concatenate(arrs)
                idx = np.unique(idx[dist[idx] == b])  # 丢弃过期条目
                if idx.size == 0:
                    continue
                i, j, k = self._unflat(idx)
                for w, DI, DJ, DK in self._move_groups:
                    ni, nj = i + DI, j + DJ            # (方向数, 前沿数)
                    ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
                    if not ok.any():
                        continue
                    nidx = ((ni * ny + nj) * nT + (k + DK) % nT)[ok]
                    src = np.broadcast_to(idx, ok.shape)[ok]
                    # 组内不同方向可能落到同一个格子（如 +i 与 -i 来自相邻的两个
                    # 前沿点）。保留首次出现，等价于原先"按 move 顺序逐个处理、
                    # 先到者定下 parent"——ok 是按方向优先展平的，顺序天然一致。
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
            b += 1

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
        # 路径的第一个点是【格心】，而障碍物真实待在 start_pose：两者最多差半个格。
        # 不把这一跳补进路径，执行时就是一次既不做碰撞检测、也不计入做功的瞬移。
        if start_pose is not None and self._pose_gap(start_pose, path[0]) > 1e-9:
            path.insert(0, start_pose)
        return path

    @staticmethod
    def _pose_gap(a, b) -> float:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(wrap_dtheta(a[2], b[2])))

    # ----------------------------------------------------------------- 规划
    def _build_corridor_mask(self, corridor_polys: List[np.ndarray]):
        """把走廊栅格化成 (窗口, 窗口内的布尔掩码)；没有任何格子被挡时返回 (None, None)。

        只在走廊的构型空间障碍物所覆盖的窗口里栅格化：一条边的走廊相对整张地图很小，
        全网格逐 theta 扫描会把 99% 的算力花在必然为 False 的格子上。
        """
        x0, y0, cell = float(self.xs[0]), float(self.ys[0]), self.cell
        layers = []                       # (k, i0, i1, j0, j1, layer)
        wi0, wi1, wj0, wj1 = self.nx, 0, self.ny, 0
        for k, th in enumerate(self.thetas):
            O = rect_corners(0.0, 0.0, self.obstacle_w, self.obstacle_h, th)
            for cp in corridor_polys:
                C = c_obstacle(cp, O)
                bb = offset_bbox(C, self.snap_margin)
                if bb is None:                       # 求解不可靠 -> 退回整层网格
                    i0, i1, j0, j1 = 0, self.nx, 0, self.ny
                else:
                    bxmin, bxmax, bymin, bymax = bb
                    i0 = max(0, int(math.ceil((bxmin - x0) / cell)))
                    i1 = min(self.nx, int(math.floor((bxmax - x0) / cell)) + 1)
                    j0 = max(0, int(math.ceil((bymin - y0) / cell)))
                    j1 = min(self.ny, int(math.floor((bymax - y0) / cell)) + 1)
                    if i0 >= i1 or j0 >= j1:         # 窗口整个落在网格之外
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
        """设置障碍物必须腾空的走廊区域。

        判定按 `snap_margin` 保守放大，作用是双向的：
          * 起点侧——真实位姿压住走廊时，吸附后的格心必定也被标记为"仍挡路"，
            于是 plan_anywhere 不可能把"原地不动"当成合法目标；
          * 终点侧——选中的目标格心在真实几何下会以余量腾空走廊，而不是刚好擦过。

        掩码只取决于走廊几何、障碍物尺寸和网格参数，这三者在整次运行中都不变，
        而同一条边会被几十个重规划周期反复查询——所以直接把结果按走廊字节缓存。
        存的是 packbits 压缩的窗口掩码（比整格布尔数组小两个数量级），命中时
        只需一次 unpackbits，省掉 n_theta 次 Minkowski 和 + 凸包 + 栅格化。
        """
        key = (self.obstacle_w, self.obstacle_h, self.n_theta, self.cell,
               self.nx, self.ny, float(self.xs[0]), float(self.ys[0]),
               round(self.snap_margin, 12),
               tuple(np.asarray(cp, dtype=float).tobytes() for cp in corridor_polys))
        hit = _CORRIDOR_MASK_CACHE.get(key)
        if hit is not None:
            _CORRIDOR_MASK_CACHE[key] = _CORRIDOR_MASK_CACHE.pop(key)  # 标记最近使用
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

    def plan_anywhere(self,
                      start_pose: Tuple[float, float, float]) -> PushPlanResult:
        """规划最小代价推动到【任意】满足腾空 set_corridor() 所设走廊的位姿"""
        start_idx = self._snap(*start_pose)

        if not self.in_disk[start_idx]:
            return PushPlanResult(False, "起点在工作圆之外")
        # 检查起点与墙壁的碰撞（不带 margin）：障碍物实际上已经在那里了
        O_start = rect_corners(0, 0, self.obstacle_w, self.obstacle_h, start_pose[2])
        for wp in self.wall_polys:
            if sat_rect_intersect(
                O_start + np.array([start_pose[0], start_pose[1]]),
                wp, eps=1e-6):
                return PushPlanResult(False, "起点与墙壁碰撞")

        # 起点被吸附进墙 / 被 margin 封死时先解卡，否则搜索一步也走不出去
        if self._unstick_start(start_idx):
            self._cache = None            # 构型空间已变，作废上一次 Dijkstra 结果

        # 如果障碍物位置改变了则重新运行 Dijkstra
        start_flat = int(self._flat(*start_idx))
        if self._cache is None or self._cache[2] != start_flat:
            self._cache = self._search(start_idx) + (start_flat,)
        dist, parent, _cached_start = self._cache

        INF = np.int64(1) << 62
        # 选取代价最小的目标：不可达的格子在 dist 里已经是 INF，被走廊挡住的格子
        # 全部落在 _route_window 内，于是把窗口内被挡的那些临时抬到 INF，直接对
        # 整条 dist 取 argmin 即可。相比"构造 reachable / goals / float 打分数组"
        # 少了三个整格临时数组——这一步每轮规划要跑上百次，分配开销就是主要成本。
        win = self._route_window
        if win is None:
            best = int(np.argmin(dist))
        else:
            i0, i1, j0, j1 = win
            sub = dist.reshape(self.nx, self.ny, self.n_theta)[i0:i1, j0:j1, :]
            blk = self._route_mask
            saved = sub[blk]                 # 缓存的 dist 是共享的，必须原样还原
            sub[blk] = INF
            try:
                best = int(np.argmin(dist))
            finally:
                sub[blk] = saved

        if dist[best] >= INF:
            return PushPlanResult(False, "没有可达位姿能腾空该路径")

        nyT = self.ny * self.n_theta
        goal_idx = (best // nyT, (best % nyT) // self.n_theta,
                    best % self.n_theta)

        goal_pose = self._pose(*goal_idx)
        poses = self._trace(parent, best, start_pose)
        trans = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses, poses[1:]))
        rot = sum(abs(wrap_dtheta(a[2], b[2])) for a, b in zip(poses, poses[1:]))
        # 代价按【返回的这条路径】重算，而不是取 Dijkstra 的整数量化值：后者不含
        # 起点吸附那一段，会与执行期 geometry.se2_path_cost() 的结算口径对不上。
        cost = trans + self.rot_weight * rot

        return PushPlanResult(True, "", cost, trans, rot, goal_pose, poses)

    def plan_path(self,
                  start_pose: Tuple[float, float, float],
                  goal_pose: Tuple[float, float, float]) -> PushPlanResult:
        """在 SE2 中规划从起始位姿到目标位姿的最小代价推动路径"""
        start_idx = self._snap(*start_pose)
        goal_idx = self._snap(*goal_pose)

        if not self.in_disk[start_idx]:
            return PushPlanResult(False, "起始位姿在机器人工作圆之外")
        # 检查起点与墙壁的碰撞（不带 margin）
        O_start = rect_corners(0, 0, self.obstacle_w, self.obstacle_h, start_pose[2])
        for wp in self.wall_polys:
            if sat_rect_intersect(
                O_start + np.array([start_pose[0], start_pose[1]]),
                wp, eps=1e-6):
                return PushPlanResult(False, "起始位姿与墙壁碰撞")

        if not self.free[goal_idx]:
            return PushPlanResult(False, "目标位姿与墙壁碰撞（或距离过近）")

        if not self.in_disk[goal_idx]:
            return PushPlanResult(False, "目标位姿在机器人工作圆之外")

        # 起点被吸附进墙 / 被 margin 封死时先解卡（与 plan_anywhere 同一处理）
        if self._unstick_start(start_idx):
            self._cache = None

        # 如果障碍物位置改变了则重新运行 Dijkstra
        start_flat = int(self._flat(*start_idx))
        if self._cache is None or self._cache[2] != start_flat:
            self._cache = self._search(start_idx) + (start_flat,)
        dist, parent, _cs = self._cache

        goal_flat = int(self._flat(*goal_idx))
        INF = np.int64(1) << 62

        if dist[goal_flat] >= INF:
            return PushPlanResult(False, "离散搜索空间中从起点到目标无可行的路径")

        poses = self._trace(parent, goal_flat, start_pose)
        trans = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses, poses[1:]))
        rot = sum(abs(wrap_dtheta(a[2], b[2])) for a, b in zip(poses, poses[1:]))
        cost = trans + self.rot_weight * rot     # 同 plan_anywhere：以路径为准

        # goal 与 path 必须分别填对：漏掉 goal 会让 poses 落进 goal 字段、path 留空，
        # 而调用方判的是 `if se2_feasible and se2_path`——空列表为假，于是明明规划
        # 成功却被当成失败，该障碍物直接被判定为不可推动。
        return PushPlanResult(True, "", cost, trans, rot, self._pose(*goal_idx), poses)

# ============ 工厂函数 ============= #
def polygon_exterior_coords(polygon) -> np.ndarray:
    """提取 shapely Polygon 的外边界顶点为 numpy 数组（去除闭合重复点）。"""
    # shapely Polygon 的外边界坐标包含闭合重复点；将其去除
    coords = list(polygon.exterior.coords)
    # 若坐标为 [(x0,y0), ..., (xn,yn), (x0,y0)]，若末尾回路则丢弃最后一点
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return np.array(coords, dtype=float)

def build_push_planner(wall_polys,             # shapely Polygon 列表 — 不可穿越区域轮廓
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
                       verbose: bool = False) -> PushPlanner:
    """从 CA-NAMO 风格的数据构建 PushPlanner。"""
    wall_verts = [polygon_exterior_coords(p) for p in wall_polys]

    return PushPlanner(
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
        verbose=verbose,
    )
