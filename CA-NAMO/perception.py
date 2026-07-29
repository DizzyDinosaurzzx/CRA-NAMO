"""感知与信念状态维护与更新"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from obstacle import MovableObstacle
from roadmap import Roadmap, EdgeKey
from config import Config


class Belief:
    def __init__(self, roadmap: Roadmap, cfg: Config):
        self.roadmap = roadmap
        self.cfg = cfg
        # oid -> 已感知障碍物的【副本】。持有副本而非世界对象的引用，否则
        # `obstacle(oid).difficulty` 会直接返回 ground truth，绕开触摸/估计机制。
        self.perceived: Dict[int, MovableObstacle] = {}
        self.edge_blockers: Dict[EdgeKey, Set[int]] = {}    # 边 -> 阻挡该边的障碍物 oid 集合
        self.newly_revealed: List[int] = []
        # 通过"推动碰撞"发现的匿名占据区域：只知道"这里有东西"，不知其身份/几何/难度。
        # 仅用作放置/推动的硬约束；无法据此规划"移除"它（不知道怎么移、多难移）。
        self.contacts: List[Polygon] = []
        # 需求4: 已触摸获知真实难度的障碍物
        self.touched: Set[int] = set()
        self.touched_difficulty: Dict[int, float] = {}

    # -------------------- 感知 ----------------------
    def perceive(self, world_obstacles: List[MovableObstacle],
                 robot_pos: Tuple[float, float]) -> List[int]:
        """揭示 `robot_pos` 周围所有可见障碍物；已知的则刷新其位姿。

        已知障碍物必须重新对齐：信念里的位姿只在机器人自己推动时更新，一旦推动
        中途被撞停（世界停在半路、信念以为没动），二者就会永久脱节。看得见就同步，
        是让信念自愈的最低成本手段。
        """
        self.newly_revealed = []
        rp = Point(robot_pos)
        for w in world_obstacles:
            known = self.perceived.get(w.oid)
            # 已知且位姿一致 —— 没有任何可同步的信息，直接跳过昂贵的可见性判定
            if known is not None and self._pose_matches(known, w):
                continue
            if rp.distance(Point(w.center())) > self.cfg.R_perc:
                continue
            if not self._visible(robot_pos, w, world_obstacles):
                continue
            if known is not None:
                self._sync_pose(known, w)
                continue
            obs = w.perceived_copy()
            self.perceived[w.oid] = obs
            self.newly_revealed.append(w.oid)
            self._update_edges_for(obs)
            # 该物体现在被完整感知，之前对它的匿名"接触"记录可以清除（由真实 footprint 取代）
            self._clear_contacts_overlapping(obs.polygon)
        return self.newly_revealed

    @staticmethod
    def _pose_matches(a: MovableObstacle, b: MovableObstacle) -> bool:
        return (abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9
                and abs(a.theta - b.theta) < 1e-9)

    def _sync_pose(self, known: MovableObstacle, world_obs: MovableObstacle):
        """把已感知副本的位姿对齐到世界真实位姿。"""
        old_footprint = known.polygon
        self._forget_edges(known.oid)
        known.x, known.y, known.theta = world_obs.x, world_obs.y, world_obs.theta
        known.removed = world_obs.removed
        self._update_edges_for(known)
        # 机器人亲眼看到它已经不在原处了，压在【旧】footprint 上的匿名接触区随之作废。
        # 这些 blob 多半就是当初撞到它时留下的记录；不清理的话，一个已经让开的障碍物
        # 会在原地留下一堵永久的幽灵墙，把那条通道对所有后续规划封死。
        self._clear_contacts_overlapping(old_footprint)

    # -------------------- 可见性（多点采样 + 墙体遮挡） ----------------------
    def _half_edge_samples(self, obs: MovableObstacle):
        """矩形的八条半边，每条半边用 (端点, 中点, 端点) 三个采样点表示。
        半边 = 一条边从角点到边中点的那一半。
        只要任意一条半边完全可见，就认为机器人看清了该障碍物。
        """
        coords = list(obs.polygon.exterior.coords)[:-1]   # 4 个角（去掉闭合重复点）
        n = len(coords)
        halves = []
        for i in range(n):
            a = coords[i]
            b = coords[(i + 1) % n]
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            for p, q in ((a, mid), (mid, b)):
                c = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
                halves.append((p, c, q))
        return halves

    def _point_visible(self, robot_pos, p,
                       target: MovableObstacle,
                       world_obstacles: List[MovableObstacle]) -> bool:
        """从机器人到采样点 p 的视线是否畅通（不被墙体或其他障碍物遮挡）
        cfg.sight_width>0 时把视线视为具有该宽度的走廊（缓冲后判断），窄于该宽度的缝隙无法看穿；为 0 时退化为零宽度射线。
        """
        seg = LineString([robot_pos, p])
        width = self.cfg.sight_width
        sight = seg.buffer(width / 2.0, cap_style=2) if width > 0 else seg
        # 墙体遮挡：视线一旦离开静态自由空间（= 工作空间挖去墙体）即被墙挡住
        if not self.roadmap.static_free_prep.contains(sight):
            return False
        # 其他可移动障碍物遮挡
        for w in world_obstacles:
            if w.oid == target.oid:
                continue
            if sight.intersects(w.polygon):
                return False
        return True

    def _visible(self, robot_pos, target: MovableObstacle,
                 world_obstacles: List[MovableObstacle]) -> bool:
        """需求1+2: 至少半条边完整可见即认为感知到该障碍物的全部信息。"""
        for p, c, q in self._half_edge_samples(target):
            if (self._point_visible(robot_pos, p, target, world_obstacles)
                    and self._point_visible(robot_pos, c, target, world_obstacles)
                    and self._point_visible(robot_pos, q, target, world_obstacles)):
                return True
        return False

    # -------------------- 更新 ----------------------
    def _update_edges_for(self, obs: MovableObstacle):
        """Incrementally (re)compute which edges `obs` blocks."""
        poly = obs.polygon
        minx, miny, maxx, maxy = poly.bounds
        pad = 1.0
        for key, corridor in self.roadmap.edge_corridor.items():
            cminx, cminy, cmaxx, cmaxy = corridor.bounds
            if cmaxx < minx - pad or cminx > maxx + pad:      # 低成本的包围盒快速排除
                continue
            if cmaxy < miny - pad or cminy > maxy + pad:
                continue
            blockers = self.edge_blockers.setdefault(key, set())
            if corridor.intersects(poly):
                blockers.add(obs.oid)
            else:
                blockers.discard(obs.oid)

    # -------------------- 碰撞接触（部分信息） ----------------------
    def register_contact(self, region: Polygon):
        """记录一块通过推动碰撞发现的匿名占据区域（不含身份/难度信息）。
        先扣掉已感知障碍物的 footprint：那部分不是"新信息"，而且此后不会再被
        `_clear_contacts_overlapping` 清理（该障碍物不会二次 newly_revealed）
        会退化成一块永久压在它身上的幽灵区域，把它自己锁成不可推动。
        """
        if region is None or region.is_empty:
            return
        for obs in self.perceived.values():
            region = region.difference(obs.polygon)
            if region.is_empty:
                return
        if region.area <= 1e-9:      # 只剩浮点碎片
            return
        # 与已有接触区相接的并进去，不要另起一块。反复顶同一个未知物体会一次留下
        # 一小片薄条，只追加的话 contacts 会无限增长：`_removal` 每次都要对它做
        # unary_union，SE2 构型空间的缓存键（others 的 WKB）也每次都变、永不命中。
        merged = [region]
        rest = []
        for c in self.contacts:
            (merged if c.intersects(region) else rest).append(c)
        rest.append(unary_union(merged) if len(merged) > 1 else region)
        self.contacts = rest

    def _clear_contacts_overlapping(self, poly: Polygon):
        """某障碍物被完整感知后，删除与其重叠的匿名接触记录（已被真实 footprint 取代）。"""
        if not self.contacts:
            return
        self.contacts = [c for c in self.contacts if not c.intersects(poly)]

    def force_reveal(self, world_obs: MovableObstacle) -> List[int]:
        """物理接触导致的强制揭示：无视感知半径与遮挡，直接把障碍物纳入信念。

        用于机器人推动某障碍物时撞上一个尚未感知的物体——碰撞本身就暴露了它。
        """
        if world_obs.oid in self.perceived:
            return []
        obs = world_obs.perceived_copy()
        self.perceived[obs.oid] = obs
        self.newly_revealed.append(obs.oid)
        self._update_edges_for(obs)
        # 真实 footprint 已知，覆盖它的匿名接触区必须清掉，否则会退化成一块永久
        # 压在它身上的幽灵区域，把它自己锁成不可推动（见 register_contact 的说明）。
        self._clear_contacts_overlapping(obs.polygon)
        return [obs.oid]

    def _forget_edges(self, oid: int):
        """清除某障碍物在【所有】边上的阻挡关系。

        不能只清 `_update_edges_for` 扫描的那个包围盒邻域：障碍物移动后新旧位置
        的邻域可能完全不相交，旧位置上的阻挡记录会残留下来。
        """
        for blockers in self.edge_blockers.values():
            blockers.discard(oid)

    def relocate(self, obs: MovableObstacle, x: float, y: float, theta: float):
        """Apply an executed push: move the obstacle and refresh its blocked edges."""
        self._forget_edges(obs.oid)
        obs.x, obs.y, obs.theta = x, y, theta
        obs.removed = True
        self._update_edges_for(obs)

    # -------------------- 需求3: 机器人自身碰撞感知 ----------------------
    @staticmethod
    def _first_contact_t(from_pos, to_pos, poly: Polygon, radius: float,
                         coarse: int = 64, refine: int = 20) -> float:
        """机器人圆盘沿 from->to 平移时，首次碰到 `poly` 的行程比例 t ∈ [0, 1]。

        先粗扫定位到第一个接触样本，再在它与上一个未接触样本之间二分。不能直接对
        [0, 1] 二分：接触区间可能只是路径中段的一小段，终点未必仍在接触，二分会
        整段错过。粗扫漏掉的极短擦碰按 1.0 处理（相当于走完整条边，偏保守）。
        """
        fx, fy = from_pos
        dx, dy = to_pos[0] - fx, to_pos[1] - fy

        def touching(t: float) -> bool:
            return poly.distance(Point(fx + dx * t, fy + dy * t)) <= radius

        if touching(0.0):
            return 0.0
        lo = 0.0
        for i in range(1, coarse + 1):
            t = i / coarse
            if touching(t):
                hi = t
                for _ in range(refine):
                    mid = 0.5 * (lo + hi)
                    if touching(mid):
                        hi = mid
                    else:
                        lo = mid
                return hi
            lo = t
        return 1.0

    def check_robot_collision(
        self, from_pos, to_pos,
        world_obstacles: List[MovableObstacle],
        cfg: Config,
    ) -> Tuple[List[int], float]:
        """检查机器人沿 from->to 是否撞上未感知障碍物，撞到则 force_reveal。

        返回 `(revealed_oids, t_contact)`，`t_contact` 是首次接触处的行程比例，
        无碰撞时为 1.0。**只揭示最先接触到的那些障碍物**：机器人在接触点就停住了，
        走廊后半段扫到的东西它根本没走到，一并揭示等于白送信息。
        """
        corridor = LineString([from_pos, to_pos]).buffer(cfg.robot_radius, cap_style=1)
        candidates = [w for w in world_obstacles
                      if w.oid not in self.perceived
                      and corridor.intersects(w.polygon)]
        if not candidates:
            return [], 1.0

        ts = [self._first_contact_t(from_pos, to_pos, w.polygon, cfg.robot_radius)
              for w in candidates]
        t_hit = min(ts)
        revealed: List[int] = []
        for w, t in zip(candidates, ts):
            if t <= t_hit + 1e-6:          # 同时撞上的算一起
                self.force_reveal(w)
                revealed.append(w.oid)
        return revealed, t_hit

    # -------------------- 需求4: 触摸感知难度 ----------------------
    def touch_check(
        self, robot_pos,
        world_obstacles: List[MovableObstacle],
        cfg: Config,
    ) -> List[int]:
        """触摸圆接触障碍物则揭示真实 difficulty。"""
        touch_radius = cfg.robot_radius + cfg.touch_margin
        rp = Point(robot_pos)
        revealed: List[int] = []
        for w in world_obstacles:
            if w.oid in self.touched:
                continue
            if rp.distance(Point(w.center())) > touch_radius + max(w.l, w.d):
                continue
            if rp.buffer(touch_radius).intersects(w.polygon):
                self.touched.add(w.oid)
                self.touched_difficulty[w.oid] = w.difficulty
                revealed.append(w.oid)
        return revealed

    def get_difficulty(self, oid: int, estimator) -> float:
        """搜索阶段获取难度: 触摸过用真实值，否则用 LLM/启发式估计。"""
        if oid in self.touched_difficulty:
            return self.touched_difficulty[oid]
        return estimator.estimate(self.perceived[oid].observation())

    # -------------------- 查询 ----------------------
    def blockers_of(self, key: EdgeKey) -> Set[int]:
        return self.edge_blockers.get(key, set())

    def obstacle(self, oid: int) -> MovableObstacle:
        return self.perceived[oid]
