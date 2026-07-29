"""仿真世界与 SE2 推动规划器之间的适配层。

push_planner 是一个纯 numpy 的离散搜索引擎，只认识"矩形尺寸 + 墙体顶点数组 +
网格参数"。本模块负责把 shapely / Config / MovableObstacle 翻译成它要的形式，
把它给出的离散答案拿回真实几何复核，并把路径换算成做功：

1. push_plan_se2()  : 唯一的推动规划入口——建/取规划器实例、设走廊、取最优位姿与
                      到达它的连续路径，并用 shapely 复核该位姿真的腾空了走廊
2. se2_path_cost()  : 一段 SE2 位姿序列的推动代价（口径与规划器内部一致）
   push_work()      : 代价 x 难度 = 操作做功
3. _swept_region()  : 矩形位姿间的扫掠区。供 planner.py 在【执行期】做真值碰撞
                      校验，不参与推动规划本身
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
from shapely.geometry import Polygon
from shapely.ops import unary_union
from obstacle import MovableObstacle
from config import Config
import push_planner

# -------------计算器 1：推动规划--------------- #

# 扫掠区离散化时允许的最大单步转角（15°）。步长越小越贴近真实扫掠区，
# 代价是更多次 shapely 布尔运算。
_SWEPT_MAX_DTHETA = math.pi / 12.0


def _swept_region(obs: MovableObstacle, nx: float, ny: float,
                  theta: Optional[float] = None) -> Polygon:
    """矩形从当前位姿运动到 (nx, ny, theta) 所扫过的区域"""
    start = obs.polygon
    dtheta = 0.0 if theta is None else push_planner.wrap_dtheta(obs.theta, theta)
    if abs(dtheta) < 1e-9:
        return unary_union([start, obs.polygon_at(nx, ny)]).convex_hull
    steps = max(2, int(math.ceil(abs(dtheta) / _SWEPT_MAX_DTHETA)))
    poses = [obs.polygon_at(obs.x + (nx - obs.x) * i / steps,
                            obs.y + (ny - obs.y) * i / steps,
                            obs.theta + dtheta * i / steps)
             for i in range(steps + 1)]
    return unary_union([unary_union([a, b]).convex_hull
                        for a, b in zip(poses, poses[1:])])


# ---------- 计算器 1b：SE2 推动路径规划 ---------- #

_MAX_PUSH_STATES = 200_000
# 构型空间的构建是全流程最贵的一步，故按"决定该 C-space 的全部参数"缓存规划器实例。
# 每个实例约 4 MB（若干 200k 元素的布尔网格 + Dijkstra 结果），而缓存键现在是精确的，
# 命中率低于从前的模糊键 —— 必须限容，否则长时间运行会把内存吃光。按 LRU 淘汰。
_PLANNER_CACHE: Dict[tuple, push_planner.PushPlanner] = {}
_PLANNER_CACHE_MAX = 32


def _polygon_parts(geom):
    """把任意 shapely 几何摊平成多边形列表（丢弃线/点等退化部件）。

    `others` 里混有 `swept.intersection(wall)` 产生的接触区，退化情况下会是
    LineString 或 GeometryCollection；它们没有 `.exterior`，直接喂给规划器会抛异常
    并被吞掉，表现为"这个障碍物忽然不可推动"。
    """
    if geom is None or geom.is_empty:
        return []
    parts = getattr(geom, "geoms", None)
    parts = list(parts) if parts is not None else [geom]
    return [p for p in parts if p.geom_type == "Polygon" and not p.is_empty]


def _geometry_signature(geom) -> tuple:
    """几何体的精确缓存标识。

    曾用 `(面积, bounds[0])` 概括 `others`，但障碍物做竖直平移时这两个量都不变，
    不同布局会命中同一条缓存 —— 于是复用了按【旧位置】构建的构型空间，规划出的
    推动路径直接穿过实际存在的障碍物。这里改用精确的 WKB 字节：命中即等价，
    不命中最多是重建一次（安全），绝不会张冠李戴。
    """
    if geom is None or geom.is_empty:
        return ("none",)
    return ("wkb", geom.wkb)


def _resolve_cell(bounds: Tuple[float, float, float, float], cfg: Config) -> float:
    """在状态总数上限内自适应放大网格单元。"""
    xmin, xmax, ymin, ymax = bounds
    bw, bh = xmax - xmin, ymax - ymin
    cell = cfg.push_cell
    if int(bw / cell) * int(bh / cell) * cfg.push_n_theta > _MAX_PUSH_STATES:
        cell = max(cell, math.sqrt(bw * bh * cfg.push_n_theta / _MAX_PUSH_STATES))
    return cell


def _get_push_planner(obs: MovableObstacle, static_obstacles,
                      bounds: Tuple[float, float, float, float],
                      robot_pos: Tuple[float, float], cfg: Config,
                      others_polys=None) -> push_planner.PushPlanner:
    """取得（或构建）与这组参数对应的 PushPlanner。"""
    cell = _resolve_cell(bounds, cfg)
    # 工作圆半径 = 机器人到障碍物距离 + R_push（障碍物始终在圆内且有推送空间）
    dist_to_obs = math.hypot(robot_pos[0] - obs.x, robot_pos[1] - obs.y)
    work_radius = (float("inf") if cfg.push_containment == "none"
                   else dist_to_obs + cfg.R_push + 1.0)

    key = (obs.l, obs.d, bounds, robot_pos, round(work_radius, 6), cell,
           cfg.push_n_theta, cfg.push_connectivity, cfg.push_rot_weight,
           cfg.push_containment, cfg.push_forward_penalty,
           _geometry_signature(others_polys))
    planner = _PLANNER_CACHE.get(key)
    if planner is not None:
        _PLANNER_CACHE[key] = _PLANNER_CACHE.pop(key)   # 标记为最近使用
        return planner

    # 静态墙体 + 其他可移动障碍物，一起作为不可穿越区域
    walls = [so.polygon for so in static_obstacles] + _polygon_parts(others_polys)
    planner = push_planner.build_push_planner(
        wall_polys=walls,
        obstacle_w=obs.l, obstacle_h=obs.d,
        bounds=bounds, robot_pos=robot_pos,
        work_radius=work_radius,
        cell=cell, n_theta=cfg.push_n_theta,
        connectivity=cfg.push_connectivity,
        rot_weight=cfg.push_rot_weight,
        containment=cfg.push_containment,
        forward_penalty=cfg.push_forward_penalty,
        verbose=cfg.verbose,
    )
    _PLANNER_CACHE[key] = planner
    while len(_PLANNER_CACHE) > _PLANNER_CACHE_MAX:
        _PLANNER_CACHE.pop(next(iter(_PLANNER_CACHE)))   # 淘汰最久未用的
    return planner


def push_plan_se2(
    obs: MovableObstacle,
    must_clear_polys,                   # 必须被腾空的走廊多边形
    static_obstacles,                   # StaticObstacle 列表
    bounds: Tuple[float, float, float, float],
    robot_pos: Tuple[float, float],
    cfg: Config,
    others_polys=None,                  # 需要避开的其他可移动障碍物
) -> Tuple[bool, Optional[list], float, Optional[Tuple[float, float, float]]]:
    """在 SE2 全空间搜索能腾空走廊的最省代价位姿。

    位姿与到达它的路径由同一次 Dijkstra 一并给出，旋转与不旋转的候选目标平等竞争。
    这是唯一的推动规划入口：搜不出来就是不可推动，不存在"瞬移到落点"的兜底。
    返回 (feasible, path, push_cost, goal_pose)。
    """
    if not cfg.push_use_planner:
        return (False, None, math.inf, None)
    try:
        planner = _get_push_planner(obs, static_obstacles, bounds, robot_pos,
                                    cfg, others_polys)
        # 每次都要重设走廊：被阻挡的边逐周期不同
        corridor_verts = [push_planner.polygon_exterior_coords(p)
                          for p in must_clear_polys] if must_clear_polys else []
        planner.set_corridor([c for c in corridor_verts if len(c) >= 3])

        result = planner.plan_anywhere((obs.x, obs.y, obs.theta))
        if not result.success:
            return (False, None, math.inf, None)
        # 精确几何复核：规划器是在离散构型空间里判断"腾空"的，起点还会被吸附到格心，
        # 于是可能出现真实位姿压着走廊、格心却不压的错位——那样它会把"原地不动"当成
        # 零代价解返回。搜索侧据此认定该障碍物免费让开，机器人过去执行一次零位移的
        # 推动，走廊纹丝不动，下一轮再得出同样结论，就此无限空转。
        # 这里用 shapely 按真实几何再判一次，通不过就判该障碍物本轮不可推动。
        end_poly = obs.polygon_at(*result.goal)
        if any(end_poly.intersects(p) for p in (must_clear_polys or [])):
            cfg.log(f"[push_plan_se2] oid={obs.oid} 目标位姿实际未腾空走廊，判为无解")
            return (False, None, math.inf, None)
        return (True, result.path, result.cost, result.goal)
    except Exception as e:
        cfg.log(f"[push_plan_se2] error: {e}")
        return (False, None, math.inf, None)


# ---------- 计算器 1c：推动做功 ---------- #

def se2_path_cost(obs: MovableObstacle, poses, cfg: Config) -> float:
    """一段 SE2 位姿序列的推动代价 = 平移弧长 + r̄ * 累计转角。

    与 push_planner 内部的代价定义（rot_weight = r̄）完全一致，因此返回值可以直接
    当作 `push_work()` 的 `push_distance`。

    用途是按【实际执行的那段轨迹】结算做功：推动中途被撞停时，障碍物只走完了
    规划路径的一个前缀，不能再拿规划时的全程代价去记账，也不能当作没发生过。
    """
    if poses is None or len(poses) < 2:
        return 0.0
    rot_weight = (push_planner.mean_rotation_radius(obs.l, obs.d)
                  if cfg.push_rot_weight is None else float(cfg.push_rot_weight))
    total = 0.0
    for a, b in zip(poses, poses[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        total += rot_weight * abs(push_planner.wrap_dtheta(a[2], b[2]))
    return total


def push_work(difficulty: float, push_distance: float) -> float:
    """推开障碍物的操作功。difficulty 即物理模型中的 μmg。

    刚体在均匀压力下被推动，做功分为平移与绕形心转动两部分：

        J1 = μmg · d                 (平移距离 d)
        J2 = μmg · θ · r̄             (转动 θ，r̄ = 摩擦力矩臂)

        W  = J1 + J2 = difficulty · (d + θ·r̄)

    `push_distance` 传入的正是括号里那个 `d + θ·r̄`：SE2 规划器以
    rot_weight = r̄ 累加代价（见 push_planner.mean_rotation_radius），
    所以它返回的 cost 天然就是这个和，无需在此再拆分。

    显式传入 difficulty 而不是从障碍物对象里读：规划期用的是估计值（来自信念），
    结算期用的是 ground truth（来自世界），两者绝不能混。
    """
    return difficulty * push_distance
