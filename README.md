# CA-NAMO：基于 LLM 操作难度估计的代价感知可移动障碍物导航

本仓库实现了 *Cost-Aware Online Navigation Among Movable Obstacles* 的参考原型。机器人在线地将行驶距离与操作做功统一为一个代价函数，利用 LLM（带离线启发式回退）估计每个障碍物的推动难度，从而在"绕路"与"推开"之间做定量的权衡决策——全程在线运行，感知到新障碍物后即时重规划。

```
J = λ × D + W

其中：
  λ = lambda_distance（运动代价权重）
  D = 总行驶距离
  W = Σ(真实 difficulty × 实际推动距离)（操作做功）
```

## 快速开始

```bash
pip install -r requirements.txt

# 默认场景 two_doors，启发式难度估计
python main.py

# 切换场景
python main.py --scenario maze_doors

# 调大 λ 使机器人更倾向推开障碍物而非绕路（λ 越大，绕路越"贵"）
python main.py --lambda 5.0

# 三种规划策略
python main.py --strategy normal      # 最优 J = λD + W（默认）
python main.py --strategy shortest    # 最短路径优先，忽略 W
python main.py --strategy easiest     # 尽量绕路，少搬东西

# 启用 LLM 难度估计（需要 DeepSeek API Key）
DEEPSEEK_API_KEY=sk-xxxx python main.py
```
汇总图输出到 `img/summary_<场景名>.png`，逐帧图输出到 `img/frames_<场景名>/`。
非 `normal` 策略会在文件名加后缀，如 `summary_two_doors_shortest.png`。
## 系统架构
```
┌─────────────────────────────────────────────────────┐
│                  OnlineNAMO (planner.py)             │
│  维护两套世界模型：self.world（真实世界）               │
│  + self.belief（机器人信念，仅部分可观测）               │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ perceive │ → │   plan   │ → │ execute (move or  │ │
│  │ (感知)    │   │ (A* 搜索) │   │  push obstacle)  │ │
│  └──────────┘   └──────────┘   └──────────────────┘ │
│       ↑                              │               │
│       └──────── 重规划循环 ──────────┘               │
└─────────────────────────────────────────────────────┘
```
### 核心循环（每次 replan cycle）
1. **感知**（`perception.py`）：以机器人当前位置为圆心、`R_perc=10m` 为半径，多点采样视线检测（考虑墙体遮挡和其他障碍物遮挡），发现并登记可见障碍物
2. **规划**（`search.py`）：在增广路网上运行 A* 分支限界搜索，对每条被阻挡的边调用 SE2 推动规划器计算推开障碍物的代价
3. **执行**（`planner.py`）：逐步执行移动或推动动作。推动过程中检测与其他障碍物的碰撞，碰撞则原地感知并重规划
4. **重规划**：每次移动后重新感知（频率由 `step_execute_edges` 控制），发现新障碍物立即重规划

## 代码与算法对照
### 代价函数 `J = λ × D + W`
- **`config.py`**：`lambda_distance` 权重
- 运动代价 `λ × D`：在 `planner.py` 执行阶段累加真实行驶距离
- 操作代价 `W`：在 `planner.py` 执行阶段用**真实 difficulty** × 实际推动距离结算
- J 的计算公式在三种策略下不变，策略只影响搜索决策

### 三种规划策略（`search.py` — `strategy` 参数）

| 策略 | `--strategy` | 搜索行为 | 适用场景 |
|---|---|---|---|
| **normal** | `normal`（默认） | 完整 `J = λD + W`，平衡路径长度与操作代价 | 最优决策 |
| **shortest** | `shortest` | 搜索时忽略 W（`work_mult=0`），始终选几何最短路径 | 对比基线：不绕路 |
| **easiest** | `easiest` | 每个障碍物加 +50 惩罚（`work_bias=50`），倾向绕路但不禁止推动 | 对比基线：尽量不搬 |
> J 的计算公式在三种策略下完全一致，策略只改变 A* 搜索时对边代价的计算方式。执行阶段始终按真实物理代价结算。

### 增广路网（`roadmap.py`）
- **构建一次**，基于静态墙体。在自由空间内以 `grid_step=1m` 间距均匀采样节点，用 `conn_radius=2m` 连接邻近节点
- 可移动障碍物**不改变图结构**，只标记哪些边当前被阻挡（"付费解锁"边）
- 起点/终点动态插入路网（`add_terminal`），无法放入机器人圆盘时退化到最近合法节点
- 最近邻查询使用 `scipy.spatial.KDTree`（O(log N)）

### SE2 推动路径规划器（`push_planner.py` + `geometry.py`）
这是整个系统的**计算核心**——为障碍物规划一条连续的 SE(2) 路径，使其从当前位置移动到不阻挡走廊的位置。
**构型空间**：
- 将工作空间离散为 `push_cell=0.15m` 的平移网格 × `push_n_theta=12` 的角度层
- 状态总数约 20 万（自动限制在 `_MAX_PUSH_STATES` 以内）
- 用 Minkowski 和计算每个角度层对每面墙的 C-障碍物

**Dial-bucket Dijkstra 搜索**：
- 平移边权重按轴/对角/马步分级（`_W_AXIS=990`, `_W_DIAG=1400`, `_W_KNIGHT=2214`）
- 旋转边权重基于障碍物平均旋转半径
- **提前终止**：根据走廊范围自动计算最大搜索距离（走廊对角线 + 障碍物对角线），超过该距离的状态不再扩展
- 搜索结果缓存在 PushPlanner 实例中（按起点位姿索引）

**自适应旋转扫掠体计算**（`geometry.py — _swept_between`）：
- 旋转步长根据障碍物尺寸自适应缩放的逆关系）：长障碍物的旋转凸包膨出更大，需要更细的角分辨率
- `_PATH_CONTACT_AREA_EPS=1e-4`（约 1 cm²）吸收凸包近似的亚厘米浮点误差

**落点偏好**：
- `_forward_bias`：倾向于向前（远离机器人）放置障碍物，避免推到身后挡住来路
- `_select_goal`：在走廊掩码之外的候选落点中选最优（距离最短 + 前向偏置最小）

**缓存策略**：
- `_PLANNER_CACHE`（LRU, max 32）：缓存 PushPlanner 实例，按障碍物尺寸、机器人位置、其他障碍物几何签名等作为 key
- `_CORRIDOR_MASK_CACHE`（LRU, max 4096）：缓存走廊掩码的压缩位图
- `_persistent_removal_cache`（search.py）：跨 replan cycle 持久化推动规划结果，key 为 `(push_signature(obs), edge_key)`，新感知到障碍物时清空

### 推动路径验证（`geometry.py — push_plan_se2`）
1. Dijkstra 搜索返回离散候选路径
2. **扫掠体验证**（`_path_is_clear_against`）：逐段计算障碍物在两相邻位姿间的扫掠体（包含旋转），与所有静态墙体 + 其他可移动障碍物做碰撞检测
3. **落点验证**：确保目标位姿确实不再与走廊相交

### 感知系统（`perception.py`）
- **可见性判定**：对障碍物轮廓的 8 条半边各取 3 个采样点（共 24 点），逐点做视线检测
- **墙体遮挡**：视线离开静态自由空间即被遮挡
- **其他障碍物遮挡**：视线穿过其他可移动障碍物即被遮挡
- **增量更新**：已知障碍物且位姿一致 → 跳过；位姿变化 → 同步位姿并更新阻挡的边
- **碰撞接触**（`register_contact`）：匿名接触记录，后续完整感知到该障碍物时自动清除
- **碰撞感知**（`check_robot_collision`）：二分搜索 + 精细细化定位机器人与未感知障碍物的碰撞时刻
- **触摸感知**（`touch_check`）：机器人圆盘与障碍物相交时揭示其真实 difficulty

### 难度估计（`llm_difficulty.py`）
- **启发式方法**（默认）：`difficulty = material_density × footprint_area`，基于材质密度表（泡沫 → 工业机械，共 18 种材质 + 同义词典）
- **LLM 方法**（需 API Key）：DeepSeek 根据材质名称和尺寸估计难度系数，失败时自动回退到启发式
- **用途**：LLM 估计值仅影响 A* 展开顺序（`_llm_bias`），不影响 g 值和最终代价——错误的 LLM 估计只改变收敛速度，不改变规划结果的最优性

### A* 搜索过程（`search.py`）
- **状态**：路网节点
- **g 值**：从起点到当前节点的累计代价（λ × 已行驶距离 + 已规划的操作做功）
- **h 值**：`λ × 到目标的欧氏距离`（可采纳的下界）
- **分支限界**：`f > incumbent` 时剪枝
- **LLM 偏置**（可选）：对包含高难度障碍物的边加偏置，优先探索"容易"的路径

### 在线执行循环（`planner.py`）
- 每次 replan 执行规划路径的前 `step_execute_edges` 条边
- 移动前做碰撞感知（检测路径上是否有未感知障碍物）
- 推动前做触摸感知（获取真实 difficulty）
- 推动过程逐步执行 SE2 路径，每步用 STRtree 空间索引检测碰撞
- 碰撞/新发现 → 立即重规划
- 保底机制（`failed_pushes`）：已确认不可行的推动方案不再重试

## 场景列表

| 场景 | 特点 | 障碍物数 |
|---|---|---|
| `corridor` | 狭长走廊单障碍物 | 1 |
| `two_doors` | 双门经典场景（近门有遮挡物） | 4 |
| `hidden_obstacle` | 遮挡障碍物测试（B 完全遮挡 C） | 3 |
| `hidden_obstacle_backtrack` | 折返场景：近门有重物，远门被诱饵和重物堵住 | 3 |
| `maze_soft` | 30×30 迷宫，3 个轻障碍物 | 3 |
| `maze_hard` | 30×30 迷宫，3 个重障碍物 | 3 |
| `maze_mixed` | 30×30 迷宫，5 个混合难度障碍物 | 5 |
| `maze_doors` | 28 扇门洞障碍物 + 固定障碍物 | 48 |
| `maze_doors_complex` | 28 扇门洞 + 8 斜向 + 8 小方块 + 固定障碍物 | 63 |

## 命令行参数
```
python main.py [--scenario NAME] [--lambda VALUE] [--strategy NAME]
               [--no-llm-order] [--frames]
```

| 参数 | 说明 |
|---|---|
| `--scenario` | 场景名（默认 `two_doors`），可选值见上表 |
| `--lambda` | 运动代价 λ 权重（默认 1.0），越大越倾向推开障碍物 |
| `--strategy` | 规划策略：`normal`（默认）/ `shortest` / `easiest` |
| `--no-llm-order` | 禁用 LLM 排序（纯启发式展开） |
| `--frames` | 保存每一步运动的逐帧图片 |

## 输出指标
运行结束后打印：
```
Success                : True   (Reached goal.)
Total cost J           : 24.57
motion lambda*D        : 23.68
obstacle work W        : 0.89
Obstacles moved        : [1, 2]
Replan cycles          : 10
Total plan time (s)    : 2.07
A* expansions          : 163
LLM calls              : 0  (mode=heuristic)
```

- **J** = λD + W（总代价）
- **motion lambda times D** = λ × 总行驶距离（运动代价）
- **obstacle work W** = Σ(真实 difficulty × 实际推动距离)（操作代价）
- **Obstacles moved** = 被推开的障碍物 ID 列表
- **Replan cycles** = 重规划次数
- **Total plan time** = 全部规划耗时之和
- **A\* expansions** = A* 节点展开总数
- **LLM calls** = LLM API 调用次数（heuristic 模式下为 0）

## 文件结构
```
config.py          所有可调参数（λ、感知半径、推动半径、网格步长等）
obstacle.py        StaticObstacle / MovableObstacle 数据结构
roadmap.py         一次性构建的增广路网
geometry.py        SE2 推动规划适配层（扫掠体、路径验证、代价计算）
push_planner.py    SE2 网格 Dial-bucket Dijkstra 推动路径搜索器
llm_difficulty.py  DeepSeek 难度估计 + 离线启发式回退
perception.py      感知、遮挡、碰撞检测、信念状态维护
search.py          A* f=g+h 分支限界搜索 + 三种策略
planner.py         在线"规划-执行-感知-重规划"主循环
main.py            入口、可视化、指标输出
scenarios/         场景定义（9 个场景）
```

---

## 推动规划的性能优化
以下优化已实装，显著提升了推动路径规划的速度：
1. **跨 cycle 持久化推动方案缓存**（`search.py — _persistent_removal_cache`）：同一障碍物在同一条边上的推动方案在多次重规划之间复用，避免重复跑 SE2 Dijkstra（实测命中率 ~70%）
2. **Dijkstra 提前终止**（`push_planner.py — _search max_bucket`）：根据走廊范围自动限制搜索半径，避免探索障碍物不可能到达的远端状态
3. **KDTree 最近邻查询**（`roadmap.py`）：替代 O(N) 线性扫描
4. **STRtree 空间索引**（`planner.py — _execute_push_path`）：推动执行阶段每步碰撞检测从 O(N) 降至 O(log N)
5. **自适应旋转扫掠体**（`geometry.py — _swept_between`）：旋转步长按障碍物尺寸反比缩放，长障碍物自动获得更细的角度分辨率

## 正确性保证
LLM 难度估计仅影响 A* 的**展开顺序**（`search.py — _llm_bias`），不影响 g 值计算和最终解的代价。
g 值来自几何计算器使用真实几何得出的结果，h 是可采纳的下界，因此第一个被弹出的目标状态在当前信念下是代价最优的。
这里的"最优"指在机器人当前知识下的最优；结合持续在线重规划——不是一次性全局最优（因为障碍物几何先验未知，符合问题定义）
