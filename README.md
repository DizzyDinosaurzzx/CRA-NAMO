# CA-NAMO：基于 LLM 操作难度估计的代价感知可移动障碍物导航

本仓库实现了 *Cost-Aware Online Navigation Among Movable Obstacles* 的参考原型。机器人在线地将行驶距离与操作做功统一为一个代价函数，利用 LLM（带离线启发式回退）估计每个障碍物的推动难度，从而在"绕路"与"推开"之间做定量的权衡决策——全程在线运行，感知到新障碍物后即时重规划。
```
J = λ × D + W

其中：
  λ = lambda_distance（运动代价权重）
  D = 总行驶距离
  W = Σ(真实 difficulty × 实际推动距离)（操作做功）
```
**估计值用于决策，真值用于结算**：规划时用估计难度算 W，执行时一旦与障碍物发生物理接触（碰撞或推动）就换成真实难度重新结算，并在后续重规划中沿用真值。
## 快速开始
```bash
pip install -r requirements.txt

# 默认场景 two_doors
python main.py

# 切换场景
python main.py --scenario maze_doors

# 调大 λ 使机器人更倾向推开障碍物而非绕路（λ 越大，绕路越"贵"）
python main.py --lambda 5.0

# 三种规划策略
python main.py --strategy normal      # 最优 J = λD + W（默认）
python main.py --strategy shortest    # 最短路径优先，忽略 W
python main.py --strategy easiest     # 尽量绕路，少搬东西

# 保存过程动图
python main.py --frames
```
汇总图输出到 `img/summary_<场景名>.png`，逐帧过程动图输出到 `img/frames_<场景名>.gif`。
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
3. **执行**（`planner.py`）：逐步执行移动或推动动作。推动前揭示真实难度，推动中检测与其他障碍物的碰撞，碰撞则原地感知并重规划
4. **重规划**：每次移动后重新感知（频率由 `step_execute_edges` 控制），发现新障碍物立即重规划

## 代码与算法对照
### 代价函数 `J = λ × D + W`
- **`config.py`**：`lambda_distance` 权重
- 运动代价 `λ × D`：在 `planner.py` 执行阶段累加真实行驶距离
- 操作代价 `W`：在 `planner.py` 执行阶段用**真实 difficulty** × 实际推动距离结算（`planner.py — _world_obstacle(...).difficulty`），只对**实际推完的那一段**计费；推到一半撞上东西，只结算已推距离
- J 的计算公式在三种策略下不变，策略只影响搜索决策

### 三种规划策略（`search.py` — `strategy` 参数）

| 策略 | `--strategy` | 搜索行为 | 适用场景 |
|---|---|---|---|
| **normal** | `normal`（默认） | 完整 `J = λD + W`，平衡路径长度与操作代价 | 最优决策 |
| **shortest** | `shortest` | 搜索时忽略 W（`work_mult=0`），始终选几何最短路径 | 对比基线：不绕路 |
| **easiest** | `easiest` | 每个障碍物加 +50 惩罚（`work_bias=50`），倾向绕路但不禁止推动 | 对比基线：尽量不搬 |
> J 的计算公式在三种策略下完全一致，策略只改变 A* 搜索时对边代价的计算方式。执行阶段始终按真实物理代价结算。

### 增广路网（`roadmap.py`）
- **构建一次**，基于静态墙体。在自由空间内以 `grid_step=0.3m` 间距均匀采样节点，用 `conn_radius=0.6m` 连接邻近节点（two_doors 约 6000 节点 / 34000 边）
- 可移动障碍物**不改变图结构**，只标记哪些边当前被阻挡（"付费解锁"边）
- 起点/终点动态插入路网（`add_terminal`），无法放入机器人圆盘时退化到最近合法节点
- 最近邻查询使用 `scipy.spatial.KDTree`（O(log N)）

### SE2 推动路径规划器（`push_planner.py` + `geometry.py`）
这是整个系统的**计算核心**——为障碍物规划一条连续的 SE(2) 路径，使其从当前位置移动到不阻挡走廊的位置。
**构型空间**：
- 将工作空间离散为 `push_cell=0.15m` 的平移网格 × `push_n_theta=12` 的角度层
- 状态总数自动限制在 `_MAX_PUSH_STATES=200000` 以内（超了就放大 cell）
- 用 Minkowski 和计算每个角度层对每面墙的 C-障碍物

**Dial-bucket Dijkstra 搜索**：
- 平移边权重按轴/对角/马步分级（`_W_AXIS=990`, `_W_DIAG=1400`, `_W_KNIGHT=2214`）
- 旋转边权重基于障碍物平均旋转半径
- **提前终止**：根据走廊范围自动计算最大搜索距离（走廊对角线 + 障碍物对角线），超过该距离的状态不再扩展
- 搜索结果缓存在 PushPlanner 实例中（按起点位姿索引）

**落点偏好**：
- `_forward_bias`：倾向于向前（远离机器人）放置障碍物，避免推到身后挡住来路
- `_select_goal`：在走廊掩码之外的候选落点中选最优（距离最短 + 前向偏置最小）

**缓存策略**：
- `_PLANNER_CACHE`（LRU, max 32）：缓存 PushPlanner 实例，按障碍物尺寸、机器人位置、其他障碍物几何签名等作为 key
- `_CORRIDOR_MASK_CACHE`（LRU, max 4096）：缓存走廊掩码的压缩位图
- `_persistent_removal_cache`（search.py）：跨 replan cycle 持久化推动规划结果，key 为 `(push_signature(obs), edge_key)`，新感知到障碍物时清空

### 自适应旋转扫掠体（`geometry.py — _swept_between`）
扫掠体（障碍物从位姿 A 连续移动到位姿 B 扫过的区域）是碰撞检测的依据。做法是把 A→B 插值成若干中间位姿，**相邻两个位姿取凸包**再并起来。
问题出在旋转上：凸包用一条**弦**去近似旋转扫出的**圆弧**，弦与弧之间的那块"膨出"区域被漏掉了。漏掉的高度就是弓形矢高
```
bulge = R × (1 − cos(Δθ / 2))          R = ½·√(l² + d²)（障碍物半对角）
```
它与障碍物尺寸 **成正比**——固定角步长时，物体越长漏得越多，而漏掉的是**真实会扫到、但检测认为空着**的区域，属于假阴性（规划期认为安全，执行期擦碰）。
所以角步长按尺寸反比缩放：
```python
max_dtheta = (π/12) × 0.5 / max(R, 0.5)     # 0.5 m 半对角为基准，小于它的一律 15° 封顶
steps      = ceil(|Δθ| / max_dtheta)
```

| 障碍物 `l×d` | 半对角 R | 自适应 Δθ | 90° 旋转切分 | 固定 15° 的膨出 | 自适应后膨出 |
|---|---|---|---|---|---|
| 0.7×6.0（corridor 长板） | 3.02 m | 2.48° | 37 段 | **25.8 mm** | 0.71 mm |
| 2.0×2.0 | 1.41 m | 5.30° | 17 段 | 12.1 mm | 1.51 mm |
| 1.5×1.5 | 1.06 m | 7.07° | 13 段 | 9.1 mm | 2.02 mm |
| 半对角 ≤0.5 m 的小件 | ≤0.5 m | 15°（封顶） | 6 段 | 4.3 mm | 4.3 mm |
小物体保持 15°（不为精度浪费段数），大物体自动加密；代价是长条形障碍物的扫掠体计算变慢，但换来的是**规划期判定与执行期实际扫掠不再失配**——这一类失配正是"规划说能推、执行时撞上"的主要来源。
`push_planner.py` 内部也用同一个矢高公式（`self.bulge`）膨胀 C-障碍物，两处口径一致。
**接触容差**：`_PATH_CONTACT_AREA_EPS = 1e-6`（约 1 mm²）。扫掠体与障碍物相交面积小于它才判为"未碰撞"——纯粹用来吸收凸包近似与浮点运算的残差，不是真的允许 1 mm² 的物理重叠。
### 推动路径验证（`geometry.py — push_plan_se2`）
1. Dijkstra 搜索返回离散候选路径
2. **扫掠体验证**（`_path_is_clear_against`）：逐段计算障碍物在两相邻位姿间的扫掠体（含上面的自适应旋转细分），与所有静态墙体 + 其他可移动障碍物做碰撞检测，先用包围盒粗筛再算精确相交
3. **落点验证**：确保目标位姿确实不再与走廊相交

### 感知系统（`perception.py`）
- **可见性判定**：把障碍物轮廓的 4 条边各切成 2 段共 8 条半边，每条半边取首/中/尾 3 个点（共 24 点）；**同一条半边的 3 个点全部通视**才算这个障碍物可见
- **墙体遮挡**：视线离开静态自由空间即被遮挡
- **其他障碍物遮挡**：视线穿过其他可移动障碍物即被遮挡
- **增量更新**：已知障碍物且位姿一致 → 跳过；位姿变化 → 同步位姿并更新阻挡的边
- **碰撞接触**（`register_contact`）：匿名接触记录，后续完整感知到该障碍物时自动清除
- **碰撞感知**（`check_robot_collision`）：二分搜索 + 精细细化定位机器人与未感知障碍物的碰撞时刻
- **触摸感知**（`touch_check`）：机器人圆盘与障碍物相交时揭示其真实 difficulty
- **交互揭示**（`reveal_by_interaction`）：**推动本身就是接触**，被推的障碍物在推动开始前无条件揭示真值。不能出现"已经把东西推走了却仍不知道它有多难推"的状态——而只靠 `touch_check` 的邻近采样是会漏的：推动阶段机器人停在路网节点上不动，与障碍物的距离可能大于 `robot_radius`

### 难度估计（`llm_difficulty.py`）
```
difficulty = density(material) × footprint_area
```
- **锚点表**：22 种材质（styrofoam_box 0.004 → industrial_machine 37.5）+ 12 条同义词映射。材质名精确命中锚点时**直接查表，根本不调 LLM**
- **LLM 方法**：材质名不在锚点表里时，才请求 DeepSeek 估一个 density；prompt 里给出完整锚点表要求模型插值。失败自动回退启发式（`material_density` 的分词模糊匹配）
- **缓存**：按 oid 缓存最终 difficulty，按材质名缓存 density，同材质只问一次
- **用途**：估计值进入 A* 的 **g 值**（`search.py — _removal`：`work = push_work(estimated_diff, push_dist)`），因此**会**影响选哪条路、推哪个障碍物；另有一个只影响展开顺序的偏置 `_llm_bias`（`--no-llm-order` 可关）。但最终 J 始终按真值结算，估计错了只会让决策次优，不会让账算错

### A* 搜索过程（`search.py`）
- **状态**：路网节点
- **g 值**：从起点到当前节点的累计代价（λ × 已行驶距离 + 按**估计难度**算的操作做功）
- **h 值**：`λ × 到目标的欧氏距离`（可采纳的下界）
- **分支限界**：`f > incumbent` 时剪枝
- **LLM 偏置**（可选）：对包含高难度障碍物的边加偏置，优先探索"容易"的路径

### 在线执行循环（`planner.py`）
- 每次 replan 执行规划路径的前 `step_execute_edges` 条边
- 移动前做碰撞感知（检测路径上是否有未感知障碍物）
- 推动前做触摸感知 + 交互揭示（获取真实 difficulty）
- 推动过程逐步执行 SE2 路径，每步用 STRtree 空间索引检测碰撞
- 碰撞/新发现 → 立即重规划
- 保底机制（`failed_pushes`）：已确认不可行的推动方案不再重试

## 可视化
**汇总图** `img/summary_<场景>.png`：终态 + 机器人轨迹走廊 + 障碍物初始位置（红色虚线）。
**过程动图** `img/frames_<场景>.gif`：每个事件（一次移动 / 一次推动子步 / 一次揭示）一帧，合成单个循环 GIF。全部帧共用一张调色板，Pillow 因此只存帧间差异——路网背景每帧都重画且完全相同，不共享调色板的话文件会大 20 倍以上（maze_doors 72 帧、dpi=110 实测：0.38 MB vs 8.6 MB）。

| 参数（`config.py`） | 默认 | 作用 |
|---|---|---|
| `gif_fps` | 5.0 | 播放速度。总时长 = 帧数 / fps；只影响节奏，不影响体积 |
| `gif_end_hold_s` | 2 | 末帧停留秒数，循环时给眼睛一个"终态"锚点 |
| `gif_dpi` | 300 | 每帧渲染分辨率。体积随 dpi 近似线性（差分编码后变化区域很小）；标签字号只有 7pt，低于 100 会糊 |
**障碍物标签**：`oid / est=… / true=…`，其中真值一行在**与估计不一致时**显示为 `T=…`，用来一眼看出 LLM/启发式估错了哪些。没触碰过的障碍物只有 `est=`。
## 场景列表

| 场景 | 特点 | 障碍物数 |
|---|---|---|
| `corridor` | 狭长走廊单障碍物（0.7×6 长板，考验自适应旋转扫掠） | 1 |
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
| `--no-llm-order` | 禁用 LLM 排序偏置（`_llm_bias`，只影响展开顺序） |
| `--frames` | 把每一步运动保存为动图 `img/frames_<场景名>.gif` |

## 输出指标
运行结束后打印（示例为默认场景 two_doors）：
```
Success                : True   (Reached goal.)
Total cost J           : 20.9254
motion lambda*D        : 20.0
obstacle work W        : 0.9254
Obstacles moved        : [1, 2]
Replan cycles          : 40
Total plan time (s)    : 1.6741
A* expansions          : 7002
LLM calls              : 0  (mode=deepseek)
```

- **J** = λD + W（总代价）
- **motion lambda times D** = λ × 总行驶距离（运动代价）
- **obstacle work W** = Σ(真实 difficulty × 实际推动距离)（操作代价）
- **Obstacles moved** = 被推开的障碍物 ID 列表
- **Replan cycles** = 重规划次数
- **Total plan time** = 全部规划耗时之和
- **A\* expansions** = A* 节点展开总数
- **LLM calls** = LLM API 调用次数（材质全部命中锚点表时为 0，即使处于 deepseek 模式）

## 文件结构
```
config.py          所有可调参数（λ、感知半径、推动半径、网格步长、GIF 参数等）
obstacle.py        StaticObstacle / MovableObstacle 数据结构
roadmap.py         一次性构建的增广路网
geometry.py        SE2 推动规划适配层（自适应旋转扫掠体、路径验证、代价计算）
push_planner.py    SE2 网格 Dial-bucket Dijkstra 推动路径搜索器
llm_difficulty.py  DeepSeek 难度估计 + 离线启发式回退
perception.py      感知、遮挡、碰撞检测、触摸/交互揭示、信念状态维护
search.py          A* f=g+h 分支限界搜索 + 三种策略
planner.py         在线"规划-执行-感知-重规划"主循环
main.py            入口、可视化（汇总图 + GIF）、指标输出
scenarios/         场景定义（9 个场景）
```

依赖：`shapely` / `numpy` / `scipy` / `matplotlib` / `pillow`（GIF 合成）/ `requests`（LLM）。

## 推动规划的性能优化
以下优化已实装，显著提升了推动路径规划的速度：
1. **跨 cycle 持久化推动方案缓存**（`search.py — _persistent_removal_cache`）：同一障碍物在同一条边上的推动方案在多次重规划之间复用，避免重复跑 SE2 Dijkstra（实测命中率 ~70%）
2. **Dijkstra 提前终止**（`push_planner.py — _search max_bucket`）：根据走廊范围自动限制搜索半径，避免探索障碍物不可能到达的远端状态
3. **KDTree 最近邻查询**（`roadmap.py`）：替代 O(N) 线性扫描
4. **STRtree 空间索引**（`planner.py — _execute_push_path`）：推动执行阶段每步碰撞检测从 O(N) 降至 O(log N)
5. **自适应旋转扫掠体**（`geometry.py — _swept_between`）：小障碍物保持 15° 粗步长不浪费算力，只对长条形障碍物加密角分辨率——既是精度手段也是性能手段
6. **扫掠体包围盒粗筛**（`geometry.py — _path_is_clear_against`）：精确相交计算前先比对 AABB

## 正确性保证
- **代价结算是准的**：W 恒用真实 difficulty × 实际推动距离结算，与估计是否准确无关。任何物理接触（碰撞或推动）都会把该障碍物的真值写入信念，后续重规划直接使用真值
- **搜索在当前信念下是最优的**：g 值来自几何计算器用真实几何算出的推动距离，h 是可采纳的下界，因此第一个被弹出的目标状态在当前信念下代价最优
- **估计误差影响的是决策质量，不是账目**：难度估计进入 g 值，估错会让机器人选到次优的推动对象或路线；但一旦接触就修正，且 J 始终按真值计。实测把全部场景的真实 difficulty 随机扰动 0.5×–2×（平均偏差 34%）后，9 个场景仍全部成功，推动对象与重规划轮数不变，只有 W 项随真值变化
- 这里的"最优"指在机器人当前知识下的最优；结合持续在线重规划——不是一次性全局最优（因为障碍物几何先验未知，符合问题定义）
