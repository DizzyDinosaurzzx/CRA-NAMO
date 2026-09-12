# CRA-NAMO

**Cost- and Risk-Aware Online Navigation in Dynamic Maps with Movable Obstacles**

CRA-NAMO 是一个二维在线导航仿真项目。机器人需要在信息不完整的动态地图中前往目标点，并在遇到可移动障碍物时选择绕行或将其移开。

项目重点不是单纯寻找最短路径，而是同时考虑：

- 机器人行驶距离；
- 搬移障碍物所需的功；
- 运动与在线规划时间；
- 移动危险物体带来的风险；
- 障碍物自主运动和环境变化。

## 算法简介

系统采用“感知—规划—执行—再规划”的在线闭环：

```text
局部感知 → 更新机器人认知 → 规划路径和搬移动作 → 执行整条计划
    ↑                                             ↓
    └────── belief 更新后重新规划 ──────┘
```

只有 belief 真的发生更新才会重新规划：看到新的障碍物、已知障碍物换了位置或形状、
接触测出真实难度、或者机器人自己搬动了某个东西。belief 没变时同样的认知只会得到
同样的计划，因此机器人继续执行手上这条计划，而不是每走一条边就重算一次。

### 1. 局部感知

机器人只知道感知范围内且没有被遮挡的障碍物。代码分别保存：

- `world`：仿真中的真实地图；
- `belief`：机器人当前知道的地图。

规划器只能读取 `belief`。隐藏障碍物、标签变化和难度变化需要机器人通过观察、接触或碰撞才能发现。

### 2. 路径与搬移联合规划

静态自由空间首先被离散为 roadmap。搜索时，每条边可能有两种处理方式：

- 边没有被挡住：机器人直接通过；
- 边被可移动障碍物挡住：计算将障碍物移开后再通过的代价。

搜索算法比较绕行和搬移的总代价，从中选择当前认知下更合适的方案。

### 3. 障碍物搬移

障碍物的运动在 SE(2) 空间中规划，包括：

- 平移和旋转；
- 障碍物运动过程中的扫掠区域；
- 与墙体及其他障碍物的碰撞；
- 机器人到达接触位置并在搬移过程中保持接触；
- 搬移结束后重新连接到 roadmap。

落点不只看推动障碍物本身要花多少，还要看放下之后机器人剩下的路要走多久：对每个候选
姿态，先算出障碍物停在那里以后机器人从这条边继续走到终点的代价，再和搬移代价加在一起
比较，取总和最小的那个。会把去路堵死的落点直接作废——除非没有任何落点能留出通路，那时
只能先把它搬开再重新规划。`--no-lookahead` 可以关掉这一步，只按搬移本身的代价选落点。

### 4. 成本与风险

当前目标函数为：

```text
C = (1 - w)J + w · time_value · T + R
J = λD + W
```

- `D`：机器人总行驶距离；
- `λD`：机器人行驶成本；
- `W`：搬移障碍物所做的功；
- `T`：机器人运动、转向和在线规划所用时间；
- `w`：能量与时间的权衡，范围为 `[0, 1]`；
- `R`：移动危险物体产生的风险附加成本。

机器人第一次看到障碍物时会根据标签评估风险，实际接触后再根据新信息重新评估。障碍物搬移难度可由 DeepSeek 估计，也可以使用本地启发式规则。

### 5. 动态地图

动态事件和机器人使用同一条模拟时间轴。当前支持：

- 障碍物沿 SE(2) 路径自主平移和旋转；
- 在指定时间触发事件；
- 机器人到达指定区域后触发事件；
- 机器人搬动某个障碍物后触发事件；
- 动态修改障碍物的标签、尺寸和搬移难度；
- 移动物体暂时挡路时等待或重新规划。

动态事件只修改真实世界，机器人不能提前知道。当前规划仍基于地图的当前几何状态，尚未实现未来轨迹预测和不确定性建模。

## 安装

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

主要依赖为 Shapely、NumPy、SciPy、Matplotlib、Pillow 和 Requests。

## 运行

所有命令从仓库根目录执行。

```bash
# 查看参数和可用场景
python CRA-NAMO/main.py --help

# 基础障碍物旋转与搬移
python CRA-NAMO/main.py --scenario corridor

# 成本、风险、隐藏障碍物和接触后重新评估
python CRA-NAMO/main.py --scenario strategy_demo

# 动态障碍物和事件触发
python CRA-NAMO/main.py --scenario moving_depot

# 大型迷宫场景
python CRA-NAMO/main.py --scenario maze
```

常用参数：

```bash
# 调整能量和时间的权衡
python CRA-NAMO/main.py --scenario corridor --time-importance 0.5

# 关闭 LLM 搜索排序
python CRA-NAMO/main.py --scenario corridor --no-llm-order

# 落点只按搬移代价选，不预估之后还要走的路
python CRA-NAMO/main.py --scenario corridor --no-lookahead

# 保存按模拟时间采样的 GIF
python CRA-NAMO/main.py --scenario corridor --frames
```

图片和 GIF 默认保存在 `img/`。其他算法、机器人和动态地图参数集中在 `CRA-NAMO/config.py`。

## 场景

| 场景 | 主要用途 |
| --- | --- |
| `corridor` | 展示大型障碍物的平移、旋转和接触搬移 |
| `strategy_demo` | 展示绕行/搬移权衡、风险评估和隐藏信息 |
| `moving_depot` | 展示自主移动障碍物和事件触发 |
| `maze` | 自助仓储迷宫：外观相同的纸箱内容物差别极大，只有接触后测到的力才能区分 |
| `ten_doors` | 十道门，每道给出「搬 A / 搬 B / 绕行」三选一；用于量化 LLM 估计误差对决策的影响 |
| `earthquake` | 震后救援：三组耦合危险物，看似无害的推车实际支撑着开裂的梁 |
| `home` | 搬家中的大型住宅：门口堵着衣柜、书架、床垫和纸箱，推动难度相差一个数量级 |
| `hospital` | 医院：可推的病床与推车、上了刹车的移动 X 光机，配合定时发生的运送事件 |
| `warehouse` | 仓库：托盘、笼车、叉车和 AGV 送货，验证大场地和动态障碍物 |

新场景放在 `CRA-NAMO/scenarios/` 中，并提供无参数的 `create()` 函数。场景模块会被自动发现，文件名就是 `--scenario` 使用的名称；以下划线开头的模块不会被当作场景。

### 可复现随机地图

`seeded_random` 使用“房间连接图 → 墙体与门洞 → 决策点挖掘 → 背景障碍物 → 动态事件 → 决策难度校准 → 分级校验”的流水线生成地图。随机性按墙体、障碍物几何、物理属性、隐藏状态、事件和校准分别派生，因此同一 seed 可以稳定重放，也不会因某个采样器增加一次随机调用而改变整张地图。

随机地图批量实验的全部选项统一在 `CRA-NAMO/config.py` 中调整：

```python
random_map_obstacle_count = 10
random_map_dynamic_obstacle_count = 5
random_map_experiment_count = 10
random_map_generate_images = True
random_map_run_strategies = ("no-llm", "shortest", "llm-cost-risk")
random_map_timeout_seconds = 300
random_map_resume = True
random_map_seed_start = 0
random_map_output_dir = "img/random_experiments"
```

批量入口不再接收 `--run`、`--seeds` 或 `--out` 等实验参数，运行时只读取以上配置。默认三组分别是启发式 CRA-NAMO（`no-llm`）、最短路径基线（`shortest`）和同时使用 LLM 成本与风险估计的完整方法（`llm-cost-risk`）。

默认拓扑会随机生成 3–4 行、4–5 列的不等尺寸房间，然后从网格邻接关系中删除部分连接，同时保留整体连通性。不同 seed 会产生不同的环路、死路、岔路和最短路径长度。决策障碍物不是绑定在固定编号墙上，而是放到最短路线中具有反事实绕行路径的门边；验证器会确认每个关键门边被移除后仍有替代路线。房间内部还会生成不占用图通道的随机斜墙。

```bash
# 生成并运行一张均衡随机地图
python CRA-NAMO/main.py --scenario seeded_random --seed 42 \
  --map-profile balanced --strategy no-llm

# 保存完整地图描述
python CRA-NAMO/main.py --scenario seeded_random --seed 42 \
  --map-profile showcase --save-map-manifest --no-frames

# 从 manifest 精确重放
python CRA-NAMO/main.py --scenario seeded_random \
  --map-manifest img/maps/experiment_0042_showcase_map_<id>.json \
  --no-frames
```

可用 profile 包括 `balanced`、`dynamic`、`risk`、`manipulation`、`adversarial`、`showcase` 和 `benchmark`。动态事件会在临时门洞封锁、搬移后响应和物体属性突变之间采样；`benchmark` 只排除几何非法、静态不可达、事件关闭全部路线或缺少反事实决策的地图，不按算法输赢筛选 seed。

校准阶段保持房间图和墙体不变，根据门边的反事实绕行长度调节关键障碍物的真实推动阻力，并根据替代路线代价调节临时封路的等待窗口。定量决策的最优与次优代价差控制在 5%–30%，且由独立随机流决定是搬移、绕行、安全搬移、等待还是重规划占优，避免所有 seed 都给出同一种答案。隐藏难度决策同时保存接触前 belief 与接触后真值标签，用来检查算法能否因新信息改变选择。

一键生成随机地图并运行配置的全部对照策略：

```bash
python3 CRA-NAMO/benchmarks/random_maps.py
```

所有批量产物保存在 `img/random_experiments/`。地图按生成顺序分入 `experiment_0001/`、`experiment_0002/` 等目录；每个目录包含地图 JSON、三种策略的结果 JSON、三张 PNG、三张 GIF 和该地图的合并结果。文件名同时包含实验编号与策略，例如 `experiment_0001_llm-cost-risk.png` 和 `experiment_0001_llm-cost-risk.gif`。

每个策略完成后会立即原子写入当前实验目录，并同步更新根目录下的 `results.json`、`results.csv` 和 `progress.json`；一张地图的全部策略完成后，会立即写入 `experiment_XXXX_results.json` 和当前 `coverage.json`。`progress.json` 包含已完成地图/运行数、当前 seed 与策略、运行时间和 ETA。开启 `random_map_resume` 后，重启会跳过已有且 PNG、GIF、结果 JSON 均完整的策略；单个策略超过 `random_map_timeout_seconds` 时，其子进程会被终止并记录为 `timeout`，同时生成对应的状态 PNG 和 GIF。

单次 PNG 标出实验编号、seed、策略、成功/失败/超时状态、成本、墙钟/仿真时间、重规划次数、起终点、机器人实际轨迹、机器人搬动的障碍物、动态障碍物及其实际轨迹。全部实验结束后自动生成 `experiment_summary.png`，包含成功率、平均成本、平均运行时间、平均重规划次数、障碍物数量与成功率、动态障碍物数量与耗时六项对比。

单地图入口仍按 seed 和内容指纹命名；批量入口则严格按本次地图生成顺序编号，并为每种策略保存静态结果图和完整动画。

每张 manifest 保存墙体多边形、障碍物真实物理属性、动态事件、决策点、校准摘要、配置、校验指标和内容指纹；每个决策点的 `metadata.oracle` 保存候选代价、最优动作、相对差距，以及可用时的接触前 belief。随机生成器实现位于 `CRA-NAMO/scenario_generation/`。

### 障碍物数据约定

`scenarios/_realism.py` 提供两个工具：

- `push_force(mass_kg, mu)`：真实搬移阻力按 `mu * m * g` 计算。场景写的是物体的**真实质量**和**地面摩擦系数**，不是直接写一个牛顿数——这样每个障碍物的数据都可以按“这台冰箱真有 130 kg 吗”来核对。
- `check_layout(...)`：加载场景时检查障碍物是否与墙体或彼此重叠、是否超出边界、起点终点是否被占用，以及是否窄到机器人一推就会把它推倒。

难度估计器只能看到标签和包围盒，需要自己从体积和堆密度反推阻力。真实值与估计值之间的差距正是本项目要测量的估计误差，所以场景不应该用启发式公式反算 `difficulty`。

## LLM 配置

LLM 不是运行几何规划所必需的。没有 API Key 时，难度和风险评估会自动使用本地启发式规则。

如需使用 DeepSeek：

```bash
export DEEPSEEK_API_KEY="your-key"
```

请不要将真实密钥写入代码或提交到 Git。为了得到可复现的对比结果，可以关闭 LLM：

```python
cfg.deepseek_api_key = ""
cfg.use_llm_ordering = False
```

模拟时钟只由行驶、转向和等待推进；规划耗时会被测量并报告（`plan_time`），
但不推进世界，所以动态场景不会因为机器负载不同而给出不同结果。

## 代码结构

```text
CRA-NAMO/
├── main.py             # 命令行入口
├── executor.py         # 在线执行与重新规划
├── search.py           # 路径和搬移动作搜索
├── perception.py       # 局部感知与 belief 更新
├── roadmap.py          # 机器人 roadmap
├── se2_planner.py      # 障碍物 SE(2) 路径规划
├── contact.py          # 机器人—障碍物接触规划
├── dynamics.py         # 动态事件和障碍物自主运动
├── cost.py             # 成本函数
├── risk.py             # 风险评估
├── scenarios/          # 仿真场景
└── LLM_benchmark/      # LLM 估计实验
```

## 当前阶段

当前版本已经具备在线感知、路径与搬移联合决策、风险评估和基础动态地图事件。后续工作主要是加入移动障碍物轨迹预测、不确定性建模、时空碰撞检查，以及更完整的动态场景评估指标。
