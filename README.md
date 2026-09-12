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
| `ten_doors` | 十道门，每道给出「搬 A / 搬 B / 绕行」三选一；`LLM_benchmark` 的 Gap 扫描用它量化估计误差对决策和 C 的影响 |
| `earthquake` | 震后救援：三组耦合危险物，看似无害的推车实际支撑着开裂的梁 |
| `home` | 搬家中的大型住宅：门口堵着衣柜、书架、床垫和纸箱，推动难度相差一个数量级 |
| `hospital` | 医院：可推的病床与推车、上了刹车的移动 X 光机，配合定时发生的运送事件 |
| `warehouse` | 仓库：托盘、笼车、叉车和 AGV 送货，验证大场地和动态障碍物 |

新场景放在 `CRA-NAMO/scenarios/` 中，并提供无参数的 `create()` 函数。场景模块会被自动发现，文件名就是 `--scenario` 使用的名称；以下划线开头的模块不会被当作场景。

### 可复现随机地图

`seeded_random` 使用“房间连接图 → 墙体与门洞 → 决策点挖掘 → 背景障碍物 → 动态事件 → 决策难度校准 → 分级校验”的流水线生成地图。随机性按墙体、障碍物几何、物理属性、隐藏状态、事件和校准分别派生，因此同一 seed 可以稳定重放，也不会因某个采样器增加一次随机调用而改变整张地图。

随机地图批量实验的全部选项统一在 `CRA-NAMO/config.py` 中调整：

```python
random_map_obstacle_count = 16
random_map_dynamic_obstacle_count = 1
random_map_experiment_count = 10
random_map_generate_images = True
random_map_run_strategies = ("no-llm", "shortest", "llm-cost-risk", "llm-choice")
random_map_profiles = ("depot", "home", "hospital", "earthquake")
random_map_timeout_seconds = 300
random_map_resume = True
random_map_seed_start = 0
random_map_output_dir = "img/random_experiments"
```

批量入口不再接收 `--run`、`--seeds` 或 `--out` 等实验参数，运行时只读取以上配置。默认四组分别是启发式 CRA-NAMO（`no-llm`）、最短路径基线（`shortest`）、同时使用 LLM 成本与风险估计的完整方法（`llm-cost-risk`），以及由 LLM 直接在候选方案中选择的 `llm-choice`。后两组需要 DeepSeek key，否则会退化成启发式；`llm-choice` 每个决策要多跑 k+2 次 A\*，明显更慢，可能需要调大 `random_map_timeout_seconds`。

默认拓扑会随机生成 4–5 行、6–7 列的不等尺寸房间，然后从网格邻接关系中删除部分连接，同时保留整体连通性。不同 seed 会产生不同的环路、死路、岔路和最短路径长度。决策障碍物不是绑定在固定编号墙上，而是放到最短路线中具有反事实绕行路径的门边；验证器会确认每个关键门边被移除后仍有替代路线。房间内部还会生成不占用图通道的随机斜墙。

每张地图安排 6 个门决策加最多 1 个动态决策。只堵一扇门的决策类型，其所在墙只开一个门洞，否则机器人从旁边那扇门径直走过去，这个门就不构成选择。验证器另外要求"一次性绕开全部决策"的代价不低于直达路线的 35%，避免六个决策被同一条免费走廊全部绕过。

`scenario_generation/profiles.py` 里的 profile 按主题划分，障碍物标签取自对应的手写场景：`depot` 对应 moving_depot 和 warehouse，`home` 对应 home 和 maze，`hospital` 对应 hospital，`earthquake` 对应 earthquake，`benchmark` 混合四套词表。批量实验按 seed 轮换 `random_map_profiles` 里的主题。

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

可用 profile 包括 `depot`、`home`、`hospital`、`earthquake` 和 `benchmark`。动态事件会在临时门洞封锁、搬移后响应和物体属性突变之间采样；验证器只排除几何非法、静态不可达、事件关闭全部路线、缺少反事实决策或决策可被免费绕过的地图，不按算法输赢筛选 seed。

七种门决策里有四种专门用来拉开 LLM 与离线启发式的差距，依据是两张启发式表都只做词法匹配：`risk.keyword_level` 匹配不上就返回 low，`llm_difficulty` 匹配不上就回落到 unknown（mu*rho = 40），匹配上任一个词则取最大值。

- `blind_risk`：标签在词表里读作无害，实际不能碰。例如 `crash_cart` 被读成 cart，估成 mu*rho=4.5 的免费一推，实际是抢救车。接触时通过 `contact_reveals` 交出一个词表读得懂的标签，于是盲推的一方照样被记上风险附加项。
- `blind_weight`：标签读着轻，实际很重。例如 `lead_shielding_screen` 估成 40，真实接近 900，执行器按真值计费。
- `false_alarm`：反向陷阱。`beam_offcut_carton` 只是一箱木料边角料，但 beam 是 extreme 关键词，离线一方会为了躲一个纸箱去绕路。没有这一类，"什么都不推"就能拿高分。
- `risk_or_risk`：两扇门都堵着，一边词法上危险，一边只有语义上危险。

标签目录在 `scenario_generation/materials.py`，`tests/scenario_generation/test_materials.py` 会在任何一张启发式表的改动让陷阱重新可见时失败。

校准阶段保持房间图和墙体不变，根据门边的反事实绕行长度调节关键障碍物的真实推动阻力，并根据替代路线代价调节临时封路的等待窗口。定量决策的最优与次优代价差控制在 5%–30%，且由独立随机流决定是搬移、绕行、安全搬移、等待还是重规划占优，避免所有 seed 都给出同一种答案。风险陷阱的推动代价会被压到刚好比绕行便宜，只有算上真实风险附加项之后绕行才占优，所以读不懂标签的一方确实会被引诱过去。

每个决策同时保存两套代价：`option_costs` 是世界会真正收取的真值，`belief_option_costs` 是只看标签、只查离线表能算出来的那一套。两者给出不同最优动作的次数记在校验指标 `belief_flip_decisions` 里，当前配置下平均每张图有 4.5 个。这个数字就是一张地图能测出多少语义理解。

一键生成随机地图并运行配置的全部对照策略：

```bash
python3 CRA-NAMO/benchmarks/random_maps.py
```

所有批量产物保存在 `img/random_experiments/`。地图按生成顺序分入 `experiment_0001/`、`experiment_0002/` 等目录；每个目录包含地图 JSON、每种策略的结果 JSON、PNG、GIF 和该地图的合并结果。文件名同时包含实验编号与策略，例如 `experiment_0001_llm-cost-risk.png` 和 `experiment_0001_llm-cost-risk.gif`。

每个策略完成后会立即原子写入当前实验目录，并同步更新根目录下的 `results.json`、`results.csv` 和 `progress.json`；一张地图的全部策略完成后，会立即写入 `experiment_XXXX_results.json` 和当前 `coverage.json`。`progress.json` 包含已完成地图/运行数、当前 seed 与策略、运行时间和 ETA。开启 `random_map_resume` 后，重启会跳过已有且 PNG、GIF、结果 JSON 均完整的策略；单个策略超过 `random_map_timeout_seconds` 时，其子进程会被终止并记录为 `timeout`，同时生成对应的状态 PNG 和 GIF。

单次 PNG 标出实验编号、seed、策略、成功/失败/超时状态、成本、墙钟/仿真时间、重规划次数、起终点、机器人实际轨迹、机器人搬动的障碍物、动态障碍物及其实际轨迹。全部实验结束后在终端打印策略对比表，并写入 `experiment_summary.md` 和 `experiment_summary.csv`；每跑完一张地图也会刷新一次，中途停下同样能看到当前对比。表格按策略分行，列为运行数、成功率、LLM 调用次数、J、搬移障碍物做的功 W、仿真时间 T、移动时间和等待时间。成功率与 LLM 调用次数统计全部运行，其余各列只在成功的运行上取平均。

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

## LLM 估计实验

`CRA-NAMO/LLM_benchmark/llm_accuracy.py` 单独评估两个估计器：难度（`mu*rho`）和风险等级。
它既量估计器有多准，也量这份误差换算成路线代价是多少。

```bash
cd CRA-NAMO/LLM_benchmark
python3 llm_accuracy.py accuracy          # 估计值 vs 参考值，需要 API Key
python3 llm_accuracy.py risk              # 风险等级，视觉与接触两种观测
python3 llm_accuracy.py size              # 尺寸无关性检查
python3 llm_accuracy.py order             # 锚点顺序检查：是估计还是抄表
python3 llm_accuracy.py doors             # 十门 Gap 扫描，离线，不调 API
python3 llm_accuracy.py doors-calibrate   # 逐门测出三个选项各自的真实代价
python3 llm_accuracy.py report            # 汇总成 report.md 和全部图表
python3 llm_accuracy.py all               # 除 doors-calibrate 外的全部阶段
```

结果写在 `LLM_benchmark/llm_test_out/`。

### 十门 Gap 扫描

`doors` 阶段不调 API，而是把「估计值 / 真实值 = F」直接写进 belief，再看规划器的
选择和真实代价怎么变。比例是构造出来的而不是采样出来的，所以横轴上的 `F` 就是
真实的 Gap，「多大的误差换来多少 C」可以直接读。

| 轴 | 含义 |
| --- | --- |
| 代价梯度 `F` | 1.15x 到 10x，belief 中的难度是真实值的 `F` 倍或 `1/F` 倍 |
| 风险梯度 `K` | 0 到 4 级，belief 中的等级相对真实等级偏移 `K` 级后截断 |
| `under` | 每个估计都更便宜、更安全（乐观），确定性 |
| `over` | 每个估计都更贵、更危险（悲观），确定性 |
| `mixed` | 每个障碍物一个固定随机方向，方向在整条梯度上保持不变，因此各梯级是配对样本 |

**C 按真实风险等级重新计价。** 执行器按 belief 的等级收风险附加项，低估风险的运行
本来会因此白拿一笔折扣、看上去比 `exact` 还便宜。报告里的每一个 C 都是 `J` 加上被
搬走的障碍物真正值那么多的附加项，所以错误决策显示为代价而不是节省。

常用开关：

```bash
python3 llm_accuracy.py doors --doors-seeds 5 --doors-workers 6
python3 llm_accuracy.py doors --doors-quick --no-doors-shots   # 冒烟测试
```

`--doors-workers` 只影响耗时：每次运行独立且带种子，并行不改变任何一个结果。

输出：

| 文件 | 内容 |
| --- | --- |
| `doors.json` | 每次运行的完整记录，含 `C_true`、`C_believed`、逐门选择 |
| `doors_gap_vs_cost.csv` | Gap 比例 → C 差别比例的表格，可直接引用 |
| `doors_gap.png` | 四格图：C 的变化、决策改动数、风险梯度、多出来的代价花在哪 |
| `doors_gates.png` | 逐门热力图：哪一道门在多大的 Gap 上开始改主意 |
| `ten_doors_*.png` | `exact` 和几个极端 Gap 点的路线截图 |

### 先校准，再扫描

`doors-calibrate` 把一道门的绕行开口砌死、再把其中一扇门的 belief 抬到机器人推不动，
于是规划器只剩一个选项，三次运行（共 30 次）就测出「搬 A」「搬 B」「绕行」各自的真实
代价。一次搬移的计价里只有「难度 x 移动距离」随 belief 变化，所以每道门的翻转比例可以
解析求出，正好对照扫描里实测的首次翻转点；它还会给出每扇门正好与绕行持平所需的难度。

**当前的 `ten_doors` 需要按这一列重新配平。** 用 exact belief 跑一次的结果是十道门里
九道绕行、只搬走一扇 400 N 的门，W 只占 C 的 0.9%。决策这样一边倒时没有第二个选项可换，
Gap 扫描量到的会接近平坦。`doors` 阶段会在 exact 运行过于一边倒时直接告警。

改过 `scenarios/ten_doors.py` 里的难度之后才需要重跑校准阶段。

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
└── LLM_benchmark/      # LLM 估计实验与十门 Gap 扫描
```

## 当前阶段

当前版本已经具备在线感知、路径与搬移联合决策、风险评估和基础动态地图事件。后续工作主要是加入移动障碍物轨迹预测、不确定性建模、时空碰撞检查，以及更完整的动态场景评估指标。
