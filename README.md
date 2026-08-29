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
局部感知 → 更新机器人认知 → 规划路径和搬移动作 → 执行一小段
    ↑                                             ↓
    └────── 发现新障碍物或地图发生变化后重新规划 ──────┘
```

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
