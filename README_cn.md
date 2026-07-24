# 基于 LLM 估计操作难度的代价感知 NAMO

本项目是提案 *Cost-Aware Navigation Among Movable Obstacles with LLM-Estimated Manipulation Difficulty*（基于 LLM 估计操作难度的可移动障碍物间代价感知导航）的参考实现。机器人将行走距离与操作代价统一为单一目标函数，使用 **DeepSeek** 大语言模型（以及离线启发式备用方案）估计每个障碍物的移动难度，并将"绕行 vs. 推开"问题转化为可量化的权衡——在线运行，随感知发现新障碍物时持续重规划。

```
J = lambda_d * (行走距离) + lambda_w * (移除路径上障碍物所做的功)
```

## 快速开始

```bash
pip install -r requirements.txt

python main.py                      # two_doors 场景，使用离线启发式难度估计
python main.py --lambda_w 8         # 提高推障代价 -> 机器人倾向于绕行
python main.py --no-llm-order       # 消融实验：禁用 LLM 搜索排序

# 使用 DeepSeek 估计难度（仅影响搜索顺序）：
DEEPSEEK_API_KEY=sk-xxxx python main.py
```

结果摘要图像保存至 `img/summary_<scenario>.png`。

## 代码与提案的对应关系

| 提案章节 | 模块 | 说明 |
|---|---|---|
| 代价函数 `J` | `config.py`（`lambda_d`, `lambda_w`）| 权重按平台设置 |
| 增广路网 | `roadmap.py` | 在静态墙体上**一次性**构建；可移动障碍物仅标记哪些边当前需要"付费解锁"，不改变图结构 |
| 两个几何计算器 | `geometry.py` | `push_plan` = 基于采样的重定位（可行性 + 实际推动距离，防止障碍物反向阻路）；`walking_distance` = 通过静态自由空间的实际行走距离 |
| LLM 难度估计 | `llm_difficulty.py` | 通过与基础仓库相同的 API 接口调用 DeepSeek-V3，或使用离线的"材质密度 × 面积"启发式。**仅用于排列搜索顺序** |
| 感知与在线重规划 | `perception.py` | 感知半径 `R_perc`，遮挡处理（移动障碍物后可发现被遮挡的物体），**增量式**边更新，对未探索空间保持乐观假设 |
| 最佳优先 `f=g+h` + 分支定界 | `search.py` | `g` = 累计代价，可接受启发值 `h = lambda_d * 欧氏距离`，当前最优解剪枝；障碍物移除限制在半径 `R_push` 内 |
| 规划-执行-感知-重规划循环 | `planner.py` | 每轮执行一条边，然后重新感知并重规划 |
| 场景 / 评估框架 | `scenarios.py`, `main.py` | `two_doors` 演示 + 指标（J、规划时间、成功率）及可视化 |

## 正确性保证（为什么 LLM 的错误估计不会造成损害）

LLM 难度估计仅影响搜索的**扩展顺序**（`search.Planner._llm_bias`）。进入 `g` 的代价来自使用真实几何数据的几何计算器，`h` 是可接受的下界，因此第一个弹出的目标状态在**当前信念下**是代价最优的。因此，错误的 LLM 估计只会影响找到好的当前最优解的速度（分支定界），而不会影响返回的规划方案或其代价。

"最优"是指在机器人当前认知下的最优；结合感知过程中的持续在线重规划——这并非一次性的全局最优（根据问题定义，障碍物的几何形状是先验未知的）。

## `two_doors` 演示

两个房间由两个门洞相连。近侧门洞（与起点和终点对齐）被容易移动的障碍物 `A` 堵住，其后方还隐藏着中等难度的障碍物 `B`；远侧门洞畅通无阻，但需要绕远路。`A` 和 `B` 只有靠近时才能被感知，且 `B` 在 `A` 被移走之前始终被遮挡。

* `python main.py`（λ_w = 1）：推障代价低 → 机器人推开 `A` 和 `B` 直接穿过（`J ≈ 26.6`，做功 = 4.5）。
* `python main.py --lambda_w 8`：推障代价高 → 发现代价较高的 `B` 后，机器人选择从上方绕行（`J ≈ 45.8`）。

## 文件说明

```
config.py          所有可调参数（lambda_d、lambda_w、R_perc、R_push 等）
obstacle.py        StaticObstacle / MovableObstacle（材质 + 真实难度）
geometry.py        push_plan() 和 walking_distance() — 两个几何计算器
roadmap.py         构建一次的增广路网，含付费解锁边
llm_difficulty.py  DeepSeek 难度估计 + 离线启发式备用方案
perception.py      感知圆、遮挡处理、增量式信念更新
search.py          带分支定界的最佳优先 f=g+h 搜索
planner.py         在线规划-执行-感知-重规划循环
scenarios.py       演示环境
main.py            运行 + 可视化 + 打印指标
```

## 备注 / 简化说明（本项目为原型实现）

* 障碍物重定位为纯平移（推动过程中朝向固定）；扫掠区域为精确凸包，因此墙壁碰撞检测是精确的。
* 防止反向阻路的实现方式：强制清除障碍物当前阻塞的通道，并软性偏好不重新阻塞其他自由通道的放置位置。由于稠密路网覆盖了自由空间，严格的"不阻塞任何通道"规则会禁止所有放置位置；任何残余的新阻塞均由重感知捕获并在下次重规划时处理。
* 基线方法（纯绕行、通过 `--no-llm-order` 进行的 LLM 消融实验）及多场景评估为后续预留接口；消融标志已可正常使用。
