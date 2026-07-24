# Cost-Aware NAMO with LLM-Estimated Manipulation Difficulty

A reference implementation of the proposal *Cost-Aware Navigation Among Movable
Obstacles with LLM-Estimated Manipulation Difficulty*. The robot unifies travel
distance and manipulation work into one cost, uses a **DeepSeek** LLM (with an offline
heuristic fallback) to estimate how hard each obstacle is to move, and decides
"detour vs. push" as a quantitative trade-off — online, replanning as its perception
reveals new obstacles.

```
J = lambda_d * (distance walked) + lambda_w * (work to remove obstacles on the path)
```

## Quick start

```bash
pip install -r requirements.txt

python main.py                      # two_doors scenario, offline heuristic difficulty
python main.py --lambda_w 8         # make pushing expensive -> robot prefers to detour
python main.py --no-llm-order       # ablation: disable LLM search ordering

# Use DeepSeek to estimate difficulty (drives search ORDER only):
DEEPSEEK_API_KEY=sk-xxxx python main.py
```

A summary figure is written to `img/summary_<scenario>.png`.

## How the code maps to the proposal

| Proposal section | Module | Notes |
|---|---|---|
| Cost function `J` | `config.py` (`lambda_d`, `lambda_w`) | weights set per platform |
| Augmented roadmap (增广路网) | `roadmap.py` | built **once** on the static walls; movable obstacles only mark which edges are currently "pay-to-unlock", never change the graph |
| Two geometry calculators (两个几何计算器) | `geometry.py` | `push_plan` = sampling-based relocation (feasibility + real push distance, non-backfiring drop); `walking_distance` = real travel distance through static free space |
| LLM difficulty estimate | `llm_difficulty.py` | DeepSeek-V3 via the same API pattern as the base repo, or an offline `material-density x area` heuristic. Used **only to order the search** |
| Perception + online replanning (感知与在线重规划) | `perception.py` | perception circle `R_perc`, occlusion so moving an obstacle reveals what was behind it, **incremental** edge updates, optimism about unexplored space |
| Best-first `f=g+h` + branch & bound (搜索过程) | `search.py` | `g` = accumulated cost, admissible `h = lambda_d * Euclid-to-goal`, incumbent pruning; removal restricted to radius `R_push` |
| plan–execute–perceive–replan loop | `planner.py` | one edge executed per cycle, then re-perceive and replan |
| Scenario / evaluation harness | `scenarios.py`, `main.py` | `two_doors` demo + metrics (J, plan time, success) and visualisation |

## Correctness guarantee (why a wrong LLM guess can't hurt)

The LLM difficulty estimate feeds only the **expansion order** of the search
(`search.Planner._llm_bias`). The costs that go into `g` come from the geometry
calculators using ground-truth geometry, and `h` is an admissible lower bound, so the
first goal state popped is cost-optimal **under the current belief**. A wrong LLM
estimate therefore only changes how fast a good incumbent is found (branch-and-bound),
never the returned plan or its cost.

"Optimal" means optimal under the robot's current knowledge; combined with continuous
online replanning as perception reveals obstacles — not a one-shot global optimum
(the geometry of obstacles is unknown a-priori, per the Problem Formulation).

## The `two_doors` demo

Two rooms joined by two doorways. The near doorway (aligned with start and goal) is
blocked by an easy obstacle `A` and, hidden behind it, a medium obstacle `B`; the far
doorway is open but forces a long detour. `A` and `B` are only seen up close, and `B`
is occluded by `A` until `A` is moved.

* `python main.py` (λ_w = 1): pushing is cheap → the robot pushes `A` and `B` aside and
  goes straight through (`J ≈ 26.6`, work = 4.5).
* `python main.py --lambda_w 8`: pushing is expensive → after uncovering the costly `B`
  the robot detours over the top (`J ≈ 45.8`).

## Files

```
config.py          all tunable parameters (lambda_d, lambda_w, R_perc, R_push, ...)
obstacle.py        StaticObstacle / MovableObstacle (material + ground-truth difficulty)
geometry.py        push_plan() and walking_distance() — the two geometry calculators
roadmap.py         build-once augmented roadmap with pay-to-unlock edges
llm_difficulty.py  DeepSeek difficulty estimate + offline heuristic fallback
perception.py      perception circle, occlusion, incremental belief update
search.py          best-first f=g+h search with branch-and-bound
planner.py         the online plan-execute-perceive-replan loop
scenarios.py       demo environments
main.py            run + visualise + print metrics
```

## Notes / simplifications (this is a "本体" prototype)

* Obstacle relocation is a pure translation (orientation fixed during a push); the swept
  region is then the exact convex hull, so wall-collision checks are exact.
* Non-backfiring is enforced as: hard-clear the corridors the obstacle currently blocks,
  and softly prefer drops that don't re-block other free corridors. A dense roadmap
  tiles the free space, so a hard "block nothing" rule would forbid every drop; any
  residual new blockage is caught by re-perception and handled on the next replan.
* Baselines (pure-detour, LLM ablation via `--no-llm-order`) and multi-scenario
  evaluation are stubbed for later; the ablation flag already works.
