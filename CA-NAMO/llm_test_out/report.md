# LLM mu*rho accuracy and its cost in navigation
Model `deepseek-v4-flash`, 3 repeats per item, 61 items, 1357.8s of API time.

## Headline

- The estimator is off by a typical factor of **1.4x**; only **84%** of objects land within 2x of the reference, and it over-estimates (1.3x median bias).
- Cause: **2% of replies are exactly 1440**, the largest anchor in the prompt. It is copying a table row, not estimating.
- Navigation still mostly survives it: over 40 runs at this error level the mean penalty is **+1.7%** of J and the worst is **+51.9%**. The planner's push-or-detour decision has wide margins, and touch sensing corrects the estimate on contact — so a bad mu*rho costs a wrong first choice, not a wrong final path.
- Best request setting tested is `v4-flash, thinking on, 32k tok` at **1.3x** typical error (88% within 2x, 61/61 answered). Note `_deepseek` hardcodes `max_tokens=32` and `thinking=disabled`, so any setting that needs a reasoning budget is unreachable without a code change.

## 1. Estimator accuracy
`median_abs_factor` is the typical multiplicative miss: 1.0 is exact, 2.0 means the usual answer is off by a factor of two in either direction. `median_ratio` separates bias from spread — above 1.0 the model systematically over-estimates.

| group | n | LLM typ. factor | LLM bias | LLM <=2x | LLM <=3x | LLM Spearman | heuristic typ. factor | heuristic <=2x |
|---|---|---|---|---|---|---|---|---|
| paraphrase | 22 | 1.447 | 1.225 | 86% | 96% | 0.918 | 1.441 | 64% |
| novel | 25 | 1.364 | 1.344 | 84% | 96% | 0.968 | 2.39 | 40% |
| state | 8 | 1.504 | 1.328 | 62% | 100% | 0.905 | 3.904 | 12% |
| brand | 6 | 1.305 | 0.923 | 100% | 100% | 0.812 | 2.497 | 0% |
| ALL | 61 | 1.371 | 1.316 | 84% | 97% | 0.968 | 2.381 | 41% |

**Anchor snapping.** 28% of replies are a verbatim row of the anchor table rather than an interpolation, and 2% are exactly 1440 — the largest anchor (`concrete_block`) and the last line of the table in the prompt. Across 61 distinct objects the model produced only 51 distinct numbers. This is the whole story behind the error above: the estimator is not estimating, it is picking a row, and it disproportionately picks the heaviest one.

**Repeatability** at temperature 0: 16% of items returned an identical value on every repeat; median spread across repeats 1.25x, worst 10.00x.

### Worst 10 items
| label | group | reference | LLM | factor | why the reference says so |
|---|---|---|---|---|---|
| waste receptacle | paraphrase | 16.8 | 4.5 | 0.27x | restatement of anchor 'trash_bin' (mu*rho=16.8) |
| large potted ficus in ceramic planter | novel | 34.7 | 120.0 | 3.46x | 45 kg / 0.648 m^3, mu=0.5; planter + wet soil ~40 kg; unglazed ceramic on concrete |
| commercial chest freezer | novel | 43.2 | 16.0 | 0.37x | 90 kg / 0.833 m^3, mu=0.4; empty; sheet-steel cabinet on plastic feet |
| wheelie bin filled with waste | state | 8.2 | 3.3 | 0.40x | 75 kg / 0.4592 m^3, mu=0.05; 240 L bin 15 kg + 60 kg refuse; two wheels |
| stocked warehouse racking bay | paraphrase | 75.2 | 180.0 | 2.40x | restatement of anchor 'shelf' (mu*rho=75.15) |
| bookshelf packed full of books | state | 75.4 | 180.0 | 2.39x | 95 kg / 0.567 m^3, mu=0.45; frame 25 kg + ~70 kg of books |
| upholstered armchair | paraphrase | 14.0 | 32.5 | 2.33x | restatement of anchor 'chair' (mu*rho=13.95) |
| rolled up broadloom carpet four metres wide | novel | 48.0 | 100.0 | 2.08x | 80 kg / 1 m^3, mu=0.6; carpet backing on concrete grips hard |
| wooden church pew | novel | 21.6 | 45.0 | 2.08x | 60 kg / 1.25 m^3, mu=0.45; long but light; bbox is mostly the empty seat volume |
| cardboard box of packing peanuts | state | 3.1 | 6.3 | 2.06x | 0.7 kg / 0.08 m^3, mu=0.35; loose fill ~5 kg/m^3 plus the carton itself |

## 2. Size independence
mu*rho must not depend on the object's size — the caller multiplies by volume afterwards, so any size response is counted twice.

1/10 items returned the same number at 0.5x, 1x and 2x linear scale (8x volume range).

| label | 0.5x | 1x | 2x | spread |
|---|---|---|---|---|
| empty steel drum | 68.0 | 34.0 | 40.0 | 2.00x |
| IKEA BILLY bookcase, empty | 56.0 | 30.8 | 30.0 | 1.87x |
| empty 240 litre wheelie bin | 1.5 | 1.0 | 0.9 | 1.67x |
| expanded polystyrene packing box | 5.25 | 5.25 | 3.5 | 1.50x |
| wooden shipping crate | 60.0 | 45.0 | 40.5 | 1.48x |
| steel storage rack with stock | 250.0 | 225.0 | 175.0 | 1.43x |
| unloaded push trolley | 2.0 | 1.5 | 1.5 | 1.33x |
| granite countertop slab | 1350.0 | 1620.0 | 1350.0 | 1.20x |
| cardboard box packed with hardcover books | 200.0 | 210.0 | 210.0 | 1.05x |
| solid concrete cube | 1440.0 | 1440.0 | 1440.0 | 1.00x |

## 3. Does the error reach the planner?
The search only ever asks one question per obstacle: is `difficulty x push_distance` cheaper than `lambda x detour`? With lambda=350 N and a 2 m push, an error only matters if it moves an obstacle across that line.

| detour available | obstacles | decisions flipped by the LLM error |
|---|---|---|
| 2 m | 61 | 8% |
| 5 m | 61 | 10% |
| 10 m | 61 | 3% |
| 20 m | 61 | 3% |
| 50 m | 61 | 2% |

## 4. Navigation cost
Same maps, same physics, same true difficulties — only the estimator's belief differs. `oracle` knows the reference mu*rho exactly and is the floor; the penalty column is the extra J each arm pays over it.

| map | arm | goal reached | J | lambda*D | W | pushes | replans | penalty vs oracle |
|---|---|---|---|---|---|---|---|---|
| two_doors | oracle | True | 7154 | 7000 | 154 | 2 | 37 | - |
| two_doors | llm | True | 7154 | 7000 | 154 | 2 | 37 | +0.0% |
| two_doors | heuristic | True | 10871 | 10871 | 0 | 0 | 69 | +51.9% |
| hidden_obstacle | oracle | True | 40335 | 19733 | 20602 | 2 | 117 | - |
| hidden_obstacle | llm | True | 40335 | 19733 | 20602 | 2 | 117 | +0.0% |
| hidden_obstacle | heuristic | True | 40335 | 19733 | 20602 | 2 | 117 | +0.0% |
| maze_mixed | oracle | True | 21137 | 20717 | 420 | 2 | 122 | - |
| maze_mixed | llm | True | 21137 | 20717 | 420 | 2 | 122 | +0.0% |
| maze_mixed | heuristic | True | 21101 | 20792 | 309 | 2 | 122 | -0.2% |
| corridor | oracle | True | 13588 | 12327 | 1261 | 1 | 65 | - |
| corridor | llm | True | 13588 | 12327 | 1261 | 1 | 65 | +0.0% |
| corridor | heuristic | True | 13588 | 12327 | 1261 | 1 | 65 | +0.0% |

### Sensitivity sweep
Every estimate multiplied by a fixed factor. A flat row means the map's decisions are not close to the break-even line and accuracy is free; a step means it is.

| map | x0.05 | x0.125 | x0.25 | x0.5 | x1 | x2 | x4 | x8 | x20 |
|---|---|---|---|---|---|---|---|---|---|
| two_doors | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +7.8% | +7.8% |
| hidden_obstacle | +35.6% | +8.3% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +1.5% |
| maze_mixed | +58.9% | +0.0% | +0.0% | +0.0% | +0.0% | -0.2% | -0.2% | -0.2% | -0.2% |
| corridor | +0.5% | +0.5% | +0.5% | +0.5% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |

### Realistic-error replicates
The single `llm` row above is one draw, and one draw on four maps is not a measurement. These re-run each map with an independent per-obstacle lognormal error at the spread measured over all 61 items (sigma=0.195 in log10, a typical 1.57x miss), 5 seeds per map — what an estimator this noisy costs on average, rather than on the labels it happened to get right.

| map | mean penalty | worst penalty | failed runs |
|---|---|---|---|
| two_doors | +0.0% | +0.0% | 0 |
| hidden_obstacle | +0.0% | +0.0% | 0 |
| maze_mixed | +0.0% | +0.0% | 0 |
| corridor | +0.0% | +0.0% | 0 |

### Is reasoning mode worth it, on the robot?
The shipped estimator now reasons; the previous one did not (1.57x spread vs 5.71x). Accuracy is only worth buying if it changes what the robot does. `llm` and `llm_nothink` are the two estimators' actual predictions; the noise columns are 5 draws from each one's measured error distribution, which is the fairer comparison — a single point estimate can be lucky.

| map | llm (reasoning, current) | llm (no reasoning, previous) | current noise mean/worst | previous noise mean/worst |
|---|---|---|---|---|
| two_doors | +0.0% | +0.0% | +0.0% / +0.0% | +13.5% / +51.9% |
| hidden_obstacle | +0.0% | +0.0% | +0.0% / +0.0% | +0.0% / +0.0% |
| maze_mixed | +0.0% | +0.0% | +0.0% / +0.0% | +0.0% / +0.0% |
| corridor | +0.0% | +0.0% | +0.0% / +0.0% | +0.0% / +0.0% |

## 4b. When does the error start to cost anything?
`lambda_distance` is the exchange rate between detouring and pushing, so it decides how many obstacles sit near the break-even where a wrong mu*rho changes the answer. Each cell is the mean penalty over 5 noisy-estimate runs against the oracle at the same lambda. The shipped value is 350.

| map | lambda=50 | lambda=100 | lambda=200 | lambda=350 | lambda=700 | lambda=1400 |
|---|---|---|---|---|---|---|
| two_doors | +18.6% | +12.6% | +12.8% | +13.5% | +2.5% | +0.0% |
| maze_mixed | +0.5% | +0.8% | -0.2% | +0.0% | +7.5% | +2.7% |

## 4c. Proof that it is copying, not estimating
The anchor table's row order is rewritten; every other byte of the prompt is unchanged, and only the 39 off-table objects are asked (a paraphrase item's correct answer *is* an anchor, so it cannot distinguish copying from being right). If the model were estimating, row order could not matter.

| table order | first row | last row | modal answer | share | = first row | = largest | distinct answers |
|---|---|---|---|---|---|---|---|
| ascending | nan | 1440 | 1440 | 59% | 0% | 59% | 8/39 |
| descending | nan | 1 | 1440 | 59% | 0% | 59% | 9/39 |
| shuffled1 | nan | 10.4 | 350 | 85% | 0% | 5% | 6/39 |
| shuffled2 | nan | 4.5 | 174 | 41% | 0% | 5% | 13/39 |

Reordering moves the collapse target — the same objects are answered 1440, then 350, then 174 — so the answer is a function of prompt layout rather than of the object. Note this also rules out the two obvious single-cause stories: it is not 'the last row' (descending keeps 1440 while moving it to the top) and not 'the largest value' (shuffling drops the largest to a few percent). **Reordering is not a fix either** — it relocates the collapse rather than removing it. Only giving the model room to reason does that.

## 5. Is it the model or the way it is asked?
Same prompt, same items, different request settings. `parsed` counts replies a number could be read out of — anything unparsed falls back to the heuristic in production, so a low count is itself a failure mode. `cut` counts replies truncated by the token budget; **scores on a variant with a non-zero `cut` are optimistic**, because the items that get truncated are the ones the model reasons longest about.

| setting | parsed | cut | typ. factor | bias | <=2x | <=3x | Spearman | wall time |
|---|---|---|---|---|---|---|---|---|
| shipped: v4-flash, no thinking, 32 tok | 61/61 | 0 | 5.25 | 5.25 | 26% | 34% | 0.442 | 10s |
| v4-flash, thinking on, 32k tok | 61/61 | 0 | 1.333 | 1.147 | 88% | 95% | 0.95 | 528s |
| deepseek-chat, no thinking, 32 tok | 60/61 | 0 | 5.005 | 5.005 | 27% | 35% | 0.432 | 14s |
