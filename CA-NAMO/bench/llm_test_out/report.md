# LLM mu*rho estimator accuracy
Model `deepseek-v4-flash`, 3 repeats per item, 61 items, 1357.8s of API time.

## Headline

- The estimator is off by a typical factor of **1.4x**; only **84%** of objects land within 2x of the reference, and it over-estimates (1.3x median bias).
- Cause: **2% of replies are exactly 1440**, the largest anchor in the prompt. It is copying a table row, not estimating.
- Best request setting tested is `v4-flash, thinking on, 32k tok` at **1.3x** typical error (88% within 2x, 61/61 answered). Set `Config.deepseek_model` / `deepseek_thinking` / `llm_max_tokens` to match — `llm_difficulty._deepseek` reads those fields directly, no code change needed.

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

## 3. Proof that it is copying, not estimating
The anchor table's row order is rewritten; every other byte of the prompt is unchanged, and only the 39 off-table objects are asked (a paraphrase item's correct answer *is* an anchor, so it cannot distinguish copying from being right). If the model were estimating, row order could not matter.

| table order | first row | last row | modal answer | share | = first row | = largest | distinct answers |
|---|---|---|---|---|---|---|---|
| ascending | nan | 1440 | 1440 | 59% | 0% | 59% | 8/39 |
| descending | nan | 1 | 1440 | 59% | 0% | 59% | 9/39 |
| shuffled1 | nan | 10.4 | 350 | 85% | 0% | 5% | 6/39 |
| shuffled2 | nan | 4.5 | 174 | 41% | 0% | 5% | 13/39 |

Reordering moves the collapse target — the same objects are answered 1440, then 350, then 174 — so the answer is a function of prompt layout rather than of the object. Note this also rules out the two obvious single-cause stories: it is not 'the last row' (descending keeps 1440 while moving it to the top) and not 'the largest value' (shuffling drops the largest to a few percent). **Reordering is not a fix either** — it relocates the collapse rather than removing it. Only giving the model room to reason does that.

## 4. Is it the model or the way it is asked?
Same prompt, same items, different request settings. `parsed` counts replies a number could be read out of — anything unparsed falls back to the heuristic in production, so a low count is itself a failure mode. `cut` counts replies truncated by the token budget; **scores on a variant with a non-zero `cut` are optimistic**, because the items that get truncated are the ones the model reasons longest about.

| setting | parsed | cut | typ. factor | bias | <=2x | <=3x | Spearman | wall time |
|---|---|---|---|---|---|---|---|---|
| shipped: v4-flash, no thinking, 32 tok | 61/61 | 0 | 5.25 | 5.25 | 26% | 34% | 0.442 | 10s |
| v4-flash, thinking on, 32k tok | 61/61 | 0 | 1.333 | 1.147 | 88% | 95% | 0.95 | 528s |
| deepseek-chat, no thinking, 32 tok | 60/61 | 0 | 5.005 | 5.005 | 27% | 35% | 0.432 | 14s |
