# LLM estimator benchmark: difficulty and risk
Model `deepseek-flash`, 2 repeats per item, 110 items, 563.6s of API time.

## Headline

- The estimator is off by a typical factor of **1.4x**; only **83%** of objects land within 2x of the reference, and it over-estimates (1.2x median bias).
- It is interpolating rather than copying: only 28% of replies land exactly on an anchor value, and 84 distinct numbers come back for 110 objects.
- On **risk** the model agrees with the reference level on **64%** of objects against the keyword fallback's 68%, and calls **6% of them safer than they are** (fallback 27%). That direction is the one that matters: an under-called obstacle is one the planner is willing to push.
- On the ten-gate corridor, **every doubling of the cost estimate's error costs +1.6% of C**. At the ladder's far end (10x) the route is +5.3% dearer on average, worst +7.0%, with 0.0 of 10 gates decided differently.
- Read the measured accuracy off that ladder and the estimator's typical **1.4x miss is worth +1.2% of C**; its p90 2.3x miss is worth +1.0%. That is the price of the error in the only unit the planner cares about.

## 1. Estimator accuracy
`median_abs_factor` is the typical multiplicative miss: 1.0 is exact, 2.0 means the usual answer is off by a factor of two in either direction. `median_ratio` separates bias from spread — above 1.0 the model systematically over-estimates.

| group | n | LLM typ. factor | LLM bias | LLM <=2x | LLM <=3x | LLM Spearman | heuristic typ. factor | heuristic <=2x |
|---|---|---|---|---|---|---|---|---|
| object | 87 | 1.342 | 1.181 | 79% | 94% | 0.972 | 2.667 | 39% |
| - anchor paraphrase | 22 | 1.0 | 1.0 | 91% | 100% | 0.945 | 1.441 | 64% |
| - off-table | 65 | 1.449 | 1.238 | 75% | 92% | 0.967 | 4.34 | 31% |
| state | 12 | 1.498 | 1.336 | 92% | 92% | 0.942 | 4.87 | 17% |
| brand | 11 | 1.19 | 0.935 | 100% | 100% | 0.936 | 2.589 | 9% |
| ALL | 110 | 1.355 | 1.174 | 83% | 94% | 0.97 | 2.838 | 34% |

**Anchor snapping.** 28% of replies are a verbatim row of the anchor table rather than an interpolation, and 2% are exactly 1440 — the largest anchor (`concrete_block`) and the last line of the table in the prompt. Across 110 distinct objects the model produced 84 distinct numbers. The prompt asks for an interpolation between anchors and mostly gets one: the replies are spread over the range rather than piled on the table's last line, so the error above is estimation error, not a lookup wearing its clothes.

**Repeatability** at temperature 0: 33% of items returned an identical value on every repeat; median spread across repeats 1.12x, worst 22.73x.

### Worst 10 items
| label | category | reference | LLM | factor | why the reference says so |
|---|---|---|---|---|---|
| inflated exercise ball | fitness | 2.6 | 0.2 | 0.10x | 1.2 kg / 0.2746 m^3, mu=0.6; PVC shell around air, ~4 kg/m^3 bulk; grippy on a hard floor |
| commercial chest freezer | appliance | 43.2 | 6.5 | 0.15x | 90 kg / 0.833 m^3, mu=0.4; empty; sheet-steel cabinet on plastic feet |
| tool chest filled with hand tools | facilities | 16.0 | 87.5 | 5.47x | 90 kg / 0.2815 m^3, mu=0.05; steel roller cabinet packed with steel tools |
| drum of corrosive chemical on a spill pallet | hazmat | 53.8 | 220.0 | 4.09x | 250 kg / 1.859 m^3, mu=0.4; 190 kg drum on a 60 kg bunded polyethylene pallet |
| wheelbarrow of wet sand resting on its legs | construction | 66.3 | 265.0 | 4.00x | 120 kg / 0.6338 m^3, mu=0.35; barrow 15 kg + 105 kg wet sand; parked on legs, not on its wheel |
| marble statue on a stone plinth | valuables | 225.6 | 760.0 | 3.37x | 350 kg / 0.931 m^3, mu=0.6; carved marble ~2700 kg/m^3 but the bbox is mostly air around the figure |
| large potted ficus in ceramic planter | outdoor | 34.7 | 92.5 | 2.66x | 45 kg / 0.648 m^3, mu=0.5; planter + wet soil ~40 kg; unglazed ceramic on concrete |
| concrete planter with a semi-mature tree | outdoor | 300.0 | 750.0 | 2.50x | 600 kg / 1.2 m^3, mu=0.6; precast planter ~350 kg + 250 kg of wet soil and root ball |
| stocked refrigerated display cabinet | retail | 49.3 | 120.0 | 2.43x | 250 kg / 2.28 m^3, mu=0.45; cabinet ~180 kg + 70 kg stock; glass doors, levelling feet |
| stocked warehouse racking bay | warehouse | 75.2 | 175.0 | 2.33x | restatement of anchor 'shelf' (mu*rho=75.15) |

## 1b. Risk assessment
A different question from mu*rho, on the same objects: not how hard it is to push, but what happens to people and to the building if it is pushed. Model `deepseek-flash`, 2 replies per item on sight and 1 after contact.

The two `m` columns price the mistake in the planner's own units. `risk.RISK_DETOUR_EQUIV_M` says what each level is worth as a detour (low 0 m, medium 20 m, medium_high 80 m, high 400 m, extreme 5000 m), so **shortfall** is the protection the estimator dropped and **excess** is the detour it invented, averaged over every item. Shortfall is the one that hurts someone; excess only costs distance.

| arm | n | exact | within 1 level | called too safe | called too dangerous | shortfall m | excess m |
|---|---|---|---|---|---|---|---|
| keyword fallback | 110 | 68% | 88% | 27% | 4% | 28 | 46 |
| LLM on sight | 110 | 64% | 96% | 6% | 30% | 49 | 9 |
| LLM after contact | 110 | 69% | 97% | 4% | 26% | 48 | 8 |

**Does touching it help?** Handing the model the measured push force moves exact agreement from 64% to 69%, and the too-safe rate from 6% to 4%: on the direction that matters, contact helps. The reference level is identical in both arms, so all of that movement is the force number changing the model's mind.

### Worst under-calls on sight
Every row is an obstacle the planner would have been willing to push.

| label | reference | LLM | keyword | why the reference says so |
|---|---|---|---|---|
| fallen roof beam wedged against a wall | extreme | medium_high | extreme | already load-bearing by accident; whatever came down on it comes down again when it moves |
| empty 19 kilogram propane cylinder | medium_high | low | low | half the mass and none of the safety margin: a nominally empty cylinder still holds vapour under pressure, so the risk label must NOT follow the weight down |
| unoccupied hospital bed | medium | low | low | no patient on it, so not high; still a costly asset that a ward may be about to need |
| resuscitation trolley stocked for emergencies | high | medium_high | low | nobody is on it, but somebody's life depends on it being where the staff left it |
| empty steel drum | medium | low | low | drained but not purged: the vapour left inside is the flammable part, so it does not fall all the way to low |
| empty 200 litre aquarium | medium | low | low | nothing to spill, but it is still a 1.2 m box of 8 mm glass |

## 2. Size independence
mu*rho must not depend on the object's size — the caller multiplies by volume afterwards, so any size response is counted twice.

4/10 items returned the same number at 0.5x, 1x and 2x linear scale (8x volume range).

| label | 0.5x | 1x | 2x | spread |
|---|---|---|---|---|
| unloaded push trolley | 2.5 | 1.0 | 1.0 | 2.50x |
| empty steel drum | 50.0 | 31.0 | 20.0 | 2.50x |
| IKEA BILLY bookcase, empty | 50.0 | 36.0 | 25.4 | 1.97x |
| steel storage rack with stock | 220.0 | 160.0 | 200.0 | 1.38x |
| granite countertop slab | 1350.0 | 1620.0 | 1620.0 | 1.20x |
| cardboard box packed with hardcover books | 200.0 | 175.0 | 175.0 | 1.14x |
| expanded polystyrene packing box | 5.25 | 5.25 | 5.25 | 1.00x |
| wooden shipping crate | 27.0 | 27.0 | 27.0 | 1.00x |
| solid concrete cube | 1440.0 | 1440.0 | 1440.0 | 1.00x |
| empty 240 litre wheelie bin | 1.0 | 1.0 | 1.0 | 1.00x |

## 3. Proof that it is copying, not estimating
The anchor table's row order is rewritten; every other byte of the prompt is unchanged, and only the 88 off-table objects are asked (a paraphrase item's correct answer *is* an anchor, so it cannot distinguish copying from being right). If the model were estimating, row order could not matter.

| table order | first row | last row | modal answer | share | = first row | = largest | distinct answers |
|---|---|---|---|---|---|---|---|
| ascending | 1 | 1440 | 3.5 | 7% | 3% | 1% | 53/73 |
| descending | 1440 | 1 | 3 | 6% | 2% | 2% | 55/67 |
| shuffled1 | 350 | 10.4 | 270 | 4% | 0% | 3% | 61/74 |
| shuffled2 | 173.6 | 4.5 | 3 | 7% | 0% | 3% | 56/68 |

Reordering moves the collapse target — the same objects are answered 1440, then 350, then 174 — so the answer is a function of prompt layout rather than of the object. Note this also rules out the two obvious single-cause stories: it is not 'the last row' (descending keeps 1440 while moving it to the top) and not 'the largest value' (shuffling drops the largest to a few percent). **Reordering is not a fix either** — it relocates the collapse rather than removing it. Only giving the model room to reason does that.

## 4. What the error costs a route
The `ten_doors` map is 10 walls in a row, each with three ways past it: move the obstacle in door A, move the one in door B, or walk around through a third opening placed far enough off the axis to cost a real detour. Every gate is one independent three-way decision, so a run is 10 of them and the arms differ only in what the planner was told to believe.

No API is called here. The belief is written directly: at Gap ratio `F` every cost estimate is exactly `true * F` (the **over-estimate** arm), exactly `true / F` (the **under-estimate** arm), or one of the two with the direction drawn once per obstacle and then held fixed for the whole ladder (the **random-direction** arm). Because `F` is constructed rather than sampled, the realized gap equals the axis label, and the rows below can be read as 'an estimator this wrong costs this much'. Holding each obstacle's direction fixed across the ladder also makes the rungs paired samples, so the curve's shape is the Gap growing rather than a fresh set of random numbers.

Risk is perturbed the same way: `K` levels toward safe in the under-estimate arm, `K` toward dangerous in the over-estimate arm, clipped to the ladder.

**C is re-priced at the reference risk level.** The executor charges the risk surcharge at the level it believed, so a run that called a hazard safe would otherwise book a discount for the mistake and look cheaper than `exact`. Every C below is `J` plus the surcharge the pushed obstacles were really worth, which is what makes a wrong decision show up as a cost rather than a saving.

Beliefs are seeded per obstacle, but the corrections a real run earns are left in: touching an obstacle reveals its true difficulty and re-rates its risk. What the perturbation buys is therefore a wrong *decision*, taken before the robot could know better, which is exactly what a bad estimate costs in practice.

The ladder straddles one threshold worth knowing about. `Config.contact_replan_ratio` is 1.25, so an error smaller than that is never noticed: the robot takes hold, finds the force close enough to what it planned for, and carries on. Above it the mismatch triggers a re-plan mid-push, which is why the curve is flatter at the first rung or two than a straight line through the rest would predict.

`exact` is the floor at C = 42,562 J with 1 obstacles pushed.


### 4.1 Gap ratio against the change in C
The headline table. One row per rung of the cost ladder and arm; `dC` is the percentage change from the exact run's C.

| Gap F | arm | runs | reached goal | dC mean | dC median | dC p90 | dC worst | gates changed (of 10) | worst | extra risky pushes | mean C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.15x | under-estimate | 1 | 1/1 | +1.5% | +1.5% | +1.5% | +1.5% | 0.0 | 0 | +0.0 | 43,203 |
| 1.15x | over-estimate | 1 | 1/1 | +0.0% | +0.0% | +0.0% | +0.0% | 0.0 | 0 | +0.0 | 42,562 |
| 1.15x | random direction | 3 | 3/3 | +0.9% | +1.3% | +1.5% | +1.5% | 0.0 | 0 | +0.0 | 42,966 |
| 1.3x | under-estimate | 1 | 1/1 | +2.0% | +2.0% | +2.0% | +2.0% | 0.0 | 0 | +0.0 | 43,413 |
| 1.3x | over-estimate | 1 | 1/1 | +0.5% | +0.5% | +0.5% | +0.5% | 0.0 | 0 | +0.0 | 42,772 |
| 1.3x | random direction | 3 | 3/3 | +1.4% | +1.8% | +2.0% | +2.0% | 0.0 | 0 | +0.0 | 43,176 |
| 1.5x | under-estimate | 1 | 1/1 | +2.0% | +2.0% | +2.0% | +2.0% | 0.0 | 0 | +0.0 | 43,413 |
| 1.5x | over-estimate | 1 | 1/1 | -0.6% | -0.6% | -0.6% | -0.6% | 0.0 | 0 | +0.0 | 42,316 |
| 1.5x | random direction | 3 | 3/3 | +0.8% | +0.9% | +1.8% | +2.0% | 0.0 | 0 | +0.0 | 42,895 |
| 1.75x | under-estimate | 1 | 1/1 | +2.0% | +2.0% | +2.0% | +2.0% | 0.0 | 0 | +0.0 | 43,413 |
| 1.75x | over-estimate | 1 | 1/1 | -0.6% | -0.6% | -0.6% | -0.6% | 0.0 | 0 | +0.0 | 42,316 |
| 1.75x | random direction | 3 | 3/3 | +0.8% | +0.9% | +1.8% | +2.0% | 0.0 | 0 | +0.0 | 42,895 |
| 2x | under-estimate | 1 | 1/1 | +2.4% | +2.4% | +2.4% | +2.4% | 0.0 | 0 | +0.0 | 43,572 |
| 2x | over-estimate | 1 | 1/1 | -0.6% | -0.6% | -0.6% | -0.6% | 0.0 | 0 | +0.0 | 42,316 |
| 2x | random direction | 3 | 3/3 | +1.0% | +1.1% | +2.1% | +2.4% | 0.0 | 0 | +0.0 | 43,004 |
| 2.5x | under-estimate | 1 | 1/1 | +2.2% | +2.2% | +2.2% | +2.2% | 0.0 | 0 | +0.0 | 43,501 |
| 2.5x | over-estimate | 1 | 1/1 | -0.6% | -0.6% | -0.6% | -0.6% | 0.0 | 0 | +0.0 | 42,316 |
| 2.5x | random direction | 3 | 3/3 | +1.0% | +1.1% | +2.0% | +2.2% | 0.0 | 0 | +0.0 | 42,978 |
| 3x | under-estimate | 1 | 1/1 | +2.9% | +2.9% | +2.9% | +2.9% | 0.0 | 0 | +0.0 | 43,792 |
| 3x | over-estimate | 1 | 1/1 | +4.5% | +4.5% | +4.5% | +4.5% | 1.0 | 1 | +0.0 | 44,477 |
| 3x | random direction | 3 | 3/3 | +1.4% | +1.8% | +2.7% | +2.9% | 0.0 | 0 | +0.0 | 43,171 |
| 4x | under-estimate | 1 | 1/1 | +6.0% | +6.0% | +6.0% | +6.0% | 0.0 | 0 | +0.0 | 45,112 |
| 4x | over-estimate | 1 | 1/1 | +4.5% | +4.5% | +4.5% | +4.5% | 1.0 | 1 | +0.0 | 44,477 |
| 4x | random direction | 3 | 3/3 | +2.9% | +4.3% | +4.7% | +4.8% | 0.0 | 0 | +0.0 | 43,795 |
| 6x | under-estimate | 1 | 1/1 | +9.8% | +9.8% | +9.8% | +9.8% | 0.0 | 0 | +0.0 | 46,742 |
| 6x | over-estimate | 1 | 1/1 | +4.5% | +4.5% | +4.5% | +4.5% | 1.0 | 1 | +0.0 | 44,477 |
| 6x | random direction | 3 | 3/3 | +5.2% | +6.8% | +6.9% | +7.0% | 0.0 | 0 | +0.0 | 44,794 |
| 10x | under-estimate | 1 | 1/1 | +10.5% | +10.5% | +10.5% | +10.5% | 0.0 | 0 | +0.0 | 47,043 |
| 10x | over-estimate | 1 | 1/1 | +4.5% | +4.5% | +4.5% | +4.5% | 1.0 | 1 | +0.0 | 44,477 |
| 10x | random direction | 3 | 3/3 | +5.3% | +7.0% | +7.0% | +7.0% | 0.0 | 0 | +0.0 | 44,823 |

**Sensitivity.** Fitting the mean `dC` against `log2(F)` gives the cost of each doubling of estimate error: under-estimate **+3.1% of C per doubling**, over-estimate **+1.9% of C per doubling**, random direction **+1.6% of C per doubling**. The two deterministic arms bracket the random one: a uniform bias moves every option the same way and only changes push-against-detour, while a random direction also reverses door A against door B, which is the cheaper mistake to make but the easier one to make often.

### 4.2 Where the measured estimator lands on that ladder
Section 1 measured how wrong `deepseek-flash` is on the reference objects. Reading that number off the curve above turns an accuracy figure into a route cost, which is the only unit that matters to the planner. Values are interpolated on the log-ratio axis between the rungs that were actually run.

| estimator accuracy | as a Gap ratio | dC mean | dC worst | gates changed | extra risky pushes |
|---|---|---|---|---|---|
| typical miss (median) | 1.35x | +1.2% | +2.0% | 0.0 | 0.0 |
| bad-case miss (p90) | 2.28x | +1.0% | +2.3% | 0.0 | 0.0 |

### 4.3 The same sweep on risk levels
A risk level is not a ratio, so this ladder is in levels: `K` steps from the reference level, clipped at `low` and `extreme`. Under-calling is the direction that matters, because it is the one that lets the planner push something it should have walked around.

| Gap K | arm | runs | reached goal | dC mean | dC worst | gates changed | extra risky pushes |
|---|---|---|---|---|---|---|---|
| +/-1 | under-estimate | 1 | 1/1 | +2.5% | +2.5% | 0.0 | +0.0 |
| +/-1 | over-estimate | 1 | 1/1 | +4.5% | +4.5% | 1.0 | +0.0 |
| +/-1 | random direction | 3 | 3/3 | +0.9% | +2.1% | 0.0 | +0.0 |
| +/-2 | under-estimate | 1 | 1/1 | +2.5% | +2.5% | 0.0 | +0.0 |
| +/-2 | over-estimate | 1 | 1/1 | +4.5% | +4.5% | 1.0 | +0.0 |
| +/-2 | random direction | 3 | 3/3 | +0.9% | +2.1% | 0.0 | +0.0 |
| +/-3 | under-estimate | 1 | 1/1 | +2.5% | +2.5% | 0.0 | +0.0 |
| +/-3 | over-estimate | 1 | 1/1 | +4.5% | +4.5% | 1.0 | +0.0 |
| +/-3 | random direction | 3 | 3/3 | +0.9% | +2.1% | 0.0 | +0.0 |
| +/-4 | under-estimate | 1 | 1/1 | +2.5% | +2.5% | 0.0 | +0.0 |
| +/-4 | over-estimate | 1 | 1/1 | +4.5% | +4.5% | 1.0 | +0.0 |
| +/-4 | random direction | 3 | 3/3 | +0.9% | +2.1% | 0.0 | +0.0 |

### 4.4 Both at once
Cost and risk perturbed together, so the two contributions can be compared against their sum.

| Gap F / K | arm | runs | dC together | dC cost alone | dC risk alone | sum of the two | gates changed |
|---|---|---|---|---|---|---|---|
| 1.5x / +/-1 | under-estimate | 1 | +2.1% | +2.0% | +2.5% | +4.5% | 0.0 |
| 1.5x / +/-1 | over-estimate | 1 | +4.5% | -0.6% | +4.5% | +3.9% | 1.0 |
| 1.5x / +/-1 | random direction | 3 | +1.2% | +0.8% | +0.9% | +1.7% | 0.0 |
| 2x / +/-2 | under-estimate | 1 | +4.8% | +2.4% | +2.5% | +4.9% | 0.0 |
| 2x / +/-2 | over-estimate | 1 | +4.5% | -0.6% | +4.5% | +3.9% | 1.0 |
| 2x / +/-2 | random direction | 3 | +2.0% | +1.0% | +0.9% | +1.9% | 0.0 |
| 4x / +/-3 | under-estimate | 1 | +7.5% | +6.0% | +2.5% | +8.5% | 0.0 |
| 4x / +/-3 | over-estimate | 1 | +4.5% | +4.5% | +4.5% | +9.0% | 1.0 |
| 4x / +/-3 | random direction | 3 | +3.8% | +2.9% | +0.9% | +3.8% | 0.0 |
| 10x / +/-4 | under-estimate | 1 | +12.7% | +10.5% | +2.5% | +13.1% | 0.0 |
| 10x / +/-4 | over-estimate | 1 | +4.5% | +4.5% | +4.5% | +9.0% | 1.0 |
| 10x / +/-4 | random direction | 3 | +6.0% | +5.3% | +0.9% | +6.2% | 0.0 |

### 4.5 Which decisions move first
A gate flips when the error exceeds its own margin, so the ladder rung at which each gate first changes its mind is a direct reading of how much slack that decision had. `A/B ratio` is the true difficulty ratio between the two doors: a random-direction error has to exceed roughly that ratio before door A and door B swap places.

| gate | kind | detour | A/B ratio | door A | door B | exact chose | first cost flip | first risk flip | runs that chose otherwise |
|---|---|---|---|---|---|---|---|---|---|
| 0 | cost | 9.3 m | 1.625x | 1600 N low | 2600 N low | detour | never | never | 0/90 |
| 1 | cost | 9.9 m | 7.5x | 400 N low | 3000 N low | A | 3x | +/-1 | 12/90 |
| 2 | cost | 11.3 m | 1.222x | 2200 N low | 1800 N low | detour | never | never | 0/90 |
| 3 | cost | 11.9 m | 1.172x | 3400 N low | 2900 N low | detour | never | never | 0/90 |
| 4 | risk | 8.5 m | 3x | 800 N medium | 2400 N low | detour | never | never | 0/90 |
| 5 | risk | 13.9 m | 1.333x | 1200 N low | 900 N medium | detour | never | never | 0/90 |
| 6 | cost | 12.9 m | 1.3x | 2600 N low | 2000 N low | detour | never | never | 0/90 |
| 7 | cost | 8.7 m | 1.111x | 5000 N low | 4500 N low | detour | never | never | 0/90 |
| 8 | risk | 14.9 m | 2.6x | 1000 N medium | 2600 N low | detour | never | never | 0/90 |
| 9 | risk | 14.3 m | 1.333x | 2000 N low | 1500 N medium | detour | never | never | 0/90 |

### 4.7 Route screenshots
| run | image |
|---|---|
| cost_10x_0lvl_over | `ten_doors_cost_over.png` |
| cost_10x_0lvl_under | `ten_doors_cost_under.png` |
| exact | `ten_doors_exact.png` |
| joint_10x_4lvl_under | `ten_doors_joint_under.png` |
| risk_1x_4lvl_under | `ten_doors_risk_under.png` |

The machine-readable version of section 4.1 is `doors_gap_vs_cost.csv`; `doors_gap.png` plots it and `doors_gates.png` breaks it down per gate.
