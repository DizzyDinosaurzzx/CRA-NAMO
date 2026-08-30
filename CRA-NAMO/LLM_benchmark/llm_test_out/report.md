# LLM estimator benchmark: difficulty and risk
Model `deepseek-v4-flash-vision-exp`, 2 repeats per item, 110 items, 1820.7s of API time.

## Headline

- The estimator is off by a typical factor of **1.4x**; only **84%** of objects land within 2x of the reference, and it over-estimates (1.1x median bias).
- It is interpolating rather than copying: only 16% of replies land exactly on an anchor value, and 100 distinct numbers come back for 110 objects.
- On **risk** the model agrees with the reference level on **72%** of objects against the keyword fallback's 68%, and calls **6% of them safer than they are** (fallback 27%). That direction is the one that matters: an under-called obstacle is one the planner is willing to push.
- At the largest simulated Gap, across 1 crossings of the ten-gate map it changes **1.0 of 10 decisions** (worst 1) and costs **+8.6% of C** on average, worst +8.6%.

## 1. Estimator accuracy
`median_abs_factor` is the typical multiplicative miss: 1.0 is exact, 2.0 means the usual answer is off by a factor of two in either direction. `median_ratio` separates bias from spread — above 1.0 the model systematically over-estimates.

| group | n | LLM typ. factor | LLM bias | LLM <=2x | LLM <=3x | LLM Spearman | heuristic typ. factor | heuristic <=2x |
|---|---|---|---|---|---|---|---|---|
| object | 87 | 1.389 | 1.087 | 84% | 94% | 0.974 | 2.667 | 39% |
| - anchor paraphrase | 22 | 1.218 | 1.0 | 91% | 100% | 0.951 | 1.441 | 64% |
| - off-table | 65 | 1.418 | 1.095 | 82% | 92% | 0.971 | 4.34 | 31% |
| state | 12 | 1.396 | 1.216 | 83% | 100% | 0.942 | 4.87 | 17% |
| brand | 11 | 1.32 | 1.008 | 91% | 91% | 0.882 | 2.589 | 9% |
| ALL | 110 | 1.387 | 1.076 | 84% | 94% | 0.969 | 2.838 | 34% |

**Anchor snapping.** 16% of replies are a verbatim row of the anchor table rather than an interpolation, and 2% are exactly 1440 — the largest anchor (`concrete_block`) and the last line of the table in the prompt. Across 110 distinct objects the model produced 100 distinct numbers. The prompt asks for an interpolation between anchors and mostly gets one: the replies are spread over the range rather than piled on the table's last line, so the error above is estimation error, not a lookup wearing its clothes.

**Repeatability** at temperature 0: 26% of items returned an identical value on every repeat; median spread across repeats 1.14x, worst 8.00x.

### Worst 10 items
| label | category | reference | LLM | factor | why the reference says so |
|---|---|---|---|---|---|
| inflated exercise ball | fitness | 2.6 | 0.2 | 0.06x | 1.2 kg / 0.2746 m^3, mu=0.6; PVC shell around air, ~4 kg/m^3 bulk; grippy on a hard floor |
| loaded Pelican 1650 case | container | 72.1 | 8.9 | 0.12x | 30 kg / 0.1665 m^3, mu=0.4; case 12 kg + ~18 kg kit; polymer shell on concrete |
| wheelbarrow of wet sand resting on its legs | construction | 66.3 | 330.0 | 4.98x | 120 kg / 0.6338 m^3, mu=0.35; barrow 15 kg + 105 kg wet sand; parked on legs, not on its wheel |
| stocked refrigerated display cabinet | retail | 49.3 | 11.5 | 0.23x | 250 kg / 2.28 m^3, mu=0.45; cabinet ~180 kg + 70 kg stock; glass doors, levelling feet |
| commercial chest freezer | appliance | 43.2 | 11.2 | 0.26x | 90 kg / 0.833 m^3, mu=0.4; empty; sheet-steel cabinet on plastic feet |
| large potted ficus in ceramic planter | outdoor | 34.7 | 122.5 | 3.53x | 45 kg / 0.648 m^3, mu=0.5; planter + wet soil ~40 kg; unglazed ceramic on concrete |
| marble statue on a stone plinth | valuables | 225.6 | 660.0 | 2.93x | 350 kg / 0.931 m^3, mu=0.6; carved marble ~2700 kg/m^3 but the bbox is mostly air around the figure |
| concrete planter with a semi-mature tree | outdoor | 300.0 | 850.0 | 2.83x | 600 kg / 1.2 m^3, mu=0.6; precast planter ~350 kg + 250 kg of wet soil and root ball |
| glass display cabinet of museum exhibits | valuables | 41.7 | 101.2 | 2.43x | 120 kg / 1.296 m^3, mu=0.45; steel frame and glazing 90 kg + 30 kg of exhibits |
| fallen roof beam wedged against a wall | disaster | 685.7 | 285.0 | 0.42x | 400 kg / 0.35 m^3, mu=0.6; steel section resting on rubble at one end |

## 1b. Risk assessment
A different question from mu*rho, on the same objects: not how hard it is to push, but what happens to people and to the building if it is pushed. Model `deepseek-v4-flash-vision-exp`, 2 replies per item on sight and 1 after contact.

The two `m` columns price the mistake in the planner's own units. `risk.RISK_DETOUR_EQUIV_M` says what each level is worth as a detour (low 0 m, medium 20 m, medium_high 80 m, high 400 m, extreme 5000 m), so **shortfall** is the protection the estimator dropped and **excess** is the detour it invented, averaged over every item. Shortfall is the one that hurts someone; excess only costs distance.

| arm | n | exact | within 1 level | called too safe | called too dangerous | shortfall m | excess m |
|---|---|---|---|---|---|---|---|
| keyword fallback | 110 | 68% | 88% | 27% | 4% | 28 | 46 |
| LLM on sight | 110 | 72% | 96% | 6% | 22% | 50 | 7 |
| LLM after contact | 110 | 66% | 96% | 12% | 22% | 55 | 6 |

**Does touching it help?** Handing the model the measured push force moves exact agreement from 72% to 66%, and the too-safe rate from 6% to 12%: on the direction that matters, contact does not help. The reference level is identical in both arms, so all of that movement is the force number changing the model's mind.

### Worst under-calls on sight
Every row is an obstacle the planner would have been willing to push.

| label | reference | LLM | keyword | why the reference says so |
|---|---|---|---|---|
| resuscitation trolley stocked for emergencies | high | medium | low | nobody is on it, but somebody's life depends on it being where the staff left it |
| fallen roof beam wedged against a wall | extreme | medium_high | extreme | already load-bearing by accident; whatever came down on it comes down again when it moves |
| pile of collapsed masonry rubble | medium_high | low | low | unstable and it may be holding a void open; a rescue robot does not get to find out by pushing |
| empty 19 kilogram propane cylinder | medium_high | low | low | half the mass and none of the safety margin: a nominally empty cylinder still holds vapour under pressure, so the risk label must NOT follow the weight down |
| empty steel drum | medium | low | low | drained but not purged: the vapour left inside is the flammable part, so it does not fall all the way to low |
| catering trolley carrying hot food pans | medium_high | medium | low | open pans of food at 80 C at waist height: a tip is a scald, which is an injury rather than a mess |
| Igloo 150 quart cooler packed with ice | medium | low | low | 50 kg of ice water if the lid comes off; a slip hazard and a ruined load |

## 2. Size independence
mu*rho must not depend on the object's size — the caller multiplies by volume afterwards, so any size response is counted twice.

2/10 items returned the same number at 0.5x, 1x and 2x linear scale (8x volume range).

| label | 0.5x | 1x | 2x | spread |
|---|---|---|---|---|
| empty 240 litre wheelie bin | 1.0 | 1.2 | 0.12 | 10.00x |
| unloaded push trolley | 2.4 | 1.0 | 0.6 | 4.00x |
| empty steel drum | 75.0 | 35.0 | 34.0 | 2.21x |
| steel storage rack with stock | 280.0 | 200.0 | 175.0 | 1.60x |
| wooden shipping crate | 36.0 | 36.0 | 27.0 | 1.33x |
| cardboard box packed with hardcover books | 175.0 | 175.0 | 210.0 | 1.20x |
| granite countertop slab | 1620.0 | 1620.0 | 1650.0 | 1.02x |
| IKEA BILLY bookcase, empty | 34.8 | 34.8 | 35.0 | 1.01x |
| expanded polystyrene packing box | 5.25 | 5.25 | 5.25 | 1.00x |
| solid concrete cube | 1440.0 | 1440.0 | 1440.0 | 1.00x |

## 3. Proof that it is copying, not estimating
The anchor table's row order is rewritten; every other byte of the prompt is unchanged, and only the 88 off-table objects are asked (a paraphrase item's correct answer *is* an anchor, so it cannot distinguish copying from being right). If the model were estimating, row order could not matter.

| table order | first row | last row | modal answer | share | = first row | = largest | distinct answers |
|---|---|---|---|---|---|---|---|
| ascending | 1 | 1440 | 1440 | 31% | 0% | 31% | 16/88 |
| descending | 1440 | 1 | 1440 | 28% | 28% | 28% | 13/88 |
| shuffled1 | 350 | 10.4 | 173.6 | 24% | 16% | 10% | 17/88 |
| shuffled2 | 173.6 | 4.5 | 350 | 24% | 19% | 4% | 16/88 |

Reordering moves the collapse target — the same objects are answered 1440, then 350, then 174 — so the answer is a function of prompt layout rather than of the object. Note this also rules out the two obvious single-cause stories: it is not 'the last row' (descending keeps 1440 while moving it to the top) and not 'the largest value' (shuffling drops the largest to a few percent). **Reordering is not a fix either** — it relocates the collapse rather than removing it. Only giving the model room to reason does that.

## 4. What the error costs a route
The `ten_doors` map is ten walls in a row, each with three ways past it: move the obstacle in door A, move the one in door B, or walk around through a third opening placed far enough off the axis to cost a real detour. Every gate is one independent three-way decision, so a run is ten of them and the arms differ only in what the planner was told to believe.

`exact` is the floor: beliefs equal the truth. At a Gap point `F / K`, each cost estimate is a seeded log-uniform draw between `true/F` and `true*F`, while each risk estimate is shifted by a seeded random integer from `-K` to `+K` levels and clipped to the valid ladder. The tested points are 1x/±0, 1.5x/±1, 2x/±2, 4x/±3, 10x/±4. `difficulty` and `risk` perturb one term each and `both` uses the same two draws, so arm differences isolate the source rather than a different random sample.

Beliefs are seeded per obstacle and no API is called, but the corrections a real run earns are left in: touching an obstacle reveals its true difficulty and re-rates its risk. What the perturbation buys is therefore a wrong *decision*, taken before the robot could know better — which is exactly what a bad estimate costs in practice. The two route screenshots compare the exact arm with seed 0 at the maximum joint Gap.

| Gap F/±K | arm | runs | reached goal | realized cost gap | realized risk gap | mean C | C vs exact | gates changed | risky pushes |
|---|---|---|---|---|---|---|---|---|---|
| 1x/±0 | exact | 1 | 1/1 | 1.00x | 0.00 | 42,662 | - | - | 0.0 |
| 1.5x/±1 | difficulty | 1 | 1/1 | 1.26x | 0.00 | 43,232 | +1.3% | 0.0 | 0.0 |
| 1.5x/±1 | risk | 1 | 1/1 | 1.00x | 0.25 | 43,104 | +1.0% | 0.0 | 0.0 |
| 1.5x/±1 | both | 1 | 1/1 | 1.26x | 0.25 | 43,674 | +2.4% | 0.0 | 0.0 |
| 2x/±2 | difficulty | 1 | 1/1 | 1.48x | 0.00 | 42,801 | +0.3% | 0.0 | 0.0 |
| 2x/±2 | risk | 1 | 1/1 | 1.00x | 0.75 | 42,544 | -0.3% | 0.0 | 0.0 |
| 2x/±2 | both | 1 | 1/1 | 1.48x | 0.75 | 42,683 | +0.1% | 0.0 | 0.0 |
| 4x/±3 | difficulty | 1 | 1/1 | 2.25x | 0.00 | 42,881 | +0.5% | 0.0 | 0.0 |
| 4x/±3 | risk | 1 | 1/1 | 1.00x | 0.75 | 43,104 | +1.0% | 0.0 | 0.0 |
| 4x/±3 | both | 1 | 1/1 | 2.25x | 0.75 | 42,881 | +0.5% | 0.0 | 0.0 |
| 10x/±4 | difficulty | 1 | 1/1 | 4.25x | 0.00 | 44,382 | +4.0% | 0.0 | 0.0 |
| 10x/±4 | risk | 1 | 1/1 | 1.00x | 1.10 | 44,951 | +5.4% | 1.0 | 0.0 |
| 10x/±4 | both | 1 | 1/1 | 4.25x | 1.10 | 46,332 | +8.6% | 1.0 | 0.0 |

`gates changed` counts gates where the perturbed run chose differently from `exact` — the direct measure of an estimate changing a decision. The penalty is taken on **C**, not J: the risk surcharge is charged outside J, so a run that pushed something it should have avoided books a cheaper J and a dearer C, and only C tells the two apart. `pushes of a risky obstacle` counts obstacles moved that were not `low` risk to begin with; `exact` moves 0 of them, and every one above that is the planner disturbing something it should have walked around.

### Where the decisions moved
| gate | detour | door A | door B | exact chose | perturbed runs that chose otherwise |
|---|---|---|---|---|---|
| 0 | 9.3 m | 1600 N low | 2600 N low | detour | 0/12 |
| 1 | 9.9 m | 400 N low | 3000 N low | A | 2/12 |
| 2 | 11.3 m | 2200 N low | 1800 N low | detour | 0/12 |
| 3 | 11.9 m | 3400 N low | 2900 N low | detour | 0/12 |
| 4 | 8.5 m | 800 N medium | 2400 N low | detour | 0/12 |
| 5 | 13.9 m | 1200 N low | 900 N medium | detour | 0/12 |
| 6 | 12.9 m | 2600 N low | 2000 N low | detour | 0/12 |
| 7 | 8.7 m | 5000 N low | 4500 N low | detour | 0/12 |
| 8 | 14.9 m | 1000 N medium | 2600 N low | detour | 0/12 |
| 9 | 14.3 m | 2000 N low | 1500 N medium | detour | 0/12 |
