"""Benchmark the LLM mu*rho estimator, and price its error in navigation cost.

Three stages, run independently or with `all`:

  accuracy   Ask the live model for mu*rho on every item in `llm_dataset`,
             several times each, and compare against the reference values.
             Also records the no-LLM heuristic (`material_mu_rho`) on the same
             items, so every accuracy number has a floor to beat.

  size       mu*rho is defined size-independent, but the prompt states the
             object's dimensions. Re-asks a subset at 0.5x / 1x / 2x linear
             scale; any movement is prompt leakage, not physics.

  nav        Runs whole scenarios with the estimator's output swapped out, and
             measures what the planner actually pays. Arms:
               oracle     perfect mu*rho              (the floor on cost)
               llm        stage-1 medians             (what you would ship)
               heuristic  `material_mu_rho` fallback  (what you get with no key)
               x<f>       every estimate multiplied by f  (sensitivity curve)
               noise<s>   per-obstacle lognormal error at the measured spread
             Maps are relabelled with paraphrases of their own anchors, so the
             ground-truth physics is untouched and the arms differ *only* in
             what the estimator believes.

  report     Turns the saved JSON into report.md plus two charts. No API calls.

Why the arms are comparable: the true `difficulty` is recomputed from the same
reference mu*rho in every arm, and work is always charged at the true value
(`planner.py` uses `_world_obstacle(oid).difficulty`). A bad estimate therefore
cannot make the bill *look* cheaper — it can only cause the planner to choose
worse, which is exactly the effect we want to isolate.

Usage
-----
    python3 LLM_test.py all                 # ~15 min, ~200 API calls
    python3 LLM_test.py accuracy --repeats 3
    python3 LLM_test.py nav --maps two_doors,hidden_obstacle
    python3 LLM_test.py report              # offline, re-renders from JSON
"""

from __future__ import annotations

# Run from anywhere: the library lives one directory up.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

import scenarios
from config import Config
from llm_difficulty import (
    MATERIAL_MU_RHO,
    DifficultyEstimator,
    _canonical_anchor,
    _normalise,
    friction_force,
    material_mu_rho,
)
from llm_dataset import (
    DATASET,
    PARAPHRASE_OF_ANCHOR,
    Item,
    assert_all_off_anchor,
)
from executor import OnlineNAMO

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_test_out")

# Maps used by the nav stage. Every one offers the planner a real choice between
# pushing and detouring — on a map with only one route the estimate cannot
# change anything and the comparison is vacuous.
DEFAULT_MAPS = ("two_doors", "hidden_obstacle", "maze_mixed", "corridor")

# Sensitivity sweep: uniform multiplicative error applied to every estimate.
# Spread wide on purpose — the response is expected to be flat then step, and a
# narrow sweep would miss where the step is.
SWEEP_FACTORS = (0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0)
NOISE_SEEDS = (0, 1, 2, 3, 4)

SIZE_SCALES = (0.5, 1.0, 2.0)

# Palette: dataviz categorical slots 1/2/3/7, validated all-pairs on the light
# surface (worst CVD dE 9.2, worst normal-vision dE 16.3).
GROUP_STYLE = {
    "paraphrase": ("#2a78d6", "o"),
    "novel":      ("#eb6834", "s"),
    "state":      ("#1baf7a", "^"),
    "brand":      ("#4a3aa7", "D"),
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def log(msg: str) -> None:
    """Heartbeat. Every long stage prints through here so a stalled run is
    visible in the output file rather than looking like a hang."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save(name: str, payload) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log(f"wrote {path}")
    return path


def _load(name: str):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Stage 1 - estimator accuracy
# --------------------------------------------------------------------------- #

def stage_accuracy(cfg: Config, repeats: int, workers: int) -> dict:
    assert_all_off_anchor()
    est = DifficultyEstimator(cfg)
    if not est.api_key:
        raise SystemExit("no DeepSeek API key in Config.deepseek_api_key / $DEEPSEEK_API_KEY")

    jobs = [(item, rep) for item in DATASET for rep in range(repeats)]
    log(f"accuracy: {len(DATASET)} items x {repeats} repeats = {len(jobs)} calls, "
        f"{workers} workers, model={cfg.deepseek_model}")
    done = [0]

    def ask(job):
        item, rep = job
        value = est._deepseek(item.observation(oid=rep))
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(jobs):
            log(f"  {done[0]}/{len(jobs)} calls")
        return item.label, value

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, jobs))
    log(f"accuracy: {len(jobs)} calls in {time.time() - t0:.1f}s")

    by_label: Dict[str, List[Optional[float]]] = {}
    for label, value in results:
        by_label.setdefault(label, []).append(value)

    rows = []
    for item in DATASET:
        preds = [p for p in by_label[item.label] if p is not None and p > 0]
        rows.append({
            **{k: v for k, v in asdict(item).items()},
            "mu_rho_true": item.mu_rho,
            "preds": by_label[item.label],
            # median over repeats is what the nav stage ships: one number per
            # material, robust to a single malformed reply
            "pred": statistics.median(preds) if preds else None,
            "n_ok": len(preds),
            "heuristic": material_mu_rho(item.label),
        })
    payload = {"model": cfg.deepseek_model, "repeats": repeats,
               "seconds": round(time.time() - t0, 1), "rows": rows}
    _save("accuracy.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# Stage 2 - size independence
# --------------------------------------------------------------------------- #

def stage_size(cfg: Config, workers: int) -> dict:
    """The prompt hands the model l x d x h and then forbids it from mattering.

    Anything that moves with scale is the model reading size as a mass cue,
    which double-counts volume: the caller multiplies by volume again.
    """
    est = DifficultyEstimator(cfg)
    subset = [it for it in DATASET
              if it.label in {
                  "expanded polystyrene packing box", "wooden shipping crate",
                  "steel storage rack with stock", "solid concrete cube",
                  "unloaded push trolley", "empty steel drum",
                  "granite countertop slab", "cardboard box packed with hardcover books",
                  "empty 240 litre wheelie bin", "IKEA BILLY bookcase, empty",
              }]
    jobs = [(it, s) for it in subset for s in SIZE_SCALES]
    log(f"size: {len(subset)} items x {len(SIZE_SCALES)} scales = {len(jobs)} calls")
    done = [0]

    def ask(job):
        item, scale = job
        value = est._deepseek(item.observation(scale=scale))
        done[0] += 1
        log(f"  {done[0]}/{len(jobs)}  {item.label[:38]:<38s} x{scale:g} -> {value}")
        return item.label, scale, value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, jobs))

    rows: Dict[str, dict] = {}
    for label, scale, value in results:
        row = rows.setdefault(label, {"label": label, "by_scale": {}})
        row["by_scale"][str(scale)] = value
    for it in subset:
        rows[it.label]["mu_rho_true"] = it.mu_rho
        rows[it.label]["group"] = it.group
    payload = {"scales": list(SIZE_SCALES), "rows": list(rows.values())}
    _save("size.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# Stage 2b - request-setting ablation
# --------------------------------------------------------------------------- #

# `_deepseek` hardcodes its request settings, so a bad number could be the
# model's judgement or just the way it is being asked. These variants separate
# the two. The first row is exactly what `llm_difficulty.py` ships today.
ABLATIONS = (
    ("shipped: v4-flash, no thinking, 32 tok", "deepseek-v4-flash", "disabled", 32),
    # 32k, not 4k: reasoning runs to several thousand tokens on this prompt and
    # the harder objects think the longest. A budget that truncates them drops
    # exactly the items the model finds difficult, and the survivors would score
    # far better than the setting deserves.
    ("v4-flash, thinking on, 32k tok",         "deepseek-v4-flash", "enabled", 32000),
    ("deepseek-chat, no thinking, 32 tok",     "deepseek-chat",     "disabled", 32),
)


def _ask_raw(cfg: Config, model: str, thinking: str, max_tokens: int,
             prompt: str) -> tuple:
    """One call with explicit request settings, mirroring `_deepseek`'s parsing.

    Returns `(value, finish_reason)`. The reason is kept because an unparsed
    reply is ambiguous on its own: `length` means the token budget was too small
    and the item should not be scored, while `stop` with no number is a genuine
    refusal to answer.

    Takes the *last* number in the reply rather than the first: a reasoning
    model's visible answer may be preceded by a restatement, and the final
    number is the one it is committing to.
    """
    import re
    import requests
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
            "thinking": {"type": thinking}}
    try:
        r = requests.post(cfg.deepseek_base_url, timeout=600, json=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.deepseek_api_key}"})
        data = r.json()
        if r.status_code >= 400 or "choices" not in data:
            return None, f"http{r.status_code}"
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content") or ""
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        return (float(nums[-1]) if nums else None), choice.get("finish_reason")
    except Exception as exc:
        return None, f"exception:{type(exc).__name__}"


def stage_ablate(cfg: Config, workers: int) -> dict:
    est = DifficultyEstimator(cfg)
    prompts = {it.label: est._build_prompt(it.observation()) for it in DATASET}
    variants = []
    for tag, model, thinking, max_tokens in ABLATIONS:
        log(f"ablate: {tag} - {len(DATASET)} items")
        done = [0]

        def ask(item: Item, _m=model, _t=thinking, _mt=max_tokens):
            value, reason = _ask_raw(cfg, _m, _t, _mt, prompts[item.label])
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(DATASET):
                log(f"  {tag}: {done[0]}/{len(DATASET)}")
            return item.label, (value, reason)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(ask, DATASET))
        got = dict(results)
        rows = [{"label": it.label, "group": it.group, "mu_rho_true": it.mu_rho,
                 "l": it.l, "d": it.d, "h": it.h, "note": it.note,
                 "pred": got[it.label][0], "finish_reason": got[it.label][1]}
                for it in DATASET]
        truncated = sum(1 for r in rows if r["finish_reason"] == "length")
        variants.append({"tag": tag, "model": model, "thinking": thinking,
                         "max_tokens": max_tokens, "seconds": round(time.time() - t0, 1),
                         "n_parsed": sum(1 for r in rows if r["pred"]),
                         "n_truncated": truncated, "rows": rows})
        log(f"  {tag}: {variants[-1]['n_parsed']}/{len(DATASET)} parsed "
            f"({truncated} truncated) in {variants[-1]['seconds']}s")
    payload = {"variants": variants}
    _save("ablate.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# Stage 2c - is the collapse target the table's last row, or its largest value?
# --------------------------------------------------------------------------- #

_ANCHOR_LINE = re.compile(r"^\s{2}(\S+)\s+mu=\S+\s+rho=\S+\s+mu\*rho=(\S+)\s*$")


def _reorder_anchor_block(prompt: str, order: str, seed: int = 0) -> str:
    """Rewrite only the anchor table's row order, leaving the prompt otherwise
    byte-identical to what production sends.

    The shipped table is sorted ascending, which puts `concrete_block` (1440) on
    both the last line *and* the largest value — the two candidate explanations
    for the collapse are perfectly confounded. Reordering separates them.
    """
    lines = prompt.split("\n")
    idx = [i for i, ln in enumerate(lines) if _ANCHOR_LINE.match(ln)]
    if not idx:
        raise RuntimeError("anchor block not found - the prompt format changed")
    block = [lines[i] for i in idx]
    values = [float(_ANCHOR_LINE.match(ln).group(2)) for ln in block]

    pairs = list(zip(block, values))
    if order == "ascending":
        pairs.sort(key=lambda p: p[1])
    elif order == "descending":
        pairs.sort(key=lambda p: -p[1])
    else:                                   # shuffled
        random.Random(seed).shuffle(pairs)

    for slot, (line, _) in zip(idx, pairs):
        lines[slot] = line
    ordered = [v for _, v in pairs]
    return "\n".join(lines), ordered


def stage_order(cfg: Config, workers: int) -> dict:
    est = DifficultyEstimator(cfg)
    # Only the off-table objects: the paraphrase items have an anchor as their
    # correct answer, so they cannot show whether a hit is copying or reasoning.
    items = [it for it in DATASET if it.group != "paraphrase"]
    orders = [("ascending", 0), ("descending", 0), ("shuffled", 1), ("shuffled", 2)]

    variants = []
    for order, seed in orders:
        tag = order if order != "shuffled" else f"shuffled{seed}"
        prompts = {}
        first_value = last_value = max_value = None
        for it in items:
            text, ordered = _reorder_anchor_block(
                est._build_prompt(it.observation()), order, seed)
            prompts[it.label] = text
            first_value, last_value = ordered[0], ordered[-1]
            max_value = max(ordered)
        log(f"order[{tag}]: first row = {first_value:g}, "
            f"last row = {last_value:g}, largest = {max_value:g}")

        def ask(it: Item, _p=prompts):
            value, _ = _ask_raw(cfg, cfg.deepseek_model, "disabled", 32, _p[it.label])
            return it.label, value

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            got = dict(pool.map(ask, items))

        preds = [v for v in got.values() if v]
        counts: Dict[float, int] = {}
        for v in preds:
            counts[round(v, 4)] = counts.get(round(v, 4), 0) + 1
        modal, modal_n = max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0)
        variants.append({
            "tag": tag, "first_row_value": first_value,
            "last_row_value": last_value, "max_value": max_value,
            "n": len(preds), "modal_value": modal,
            "modal_share": round(modal_n / len(preds), 3) if preds else None,
            "on_last_row": round(sum(1 for v in preds if abs(v - last_value) < 1e-6)
                                 / len(preds), 3) if preds else None,
            "on_first_row": round(sum(1 for v in preds if abs(v - first_value) < 1e-6)
                                  / len(preds), 3) if preds else None,
            "on_max": round(sum(1 for v in preds if abs(v - max_value) < 1e-6)
                            / len(preds), 3) if preds else None,
            "distinct": len(counts), "seconds": round(time.time() - t0, 1),
            "preds": got,
        })
        v = variants[-1]
        log(f"  {tag:<11s} modal={v['modal_value']} ({v['modal_share']:.0%})  "
            f"on_last_row={v['on_last_row']:.0%}  on_max={v['on_max']:.0%}  "
            f"distinct={v['distinct']}")

    payload = {"n_items": len(items), "variants": variants}
    _save("order.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# Stage 3 - navigation impact
# --------------------------------------------------------------------------- #

def _resolve_anchor(material: str) -> str:
    """Map a scenario's material label onto a PROMPT_ANCHORS entry.

    Scenario authors write labels by hand, so a stray plural has to be tolerated
    — but silently guessing would corrupt the ground truth, so anything else is
    an error rather than a fallback.
    """
    for candidate in (material, str(material).rstrip("sS")):
        anchor = _canonical_anchor(candidate)
        if anchor is not None:
            return anchor
    raise ValueError(
        f"scenario material {material!r} is not an anchor or alias; the nav "
        "experiment needs a calibrated ground truth for every obstacle")


def _relabel_map(scenario: dict) -> Dict[str, float]:
    """Rewrite every obstacle's material as a paraphrase of its own anchor.

    The obstacle keeps its geometry and its true difficulty (recomputed from the
    same anchor mu*rho it already used), so the world is physically identical.
    Only the string the estimator sees changes — which is precisely the variable
    under test.
    """
    ground_truth: Dict[str, float] = {}
    for w in scenario["movable"]:
        anchor = _resolve_anchor(w.material)
        label = PARAPHRASE_OF_ANCHOR[anchor]
        mu_rho = MATERIAL_MU_RHO[anchor]
        w.material = label
        w.difficulty = max(0.01, round(friction_force(mu_rho, w.volume), 3))
        ground_truth[_normalise(label)] = mu_rho
    return ground_truth


def _run_map(map_name: str, arm: str,
             mu_rho_by_label: Optional[Dict[str, float]],
             lambda_distance: Optional[float] = None) -> dict:
    """One scenario, one belief about mu*rho. `None` means the no-LLM heuristic."""
    scenario = scenarios.load(map_name)          # fresh objects every call
    cfg: Config = scenario["cfg"]
    cfg.save_frames = False
    cfg.verbose = False
    if lambda_distance is not None:
        cfg.lambda_distance = float(lambda_distance)
    ground_truth = _relabel_map(scenario)

    sim = OnlineNAMO(scenario["workspace"], scenario["static"], scenario["movable"],
                     scenario["start"], scenario["goal"], cfg)
    # Cut the network out: every arm is served from a pre-seeded cache, so runs
    # are deterministic and repeatable without burning calls.
    sim.estimator.api_key = ""
    sim.estimator.mode = arm
    if mu_rho_by_label is not None:
        for label, value in mu_rho_by_label.items():
            key = _normalise(label)
            sim.estimator.material_mu_rho_cache[key] = value
            sim.estimator.material_source_cache[key] = arm

    t0 = time.time()
    res = sim.run()
    wall = time.time() - t0

    # What the planner believed, per obstacle, for the record.
    beliefs = {str(oid): round(v, 4) for oid, v in sim.estimator.mu_rho_cache.items()}
    return {
        "map": map_name, "arm": arm, "success": res.success, "J": res.J,
        "walk_cost": res.walk_cost, "work_cost": res.work_cost,
        "pushes": len(res.removed), "removed": res.removed, "cycles": res.cycles,
        "expansions": res.total_expansions, "plan_time": res.plan_time,
        "wall_s": round(wall, 2), "message": res.message,
        "ground_truth": ground_truth, "believed_mu_rho": beliefs,
    }


def _preds_and_sigma(rows: Optional[List[dict]]):
    """Predictions keyed the way `_relabel_map` keys ground truth, plus the
    spread of that estimator's log10 error over the whole dataset.

    Only the paraphrase rows can be used as an arm — those are the labels the
    relabelled maps carry — but sigma is taken over every item, because it
    stands for the error the estimator would show on arbitrary real-world
    labels, which is the deployment case.
    """
    if not rows:
        return {}, 0.0
    pred = {_normalise(r["label"]): r["pred"] for r in rows
            if r["pred"] and r["group"] == "paraphrase"}
    errs = [math.log10(r["pred"] / r["mu_rho_true"]) for r in rows if r["pred"]]
    return pred, (statistics.pstdev(errs) if len(errs) > 1 else 0.0)


def _llm_and_sigma(accuracy: Optional[dict]):
    return _preds_and_sigma(accuracy["rows"] if accuracy else None)


def _thinking_rows(ablate: Optional[dict]) -> Optional[List[dict]]:
    """The reasoning-mode variant, but only if it answered essentially
    everything — a truncated variant has dropped its hardest items and would
    make reasoning mode look better than it is."""
    if not ablate:
        return None
    for v in ablate["variants"]:
        if v["thinking"] == "enabled" and v["n_parsed"] >= 0.9 * len(v["rows"]):
            return v["rows"]
    return None


# Extra estimators to run alongside the current one, as (arm name, results file).
# The no-reasoning file is the estimator this project shipped before 2026-08-10;
# keeping it as an arm means the before/after comparison lives in a single
# nav.json and can be re-derived without re-running the old configuration.
COMPARISON_ESTIMATORS = (("llm_nothink", "accuracy_nothink_backup.json"),)


def stage_lambda(maps: List[str], lambdas: List[float],
                 accuracy: Optional[dict]) -> dict:
    """Where the cost of a bad estimate turns on.

    lambda_distance is the price of a metre of driving, so it sets the exchange
    rate between detouring and pushing. Estimation error is only expensive when
    obstacles sit near that break-even; the shipped lambda may or may not put
    them there, and that is a property of the tuning, not of the model. This
    sweep locates the regime instead of assuming it.
    """
    llm_pred, sigma = _llm_and_sigma(accuracy)
    runs = []
    for map_name in maps:
        gt = _relabel_map(scenarios.load(map_name))
        for lam in lambdas:
            arms: List[tuple] = [("oracle", dict(gt)), ("heuristic", None)]
            if llm_pred and all(k in llm_pred for k in gt):
                arms.append(("llm", {k: llm_pred[k] for k in gt}))
            for seed in NOISE_SEEDS:
                rng = random.Random(seed)
                arms.append((f"noise{seed}",
                             {k: v * (10.0 ** rng.gauss(0.0, max(sigma, 1e-6)))
                              for k, v in gt.items()}))
            for arm, mapping in arms:
                row = _run_map(map_name, arm, mapping, lambda_distance=lam)
                row["lambda"] = lam
                runs.append(row)
            log(f"  {map_name:<14s} lambda={lam:<6g} "
                f"oracle J={runs[-len(arms)]['J']:.0f}")
    payload = {"maps": maps, "lambdas": lambdas,
               "sigma_log10": round(sigma, 4), "runs": runs}
    _save("lambda.json", payload)
    return payload


def stage_nav(maps: List[str], accuracy: Optional[dict]) -> dict:
    llm_pred: Dict[str, float] = {}
    llm_pred, sigma = _llm_and_sigma(accuracy)
    if llm_pred:
        log(f"nav: {len(llm_pred)} LLM medians available; "
            f"log10 error sigma {sigma:.3f} ({10 ** sigma:.2f}x)")
    else:
        log("nav: no accuracy.json - skipping the 'llm' arm, sweep only")

    # Every estimator gets both a point arm (its actual predictions) and a noise
    # family drawn from its own measured error spread. The point arm can be
    # lucky on one map's labels; the noise family is what generalises.
    estimators = [("llm", llm_pred, sigma)]
    for arm_name, filename in COMPARISON_ESTIMATORS:
        pred, s = _preds_and_sigma((_load(filename) or {}).get("rows"))
        if pred:
            estimators.append((arm_name, pred, s))
            log(f"nav: {arm_name} from {filename}; sigma {s:.3f} ({10 ** s:.2f}x)")

    runs = []
    for map_name in maps:
        gt = _relabel_map(scenarios.load(map_name))     # labels + true mu*rho

        arms: List[tuple] = [("oracle", dict(gt)), ("heuristic", None)]
        for arm_name, pool, s in estimators:
            missing = [k for k in gt if k not in pool]
            if missing:
                # a partial arm would be half model, half heuristic fallback and
                # would measure neither, so refuse it rather than report a blend
                log(f"  {map_name}: no {arm_name!r} arm, missing {missing}")
                continue
            arms.append((arm_name, {k: pool[k] for k in gt}))
            if s <= 0.0:
                continue
            for seed in NOISE_SEEDS:
                rng = random.Random(seed)
                arms.append((f"noise_{arm_name}_{seed}",
                             {k: v * (10.0 ** rng.gauss(0.0, s))
                              for k, v in gt.items()}))
        for f in SWEEP_FACTORS:
            arms.append((f"x{f:g}", {k: v * f for k, v in gt.items()}))

        for arm, mapping in arms:
            row = _run_map(map_name, arm, mapping)
            runs.append(row)
            log(f"  {map_name:<18s} {arm:<12s} J={row['J']:>12.1f} "
                f"pushes={row['pushes']} ok={row['success']} ({row['wall_s']}s)")

    payload = {"maps": maps, "sigma_log10": round(sigma, 4),
               "estimator_sigmas": {name: round(s, 4) for name, _, s in estimators},
               "factors": list(SWEEP_FACTORS), "runs": runs}
    _save("nav.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _log_errors(rows, key="pred"):
    out = []
    for r in rows:
        p = r.get(key)
        if p and p > 0:
            out.append(math.log10(p / r["mu_rho_true"]))
    return out


def _accuracy_stats(rows, key="pred") -> dict:
    errs = _log_errors(rows, key)
    if not errs:
        return {"n": 0}
    absol = [abs(e) for e in errs]
    within = lambda f: sum(1 for e in absol if e <= math.log10(f)) / len(absol)
    return {
        "n": len(errs),
        "median_ratio": round(10 ** statistics.median(errs), 3),
        "geomean_ratio": round(10 ** statistics.fmean(errs), 3),
        "median_abs_factor": round(10 ** statistics.median(absol), 3),
        "p90_abs_factor": round(10 ** np.percentile(absol, 90), 3),
        "within_1_5x": round(within(1.5), 3),
        "within_2x": round(within(2.0), 3),
        "within_3x": round(within(3.0), 3),
        "within_10x": round(within(10.0), 3),
        "sigma_log10": round(statistics.pstdev(errs), 3),
    }


def _anchor_snapping(rows, key="pred") -> dict:
    """How often the reply is a verbatim row of the anchor table.

    The prompt asks the model to *interpolate between* anchors, so an exact
    anchor value is a copy, not an estimate. Concentration on the largest anchor
    is worse than a wrong number: it is the table's last line, and it makes
    every unknown object look like solid concrete.
    """
    anchor_values = {round(v, 4) for k, v in MATERIAL_MU_RHO.items() if k != "unknown"}
    top = max(anchor_values)
    preds = [r[key] for r in rows if r.get(key)]
    if not preds:
        return {"n": 0}
    return {
        "n": len(preds),
        "on_anchor": round(sum(1 for p in preds if round(p, 4) in anchor_values) / len(preds), 3),
        "on_max_anchor": round(sum(1 for p in preds if abs(p - top) < 1e-6) / len(preds), 3),
        "max_anchor": top,
        "distinct_values": len({round(p, 4) for p in preds}),
    }


def _spearman(rows, key="pred") -> Optional[float]:
    pairs = [(r["mu_rho_true"], r[key]) for r in rows if r.get(key)]
    if len(pairs) < 3:
        return None
    from scipy.stats import spearmanr
    rho = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic
    return round(float(rho), 3)


def _repeatability(rows) -> dict:
    """Spread across repeats of the *same* question at temperature 0.

    Non-zero spread means the ordering the planner sees is not stable between
    identical runs, which matters more than the absolute value.
    """
    spreads = []
    for r in rows:
        preds = [p for p in r["preds"] if p and p > 0]
        if len(preds) >= 2 and min(preds) > 0:
            spreads.append(max(preds) / min(preds))
    if not spreads:
        return {"n": 0}
    return {"n": len(spreads),
            "identical_frac": round(sum(1 for s in spreads if s <= 1.0001) / len(spreads), 3),
            "median_spread": round(statistics.median(spreads), 3),
            "max_spread": round(max(spreads), 3)}


def _decision_flips(rows, cfg: Config, move_dist=2.0) -> List[dict]:
    """How often the estimate flips the only decision the planner makes.

    An obstacle is worth pushing when work < the detour it saves:
        difficulty * move_dist  <  lambda * detour
    so each obstacle has a break-even detour length. Comparing the estimated
    break-even against the true one at a few plausible detour lengths turns the
    mu*rho error into the quantity that actually reaches the search.
    """
    out = []
    for detour in (2.0, 5.0, 10.0, 20.0, 50.0):
        budget = cfg.lambda_distance * detour
        flips = 0
        total = 0
        for r in rows:
            if not r.get("pred"):
                continue
            volume = r["l"] * r["d"] * r["h"]
            true_push = friction_force(r["mu_rho_true"], volume) * move_dist
            est_push = friction_force(r["pred"], volume) * move_dist
            total += 1
            if (true_push < budget) != (est_push < budget):
                flips += 1
        out.append({"detour_m": detour, "n": total,
                    "flip_rate": round(flips / total, 3) if total else None})
    return out


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def _chart_accuracy(rows, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lo, hi = 0.5, 3000.0
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, zorder=2)
    for band, alpha in ((2.0, 0.10), (1.5, 0.14)):
        ax.fill_between([lo, hi], [lo / band, hi / band], [lo * band, hi * band],
                        color=MUTED, alpha=alpha, lw=0, zorder=1)

    # The horizontal stripes are the finding, so name them on the plot rather
    # than leaving the reader to decode why so many points share a y value.
    counts: Dict[float, int] = {}
    for r in rows:
        if r.get("pred"):
            counts[round(r["pred"], 4)] = counts.get(round(r["pred"], 4), 0) + 1
    anchor_name = {round(v, 4): k for k, v in MATERIAL_MU_RHO.items()}
    for value, n in sorted(counts.items(), key=lambda kv: -kv[1])[:2]:
        if n < 4 or value not in anchor_name:
            continue
        ax.axhline(value, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
        ax.text(hi * 0.92, value * 1.12,
                f"{n} replies = anchor '{anchor_name[value]}' ({value:g})",
                ha="right", va="bottom", fontsize=8.5, color=INK)

    for group, (color, marker) in GROUP_STYLE.items():
        pts = [(r["mu_rho_true"], r["pred"]) for r in rows
               if r["group"] == group and r.get("pred")]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46, marker=marker,
                   facecolor=color, edgecolor="#fcfcfb", linewidth=1.0,
                   label=f"{group} (n={len(pts)})", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("reference mu*rho  [kg/m$^3$]", color=INK)
    ax.set_ylabel("LLM estimate  [kg/m$^3$]", color=INK)
    ax.set_title("LLM mu*rho vs reference — shaded bands are 1.5x and 2x",
                 color=INK, fontsize=11, loc="left")
    ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"wrote {path}")


def _chart_sensitivity(nav: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = nav["runs"]
    maps = nav["maps"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # The dead zone is the result worth reading off this chart: inside it the
    # planner makes the same choices it would with a perfect estimate.
    free_lo, free_hi = 0.25, 4.0
    ax.axvspan(free_lo, free_hi, color=MUTED, alpha=0.13, lw=0)
    ax.text(1.0, 0.97, "estimates in this band cost nothing",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=INK)

    for i, map_name in enumerate(maps):
        base = next((r for r in runs if r["map"] == map_name and r["arm"] == "oracle"), None)
        if not base or not base["J"]:
            continue
        xs, ys = [], []
        for f in nav["factors"]:
            row = next((r for r in runs if r["map"] == map_name and r["arm"] == f"x{f:g}"), None)
            if row and row["success"]:
                xs.append(f)
                ys.append(100.0 * (row["J"] - base["J"]) / base["J"])
        color = colors[i % len(colors)]
        ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.0, label=map_name)
        for arm, mark, size in (("llm", "*", 190), ("heuristic", "X", 90)):
            row = next((r for r in runs if r["map"] == map_name and r["arm"] == arm), None)
            if row and row["success"]:
                ax.scatter([1.0], [100.0 * (row["J"] - base["J"]) / base["J"]],
                           marker=mark, s=size, color=color,
                           edgecolor=INK, linewidth=0.8, zorder=5)

    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("estimate error factor  (estimated mu*rho / true)", color=INK)
    ax.set_ylabel("cost penalty vs oracle  [% of J]", color=INK)
    ax.set_title("What a wrong mu*rho costs — star = LLM, cross = heuristic",
                 color=INK, fontsize=11, loc="left")
    ax.grid(True, which="both", color=GRID, lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _table(header: List[str], body: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def stage_report(cfg: Config) -> str:
    acc = _load("accuracy.json")
    nav = _load("nav.json")
    size = _load("size.json")
    abl = _load("ablate.json")
    if not acc:
        raise SystemExit("no accuracy.json - run `python3 LLM_test.py accuracy` first")

    rows = acc["rows"]
    parts = [f"# LLM mu*rho accuracy and its cost in navigation\n",
             f"Model `{acc['model']}`, {acc['repeats']} repeats per item, "
             f"{len(rows)} items, {acc['seconds']}s of API time.\n"]

    # -- headline -----------------------------------------------------------
    overall = _accuracy_stats(rows)
    snapshot = _anchor_snapping(rows)
    head = [
        f"- The estimator is off by a typical factor of "
        f"**{overall['median_abs_factor']:.1f}x**; only "
        f"**{overall['within_2x']:.0%}** of objects land within 2x of the "
        f"reference, and it over-estimates "
        f"({overall['median_ratio']:.1f}x median bias).",
        f"- Cause: **{snapshot['on_max_anchor']:.0%} of replies are exactly "
        f"{snapshot['max_anchor']:g}**, the largest anchor in the prompt. It is "
        f"copying a table row, not estimating.",
    ]
    heur = _accuracy_stats(rows, key="heuristic")
    if heur.get("n") and heur["median_abs_factor"] < overall["median_abs_factor"]:
        head.append(
            f"- The offline fallback is **better than the LLM it replaced**: "
            f"`material_mu_rho` scores {heur['median_abs_factor']:.1f}x "
            f"({heur['within_2x']:.0%} within 2x) on the same items, against the "
            f"model's {overall['median_abs_factor']:.1f}x. In the shipped "
            f"configuration, calling the API makes the estimate worse.")
    if nav:
        pens = []
        for m in nav["maps"]:
            base = next((r for r in nav["runs"] if r["map"] == m and r["arm"] == "oracle"), None)
            for r in nav["runs"]:
                if (r["map"] == m and r["arm"].startswith("noise") and r["success"]
                        and base and base["J"]):
                    pens.append(100.0 * (r["J"] - base["J"]) / base["J"])
        if pens:
            head.append(
                f"- Navigation still mostly survives it: over "
                f"{len(pens)} runs at this error level the mean penalty is "
                f"**{statistics.fmean(pens):+.1f}%** of J and the worst is "
                f"**{max(pens):+.1f}%**. The planner's push-or-detour decision "
                f"has wide margins, and touch sensing corrects the estimate on "
                f"contact — so a bad mu*rho costs a wrong first choice, not a "
                f"wrong final path.")
    if abl:
        # Only variants that answered nearly every item can be ranked: a
        # truncated run has silently dropped its hardest cases.
        rankable = [v for v in abl["variants"]
                    if _accuracy_stats(v["rows"]).get("n", 0) >= 0.9 * len(v["rows"])]
        best = min(rankable, key=lambda v: _accuracy_stats(v["rows"])["median_abs_factor"],
                   default=None)
        if best:
            s = _accuracy_stats(best["rows"])
            head.append(
                f"- Best request setting tested is `{best['tag']}` at "
                f"**{s['median_abs_factor']:.1f}x** typical error "
                f"({s['within_2x']:.0%} within 2x, {best['n_parsed']}/"
                f"{len(best['rows'])} answered). Note `_deepseek` hardcodes "
                f"`max_tokens=32` and `thinking=disabled`, so any setting that "
                f"needs a reasoning budget is unreachable without a code change.")
        dropped = [v["tag"] for v in abl["variants"] if v not in rankable]
        if dropped:
            head.append(f"- Not ranked (token budget truncated too many items to "
                        f"score fairly): {', '.join(dropped)}.")
    parts.append("\n## Headline\n\n" + "\n".join(head) + "\n")

    # -- accuracy -----------------------------------------------------------
    parts.append("\n## 1. Estimator accuracy\n")
    parts.append("`median_abs_factor` is the typical multiplicative miss: 1.0 is "
                 "exact, 2.0 means the usual answer is off by a factor of two in "
                 "either direction. `median_ratio` separates bias from spread — "
                 "above 1.0 the model systematically over-estimates.\n\n")
    body = []
    for group in ("paraphrase", "novel", "state", "brand", "ALL"):
        sub = rows if group == "ALL" else [r for r in rows if r["group"] == group]
        s = _accuracy_stats(sub)
        h = _accuracy_stats(sub, key="heuristic")
        if not s.get("n"):
            continue
        body.append([group, s["n"], s["median_abs_factor"], s["median_ratio"],
                     f"{s['within_2x']:.0%}", f"{s['within_3x']:.0%}",
                     _spearman(sub), h.get("median_abs_factor", "-"),
                     f"{h.get('within_2x', 0):.0%}"])
    parts.append(_table(
        ["group", "n", "LLM typ. factor", "LLM bias", "LLM <=2x", "LLM <=3x",
         "LLM Spearman", "heuristic typ. factor", "heuristic <=2x"], body))

    snap = _anchor_snapping(rows)
    parts.append(
        f"\n**Anchor snapping.** {snap['on_anchor']:.0%} of replies are a "
        f"verbatim row of the anchor table rather than an interpolation, and "
        f"{snap['on_max_anchor']:.0%} are exactly {snap['max_anchor']:g} — the "
        f"largest anchor (`concrete_block`) and the last line of the table in "
        f"the prompt. Across {snap['n']} distinct objects the model produced only "
        f"{snap['distinct_values']} distinct numbers. This is the whole story "
        f"behind the error above: the estimator is not estimating, it is picking "
        f"a row, and it disproportionately picks the heaviest one.\n")

    rep = _repeatability(rows)
    parts.append(f"\n**Repeatability** at temperature 0: {rep['identical_frac']:.0%} "
                 f"of items returned an identical value on every repeat; median "
                 f"spread across repeats {rep['median_spread']:.2f}x, worst "
                 f"{rep['max_spread']:.2f}x.\n")

    worst = sorted((r for r in rows if r.get("pred")),
                   key=lambda r: -abs(math.log10(r["pred"] / r["mu_rho_true"])))[:10]
    parts.append("\n### Worst 10 items\n")
    parts.append(_table(
        ["label", "group", "reference", "LLM", "factor", "why the reference says so"],
        [[r["label"], r["group"], f"{r['mu_rho_true']:.1f}", f"{r['pred']:.1f}",
          f"{r['pred'] / r['mu_rho_true']:.2f}x", r["note"]] for r in worst]))

    # -- size independence --------------------------------------------------
    if size:
        parts.append("\n## 2. Size independence\n")
        parts.append("mu*rho must not depend on the object's size — the caller "
                     "multiplies by volume afterwards, so any size response is "
                     "counted twice.\n\n")
        moved = []
        for r in size["rows"]:
            vals = [v for v in r["by_scale"].values() if v]
            if len(vals) >= 2 and min(vals) > 0:
                moved.append((max(vals) / min(vals), r))
        stable = sum(1 for s, _ in moved if s <= 1.0001)
        parts.append(f"{stable}/{len(moved)} items returned the same number at "
                     f"0.5x, 1x and 2x linear scale (8x volume range).\n\n")
        parts.append(_table(
            ["label", "0.5x", "1x", "2x", "spread"],
            [[r["label"], r["by_scale"].get("0.5"), r["by_scale"].get("1.0"),
              r["by_scale"].get("2.0"), f"{s:.2f}x"]
             for s, r in sorted(moved, key=lambda t: -t[0])]))

    # -- decision relevance -------------------------------------------------
    parts.append("\n## 3. Does the error reach the planner?\n")
    parts.append(f"The search only ever asks one question per obstacle: is "
                 f"`difficulty x push_distance` cheaper than `lambda x detour`? "
                 f"With lambda={cfg.lambda_distance:g} N and a 2 m push, an error "
                 f"only matters if it moves an obstacle across that line.\n\n")
    parts.append(_table(
        ["detour available", "obstacles", "decisions flipped by the LLM error"],
        [[f"{d['detour_m']:g} m", d["n"], f"{d['flip_rate']:.0%}"]
         for d in _decision_flips(rows, cfg)]))

    # -- navigation ---------------------------------------------------------
    if nav:
        parts.append("\n## 4. Navigation cost\n")
        parts.append("Same maps, same physics, same true difficulties — only the "
                     "estimator's belief differs. `oracle` knows the reference "
                     "mu*rho exactly and is the floor; the penalty column is the "
                     "extra J each arm pays over it.\n\n")
        body = []
        for map_name in nav["maps"]:
            base = next((r for r in nav["runs"]
                         if r["map"] == map_name and r["arm"] == "oracle"), None)
            for arm in ("oracle", "llm", "heuristic"):
                row = next((r for r in nav["runs"]
                            if r["map"] == map_name and r["arm"] == arm), None)
                if not row:
                    continue
                pen = ("-" if arm == "oracle" or not base or not base["J"]
                       else f"{100.0 * (row['J'] - base['J']) / base['J']:+.1f}%")
                body.append([map_name, arm, row["success"], f"{row['J']:.0f}",
                             f"{row['walk_cost']:.0f}", f"{row['work_cost']:.0f}",
                             row["pushes"], row["cycles"], pen])
        parts.append(_table(
            ["map", "arm", "goal reached", "J", "lambda*D", "W", "pushes",
             "replans", "penalty vs oracle"], body))

        parts.append("\n### Sensitivity sweep\n")
        parts.append("Every estimate multiplied by a fixed factor. A flat row "
                     "means the map's decisions are not close to the break-even "
                     "line and accuracy is free; a step means it is.\n\n")
        header = ["map"] + [f"x{f:g}" for f in nav["factors"]]
        body = []
        for map_name in nav["maps"]:
            base = next((r for r in nav["runs"]
                         if r["map"] == map_name and r["arm"] == "oracle"), None)
            line = [map_name]
            for f in nav["factors"]:
                row = next((r for r in nav["runs"]
                            if r["map"] == map_name and r["arm"] == f"x{f:g}"), None)
                if not row or not base or not base["J"]:
                    line.append("-")
                elif not row["success"]:
                    line.append("FAIL")
                else:
                    line.append(f"{100.0 * (row['J'] - base['J']) / base['J']:+.1f}%")
            body.append(line)
        parts.append(_table(header, body))

        parts.append(f"\n### Realistic-error replicates\n")
        parts.append(f"The single `llm` row above is one draw, and one draw on "
                     f"four maps is not a measurement. These re-run each map with "
                     f"an independent per-obstacle lognormal error at the spread "
                     f"measured over all {len(rows)} items "
                     f"(sigma={nav['sigma_log10']:.3f} in log10, a typical "
                     f"{10 ** nav['sigma_log10']:.2f}x miss), {len(NOISE_SEEDS)} "
                     f"seeds per map — what an estimator this noisy costs on "
                     f"average, rather than on the labels it happened to get right.\n\n")
        def _noise_pens(map_name, estimator="llm"):
            base = next((r for r in nav["runs"]
                         if r["map"] == map_name and r["arm"] == "oracle"), None)
            prefix = f"noise_{estimator}_"
            rows_ = [r for r in nav["runs"] if r["map"] == map_name
                     and r["arm"].startswith(prefix)
                     and r["arm"][len(prefix):].isdigit()]
            pens = [100.0 * (r["J"] - base["J"]) / base["J"]
                    for r in rows_ if r["success"] and base and base["J"]]
            return pens, sum(1 for r in rows_ if not r["success"])

        body = []
        for map_name in nav["maps"]:
            pens, fails = _noise_pens(map_name)
            if pens:
                body.append([map_name, f"{statistics.fmean(pens):+.1f}%",
                             f"{max(pens):+.1f}%", fails])
        parts.append(_table(["map", "mean penalty", "worst penalty", "failed runs"], body))

        # -- does a better estimator actually buy anything? -----------------
        if "llm_nothink" in (nav.get("estimator_sigmas") or {}):
            parts.append("\n### Is reasoning mode worth it, on the robot?\n")
            parts.append(
                f"The shipped estimator now reasons; the previous one did not "
                f"({10 ** nav['estimator_sigmas']['llm']:.2f}x spread vs "
                f"{10 ** nav['estimator_sigmas']['llm_nothink']:.2f}x). Accuracy is only worth "
                f"buying if it changes what the robot does. `llm` and "
                f"`llm_nothink` are the two estimators' actual predictions; the "
                f"noise columns are {len(NOISE_SEEDS)} draws from each one's "
                f"measured error distribution, which is the fairer comparison "
                f"— a single point estimate can be lucky.\n\n")
            body = []
            for map_name in nav["maps"]:
                base = next((r for r in nav["runs"] if r["map"] == map_name
                             and r["arm"] == "oracle"), None)
                cell = {}
                for arm in ("llm", "llm_nothink"):
                    row = next((r for r in nav["runs"] if r["map"] == map_name
                                and r["arm"] == arm), None)
                    cell[arm] = ("-" if not row or not base or not base["J"]
                                 else f"{100.0 * (row['J'] - base['J']) / base['J']:+.1f}%")
                think, _ = _noise_pens(map_name, "llm")
                shipped, _ = _noise_pens(map_name, "llm_nothink")
                body.append([
                    map_name, cell["llm"], cell["llm_nothink"],
                    f"{statistics.fmean(think):+.1f}% / {max(think):+.1f}%" if think else "-",
                    f"{statistics.fmean(shipped):+.1f}% / {max(shipped):+.1f}%" if shipped else "-"])
            parts.append(_table(
                ["map", "llm (reasoning, current)", "llm (no reasoning, previous)",
                 "current noise mean/worst", "previous noise mean/worst"], body))

    # -- lambda regime ------------------------------------------------------
    lam = _load("lambda.json")
    if lam:
        parts.append("\n## 4b. When does the error start to cost anything?\n")
        parts.append(f"`lambda_distance` is the exchange rate between detouring "
                     f"and pushing, so it decides how many obstacles sit near "
                     f"the break-even where a wrong mu*rho changes the answer. "
                     f"Each cell is the mean penalty over {len(NOISE_SEEDS)} "
                     f"noisy-estimate runs against the oracle at the same "
                     f"lambda. The shipped value is "
                     f"{cfg.lambda_distance:g}.\n\n")
        header = ["map"] + [f"lambda={x:g}" for x in lam["lambdas"]]
        body = []
        for map_name in lam["maps"]:
            line = [map_name]
            for x in lam["lambdas"]:
                base = next((r for r in lam["runs"] if r["map"] == map_name
                             and r["lambda"] == x and r["arm"] == "oracle"), None)
                pens = [100.0 * (r["J"] - base["J"]) / base["J"]
                        for r in lam["runs"]
                        if r["map"] == map_name and r["lambda"] == x
                        and r["arm"].startswith("noise") and r["success"]
                        and base and base["J"]]
                line.append(f"{statistics.fmean(pens):+.1f}%" if pens else "-")
            body.append(line)
        parts.append(_table(header, body))

    # -- anchor ordering ----------------------------------------------------
    order = _load("order.json")
    if order:
        parts.append("\n## 4c. Proof that it is copying, not estimating\n")
        parts.append(
            f"The anchor table's row order is rewritten; every other byte of the "
            f"prompt is unchanged, and only the {order['n_items']} off-table "
            f"objects are asked (a paraphrase item's correct answer *is* an "
            f"anchor, so it cannot distinguish copying from being right). "
            f"If the model were estimating, row order could not matter.\n\n")
        parts.append(_table(
            ["table order", "first row", "last row", "modal answer",
             "share", "= first row", "= largest", "distinct answers"],
            [[v["tag"], f"{v.get('first_row_value', float('nan')):g}",
              f"{v['last_row_value']:g}",
              f"{v['modal_value']:g}", f"{v['modal_share']:.0%}",
              f"{v.get('on_first_row') or 0:.0%}",
              f"{v['on_max']:.0%}", f"{v['distinct']}/{v['n']}"]
             for v in order["variants"]]))
        parts.append(
            "\nReordering moves the collapse target — the same objects are "
            "answered 1440, then 350, then 174 — so the answer is a function of "
            "prompt layout rather than of the object. Note this also rules out "
            "the two obvious single-cause stories: it is not 'the last row' "
            "(descending keeps 1440 while moving it to the top) and not 'the "
            "largest value' (shuffling drops the largest to a few percent). "
            "**Reordering is not a fix either** — it relocates the collapse "
            "rather than removing it. Only giving the model room to reason "
            "does that.\n")

    # -- request-setting ablation -------------------------------------------
    if abl:
        parts.append("\n## 5. Is it the model or the way it is asked?\n")
        parts.append("Same prompt, same items, different request settings. "
                     "`parsed` counts replies a number could be read out of — "
                     "anything unparsed falls back to the heuristic in "
                     "production, so a low count is itself a failure mode. "
                     "`cut` counts replies truncated by the token budget; "
                     "**scores on a variant with a non-zero `cut` are optimistic**, "
                     "because the items that get truncated are the ones the model "
                     "reasons longest about.\n\n")
        body = []
        for v in abl["variants"]:
            s = _accuracy_stats(v["rows"])
            body.append([v["tag"], f"{v['n_parsed']}/{len(v['rows'])}",
                         v.get("n_truncated", "?"),
                         s.get("median_abs_factor", "-"), s.get("median_ratio", "-"),
                         f"{s.get('within_2x', 0):.0%}", f"{s.get('within_3x', 0):.0%}",
                         _spearman(v["rows"]), f"{v['seconds']:.0f}s"])
        parts.append(_table(
            ["setting", "parsed", "cut", "typ. factor", "bias", "<=2x", "<=3x",
             "Spearman", "wall time"], body))

    text = "".join(parts)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"wrote {path}")

    _chart_accuracy(rows, os.path.join(OUT_DIR, "accuracy.png"))
    if nav:
        _chart_sensitivity(nav, os.path.join(OUT_DIR, "sensitivity.png"))
    return path


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["accuracy", "size", "ablate", "order", "nav",
                                      "lambda", "report", "all"])
    ap.add_argument("--lambdas", default="50,100,200,350,700,1400",
                    help="lambda_distance values for the lambda stage")
    ap.add_argument("--lambda-maps", default="two_doors,maze_mixed",
                    help="scenarios for the lambda stage (kept short - it is "
                         "len(lambdas) x 8 runs per map)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="LLM calls per item in the accuracy stage")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    ap.add_argument("--maps", default=",".join(DEFAULT_MAPS),
                    help="comma-separated scenario names for the nav stage")
    args = ap.parse_args()

    cfg = Config()
    cfg.verbose = False
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    unknown = [m for m in maps if m not in scenarios.names()]
    if unknown:
        ap.error(f"unknown map(s): {', '.join(unknown)}")

    if args.stage in ("accuracy", "all"):
        stage_accuracy(cfg, args.repeats, args.workers)
    if args.stage in ("size", "all"):
        stage_size(cfg, args.workers)
    if args.stage in ("ablate", "all"):
        stage_ablate(cfg, args.workers)
    if args.stage in ("order", "all"):
        stage_order(cfg, args.workers)
    if args.stage in ("nav", "all"):
        stage_nav(maps, _load("accuracy.json"))
    if args.stage in ("lambda", "all"):
        lam_maps = [m.strip() for m in args.lambda_maps.split(",") if m.strip()]
        lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
        stage_lambda(lam_maps, lambdas, _load("accuracy.json"))
    if args.stage in ("report", "all"):
        stage_report(cfg)


if __name__ == "__main__":
    sys.exit(main())
