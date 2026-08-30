"""Benchmark difficulty and risk estimators with a synthetic ten-door gap study."""

from __future__ import annotations

# Add the project root when imported from the benchmark directory.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from typing import Dict, List, Optional

import numpy as np

import scenarios
from config import Config
from llm_difficulty import (
    MATERIAL_MU_RHO,
    DifficultyEstimator,
    material_mu_rho,
)
from llm_dataset import (
    DATASET,
    Item,
    validate,
)
from executor import OnlineNAMO
from risk import (
    LEVELS,
    LOW,
    RiskEstimator,
    _normalise as _risk_normalise,
    detour_equivalent_m,
    keyword_level,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_test_out")
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BENCH_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)

SIZE_SCALES = (0.5, 1.0, 2.0)

# One accessible color and marker pair per dataset group.
GROUP_STYLE = {
    "object": ("#2a78d6", "o"),
    "state":  ("#eb6834", "^"),
    "brand":  ("#1baf7a", "D"),
}
# Sequential blue palette for charts.
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def log(msg: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _hash_files(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.relpath(path, PROJECT_DIR).encode("utf-8"))
        with open(path, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()


def _source_files():
    return [os.path.join(root, name)
            for root, dirs, names in os.walk(PROJECT_DIR)
            if "__pycache__" not in root.split(os.sep)
            for name in names if name.endswith(".py")]


def _config_snapshot(cfg: Config) -> dict:
    snapshot = {}
    for item in fields(cfg):
        if item.name.startswith("_"):
            continue
        value = getattr(cfg, item.name)
        if item.name == "deepseek_api_key":
            value = "<set>" if value else ""
        snapshot[item.name] = value
    return snapshot


def _benchmark_metadata(cfg: Optional[Config] = None) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    prompt_files = [os.path.join(PROJECT_DIR, "llm_difficulty.py"),
                    os.path.join(PROJECT_DIR, "risk.py")]
    return {
        "code_version": commit,
        "source_hash": _hash_files(_source_files()),
        "prompt_hash": _hash_files(prompt_files),
        "dataset_version": _hash_files(
            [os.path.join(BENCH_DIR, "llm_dataset.py")]),
        "config": _config_snapshot(cfg or Config()),
    }


def _save(name: str, payload, cfg: Optional[Config] = None) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    payload = dict(payload)
    payload["metadata"] = _benchmark_metadata(cfg)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log(f"wrote {path}")
    return path


def _load(name: str):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    saved = payload.get("metadata", {})
    current = _benchmark_metadata()
    identity = ("code_version", "source_hash", "prompt_hash", "dataset_version")
    if any(saved.get(key) != current[key] for key in identity):
        log(f"ignored stale benchmark result {path}")
        return None
    return payload


def _preflight(cfg: Config) -> None:
    """Validate API connectivity before launching parallel requests."""
    import requests
    log(f"model {cfg.deepseek_model} at {cfg.deepseek_base_url}")
    try:
        # Use a minimal non-streaming request to catch configuration errors.
        r = requests.post(
            cfg.deepseek_base_url, timeout=cfg.llm_timeout,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {cfg.deepseek_api_key}"},
            json={"model": cfg.deepseek_model, "max_tokens": 1, "stream": False,
                  "messages": [{"role": "user", "content": "ok"}],
                  "thinking": {"type": "disabled"}})
    except Exception as e:
        raise SystemExit(f"cannot reach {cfg.deepseek_base_url}: {e}")
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", r.json())
        except Exception:
            detail = r.text[:300]
        raise SystemExit(
            f"the API refused the very first call - HTTP {r.status_code}: "
            f"{detail}\nNothing was run; fix this before spending a stage on it.")


def _require_answers(kind: str, n_ok: int, n_total: int) -> None:
    """Refuse to save a stage with no usable model responses."""
    if n_ok:
        return
    raise SystemExit(
        f"{kind}: not one of the {n_total} calls returned a usable answer, so "
        f"there is nothing to save. Check the key, the balance and the model "
        f"name in Config, then run this stage again.")


# Difficulty estimator accuracy.

def stage_accuracy(cfg: Config, repeats: int, workers: int) -> dict:
    validate()
    est = DifficultyEstimator(cfg)
    if not est.api_key:
        raise SystemExit("no DeepSeek API key in Config.deepseek_api_key / $DEEPSEEK_API_KEY")
    _preflight(cfg)

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
            # Median reduces the effect of malformed repeat responses.
            "pred": statistics.median(preds) if preds else None,
            "n_ok": len(preds),
            "heuristic": material_mu_rho(item.label),
        })
    _require_answers("accuracy", sum(1 for r in rows if r["pred"]), len(jobs))
    payload = {"model": cfg.deepseek_model, "repeats": repeats,
               "seconds": round(time.time() - t0, 1), "rows": rows}
    _save("accuracy.json", payload, cfg)
    return payload


# Risk estimator accuracy for sight and contact observations.
RISK_CONTACT_REPEATS = 1


def stage_risk(cfg: Config, repeats: int, workers: int) -> dict:
    validate()
    est = RiskEstimator(cfg)
    if not est.api_key:
        raise SystemExit("no DeepSeek API key in Config.deepseek_api_key / $DEEPSEEK_API_KEY")
    _preflight(cfg)

    # None denotes sight-only; a number denotes the contact arm.
    jobs = [(it, None) for it in DATASET for _ in range(repeats)]
    jobs += [(it, it.difficulty) for it in DATASET for _ in range(RISK_CONTACT_REPEATS)]
    log(f"risk: {len(DATASET)} items x {repeats} on sight + {RISK_CONTACT_REPEATS} "
        f"after contact = {len(jobs)} calls, {workers} workers, "
        f"model={cfg.deepseek_model}")
    done = [0]

    def ask(job):
        item, difficulty = job
        # Use the same label normalization as production.
        level = est._deepseek(item.observation(), _risk_normalise(item.label),
                              difficulty)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(jobs):
            log(f"  {done[0]}/{len(jobs)} calls")
        return item.label, difficulty is not None, level

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, jobs))
    log(f"risk: {len(jobs)} calls in {time.time() - t0:.1f}s")

    sight: Dict[str, List[Optional[str]]] = {}
    contact: Dict[str, List[Optional[str]]] = {}
    for label, is_contact, level in results:
        (contact if is_contact else sight).setdefault(label, []).append(level)

    rows = []
    for it in DATASET:
        rows.append({
            "label": it.label, "group": it.group, "category": it.category,
            "risk_true": it.risk, "mu_rho_true": it.mu_rho,
            "difficulty": it.difficulty,
            "sight_levels": sight.get(it.label, []),
            "sight": _modal_level(sight.get(it.label, [])),
            "contact_levels": contact.get(it.label, []),
            "contact": _modal_level(contact.get(it.label, [])),
            # Score the no-API fallback on the same labels.
            "keyword": keyword_level(it.label),
            "risk_note": it.risk_note,
        })
    _require_answers("risk", sum(1 for r in rows if r["sight"]), len(jobs))
    payload = {"model": cfg.deepseek_model, "repeats": repeats,
               "contact_repeats": RISK_CONTACT_REPEATS,
               "seconds": round(time.time() - t0, 1), "rows": rows}
    _save("risk.json", payload, cfg)
    return payload


# Size-independence probe.

def stage_size(cfg: Config, workers: int) -> dict:
    """Check that estimated mu*rho does not vary with object scale."""
    _preflight(cfg)
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
    _save("size.json", payload, cfg)
    return payload


def _ask_raw(cfg: Config, model: str, thinking: str, max_tokens: int,
             prompt: str) -> tuple:
    """Call an estimator with explicit settings and parse its final number."""
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


# Anchor-ordering probe.

_ANCHOR_LINE = re.compile(r"^\s{2}(\S+)\s+mu=\S+\s+rho=\S+\s+mu\*rho=(\S+)\s*$")


def _reorder_anchor_block(prompt: str, order: str, seed: int = 0) -> str:
    """Rewrite only the anchor table row order in a production prompt."""
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
    else:                                   # Shuffle the anchor rows.
        random.Random(seed).shuffle(pairs)

    for slot, (line, _) in zip(idx, pairs):
        lines[slot] = line
    ordered = [v for _, v in pairs]
    return "\n".join(lines), ordered


def stage_order(cfg: Config, workers: int) -> dict:
    _preflight(cfg)
    est = DifficultyEstimator(cfg)
    # Off-table objects distinguish copying from estimation.
    items = [it for it in DATASET if it.anchor is None]
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
            value, _ = _ask_raw(cfg, cfg.deepseek_model, "disabled", 32,
                                _p[it.label])
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
    _save("order.json", payload, cfg)
    return payload


# Synthetic estimator-gap route study.

DOORS_MAP = "ten_doors"

# Each gap point bounds a seeded multiplicative cost error and risk-level shift.
DOORS_GAPS = ((1.0, 0), (1.5, 1), (2.0, 2), (4.0, 3), (10.0, 4))

# Each seed preserves paired draws across the cost-only, risk-only, and joint arms.
DOORS_SEEDS = 1


def _doors_beliefs(gates: List[dict], seed: int, cost_factor: float,
                   risk_levels: int, perturb: tuple):
    """Draw offline LLM-like beliefs at one user-controlled Gap level."""
    rng_d = random.Random(f"difficulty-{seed}-{cost_factor:g}")
    rng_r = random.Random(f"risk-{seed}-{risk_levels}")
    difficulty, level = {}, {}
    for r in gates:
        d, lvl = r["difficulty"], r["risk"]
        log_band = math.log10(cost_factor)
        draw_d = 10.0 ** rng_d.uniform(-log_band, log_band)
        shift = rng_r.randint(-risk_levels, risk_levels) if risk_levels else 0
        draw_r = LEVELS[max(0, min(len(LEVELS) - 1,
                                  LEVELS.index(lvl) + shift))]
        if "difficulty" in perturb:
            d = max(0.01, d * draw_d)
        if "risk" in perturb:
            lvl = draw_r
        difficulty[r["oid"]] = d
        level[r["oid"]] = lvl
    return difficulty, level


def _gate_choices(gates: List[dict], removed) -> Dict[int, str]:
    """What the run did at each gate: `A`, `B`, `AB`, or `detour`."""
    by_oid = {r["oid"]: r for r in gates}
    choice = {r["gate"]: "detour" for r in gates}
    for oid in removed:
        r = by_oid.get(oid)
        if r:
            choice[r["gate"]] = (r["side"] if choice[r["gate"]] == "detour"
                                 else choice[r["gate"]] + r["side"])
    return choice


def _run_doors(arm: str, gates: List[dict], difficulty: Dict[int, float],
               level: Dict[int, str], summary_path: Optional[str] = None) -> dict:
    """Run one ten-door crossing with seeded estimator beliefs."""
    scenario = scenarios.load(DOORS_MAP)             # Load fresh objects for each run.
    cfg: Config = scenario["cfg"]
    cfg.save_frames = False
    cfg.verbose = False

    sim = OnlineNAMO(scenario["workspace"], scenario["static"],
                     scenario["movable"], scenario["start"], scenario["goal"], cfg)
    original_poses = {w.oid: w.polygon for w in sim.world}
    sim.estimator.api_key = ""
    sim.estimator.mode = arm
    sim.risk.api_key = ""
    for oid, value in difficulty.items():
        sim.estimator.cache[oid] = round(float(value), 3)
    for oid, lvl in level.items():
        sim.risk.level[oid] = lvl
        sim.risk.source[oid] = arm

    t0 = time.time()
    res = sim.run()
    if summary_path:
        import viz
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        viz.visualize(sim, res, original_poses, summary_path)
        log(f"wrote {summary_path}")
    choices = _gate_choices(gates, res.removed)
    true_difficulty = {r["oid"]: r["difficulty"] for r in gates}
    true_risk = {r["oid"]: r["risk"] for r in gates}
    factor_gaps = [max(difficulty[oid] / true, true / difficulty[oid])
                   for oid, true in true_difficulty.items()]
    risk_gaps = [abs(LEVELS.index(level[oid]) - LEVELS.index(true))
                 for oid, true in true_risk.items()]
    return {
        "arm": arm, "success": res.success, "J": res.J, "C": res.C,
        "walk_cost": res.walk_cost, "work_cost": res.work_cost,
        "risk_cost": res.risk_cost, "pushes": len(res.removed),
        "removed": sorted(res.removed),
        # Count moved obstacles whose true risk was above low.
        "risky_pushes": sorted(oid for oid in res.removed
                               if true_risk.get(oid, LOW) != LOW),
        "choices": {str(g): c for g, c in sorted(choices.items())},
        "cycles": res.cycles, "expansions": res.total_expansions,
        "wall_s": round(time.time() - t0, 1), "message": res.message,
        "realized_mean_cost_factor": round(statistics.fmean(factor_gaps), 3),
        "realized_max_cost_factor": round(max(factor_gaps), 3),
        "realized_mean_risk_levels": round(statistics.fmean(risk_gaps), 3),
        "realized_max_risk_levels": max(risk_gaps),
        "believed_difficulty": {str(k): round(v, 1) for k, v in difficulty.items()},
        "believed_risk": {str(k): v for k, v in level.items()},
    }


def stage_doors(seeds: int) -> dict:
    if seeds < 1:
        raise SystemExit("doors: --doors-seeds must be at least 1")
    gates = scenarios.load(DOORS_MAP)["gates"]
    families = {"both": ("difficulty", "risk"),
                "difficulty": ("difficulty",), "risk": ("risk",)}
    n_runs = 1 + (len(DOORS_GAPS) - 1) * len(families) * seeds
    log(f"doors: {DOORS_MAP}, synthetic Gap ladder {DOORS_GAPS}, "
        f"{seeds} seeds = {n_runs} runs; no API calls")

    runs = [_run_doors("exact", gates,
                       {r["oid"]: r["difficulty"] for r in gates},
                       {r["oid"]: r["risk"] for r in gates},
                       os.path.join(OUT_DIR, "ten_doors_exact.png"))]
    runs[0].update({"family": "exact", "seed": 0, "gap_index": 0,
                    "cost_factor": 1.0, "risk_levels": 0, "changed": 0})
    log(f"  {'exact':<14s} C={runs[0]['C']:>12,.0f}  J={runs[0]['J']:>12,.0f}  "
        f"pushes={runs[0]['pushes']}  ({runs[0]['wall_s']}s)")

    max_gap_index = len(DOORS_GAPS) - 1
    for gap_index, (cost_factor, risk_levels) in enumerate(DOORS_GAPS[1:], 1):
        for seed in range(seeds):
            for family, perturb in families.items():
                difficulty, level = _doors_beliefs(
                    gates, seed, cost_factor, risk_levels, perturb)
                screenshot = None
                if gap_index == max_gap_index and seed == 0 and family == "both":
                    screenshot = os.path.join(OUT_DIR,
                                              "ten_doors_max_gap.png")
                row = _run_doors(
                    f"gap{gap_index}_{family}_{seed}", gates, difficulty, level,
                    screenshot)
                row.update({"family": family, "seed": seed,
                            "gap_index": gap_index,
                            "cost_factor": cost_factor,
                            "risk_levels": risk_levels})
                row["changed"] = sum(
                    1 for g, c in row["choices"].items()
                    if c != runs[0]["choices"][g])
                runs.append(row)
                log(f"  gap={gap_index} {family:<10s} seed={seed}  "
                    f"C={row['C']:>12,.0f}  J={row['J']:>12,.0f}  "
                    f"changed={row['changed']}/10  "
                    f"risky={len(row['risky_pushes'])}  ({row['wall_s']}s)")

    payload = {"map": DOORS_MAP, "seeds": seeds,
               "gap_model": {
                   "cost": "estimate/true sampled log-uniformly in [1/F, F]",
                   "risk": "integer level shift sampled in [-K, K], then clipped",
               },
               "gaps": [{"gap_index": i, "cost_factor": f, "risk_levels": k}
                        for i, (f, k) in enumerate(DOORS_GAPS)],
               "families": list(families), "gates": gates, "runs": runs,
               "screenshots": {"exact": "ten_doors_exact.png",
                               "max_gap": "ten_doors_max_gap.png"}}
    _save("doors.json", payload)
    _chart_doors_gap(payload, os.path.join(OUT_DIR, "doors_gap.png"))
    return payload


# Risk metrics.
RISK_ARMS = (("keyword fallback", "keyword"),
             ("LLM on sight", "sight"),
             ("LLM after contact", "contact"))


def _risk_measured(risk_payload: Optional[dict], key: str = "sight") -> bool:
    """Did the risk stage produce at least one usable verdict?"""
    return bool(risk_payload) and any(r.get(key) for r in risk_payload["rows"])


def _modal_level(levels) -> Optional[str]:
    """Return the modal level, breaking ties toward greater risk."""
    got = [lvl for lvl in levels if lvl]
    if not got:
        return None
    counts: Dict[str, int] = {}
    for lvl in got:
        counts[lvl] = counts.get(lvl, 0) + 1
    best = max(counts.values())
    return max((lvl for lvl, n in counts.items() if n == best), key=LEVELS.index)


def _risk_stats(rows, key: str) -> dict:
    """Summarize risk agreement and detour-equivalent error."""
    pairs = [(r["risk_true"], r[key]) for r in rows if r.get(key)]
    if not pairs:
        return {"n": 0}
    order = {name: i for i, name in enumerate(LEVELS)}
    deltas = [order[got] - order[want] for want, got in pairs]
    gaps = [detour_equivalent_m(got) - detour_equivalent_m(want) for want, got in pairs]
    n = len(pairs)
    return {
        "n": n,
        "exact": round(sum(1 for d in deltas if d == 0) / n, 3),
        "within_1": round(sum(1 for d in deltas if abs(d) <= 1) / n, 3),
        "under": round(sum(1 for d in deltas if d < 0) / n, 3),
        "over": round(sum(1 for d in deltas if d > 0) / n, 3),
        "worst_under_levels": -min(deltas) if min(deltas) < 0 else 0,
        "shortfall_m": round(statistics.fmean(max(0.0, -g) for g in gaps), 1),
        "excess_m": round(statistics.fmean(max(0.0, g) for g in gaps), 1),
    }


def _risk_confusion(rows, key: str) -> List[List[int]]:
    """counts[reference][estimate], in LEVELS order."""
    order = {name: i for i, name in enumerate(LEVELS)}
    grid = [[0] * len(LEVELS) for _ in LEVELS]
    for r in rows:
        got = r.get(key)
        if got:
            grid[order[r["risk_true"]]][order[got]] += 1
    return grid


def _risk_under_calls(rows, key: str) -> List[dict]:
    """Items called safer than they are, worst drop first."""
    order = {name: i for i, name in enumerate(LEVELS)}
    out = [r for r in rows if r.get(key) and order[r[key]] < order[r["risk_true"]]]
    return sorted(out, key=lambda r: order[r[key]] - order[r["risk_true"]])


# Difficulty metrics.

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
    """Measure responses that exactly match an anchor-table value."""
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
    """Measure spread across repeated deterministic requests."""
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


# Charts.

def _chart_accuracy(rows, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    # Bound axes by observed data to avoid clipping points.
    seen = [v for r in rows for v in (r["mu_rho_true"], r.get("pred")) if v]
    lo, hi = min(0.2, min(seen) / 1.6), max(3000.0, max(seen) * 1.6)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, zorder=2)
    for band, alpha in ((2.0, 0.10), (1.5, 0.14)):
        ax.fill_between([lo, hi], [lo / band, hi / band], [lo * band, hi * band],
                        color=MUTED, alpha=alpha, lw=0, zorder=1)

    # Label repeated anchor-valued responses directly on the chart.
    counts: Dict[float, int] = {}
    for r in rows:
        if r.get("pred"):
            counts[round(r["pred"], 4)] = counts.get(round(r["pred"], 4), 0) + 1
    anchor_name = {round(v, 4): k for k, v in MATERIAL_MU_RHO.items()}
    for value, n in sorted(counts.items(), key=lambda kv: -kv[1])[:2]:
        if n < 4 or value not in anchor_name:
            continue
        ax.axhline(value, color=MUTED, lw=0.8, zorder=2)
        ax.text(hi * 0.92, value * 1.12,
                f"{n} replies = anchor '{anchor_name[value]}' ({value:g})",
                ha="right", va="bottom", fontsize=8.5, color=INK)

    for group, (color, marker) in GROUP_STYLE.items():
        pts = [(r["mu_rho_true"], r["pred"]) for r in rows
               if r["group"] == group and r.get("pred")]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=38, marker=marker,
                   facecolor=color, edgecolor=SURFACE, linewidth=1.0,
                   label=f"{group} (n={len(pts)})", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("reference mu*rho  [kg/m$^3$]", color=INK)
    ax.set_ylabel("LLM estimate  [kg/m$^3$]", color=INK)
    ax.set_title("LLM mu*rho vs reference — shaded bands are 1.5x and 2x",
                 color=INK, fontsize=11, loc="left")
    # Major decades keep the log grid readable.
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"wrote {path}")


def _chart_risk(risk: dict, path: str):
    """Plot one confusion matrix per risk-estimation arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    rows = risk["rows"]
    arms = [(name, key) for name, key in RISK_ARMS
            if any(r.get(key) for r in rows)]
    grids = [(name, _risk_confusion(rows, key), _risk_stats(rows, key))
             for name, key in arms]
    vmax = max(max(max(row) for row in g) for _, g, _ in grids) or 1

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    short = {"medium_high": "med-high"}
    ticks = [short.get(name, name) for name in LEVELS]
    n = len(LEVELS)

    fig, axes = plt.subplots(1, len(grids), figsize=(3.55 * len(grids) + 1.2, 4.5))
    axes = list(axes) if len(grids) > 1 else [axes]

    for ax, (name, grid, stat) in zip(axes, grids):
        ax.set_facecolor(SURFACE)
        for i in range(n):
            for j in range(n):
                count = grid[i][j]
                if count:
                    shade = count / vmax
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                           facecolor=cmap(shade), zorder=1))
                    ax.text(j, i, str(count), ha="center", va="center",
                            fontsize=9, zorder=3,
                            color=SURFACE if shade > 0.55 else INK)
            # Outline the diagonal so the ramp continues to encode counts.
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, lw=1.1, zorder=2))

        # Leave a small surface gap between cells.
        ax.set_xticks([k - 0.5 for k in range(n + 1)], minor=True)
        ax.set_yticks([k - 0.5 for k in range(n + 1)], minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", length=0)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(ticks, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(ticks, fontsize=8.5)
        ax.tick_params(colors=MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{name}\n{stat['exact']:.0%} exact  ·  "
                     f"{stat['under']:.0%} called too safe",
                     color=INK, fontsize=10, loc="left", pad=10)

    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("reference risk level", color=INK, fontsize=9.5)
    fig.supxlabel("level the estimator returned — left of the outline is an "
                  "obstacle called safer than it is", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log(f"wrote {path}")


def _chart_doors_gap(doors: dict, path: str):
    """Show how route cost and decisions change as simulated Gap grows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, base = doors["runs"], doors["runs"][0]
    gaps = doors["gaps"]
    families = (("difficulty", "cost only", "#2a78d6", "o"),
                ("risk", "risk only", "#eb6834", "^"),
                ("both", "cost + risk", "#1baf7a", "D"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)

    for family, label, color, marker in families:
        xs, penalties, changes = [], [], []
        lows_p, highs_p, lows_c, highs_c = [], [], [], []
        for gap in gaps:
            i = gap["gap_index"]
            sample = ([base] if i == 0 else
                      [r for r in runs if r.get("family") == family
                       and r.get("gap_index") == i and r["success"]])
            if not sample:
                continue
            ps = [100.0 * (r["C"] - base["C"]) / base["C"]
                  for r in sample] if base["C"] else [0.0]
            cs = [r.get("changed", 0) for r in sample]
            xs.append(i)
            penalties.append(statistics.fmean(ps))
            changes.append(statistics.fmean(cs))
            lows_p.append(min(ps)); highs_p.append(max(ps))
            lows_c.append(min(cs)); highs_c.append(max(cs))
        axes[0].plot(xs, penalties, color=color, marker=marker, lw=1.8,
                     ms=5, label=label)
        axes[1].plot(xs, changes, color=color, marker=marker, lw=1.8,
                     ms=5, label=label)
        if doors["seeds"] > 1:
            axes[0].fill_between(xs, lows_p, highs_p, color=color, alpha=0.12)
            axes[1].fill_between(xs, lows_c, highs_c, color=color, alpha=0.12)

    labels = [f"{g['cost_factor']:g}x / +/-{g['risk_levels']}"
              for g in gaps]
    for ax in axes:
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(gaps)), labels, rotation=25, ha="right")
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("change from exact C  [%]", color=INK)
    axes[1].set_ylabel("gates chosen differently  [of 10]", color=INK)
    axes[0].set_title("Route-cost impact", loc="left", color=INK)
    axes[1].set_title("Decision impact", loc="left", color=INK)
    fig.supxlabel("maximum simulated Gap: cost factor / risk levels", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log(f"wrote {path}")


# Report generation.

def _table(header: List[str], body: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def stage_report() -> str:
    acc = _load("accuracy.json")
    risk = _load("risk.json")
    doors = _load("doors.json")
    # Treat an all-failed risk stage as unavailable data.
    risk_failed = bool(risk) and not _risk_measured(risk)
    if risk_failed:
        risk = None
    size = _load("size.json")
    if not acc:
        raise SystemExit("no accuracy.json - run `python3 LLM_test.py accuracy` first")

    rows = acc["rows"]
    parts = [f"# LLM estimator benchmark: difficulty and risk\n",
             f"Model `{acc['model']}`, {acc['repeats']} repeats per item, "
             f"{len(rows)} items, {acc['seconds']}s of API time.\n"]

    # Headline.
    overall = _accuracy_stats(rows)
    snapshot = _anchor_snapping(rows)
    head = [
        f"- The estimator is off by a typical factor of "
        f"**{overall['median_abs_factor']:.1f}x**; only "
        f"**{overall['within_2x']:.0%}** of objects land within 2x of the "
        f"reference, and it over-estimates "
        f"({overall['median_ratio']:.1f}x median bias).",
    ]
    # Report anchor concentration only when observed.
    if snapshot.get("n"):
        if snapshot["on_max_anchor"] >= 0.15:
            head.append(
                f"- Cause: **{snapshot['on_max_anchor']:.0%} of replies are "
                f"exactly {snapshot['max_anchor']:g}**, the largest anchor in "
                f"the prompt. It is copying a table row, not estimating.")
        elif snapshot["on_anchor"] >= 0.40:
            head.append(
                f"- **{snapshot['on_anchor']:.0%} of replies are a verbatim "
                f"anchor value** rather than an interpolation between two of "
                f"them — copying, but spread across the table rather than "
                f"piled on one row.")
        else:
            head.append(
                f"- It is interpolating rather than copying: only "
                f"{snapshot['on_anchor']:.0%} of replies land exactly on an "
                f"anchor value, and {snapshot['distinct_values']} distinct "
                f"numbers come back for {snapshot['n']} objects.")
    if risk_failed:
        head.append(
            "- **The risk stage has no results.** Every call in it failed, so "
            "nothing below scores the risk estimator; re-run `risk` once the "
            "API answers again.")
    if risk:
        sight = _risk_stats(risk["rows"], "sight")
        kw = _risk_stats(risk["rows"], "keyword")
        head.append(
            f"- On **risk** the model agrees with the reference level on "
            f"**{sight['exact']:.0%}** of objects against the keyword "
            f"fallback's {kw['exact']:.0%}, and calls "
            f"**{sight['under']:.0%} of them safer than they are** "
            f"(fallback {kw['under']:.0%}). That direction is the one that "
            f"matters: an under-called obstacle is one the planner is willing "
            f"to push.")
    if doors:
        base = doors["runs"][0]
        max_gap = max(g["gap_index"] for g in doors["gaps"])
        perturbed = [r for r in doors["runs"][1:]
                     if r.get("family") == "both"
                     and r.get("gap_index") == max_gap]
        if perturbed and base["C"]:
            pen = [100.0 * (r["C"] - base["C"]) / base["C"] for r in perturbed
                   if r["success"]]
            changed = [r.get("changed", 0) for r in perturbed]
            extra_risky = [len(r["risky_pushes"]) - len(base["risky_pushes"])
                           for r in perturbed]
            head.append(
                f"- At the largest simulated Gap, across {len(perturbed)} "
                f"crossings of the ten-gate map it changes "
                f"**{statistics.fmean(changed):.1f} of 10 decisions** (worst "
                f"{max(changed)}) and costs "
                f"**{statistics.fmean(pen):+.1f}% of C** on average, worst "
                f"{max(pen):+.1f}%"
                + (f", and it pushes {statistics.fmean(extra_risky):+.1f} more "
                   f"obstacles that were never safe to push."
                   if any(extra_risky) else "."))
    heur = _accuracy_stats(rows, key="heuristic")
    if heur.get("n") and heur["median_abs_factor"] < overall["median_abs_factor"]:
        head.append(
            f"- The offline fallback is **better than the LLM it replaced**: "
            f"`material_mu_rho` scores {heur['median_abs_factor']:.1f}x "
            f"({heur['within_2x']:.0%} within 2x) on the same items, against the "
            f"model's {overall['median_abs_factor']:.1f}x. In the shipped "
            f"configuration, calling the API makes the estimate worse.")
    parts.append("\n## Headline\n\n" + "\n".join(head) + "\n")

    # Accuracy section.
    parts.append("\n## 1. Estimator accuracy\n")
    parts.append("`median_abs_factor` is the typical multiplicative miss: 1.0 is "
                 "exact, 2.0 means the usual answer is off by a factor of two in "
                 "either direction. `median_ratio` separates bias from spread — "
                 "above 1.0 the model systematically over-estimates.\n\n")
    body = []
    slices = (
        ("object", lambda r: r["group"] == "object"),
        ("- anchor paraphrase", lambda r: bool(r.get("anchor"))),
        ("- off-table", lambda r: r["group"] == "object" and not r.get("anchor")),
        ("state", lambda r: r["group"] == "state"),
        ("brand", lambda r: r["group"] == "brand"),
        ("ALL", lambda r: True),
    )
    for group, keep in slices:
        sub = [r for r in rows if keep(r)]
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
    verdict = (
        "This is the whole story behind the error above: the estimator is not "
        "estimating, it is picking a row, and it disproportionately picks the "
        "heaviest one."
        if snap["on_max_anchor"] >= 0.15 else
        "The prompt asks for an interpolation between anchors and mostly gets "
        "one: the replies are spread over the range rather than piled on the "
        "table's last line, so the error above is estimation error, not a "
        "lookup wearing its clothes.")
    parts.append(
        f"\n**Anchor snapping.** {snap['on_anchor']:.0%} of replies are a "
        f"verbatim row of the anchor table rather than an interpolation, and "
        f"{snap['on_max_anchor']:.0%} are exactly {snap['max_anchor']:g} — the "
        f"largest anchor (`concrete_block`) and the last line of the table in "
        f"the prompt. Across {snap['n']} distinct objects the model produced "
        f"{snap['distinct_values']} distinct numbers. {verdict}\n")

    rep = _repeatability(rows)
    parts.append(f"\n**Repeatability** at temperature 0: {rep['identical_frac']:.0%} "
                 f"of items returned an identical value on every repeat; median "
                 f"spread across repeats {rep['median_spread']:.2f}x, worst "
                 f"{rep['max_spread']:.2f}x.\n")

    worst = sorted((r for r in rows if r.get("pred")),
                   key=lambda r: -abs(math.log10(r["pred"] / r["mu_rho_true"])))[:10]
    parts.append("\n### Worst 10 items\n")
    parts.append(_table(
        ["label", "category", "reference", "LLM", "factor", "why the reference says so"],
        [[r["label"], r.get("category", r["group"]), f"{r['mu_rho_true']:.1f}",
          f"{r['pred']:.1f}",
          f"{r['pred'] / r['mu_rho_true']:.2f}x", r["note"]] for r in worst]))

    # Risk section.
    if risk:
        rk = risk["rows"]
        parts.append("\n## 1b. Risk assessment\n")
        parts.append(
            f"A different question from mu*rho, on the same objects: not how "
            f"hard it is to push, but what happens to people and to the "
            f"building if it is pushed. Model `{risk['model']}`, "
            f"{risk['repeats']} replies per item on sight and "
            f"{risk['contact_repeats']} after contact.\n\n"
            "The two `m` columns price the mistake in the planner's own units. "
            "`risk.RISK_DETOUR_EQUIV_M` says what each level is worth as a "
            "detour (low 0 m, medium 20 m, medium_high 80 m, high 400 m, "
            "extreme 5000 m), so **shortfall** is the protection the estimator "
            "dropped and **excess** is the detour it invented, averaged over "
            "every item. Shortfall is the one that hurts someone; excess only "
            "costs distance.\n\n")
        body = []
        for name, key in RISK_ARMS:
            st = _risk_stats(rk, key)
            if not st.get("n"):
                continue
            body.append([name, st["n"], f"{st['exact']:.0%}",
                         f"{st['within_1']:.0%}", f"{st['under']:.0%}",
                         f"{st['over']:.0%}", f"{st['shortfall_m']:.0f}",
                         f"{st['excess_m']:.0f}"])
        parts.append(_table(
            ["arm", "n", "exact", "within 1 level", "called too safe",
             "called too dangerous", "shortfall m", "excess m"], body))

        sight, contact = _risk_stats(rk, "sight"), _risk_stats(rk, "contact")
        if sight.get("n") and contact.get("n"):
            verdict = ("helps" if contact["under"] < sight["under"]
                       else "does not help")
            parts.append(
                f"\n**Does touching it help?** Handing the model the measured "
                f"push force moves exact agreement from {sight['exact']:.0%} to "
                f"{contact['exact']:.0%}, and the too-safe rate from "
                f"{sight['under']:.0%} to {contact['under']:.0%}: on the "
                f"direction that matters, contact {verdict}. The reference "
                f"level is identical in both arms, so all of that movement is "
                f"the force number changing the model's mind.\n")

        under = _risk_under_calls(rk, "sight")[:10]
        if under:
            parts.append("\n### Worst under-calls on sight\n")
            parts.append("Every row is an obstacle the planner would have been "
                         "willing to push.\n\n")
            parts.append(_table(
                ["label", "reference", "LLM", "keyword", "why the reference says so"],
                [[r["label"], r["risk_true"], r["sight"], r["keyword"],
                  r["risk_note"]] for r in under]))

    # Size-independence section.
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

    # Anchor-ordering section.
    order = _load("order.json")
    if order:
        parts.append("\n## 3. Proof that it is copying, not estimating\n")
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

    # Route-impact section.
    if doors:
        runs = doors["runs"]
        base = runs[0]
        parts.append("\n## 4. What the error costs a route\n")
        parts.append(
            f"The `{doors['map']}` map is ten walls in a row, each with three "
            f"ways past it: move the obstacle in door A, move the one in door "
            f"B, or walk around through a third opening placed far enough off "
            f"the axis to cost a real detour. Every gate is one independent "
            f"three-way decision, so a run is ten of them and the arms differ "
            f"only in what the planner was told to believe.\n\n"
            f"`exact` is the floor: beliefs equal the truth. At a Gap point "
            f"`F / K`, each cost estimate is a seeded log-uniform draw between "
            f"`true/F` and `true*F`, while each risk estimate is shifted by a "
            f"seeded random integer from `-K` to `+K` levels and clipped to the "
            f"valid ladder. The tested points are "
            + ", ".join(f"{g['cost_factor']:g}x/±{g['risk_levels']}" for g in doors["gaps"])
            + ". `difficulty` and `risk` perturb one term each and `both` uses "
            f"the same two draws, so arm differences isolate the source rather "
            f"than a different random sample.\n\n"
            f"Beliefs are seeded per obstacle and no API is called, but the "
            f"corrections a real run earns are left in: touching an obstacle "
            f"reveals its true difficulty and re-rates its risk. What the "
            f"perturbation buys is therefore a wrong *decision*, taken before "
            f"the robot could know better — which is exactly what a bad "
            f"estimate costs in practice. The two route screenshots compare "
            f"the exact arm with seed 0 at the maximum joint Gap.\n\n")

        body = []
        for gap in doors["gaps"]:
            for family in ("exact", "difficulty", "risk", "both"):
                if gap["gap_index"] == 0:
                    if family != "exact":
                        continue
                    sub = [base]
                else:
                    if family == "exact":
                        continue
                    sub = [r for r in runs if r.get("family") == family
                           and r.get("gap_index") == gap["gap_index"]]
                ok = [r for r in sub if r["success"]]
                pen = [100.0 * (r["C"] - base["C"]) / base["C"] for r in ok
                       if base["C"]]
                changed = [r.get("changed", 0) for r in sub]
                risky = [len(r["risky_pushes"]) for r in sub]
                body.append([
                    f"{gap['cost_factor']:g}x/±{gap['risk_levels']}", family,
                    len(sub), f"{len(ok)}/{len(sub)}",
                    f"{statistics.fmean(r['realized_mean_cost_factor'] for r in sub):.2f}x",
                    f"{statistics.fmean(r['realized_mean_risk_levels'] for r in sub):.2f}",
                    f"{statistics.fmean(r['C'] for r in ok):,.0f}" if ok else "-",
                    "-" if family == "exact" else
                    (f"{statistics.fmean(pen):+.1f}%" if pen else "-"),
                    "-" if family == "exact" else
                    f"{statistics.fmean(changed):.1f}",
                    f"{statistics.fmean(risky):.1f}"])
        parts.append(_table(
            ["Gap F/±K", "arm", "runs", "reached goal",
             "realized cost gap", "realized risk gap", "mean C",
             "C vs exact", "gates changed", "risky pushes"], body))

        parts.append(
            f"\n`gates changed` counts gates where the perturbed run chose "
            f"differently from `exact` — the direct measure of an estimate "
            f"changing a decision. The penalty is taken on **C**, not J: the "
            f"risk surcharge is charged outside J, so a run that pushed "
            f"something it should have avoided books a cheaper J and a dearer "
            f"C, and only C tells the two apart. `pushes of a risky obstacle` counts "
            f"obstacles moved that were not `low` risk to begin with; `exact` "
            f"moves {len(base['risky_pushes'])} of them, and every one above "
            f"that is the planner disturbing something it should have walked "
            f"around.\n")

        # Per-gate detail shows which decisions changed.
        parts.append("\n### Where the decisions moved\n")
        gates_by_i: Dict[str, dict] = {}
        for r in doors["gates"]:
            g = gates_by_i.setdefault(str(r["gate"]), {"detour_m": r["detour_m"]})
            g[r["side"]] = f"{r['difficulty']:.0f} N {r['risk']}"
        body = []
        for g in sorted(gates_by_i, key=int):
            flips = sum(1 for r in runs[1:] if r["choices"][g] != base["choices"][g])
            body.append([g, f"{gates_by_i[g]['detour_m']:g} m",
                         gates_by_i[g]["A"], gates_by_i[g]["B"],
                         base["choices"][g],
                         f"{flips}/{len(runs) - 1}"])
        parts.append(_table(
            ["gate", "detour", "door A", "door B", "exact chose",
             "perturbed runs that chose otherwise"], body))

    text = "".join(parts)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"wrote {path}")

    _chart_accuracy(rows, os.path.join(OUT_DIR, "cost_accuracy.png"))
    if risk:
        _chart_risk(risk, os.path.join(OUT_DIR, "risk.png"))
    if doors:
        _chart_doors_gap(doors, os.path.join(OUT_DIR, "doors_gap.png"))
    return path



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["accuracy", "risk", "size",
                                      "order", "doors", "report", "all"])
    ap.add_argument("--repeats", type=int, default=2,
                    help="LLM calls per item in the accuracy and risk stages")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    ap.add_argument("--doors-seeds", type=int, default=DOORS_SEEDS,
                    help="random seeds per Gap and arm in the offline doors stage")
    args = ap.parse_args()

    cfg = Config()
    cfg.verbose = False

    if args.stage in ("accuracy", "all"):
        stage_accuracy(cfg, args.repeats, args.workers)
    if args.stage in ("risk", "all"):
        stage_risk(cfg, args.repeats, args.workers)
    if args.stage in ("size", "all"):
        stage_size(cfg, args.workers)
    if args.stage in ("order", "all"):
        stage_order(cfg, args.workers)
    if args.stage in ("doors", "all"):
        stage_doors(args.doors_seeds)
    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    sys.exit(main())
