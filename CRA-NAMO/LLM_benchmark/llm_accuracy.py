"""用合成十门误差研究评估难度和风险估计器。"""

from __future__ import annotations

# 从 benchmark 目录导入时加入项目根目录。
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import csv
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

import cost
import scenarios
from config import Config, validate_strategy
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

# 为每个数据组指定一组易读的颜色和标记。
GROUP_STYLE = {
    "object": ("#2a78d6", "o"),
    "state":  ("#eb6834", "^"),
    "brand":  ("#1baf7a", "D"),
}
# 图表使用连续蓝色调色板。
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def log(msg: str) -> None:
    """输出带时间戳的进度消息。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt_pct(value) -> str:
    """把可能缺失的百分比格式化成表格单元。"""
    return "-" if value is None else f"{value:+.1f}%"


def _fmt_num(value, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


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
    """启动并行请求前校验 API 连通性。"""
    import requests
    log(f"model {cfg.deepseek_model} at {cfg.deepseek_base_url}")
    try:
        # 使用最小非流式请求捕获配置错误。
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
    """拒绝保存没有可用模型响应的阶段结果。"""
    if n_ok:
        return
    raise SystemExit(
        f"{kind}: not one of the {n_total} calls returned a usable answer, so "
        f"there is nothing to save. Check the key, the balance and the model "
        f"name in Config, then run this stage again.")


# 难度估计器准确率。

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
            # 中位数可降低格式错误的重复响应影响。
            "pred": statistics.median(preds) if preds else None,
            "n_ok": len(preds),
            "heuristic": material_mu_rho(item.label),
        })
    _require_answers("accuracy", sum(1 for r in rows if r["pred"]), len(jobs))
    payload = {"model": cfg.deepseek_model, "repeats": repeats,
               "seconds": round(time.time() - t0, 1), "rows": rows}
    _save("accuracy.json", payload, cfg)
    return payload


# 视觉和接触观测下的风险估计准确率。
RISK_CONTACT_REPEATS = 1


def stage_risk(cfg: Config, repeats: int, workers: int) -> dict:
    validate()
    est = RiskEstimator(cfg)
    if not est.api_key:
        raise SystemExit("no DeepSeek API key in Config.deepseek_api_key / $DEEPSEEK_API_KEY")
    _preflight(cfg)

    # None 表示仅视觉；数值表示接触阶段。
    jobs = [(it, None) for it in DATASET for _ in range(repeats)]
    jobs += [(it, it.difficulty) for it in DATASET for _ in range(RISK_CONTACT_REPEATS)]
    log(f"risk: {len(DATASET)} items x {repeats} on sight + {RISK_CONTACT_REPEATS} "
        f"after contact = {len(jobs)} calls, {workers} workers, "
        f"model={cfg.deepseek_model}")
    done = [0]

    def ask(job):
        item, difficulty = job
        # 使用与生产代码相同的标签归一化。
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
            # 在相同标签上评估无 API 后备方案。
            "keyword": keyword_level(it.label),
            "risk_note": it.risk_note,
        })
    _require_answers("risk", sum(1 for r in rows if r["sight"]), len(jobs))
    payload = {"model": cfg.deepseek_model, "repeats": repeats,
               "contact_repeats": RISK_CONTACT_REPEATS,
               "seconds": round(time.time() - t0, 1), "rows": rows}
    _save("risk.json", payload, cfg)
    return payload


# 尺寸独立性检查。

def stage_size(cfg: Config, workers: int) -> dict:
    """检查估计的 mu*rho 是否随物体缩放而变化。"""
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
    """按显式设置调用估计器，并解析最终数值。"""
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


# 锚点顺序检查。

_ANCHOR_LINE = re.compile(r"^\s{2}(\S+)\s+mu=\S+\s+rho=\S+\s+mu\*rho=(\S+)\s*$")


def _reorder_anchor_block(prompt: str, order: str, seed: int = 0) -> str:
    """只改写生产提示词中的锚点表行顺序。"""
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
    else:                                   # 打乱锚点行顺序。
        random.Random(seed).shuffle(pairs)

    for slot, (line, _) in zip(idx, pairs):
        lines[slot] = line
    ordered = [v for _, v in pairs]
    return "\n".join(lines), ordered


# 回答要留够篇幅；截断的回答解析出来是草稿数字，会伪装成“照抄锚点”。
ORDER_MAX_TOKENS = 512


def stage_order(cfg: Config, workers: int) -> dict:
    _preflight(cfg)
    est = DifficultyEstimator(cfg)
    # 表外物体用于区分复制和估计。
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
            # 32 个 token 会把每次回答都截断，解析到的是草稿里的数字而不是结论。
            value, why = _ask_raw(cfg, cfg.deepseek_model, "disabled",
                                  ORDER_MAX_TOKENS, _p[it.label])
            return it.label, (value if why != "length" else None)

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


# 合成估计误差路线研究。
#
# 十门地图上每道门是一个独立的三选一：搬 A、搬 B 或绕行。这里不调 API，而是
# 直接把「估计值 / 真实值 = F」写进 belief，再看规划器的选择和真实代价如何变化。
# F 是构造出来的而不是采样出来的，所以横轴就是真实的 Gap 比例，「多大的 Gap
# 换来多少 C」可以直接读出来。

DOORS_MAP = "ten_doors"

# 代价 Gap 梯度：belief 里的难度是真实值的 F 倍或 1/F 倍。
DOORS_COST_RATIOS = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0)
# 风险 Gap 梯度：belief 里的等级相对真实等级偏移 K 级，越界后截断。
DOORS_RISK_SHIFTS = (0, 1, 2, 3, 4)
# 联合梯度：代价比例与风险偏移成对增大。
DOORS_JOINT_LADDER = ((1.5, 1), (2.0, 2), (4.0, 3), (10.0, 4))
# 冒烟测试用的短梯度。
DOORS_QUICK_COST = (1.0, 1.5, 2.0, 4.0, 10.0)
DOORS_QUICK_RISK = (0, 2, 4)
DOORS_QUICK_JOINT = ((2.0, 2), (10.0, 4))

# 三个方向。`under` 把每个估计都往「更便宜、更安全」偏，`over` 反之，
# `mixed` 给每个障碍物一个固定的随机方向。
DOORS_DIRECTIONS = ("under", "over", "mixed")
DIRECTION_LABEL = {"under": "under-estimate", "over": "over-estimate",
                   "mixed": "random direction", "exact": "exact"}
DIRECTION_STYLE = {"under": ("#eb6834", "v"), "over": ("#2a78d6", "^"),
                   "mixed": ("#1baf7a", "o")}
FAMILY_LABEL = {"cost": "cost only", "risk": "risk only", "joint": "cost + risk"}

# `mixed` 每个 Gap 点的随机方向重复数；`under` 和 `over` 是确定性的，各一次。
DOORS_SEEDS = 3
# 校准阶段用来禁掉一个选项的 belief 难度，高于 Config.robot_max_push_force。
DOORS_FORBID_N = 1e7


def _doors_signs(gates: List[dict], seed: int) -> Dict[int, int]:
    """给每个障碍物一个只取决于 seed 的偏离方向，与 Gap 大小无关。

    方向固定之后，同一个 seed 在梯度上的各点就是配对样本：曲线的起伏来自 F
    变大，而不是来自换了一组随机数。旧实现按 (seed, F) 建流，梯度点之间互相
    独立，于是 Gap 的影响和采样噪声混在一起。
    """
    rng = random.Random(f"ten-doors-signs-{seed}")
    return {r["oid"]: (1 if rng.random() < 0.5 else -1) for r in gates}


def _doors_beliefs(gates: List[dict], cost_ratio: float, risk_shift: int,
                   direction: str, signs: Dict[int, int]):
    """按给定的 Gap 比例构造离线 belief。"""
    difficulty, level = {}, {}
    fixed = {"under": -1, "over": 1}.get(direction)
    for r in gates:
        sign = signs[r["oid"]] if fixed is None else fixed
        difficulty[r["oid"]] = float(r["difficulty"]) * (cost_ratio ** sign)
        index = LEVELS.index(r["risk"]) + sign * risk_shift
        level[r["oid"]] = LEVELS[max(0, min(len(LEVELS) - 1, index))]
    return difficulty, level


def _gate_choices(gates: List[dict], removed) -> Dict[int, str]:
    """记录每个门的选择：`A`、`B`、`AB` 或 `detour`。"""
    by_oid = {r["oid"]: r for r in gates}
    choice = {r["gate"]: "detour" for r in gates}
    for oid in removed:
        r = by_oid.get(oid)
        if r:
            choice[r["gate"]] = (r["side"] if choice[r["gate"]] == "detour"
                                 else choice[r["gate"]] + r["side"])
    return choice


def _run_doors(job: dict) -> dict:
    """按给定 belief 跑一次十门穿越，并按真实风险等级重新计价 C。

    参数是一个可 pickle 的字典，因此这一步可以放进进程池并行。
    """
    gates = job["gates"]
    difficulty = {int(k): float(v) for k, v in job["difficulty"].items()}
    level = {int(k): str(v) for k, v in job["level"].items()}

    scenario = scenarios.load(DOORS_MAP,             # 每次运行加载全新的物体。
                              closed_bypasses=tuple(job.get("closed_bypasses", ())))
    cfg: Config = scenario["cfg"]
    cfg.save_frames = False
    cfg.verbose = False

    sim = OnlineNAMO(scenario["workspace"], scenario["static"],
                     scenario["movable"], scenario["start"], scenario["goal"], cfg)
    original_poses = {w.oid: w.polygon for w in sim.world}
    sim.estimator.api_key = ""
    sim.estimator.mode = "offline-gap"
    sim.risk.api_key = ""
    for oid, value in difficulty.items():
        sim.estimator.cache[oid] = round(value, 3)
    for oid, lvl in level.items():
        sim.risk.level[oid] = lvl
        sim.risk.source[oid] = "offline-gap"

    t0 = time.time()
    res = sim.run()
    wall = round(time.time() - t0, 1)

    true_difficulty = {r["oid"]: r["difficulty"] for r in gates}
    true_risk = {r["oid"]: r["risk"] for r in gates}
    moved = sorted(set(res.removed))
    # 执行器按 belief 的等级收风险附加项，所以低估风险的运行会白拿一笔折扣，
    # 甚至比 exact 还便宜。这里按真实等级重算，C_true 才是与 belief 无关的
    # 裁判分；res.C 作为 C_believed 保留，用来显示这笔折扣有多大。
    risk_true = sum(cost.risk_cost(cfg, true_risk.get(oid, LOW)) for oid in moved)
    c_true = res.C - res.risk_cost + risk_true

    if job.get("screenshot"):
        import viz
        path = os.path.join(OUT_DIR, job["screenshot"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        viz.visualize(sim, res, original_poses, path, benchmark_info={
            "strategy": job["label"], "llm_modes": "offline gap injection",
            "wall_time_seconds": wall,
            "status": ("reached goal" if res.success else "failed")})
        log(f"wrote {path}")

    factor_gaps = [max(difficulty[oid] / true, true / difficulty[oid])
                   for oid, true in true_difficulty.items()]
    risk_gaps = [abs(LEVELS.index(level[oid]) - LEVELS.index(true))
                 for oid, true in true_risk.items()]
    return {
        "label": job["label"], "family": job["family"],
        "direction": job["direction"], "seed": job["seed"],
        "cost_ratio": job["cost_ratio"], "risk_shift": job["risk_shift"],
        "closed_bypasses": tuple(job.get("closed_bypasses", ())),
        "success": res.success,
        # 与 belief 无关的裁判分，报告里的 ΔC 全部基于它。
        "C_true": round(c_true, 4),
        "C_believed": res.C,
        "risk_cost_true": round(risk_true, 4),
        "risk_cost_believed": res.risk_cost,
        "J": res.J, "walk_cost": res.walk_cost, "work_cost": res.work_cost,
        "pushes": len(moved), "removed": moved,
        # 真实风险高于 low 却仍被搬移的障碍物。
        "risky_pushes": sorted(oid for oid in moved
                               if true_risk.get(oid, LOW) != LOW),
        "choices": {str(g): c for g, c in sorted(
            _gate_choices(gates, res.removed).items())},
        "cycles": res.cycles, "expansions": res.total_expansions,
        "wall_s": wall, "message": res.message,
        "realized_mean_cost_factor": round(statistics.fmean(factor_gaps), 3),
        "realized_max_cost_factor": round(max(factor_gaps), 3),
        "realized_mean_risk_levels": round(statistics.fmean(risk_gaps), 3),
        "realized_max_risk_levels": max(risk_gaps),
        "believed_difficulty": {str(k): round(v, 1) for k, v in difficulty.items()},
        "believed_risk": {str(k): v for k, v in level.items()},
    }


def _run_doors_batch(jobs: List[dict], workers: int) -> List[dict]:
    """顺序或并行跑完一批运行；两种路径结果完全相同。"""
    log(f"doors: {len(jobs)} runs on {workers} worker(s), no API calls")
    t0 = time.time()
    rows: List[dict] = []

    def drain(stream):
        for i, row in enumerate(stream, 1):
            rows.append(row)
            log(f"  {i}/{len(jobs)} {row['label']:<28s} "
                f"C={row['C_true']:>11,.0f}  pushes={row['pushes']:>2d}  "
                f"risky={len(row['risky_pushes'])}  ({row['wall_s']}s)")

    if workers <= 1:
        drain(_run_doors(job) for job in jobs)
    else:
        # 每次运行都是独立且带种子的，并行不改变任何一个结果。
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            drain(pool.map(_run_doors, jobs))
    log(f"doors: {len(jobs)} runs in {time.time() - t0:.1f}s")
    return rows


def _doors_rungs(cost_ratios, risk_shifts, joint) -> Dict[str, list]:
    """把三个实验族都写成 (cost_ratio, risk_shift) 梯级列表。"""
    return {
        "cost": [(float(f), 0) for f in cost_ratios if f > 1.0],
        "risk": [(1.0, int(k)) for k in risk_shifts if k > 0],
        "joint": [(float(f), int(k)) for f, k in joint if f > 1.0 or k > 0],
    }


def _doors_jobs(gates, rungs: Dict[str, list], seeds: int,
                shots: bool) -> List[dict]:
    """列出整个扫描要跑的全部运行，并挑出要出图的那几次。"""
    exact_difficulty = {r["oid"]: r["difficulty"] for r in gates}
    exact_level = {r["oid"]: r["risk"] for r in gates}
    jobs = [{"label": "exact", "gates": gates, "family": "exact",
             "direction": "exact", "seed": 0, "cost_ratio": 1.0,
             "risk_shift": 0, "difficulty": exact_difficulty,
             "level": exact_level,
             "screenshot": "ten_doors_exact.png" if shots else None}]

    # 每个族最大的一级各出一张图：一张最乐观、一张最悲观。
    shot_at = {}
    if shots:
        for family, ladder in rungs.items():
            if ladder:
                shot_at[(family, ladder[-1], "under")] = \
                    f"ten_doors_{family}_under.png"
        if rungs["cost"]:
            shot_at[("cost", rungs["cost"][-1], "over")] = \
                "ten_doors_cost_over.png"

    signs = {seed: _doors_signs(gates, seed) for seed in range(seeds)}
    for family, ladder in rungs.items():
        for rung in ladder:
            ratio, shift = rung
            for direction in DOORS_DIRECTIONS:
                for seed in (range(seeds) if direction == "mixed" else (0,)):
                    difficulty, level = _doors_beliefs(
                        gates, ratio, shift, direction, signs[seed])
                    tag = f"{family}_{ratio:g}x_{shift}lvl_{direction}"
                    jobs.append({
                        "label": tag if direction != "mixed" else f"{tag}_s{seed}",
                        "gates": gates, "family": family,
                        "direction": direction, "seed": seed,
                        "cost_ratio": ratio, "risk_shift": shift,
                        "difficulty": difficulty, "level": level,
                        "screenshot": shot_at.get((family, rung, direction))})
    return jobs


def _doors_finish(runs: List[dict], base: dict) -> None:
    """把每次运行折算成相对 exact 的差额，就地写回。"""
    for row in runs:
        ref = base["C_true"]
        row["delta_C_pct"] = (round(100.0 * (row["C_true"] - ref) / ref, 3)
                              if ref else None)
        row["delta_C"] = round(row["C_true"] - ref, 1)
        row["delta_walk"] = round(row["walk_cost"] - base["walk_cost"], 1)
        row["delta_work"] = round(row["work_cost"] - base["work_cost"], 1)
        row["delta_risk"] = round(row["risk_cost_true"]
                                  - base["risk_cost_true"], 1)
        # 低估风险换来的账面折扣：规划器自己算的 C 比裁判分便宜多少。
        row["risk_discount"] = round(row["C_true"] - row["C_believed"], 1)
        row["changed_gates"] = sorted(
            (g for g, c in row["choices"].items() if c != base["choices"][g]),
            key=int)
        row["changed"] = len(row["changed_gates"])
        row["extra_risky"] = (len(row["risky_pushes"])
                              - len(base["risky_pushes"]))


def _check_exact_balance(base: dict, n_gates: int) -> None:
    """exact 一边倒时提醒：这样的地图问不出估计误差的影响。

    十道门全部绕行或全部搬走时，改变 belief 也没有第二个选项可以换，扫描
    量到的只会是零。`doors-calibrate` 的平衡点难度就是用来修这件事的。
    """
    pushed = sum(1 for choice in base["choices"].values() if choice != "detour")
    share = base["work_cost"] / base["C_true"] if base["C_true"] else 0.0
    if 2 <= pushed <= n_gates - 2 and share >= 0.05:
        return
    log(f"doors: WARNING the exact run pushes {pushed} of {n_gates} gates and "
        f"spends {share:.1%} of C on manipulation. With the decisions this "
        f"lopsided there is little for an estimate error to move, so the "
        f"ladder below will read close to flat. Run `doors-calibrate` and "
        f"retune the difficulties in scenarios/ten_doors.py around the "
        f"break-even values it reports.")


def _check_ladder_headroom(gates, rungs: Dict[str, list], cfg: Config) -> None:
    """梯度顶端高估出来的难度不能越过机器人的推力上限。

    越过之后 `search._off_limits` 会直接禁掉这次搬移，量到的就成了那条禁令，
    而不是估计误差本身。这里只在真的越界时提醒，不改动梯度。
    """
    limit = cfg.robot_max_push_force
    if limit <= 0:
        return
    top = max([f for ladder in rungs.values() for f, _ in ladder] or [1.0])
    worst = max(r["difficulty"] for r in gates) * top
    if worst > limit:
        log(f"doors: WARNING at {top:g}x the believed difficulty reaches "
            f"{worst:,.0f} N, past the robot's {limit:,.0f} N limit - those "
            f"gates measure a ban on pushing, not an estimate gap. Lower the "
            f"ladder or raise Config.robot_max_push_force.")


def stage_doors(seeds: int, workers: int, quick: bool, shots: bool) -> dict:
    if seeds < 1:
        raise SystemExit("doors: --doors-seeds must be at least 1")
    scenario = scenarios.load(DOORS_MAP)
    gates, summary = scenario["gates"], scenario["gate_summary"]
    rungs = _doors_rungs(
        DOORS_QUICK_COST if quick else DOORS_COST_RATIOS,
        DOORS_QUICK_RISK if quick else DOORS_RISK_SHIFTS,
        DOORS_QUICK_JOINT if quick else DOORS_JOINT_LADDER)
    _check_ladder_headroom(gates, rungs, scenario["cfg"])
    jobs = _doors_jobs(gates, rungs, seeds, shots)
    log(f"doors: {DOORS_MAP}, cost ladder "
        + ", ".join(f"{f:g}x" for f, _ in rungs["cost"])
        + f"; risk ladder " + ", ".join(f"+/-{k}" for _, k in rungs["risk"])
        + f"; {seeds} random-direction seeds per point")

    runs = _run_doors_batch(jobs, workers)
    base = runs[0]
    _doors_finish(runs, base)
    _check_exact_balance(base, len(summary))

    payload = {
        "map": DOORS_MAP, "seeds": seeds,
        "gap_model": {
            "cost": "believed difficulty = true * F (over) or true / F (under); "
                    "in the mixed arm the direction is drawn once per obstacle "
                    "and held across the whole ladder",
            "risk": "believed level = true level shifted K steps toward safe "
                    "(under) or dangerous (over), clipped to the ladder",
            "scoring": "C_true re-prices the risk surcharge at the reference "
                       "level, so a run that under-called risk does not get a "
                       "discount for it",
        },
        "ladders": {name: [{"cost_ratio": f, "risk_shift": k} for f, k in rung]
                    for name, rung in rungs.items()},
        "directions": list(DOORS_DIRECTIONS),
        "gates": gates, "gate_summary": summary, "runs": runs,
        "screenshots": {job["label"]: job["screenshot"]
                        for job in jobs if job.get("screenshot")},
    }
    _save("doors.json", payload)
    _write_doors_csv(payload)
    _chart_doors_gap(payload, _load("accuracy.json"), _load("risk.json"),
                     os.path.join(OUT_DIR, "doors_gap.png"))
    _chart_doors_gates(payload, os.path.join(OUT_DIR, "doors_gates.png"))
    return payload


# 逐门校准：单独测出「搬 A」「搬 B」「绕行」各自的真实代价。

def _calibration_jobs(gates: List[dict]) -> List[dict]:
    """每道门三次运行，每次只留下一个选项。"""
    exact_difficulty = {r["oid"]: r["difficulty"] for r in gates}
    exact_level = {r["oid"]: r["risk"] for r in gates}
    by_gate: Dict[int, Dict[str, int]] = {}
    for r in gates:
        by_gate.setdefault(r["gate"], {})[r["side"]] = r["oid"]

    jobs = []
    for gate, sides in sorted(by_gate.items()):
        # 砌死绕行开口后只剩两扇门，再把其中一扇的 belief 抬到机器人推不动，
        # 剩下的那扇就是规划器唯一能选的路。
        for keep in ("A", "B"):
            difficulty = dict(exact_difficulty)
            difficulty[sides["B" if keep == "A" else "A"]] = DOORS_FORBID_N
            jobs.append({"label": f"gate{gate}_only{keep}", "gates": gates,
                         "family": "calibration", "direction": keep,
                         "seed": gate, "cost_ratio": 1.0, "risk_shift": 0,
                         "difficulty": difficulty, "level": exact_level,
                         "closed_bypasses": (gate,), "screenshot": None})
        # 两扇门都推不动时，只能绕行。
        difficulty = dict(exact_difficulty)
        for oid in sides.values():
            difficulty[oid] = DOORS_FORBID_N
        jobs.append({"label": f"gate{gate}_detour", "gates": gates,
                     "family": "calibration", "direction": "detour",
                     "seed": gate, "cost_ratio": 1.0, "risk_shift": 0,
                     "difficulty": difficulty, "level": exact_level,
                     "closed_bypasses": (), "screenshot": None})
    return jobs


def _flip_factors(option_C: Dict[str, float], option_work: Dict[str, float]):
    """预测每个选项被误估到多大比例时决策才翻转。

    规划器给一次搬移的计价里，只有「难度 x 移动距离」这一项随 belief 线性
    变化，其余是行驶和风险。所以把选项 o 的估计放大 F 倍后，它的账面代价是
    `C_o + work_o * (F - 1)`；缩小 F 倍则是 `C_o - work_o * (1 - 1/F)`。
    赢家被抬到输，或者输家被压到赢，都能解析地解出那个 F。
    """
    usable = {k: v for k, v in option_C.items() if v is not None}
    if len(usable) < 2:
        return {"best": None, "margin_J": None, "flip_over": None,
                "flip_under": None, "flip_under_option": None}
    best = min(usable, key=usable.get)
    second = min((k for k in usable if k != best), key=usable.get)
    margin = usable[second] - usable[best]

    work_best = option_work.get(best) or 0.0
    # 赢家被高估：账面代价涨到超过第二名就翻转。
    flip_over = 1.0 + margin / work_best if work_best > 0 else None

    # 输家被低估：账面代价压到低于赢家就翻转，压到 0 都不够就永远不翻。
    flip_under, flip_under_option = None, None
    for name, value in usable.items():
        if name == best:
            continue
        work = option_work.get(name) or 0.0
        gap = value - usable[best]
        if work <= gap or work <= 0:
            continue
        candidate = 1.0 / (1.0 - gap / work)
        if flip_under is None or candidate < flip_under:
            flip_under, flip_under_option = candidate, name
    return {"best": best, "margin_J": round(margin, 1),
            "flip_over": round(flip_over, 3) if flip_over else None,
            "flip_under": round(flip_under, 3) if flip_under else None,
            "flip_under_option": flip_under_option}


def _break_even_difficulty(design: dict, option_C: Dict[str, float],
                           option_work: Dict[str, float]) -> Dict[str, float]:
    """每扇门要多重才与绕行正好持平，用来重新校准地图。

    搬移代价里只有「难度 x 移动距离」随难度线性变化，所以把这一项调整到
    两个选项的 C 相等，就得到平衡点上的难度。高于它这扇门会被绕开，低于
    它会被搬走；地图想问出问题，难度就得跨在这个数两侧。
    """
    detour = option_C.get("detour")
    out: Dict[str, float] = {}
    for side in ("A", "B"):
        here, work = option_C.get(side), option_work.get(side) or 0.0
        if detour is None or here is None or work <= 0:
            continue
        value = design[side]["difficulty"] * (1.0 + (detour - here) / work)
        # 平衡点算成负数，说明这扇门输给绕行的不是重量：即使它没有重量，
        # 风险附加项或者接近它要走的路也已经比绕行贵了。调难度救不回来。
        out[side] = round(value, 1) if value > 0 else None
    return out


def stage_doors_calibrate(workers: int) -> dict:
    """测出每道门三个选项的真实代价，得到决策裕度和预测翻转比例。"""
    scenario = scenarios.load(DOORS_MAP)
    gates, summary = scenario["gates"], scenario["gate_summary"]
    jobs = _calibration_jobs(gates)
    log(f"calibrate: {len(summary)} gates x 3 options = {len(jobs)} runs")
    runs = _run_doors_batch(jobs, workers)

    by_gate: Dict[int, Dict[str, dict]] = {}
    for row in runs:
        gate = int(row["label"].split("_")[0][4:])
        by_gate.setdefault(gate, {})[row["direction"]] = row

    rows = []
    for gate in sorted(by_gate):
        got = by_gate[gate]
        option_C = {k: (r["C_true"] if r["success"] else None)
                    for k, r in got.items()}
        detour = got.get("detour")
        # 绕行时这道门一点功都不做，所以两次运行的功之差就是这道门自己的功。
        base_work = detour["work_cost"] if detour else 0.0
        option_work = {"detour": 0.0}
        for side in ("A", "B"):
            if side in got:
                option_work[side] = max(0.0, got[side]["work_cost"] - base_work)
        design = next(g for g in summary if g["gate"] == gate)
        rows.append({"gate": gate, "kind": design["kind"],
                     "detour_m": design["detour_m"],
                     "ab_ratio": design["ab_ratio"],
                     "A": design["A"], "B": design["B"],
                     "C": {k: (round(v, 1) if v is not None else None)
                           for k, v in option_C.items()},
                     "work": {k: round(v, 1) for k, v in option_work.items()},
                     "break_even_difficulty": _break_even_difficulty(
                         design, option_C, option_work),
                     **_flip_factors(option_C, option_work)})
        r = rows[-1]
        over = f"{r['flip_over']:.2f}x" if r["flip_over"] else "never"
        under = f"{r['flip_under']:.2f}x" if r["flip_under"] else "never"
        even = "  ".join(
            f"{side} {value:,.0f} N" if value else f"{side} not by weight"
            for side, value in sorted(r["break_even_difficulty"].items()))
        log(f"  gate {gate} ({r['kind']:<4s})  best={r['best'] or '-':<6s} "
            f"margin={r['margin_J'] or 0:>9,.0f} J  "
            f"flips when over-estimated {over:<7s} "
            f"when under-estimated {under:<7s}  break-even: {even}")

    payload = {"map": DOORS_MAP, "gates": rows, "runs": runs}
    _save("doors_calibration.json", payload)
    return payload


# 风险指标。
RISK_ARMS = (("keyword fallback", "keyword"),
             ("LLM on sight", "sight"),
             ("LLM after contact", "contact"))


def _risk_measured(risk_payload: Optional[dict], key: str = "sight") -> bool:
    """风险阶段是否产生至少一个可用判断？"""
    return bool(risk_payload) and any(r.get(key) for r in risk_payload["rows"])


def _modal_level(levels) -> Optional[str]:
    """返回众数等级；并列时选择更高风险。"""
    got = [lvl for lvl in levels if lvl]
    if not got:
        return None
    counts: Dict[str, int] = {}
    for lvl in got:
        counts[lvl] = counts.get(lvl, 0) + 1
    best = max(counts.values())
    return max((lvl for lvl, n in counts.items() if n == best), key=LEVELS.index)


def _risk_stats(rows, key: str) -> dict:
    """汇总风险一致性和等效绕行误差。"""
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
        # 十门风险梯度按「差几级」标刻度，用这个数把实测精度落到梯度上。
        "mean_abs_levels": round(statistics.fmean(abs(d) for d in deltas), 3),
        "shortfall_m": round(statistics.fmean(max(0.0, -g) for g in gaps), 1),
        "excess_m": round(statistics.fmean(max(0.0, g) for g in gaps), 1),
    }


def _risk_confusion(rows, key: str) -> List[List[int]]:
    """counts[reference][estimate]，顺序与 LEVELS 一致。"""
    order = {name: i for i, name in enumerate(LEVELS)}
    grid = [[0] * len(LEVELS) for _ in LEVELS]
    for r in rows:
        got = r.get(key)
        if got:
            grid[order[r["risk_true"]]][order[got]] += 1
    return grid


def _risk_under_calls(rows, key: str) -> List[dict]:
    """被判断得比真实更安全的数据项，按风险下降幅度降序。"""
    order = {name: i for i, name in enumerate(LEVELS)}
    out = [r for r in rows if r.get(key) and order[r[key]] < order[r["risk_true"]]]
    return sorted(out, key=lambda r: order[r[key]] - order[r["risk_true"]])


# 难度指标。

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
    """统计与锚点表数值完全匹配的响应。"""
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
    """测量重复确定性请求之间的离散程度。"""
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


# 图表。

def _chart_accuracy(rows, path: str):
    """左图是估计值对参考值，右图是误差倍数的分布。"""
    plt = _plt()

    fig, (ax, ax_cdf) = plt.subplots(
        1, 2, figsize=(12.2, 6.0), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # 按观测数据设置坐标轴范围，避免裁剪点。
    seen = [v for r in rows for v in (r["mu_rho_true"], r.get("pred")) if v]
    lo, hi = min(0.2, min(seen) / 1.6), max(3000.0, max(seen) * 1.6)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, zorder=2)
    for band, alpha in ((2.0, 0.10), (1.5, 0.14)):
        ax.fill_between([lo, hi], [lo / band, hi / band], [lo * band, hi * band],
                        color=MUTED, alpha=alpha, lw=0, zorder=1)

    # 在图表中直接标注重复出现的锚点数值。
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
    _style(ax, xlabel="reference mu*rho  [kg/m$^3$]",
           ylabel="LLM estimate  [kg/m$^3$]",
           title="LLM mu*rho against the reference", grid="both")
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.text(0.02, 0.98, "shaded bands are 1.5x and 2x", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    # 右图把误差换算成十门实验横轴上的「Gap 比例」。
    arms = (("LLM", "pred", "#2a78d6", 14), ("offline heuristic", "heuristic",
                                             MUTED, -16))
    limit = 1.0
    for name, key, colour, dy in arms:
        factors = sorted(10 ** abs(e) for e in _log_errors(rows, key))
        if not factors:
            continue
        limit = max(limit, factors[-1])
        share = [(i + 1) / len(factors) for i in range(len(factors))]
        ax_cdf.step(factors, share, where="post", color=colour, lw=2.0,
                    label=f"{name} (n={len(factors)})", zorder=3)
        median = statistics.median(factors)
        ax_cdf.scatter([median], [0.5], s=46, color=colour, zorder=4,
                       edgecolor=SURFACE, linewidth=1.0)
        # 两条曲线的中位数常常靠得很近，上下错开标注。
        ax_cdf.annotate(f"{name} median {median:.1f}x", xy=(median, 0.5),
                        xytext=(8, dy), textcoords="offset points",
                        fontsize=8.5, color=colour)
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlim(1.0, max(limit * 1.1, 2.0))
    # 刻度直接用十门实验的梯级，两张图可以并排对读。
    rungs = [r for r in DOORS_COST_RATIOS if 1.0 <= r <= limit * 1.1]
    for rung in rungs[1:]:
        ax_cdf.axvline(rung, color=GRID, lw=0.9, zorder=1)
    ax_cdf.set_xticks(rungs)
    ax_cdf.set_xticklabels([f"{r:g}x" for r in rungs], fontsize=7.5,
                           rotation=40, ha="right")
    ax_cdf.minorticks_off()
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_cdf.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    _style(ax_cdf, xlabel="error factor  |estimate / reference|",
           ylabel="share of objects at or below",
           title="How wrong, on the ten-gate experiment's own axis")
    ax_cdf.text(0.02, 0.98,
                "vertical guides are the Gap ladder rungs in doors_gap.png",
                transform=ax_cdf.transAxes, ha="left", va="top",
                fontsize=8, color=MUTED)
    ax_cdf.legend(frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, path)


def _chart_risk(risk: dict, path: str):
    """上排是每个阶段的混淆矩阵，下排是折算成米的代价。"""
    plt = _plt()
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

    fig = plt.figure(figsize=(3.9 * len(grids) + 1.4, 7.0),
                     constrained_layout=True)
    spec = fig.add_gridspec(2, len(grids), height_ratios=[2.6, 1.0])
    axes = [fig.add_subplot(spec[0, i]) for i in range(len(grids))]
    ax_cost = fig.add_subplot(spec[1, :])

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
            # 描出对角线，使色阶继续表示计数。
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, lw=1.1, zorder=2))

        # 在单元格之间留出少量间隙。
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
        ax.set_title(f"{name}\n{stat['exact']:.0%} exact, "
                     f"{stat['under']:.0%} too safe\n"
                     f"{stat['mean_abs_levels']:.2f} levels off on average",
                     color=INK, fontsize=9.5, loc="left", pad=8)

    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("reference risk level", color=INK, fontsize=9.5)

    # 把每个阶段的误判折算成米：缺失的保护和凭空多走的路。
    names = [name for name, _, _ in grids]
    spots = list(range(len(names)))
    width = 0.36
    shortfall = [stat["shortfall_m"] for _, _, stat in grids]
    excess = [stat["excess_m"] for _, _, stat in grids]
    ax_cost.barh([s + width / 2 for s in spots], shortfall, height=width,
                 color="#eb6834", label="protection dropped (shortfall)",
                 zorder=3)
    ax_cost.barh([s - width / 2 for s in spots], excess, height=width,
                 color="#9ec5f4", label="detour invented (excess)", zorder=3)
    for s, (under, over) in enumerate(zip(shortfall, excess)):
        ax_cost.text(under, s + width / 2, f" {under:,.0f} m", va="center",
                     fontsize=8.5, color=INK)
        ax_cost.text(over, s - width / 2, f" {over:,.0f} m", va="center",
                     fontsize=8.5, color=INK)
    ax_cost.set_yticks(spots)
    ax_cost.set_yticklabels(names, fontsize=9)
    ax_cost.invert_yaxis()
    ax_cost.margins(x=0.18)
    _style(ax_cost, xlabel="mean detour-equivalent error per object  [m]",
           title="What each mistake is worth in the planner's own units",
           grid="x")
    # 图例放到标题行右侧，否则会压住最长的那根柱子。
    ax_cost.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2,
                   bbox_to_anchor=(1.0, 1.0))

    fig.supxlabel("level the estimator returned - left of the outline in the "
                  "matrices is an obstacle called safer than it is",
                  color=MUTED, fontsize=9)
    _save_fig(fig, path)


# 十门结果的汇总与导出。

def _doors_sample(runs, family: str, direction: str, ratio: float,
                  shift: int) -> List[dict]:
    """取出一个 (族, 方向, 梯级) 上的全部运行。"""
    return [r for r in runs
            if r["family"] == family and r["direction"] == direction
            and abs(r["cost_ratio"] - ratio) < 1e-9 and r["risk_shift"] == shift]


def _doors_stats(sample: List[dict]) -> dict:
    """把一组运行汇总成 ΔC 的分布和决策改动量。"""
    if not sample:
        return {"n": 0}
    ok = [r for r in sample if r["success"] and r.get("delta_C_pct") is not None]
    deltas = sorted(r["delta_C_pct"] for r in ok)
    changed = [r["changed"] for r in sample]
    risky = [r["extra_risky"] for r in sample]
    out = {"n": len(sample), "reached": len(ok),
           "mean_changed": round(statistics.fmean(changed), 2),
           "max_changed": max(changed),
           "mean_extra_risky": round(statistics.fmean(risky), 2),
           "max_extra_risky": max(risky)}
    if deltas:
        out.update({
            "mean": round(statistics.fmean(deltas), 2),
            "median": round(statistics.median(deltas), 2),
            "p10": round(float(np.percentile(deltas, 10)), 2),
            "p90": round(float(np.percentile(deltas, 90)), 2),
            "worst": round(deltas[-1], 2), "best": round(deltas[0], 2),
            "mean_C_true": round(statistics.fmean(r["C_true"] for r in ok), 1),
        })
    return out


def _doors_curve(runs, family: str, direction: str, ladder: List[dict],
                 axis: str = "cost_ratio") -> List[tuple]:
    """返回该族该方向上的 (x, 统计量) 序列，并补上 exact 处的零点。"""
    neutral = 1.0 if axis == "cost_ratio" else 0
    points = [(neutral, {"n": 1, "reached": 1, "mean": 0.0, "median": 0.0,
                         "p10": 0.0, "p90": 0.0, "worst": 0.0, "best": 0.0,
                         "mean_changed": 0.0, "max_changed": 0,
                         "mean_extra_risky": 0.0, "max_extra_risky": 0})]
    for rung in ladder:
        stat = _doors_stats(_doors_sample(
            runs, family, direction, rung["cost_ratio"], rung["risk_shift"]))
        # 一个梯级上没有一次运行到达目标时就没有 ΔC 可画，跳过而不是补零。
        if stat.get("mean") is not None:
            points.append((rung[axis], stat))
    return points


def _interp_log(points: List[tuple], key: str, x: float):
    """在 log x 轴上对曲线线性插值；x 越界时截断到端点。"""
    usable = [(a, s[key]) for a, s in points if a > 0 and s.get(key) is not None]
    if not usable or x <= 0:
        return None
    usable.sort()
    if x <= usable[0][0]:
        return usable[0][1]
    if x >= usable[-1][0]:
        return usable[-1][1]
    for (x0, y0), (x1, y1) in zip(usable, usable[1:]):
        if x0 <= x <= x1:
            span = math.log(x1 / x0)
            if span <= 0:
                return y0
            return y0 + (y1 - y0) * math.log(x / x0) / span
    return usable[-1][1]


def _doors_slope(points: List[tuple], key: str = "mean"):
    """把 ΔC% 对 log2(F) 做最小二乘：估计误差每翻一倍要多花多少 C。"""
    xs = [math.log2(a) for a, s in points if a > 0 and s.get(key) is not None]
    ys = [s[key] for a, s in points if a > 0 and s.get(key) is not None]
    if len(xs) < 2:
        return None
    return round(float(np.polyfit(xs, ys, 1)[0]), 2)


def _doors_flip_grid(doors: dict, family: str, axis: str):
    """返回 gates x 梯级 的决策改动比例，用于热力图。"""
    runs, base = doors["runs"], doors["runs"][0]
    ladder = doors["ladders"][family]
    gate_ids = sorted(base["choices"], key=int)
    grid = []
    for gate in gate_ids:
        row = []
        for rung in ladder:
            sample = [r for r in runs
                      if r["family"] == family
                      and abs(r["cost_ratio"] - rung["cost_ratio"]) < 1e-9
                      and r["risk_shift"] == rung["risk_shift"]]
            flips = sum(1 for r in sample
                        if r["choices"][gate] != base["choices"][gate])
            row.append(flips / len(sample) if sample else 0.0)
        grid.append(row)
    return gate_ids, [rung[axis] for rung in ladder], grid


def _write_doors_csv(doors: dict) -> str:
    """把「Gap 比例 -> C 差别比例」写成可直接引用的表格。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "doors_gap_vs_cost.csv")
    base = doors["runs"][0]
    header = ["family", "direction", "gap_ratio", "risk_shift", "runs",
              "reached_goal", "delta_C_mean_pct", "delta_C_median_pct",
              "delta_C_p90_pct", "delta_C_worst_pct", "gates_changed_mean",
              "gates_changed_max", "extra_risky_mean", "C_true_mean",
              "C_true_exact"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for family, ladder in doors["ladders"].items():
            for direction in DOORS_DIRECTIONS:
                for rung in ladder:
                    stat = _doors_stats(_doors_sample(
                        doors["runs"], family, direction,
                        rung["cost_ratio"], rung["risk_shift"]))
                    if not stat.get("n"):
                        continue
                    writer.writerow([
                        family, direction, f"{rung['cost_ratio']:g}",
                        rung["risk_shift"], stat["n"], stat["reached"],
                        stat.get("mean", ""), stat.get("median", ""),
                        stat.get("p90", ""), stat.get("worst", ""),
                        stat["mean_changed"], stat["max_changed"],
                        stat["mean_extra_risky"], stat.get("mean_C_true", ""),
                        base["C_true"]])
    log(f"wrote {path}")
    return path


# 图表样式。

def _plt():
    """统一初始化 matplotlib，避免每个图表函数各写一遍。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, xlabel=None, ylabel=None, title=None, grid="y"):
    """套用全篇统一的坐标轴样式。"""
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9.5)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=8)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    return ax


def _save_fig(fig, path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    _plt().close(fig)
    log(f"wrote {path}")


def _ratio_axis(ax, ratios):
    """把 x 轴设成对数 Gap 比例，并标上真正跑过的梯级。"""
    ax.set_xscale("log")
    ax.set_xticks(list(ratios))
    # 梯级在对数轴低端挨得很近，横排会互相压住。
    ax.set_xticklabels([f"{r:g}x" for r in ratios], fontsize=7.5,
                       rotation=40, ha="right")
    ax.minorticks_off()


def _measured_accuracy(accuracy: Optional[dict]) -> Optional[dict]:
    """把准确率阶段的典型误差折算成 Gap 梯度上的一个位置。"""
    if not accuracy:
        return None
    stat = _accuracy_stats(accuracy["rows"])
    if not stat.get("n"):
        return None
    return {"median": stat["median_abs_factor"], "p90": stat["p90_abs_factor"],
            "model": accuracy.get("model", "")}


def _measured_risk_gap(risk: Optional[dict]) -> Optional[float]:
    """风险阶段实测的平均等级偏差。"""
    if not risk:
        return None
    stat = _risk_stats(risk["rows"], "sight")
    return stat.get("mean_abs_levels") if stat.get("n") else None


def _chart_doors_gap(doors: dict, accuracy: Optional[dict],
                     risk: Optional[dict], path: str):
    """四格图：Gap 比例分别换来多少 C、多少决策改动和多少危险搬移。"""
    plt = _plt()
    runs = doors["runs"]
    cost_ladder = doors["ladders"]["cost"]
    risk_ladder = doors["ladders"]["risk"]
    joint_ladder = doors["ladders"]["joint"]
    measured = _measured_accuracy(accuracy)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    ax_c, ax_d, ax_r, ax_s = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # 1. ΔC 随代价 Gap 比例的变化。
    ratios = [1.0] + [r["cost_ratio"] for r in cost_ladder]
    for direction in DOORS_DIRECTIONS:
        colour, marker = DIRECTION_STYLE[direction]
        points = _doors_curve(runs, "cost", direction, cost_ladder)
        if len(points) < 2:
            continue
        xs = [a for a, _ in points]
        ys = [s["mean"] for _, s in points]
        if direction == "mixed" and doors["seeds"] > 1:
            ax_c.fill_between(xs, [s["best"] for _, s in points],
                              [s["worst"] for _, s in points],
                              color=colour, alpha=0.12, lw=0, zorder=1)
        slope = _doors_slope(points)
        label = DIRECTION_LABEL[direction]
        if slope is not None:
            label += f"  ({slope:+.1f}%/doubling)"
        ax_c.plot(xs, ys, color=colour, marker=marker, lw=1.9, ms=5,
                  label=label, zorder=3)

    if measured:
        mixed = _doors_curve(runs, "cost", "mixed", cost_ladder)
        mid = math.sqrt(ratios[0] * ratios[-1])
        for key, style in (("median", "-"), ("p90", ":")):
            x = measured[key]
            y = _interp_log(mixed, "mean", x)
            if y is None:
                continue
            ax_c.axvline(x, color=INK, lw=1.0, ls=style, zorder=2)
            # 靠近右边界时改成向左标注，否则文字会跑出坐标区。
            right = x > mid
            ax_c.annotate(f"measured {key} miss {x:.1f}x -> {y:+.1f}% C",
                          xy=(x, y), xytext=(-8 if right else 8, 10),
                          textcoords="offset points", fontsize=8.5, color=INK,
                          ha="right" if right else "left")
            ax_c.scatter([x], [y], s=44, color=INK, zorder=5)

    ax_c.axhline(0, color=INK, lw=0.9, zorder=2)
    _ratio_axis(ax_c, ratios)
    _style(ax_c, xlabel="cost Gap ratio F = estimate / truth",
           ylabel="change from exact C  [%]",
           title="1. What an estimate that is off by F costs the route")
    ax_c.legend(frameon=False, fontsize=8.5, loc="upper left")

    # 2. 改变的决策数。
    for direction in DOORS_DIRECTIONS:
        colour, marker = DIRECTION_STYLE[direction]
        points = _doors_curve(runs, "cost", direction, cost_ladder)
        if len(points) < 2:
            continue
        ax_d.plot([a for a, _ in points],
                  [s["mean_changed"] for _, s in points],
                  color=colour, marker=marker, lw=1.9, ms=5,
                  label=DIRECTION_LABEL[direction], zorder=3)
    n_gates = len(doors["gate_summary"])
    ax_d.set_ylim(0, n_gates)
    _ratio_axis(ax_d, ratios)
    _style(ax_d, xlabel="cost Gap ratio F = estimate / truth",
           ylabel=f"gates decided differently  [of {n_gates}]",
           title="2. How many of the ten decisions the error moves")
    ax_d.legend(frameon=False, fontsize=8.5, loc="upper left")

    # 3. 风险等级偏差和联合梯度。
    width = 0.26
    shifts = [0] + [r["risk_shift"] for r in risk_ladder]
    for i, direction in enumerate(DOORS_DIRECTIONS):
        colour, _ = DIRECTION_STYLE[direction]
        points = _doors_curve(runs, "risk", direction, risk_ladder,
                              axis="risk_shift")
        xs = [a + (i - 1) * width for a, _ in points]
        ax_r.bar(xs, [s["mean"] for _, s in points], width=width,
                 color=colour, label=DIRECTION_LABEL[direction], zorder=3)
    joint = _doors_curve(runs, "joint", "mixed", joint_ladder,
                         axis="risk_shift")
    if len(joint) > 1:
        ax_r.plot([a for a, _ in joint], [s["mean"] for _, s in joint],
                  color=INK, marker="s", ms=4, lw=1.4, ls="--",
                  label="cost + risk together", zorder=4)
    measured_levels = _measured_risk_gap(risk)
    if measured_levels:
        ax_r.axvline(measured_levels, color=INK, lw=1.0, ls=":", zorder=2)
        ax_r.annotate(f"measured {measured_levels:.2f} levels",
                      xy=(measured_levels, 0), xytext=(4, 6),
                      textcoords="offset points", fontsize=8.5, color=INK)
    ax_r.axhline(0, color=INK, lw=0.9, zorder=2)
    ax_r.set_xticks(shifts)
    _style(ax_r, xlabel="risk Gap K = levels between estimate and reference",
           ylabel="change from exact C  [%]",
           title="3. What a mis-rated risk level costs")
    ax_r.legend(frameon=False, fontsize=8.5, loc="upper left")

    # 4. 多出来的 C 花在了哪里。
    parts = (("delta_walk", "driving", "#9ec5f4"),
             ("delta_work", "pushing", "#2a78d6"),
             ("delta_risk", "risk surcharge", "#eb6834"))
    xs = list(range(len(cost_ladder)))
    bottoms_pos = [0.0] * len(cost_ladder)
    bottoms_neg = [0.0] * len(cost_ladder)
    for key, label, colour in parts:
        values = []
        for rung in cost_ladder:
            sample = _doors_sample(runs, "cost", "mixed",
                                   rung["cost_ratio"], rung["risk_shift"])
            ok = [r for r in sample if r["success"]]
            values.append(statistics.fmean(r[key] for r in ok) if ok else 0.0)
        bottoms = [bottoms_neg[i] if v < 0 else bottoms_pos[i]
                   for i, v in enumerate(values)]
        ax_s.bar(xs, values, bottom=bottoms, width=0.66, color=colour,
                 label=label, zorder=3)
        for i, v in enumerate(values):
            if v < 0:
                bottoms_neg[i] += v
            else:
                bottoms_pos[i] += v
    ax_s.axhline(0, color=INK, lw=0.9, zorder=2)
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels([f"{r['cost_ratio']:g}x" for r in cost_ladder],
                         fontsize=8)
    _style(ax_s, xlabel="cost Gap ratio F  (random-direction arm)",
           ylabel="change from exact  [J]",
           title="4. Where the extra cost is spent")
    ax_s.legend(frameon=False, fontsize=8.5, loc="upper left")

    subtitle = ("C is re-priced at the reference risk level, so a run that "
                "called a hazard safe pays for it here")
    if measured:
        subtitle = f"model {measured['model']} - " + subtitle
    fig.suptitle("Estimate error on the ten-gate corridor", x=0.012, ha="left",
                 color=INK, fontsize=13)
    fig.supxlabel(subtitle, color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    _save_fig(fig, path)


def _chart_doors_gates(doors: dict, path: str):
    """逐门热力图：哪一道门在多大的 Gap 上开始改主意。"""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    panels = [("cost", "cost_ratio", "cost Gap ratio F"),
              ("risk", "risk_shift", "risk Gap K  [levels]")]
    panels = [p for p in panels if doors["ladders"].get(p[0])]
    if not panels:
        return
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    base = doors["runs"][0]
    summary = {g["gate"]: g for g in doors["gate_summary"]}

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels) + 1.6, 5.2),
        gridspec_kw={"width_ratios": [len(doors["ladders"][p[0]])
                                      for p in panels]})
    axes = list(axes) if len(panels) > 1 else [axes]

    for ax, (family, axis, xlabel) in zip(axes, panels):
        gate_ids, ticks, grid = _doors_flip_grid(doors, family, axis)
        ax.set_facecolor(SURFACE)
        for i, row in enumerate(grid):
            for j, share in enumerate(row):
                if not share:                 # 没翻转过就留白。
                    continue
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                           facecolor=cmap(share), zorder=1))
                ax.text(j, i, f"{share:.0%}", ha="center", va="center",
                        fontsize=8, zorder=3,
                        color=SURFACE if share > 0.55 else INK)
        ax.set_xlim(-0.5, len(ticks) - 0.5)
        ax.set_ylim(len(gate_ids) - 0.5, -0.5)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels([f"{t:g}x" if axis == "cost_ratio" else f"+/-{t:g}"
                            for t in ticks], fontsize=8.5)
        ax.set_xticks([k - 0.5 for k in range(len(ticks) + 1)], minor=True)
        ax.set_yticks([k - 0.5 for k in range(len(gate_ids) + 1)], minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", length=0)
        ax.set_yticks(range(len(gate_ids)))
        ax.set_yticklabels([
            f"{g}  {summary[int(g)]['kind']}  "
            f"A/B {summary[int(g)]['ab_ratio']:g}x  "
            f"detour {summary[int(g)]['detour_m']:g} m  "
            f"-> {base['choices'][g]}" for g in gate_ids], fontsize=8)
        ax.tick_params(colors=MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _style(ax, xlabel=xlabel, grid=None,
               title=f"{FAMILY_LABEL[family]} Gap")

    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("gate, its design margin, and what exact chose",
                       color=INK, fontsize=9.5)
    fig.supxlabel("share of runs at that Gap that decided the gate differently "
                  "from the exact run", color=MUTED, fontsize=9)
    fig.tight_layout()
    _save_fig(fig, path)


# 报告生成。

def _table(header: List[str], body: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def _doors_headline(doors: dict, accuracy: Optional[dict]) -> List[str]:
    """把十门扫描压缩成标题里的两三句话。"""
    runs = doors["runs"]
    ladder = doors["ladders"]["cost"]
    n_gates = len(doors["gate_summary"])
    if not ladder:
        return []
    mixed = _doors_curve(runs, "cost", "mixed", ladder)
    slope = _doors_slope(mixed)
    out = []
    if slope is not None:
        worst = ladder[-1]["cost_ratio"]
        stat = _doors_stats(_doors_sample(runs, "cost", "mixed", worst, 0))
        out.append(
            f"- On the ten-gate corridor, **every doubling of the cost "
            f"estimate's error costs {slope:+.1f}% of C**. At the ladder's "
            f"far end ({worst:g}x) the route is {stat.get('mean', 0):+.1f}% "
            f"dearer on average, worst {stat.get('worst', 0):+.1f}%, with "
            f"{stat['mean_changed']:.1f} of {n_gates} gates decided "
            f"differently.")
    measured = _measured_accuracy(accuracy)
    if measured:
        typical = _interp_log(mixed, "mean", measured["median"])
        bad = _interp_log(mixed, "mean", measured["p90"])
        if typical is not None:
            out.append(
                f"- Read the measured accuracy off that ladder and the "
                f"estimator's typical **{measured['median']:.1f}x miss is "
                f"worth {typical:+.1f}% of C**; its p90 "
                f"{measured['p90']:.1f}x miss is worth "
                f"{_fmt_pct(bad)}. That is the price of the error in the only "
                f"unit the planner cares about.")
    risky = _doors_stats(_doors_sample(
        runs, "risk", "under", 1.0, doors["ladders"]["risk"][-1]["risk_shift"])
    ) if doors["ladders"].get("risk") else {}
    if risky.get("mean_extra_risky"):
        out.append(
            f"- Under-calling risk is the expensive half: at the end of the "
            f"risk ladder the planner pushes "
            f"{risky['mean_extra_risky']:+.1f} more obstacles that were never "
            f"safe to push, for {risky.get('mean', 0):+.1f}% of C once those "
            f"pushes are priced at the level they deserved.")
    return out


def _first_flip(doors: dict, gate: str, family: str = "cost"):
    """返回这道门第一次改主意的梯级，一直不改则返回 None。"""
    base = doors["runs"][0]
    for rung in doors["ladders"].get(family, ()):
        sample = [r for r in doors["runs"]
                  if r["family"] == family
                  and abs(r["cost_ratio"] - rung["cost_ratio"]) < 1e-9
                  and r["risk_shift"] == rung["risk_shift"]]
        if any(r["choices"][gate] != base["choices"][gate] for r in sample):
            return rung
    return None


def _report_doors(doors: dict, accuracy: Optional[dict],
                  risk: Optional[dict]) -> List[str]:
    """第 4 节：Gap 比例换来多少 C。"""
    runs, base = doors["runs"], doors["runs"][0]
    cost_ladder = doors["ladders"]["cost"]
    risk_ladder = doors["ladders"]["risk"]
    joint_ladder = doors["ladders"]["joint"]
    n_gates = len(doors["gate_summary"])
    parts = ["\n## 4. What the error costs a route\n"]

    parts.append(
        f"The `{doors['map']}` map is {n_gates} walls in a row, each with three "
        f"ways past it: move the obstacle in door A, move the one in door B, or "
        f"walk around through a third opening placed far enough off the axis to "
        f"cost a real detour. Every gate is one independent three-way decision, "
        f"so a run is {n_gates} of them and the arms differ only in what the "
        f"planner was told to believe.\n\n"
        f"No API is called here. The belief is written directly: at Gap ratio "
        f"`F` every cost estimate is exactly `true * F` (the **over-estimate** "
        f"arm), exactly `true / F` (the **under-estimate** arm), or one of the "
        f"two with the direction drawn once per obstacle and then held fixed "
        f"for the whole ladder (the **random-direction** arm). Because `F` is "
        f"constructed rather than sampled, the realized gap equals the axis "
        f"label, and the rows below can be read as 'an estimator this wrong "
        f"costs this much'. Holding each obstacle's direction fixed across the "
        f"ladder also makes the rungs paired samples, so the curve's shape is "
        f"the Gap growing rather than a fresh set of random numbers.\n\n"
        f"Risk is perturbed the same way: `K` levels toward safe in the "
        f"under-estimate arm, `K` toward dangerous in the over-estimate arm, "
        f"clipped to the ladder.\n\n"
        f"**C is re-priced at the reference risk level.** The executor charges "
        f"the risk surcharge at the level it believed, so a run that called a "
        f"hazard safe would otherwise book a discount for the mistake and look "
        f"cheaper than `exact`. Every C below is `J` plus the surcharge the "
        f"pushed obstacles were really worth, which is what makes a wrong "
        f"decision show up as a cost rather than a saving.\n\n"
        f"Beliefs are seeded per obstacle, but the corrections a real run earns "
        f"are left in: touching an obstacle reveals its true difficulty and "
        f"re-rates its risk. What the perturbation buys is therefore a wrong "
        f"*decision*, taken before the robot could know better, which is "
        f"exactly what a bad estimate costs in practice.\n\n"
        f"The ladder straddles one threshold worth knowing about. "
        f"`Config.contact_replan_ratio` is {Config().contact_replan_ratio:g}, "
        f"so an error smaller than that is never noticed: the robot takes "
        f"hold, finds the force close enough to what it planned for, and "
        f"carries on. Above it the mismatch triggers a re-plan mid-push, which "
        f"is why the curve is flatter at the first rung or two than a straight "
        f"line through the rest would predict.\n\n"
        f"`exact` is the floor at C = {base['C_true']:,.0f} J with "
        f"{len(base['removed'])} obstacles pushed.\n\n")

    # 4.1 主表：不同 Gap 比例对应的 C 差别比例。
    parts.append("\n### 4.1 Gap ratio against the change in C\n")
    parts.append("The headline table. One row per rung of the cost ladder and "
                 "arm; `dC` is the percentage change from the exact run's C.\n\n")
    body = []
    for rung in cost_ladder:
        for direction in DOORS_DIRECTIONS:
            stat = _doors_stats(_doors_sample(
                runs, "cost", direction, rung["cost_ratio"],
                rung["risk_shift"]))
            if not stat.get("n"):
                continue
            body.append([
                f"{rung['cost_ratio']:g}x", DIRECTION_LABEL[direction],
                stat["n"], f"{stat['reached']}/{stat['n']}",
                f"{stat.get('mean', 0):+.1f}%", f"{stat.get('median', 0):+.1f}%",
                f"{stat.get('p90', 0):+.1f}%", f"{stat.get('worst', 0):+.1f}%",
                f"{stat['mean_changed']:.1f}", stat["max_changed"],
                f"{stat['mean_extra_risky']:+.1f}",
                f"{stat.get('mean_C_true', 0):,.0f}"])
    parts.append(_table(
        ["Gap F", "arm", "runs", "reached goal", "dC mean", "dC median",
         "dC p90", "dC worst", f"gates changed (of {n_gates})", "worst",
         "extra risky pushes", "mean C"], body))

    slopes = {d: _doors_slope(_doors_curve(runs, "cost", d, cost_ladder))
              for d in DOORS_DIRECTIONS}
    known = {d: s for d, s in slopes.items() if s is not None}
    if known:
        parts.append(
            "\n**Sensitivity.** Fitting the mean `dC` against `log2(F)` gives "
            "the cost of each doubling of estimate error: "
            + ", ".join(f"{DIRECTION_LABEL[d]} **{s:+.1f}% of C per doubling**"
                        for d, s in known.items())
            + ". The two deterministic arms bracket the random one: a uniform "
              "bias moves every option the same way and only changes "
              "push-against-detour, while a random direction also reverses "
              "door A against door B, which is the cheaper mistake to make but "
              "the easier one to make often.\n")

    # 4.2 把实测的估计精度落到这条曲线上。
    measured = _measured_accuracy(accuracy)
    if measured:
        mixed = _doors_curve(runs, "cost", "mixed", cost_ladder)
        rows = []
        for name, key in (("typical miss (median)", "median"),
                          ("bad-case miss (p90)", "p90")):
            factor = measured[key]
            rows.append([name, f"{factor:.2f}x",
                         _fmt_pct(_interp_log(mixed, "mean", factor)),
                         _fmt_pct(_interp_log(mixed, "worst", factor)),
                         _fmt_num(_interp_log(mixed, "mean_changed", factor)),
                         _fmt_num(_interp_log(mixed, "mean_extra_risky", factor))])
        parts.append("\n### 4.2 Where the measured estimator lands on that "
                     "ladder\n")
        parts.append(
            f"Section 1 measured how wrong `{measured['model']}` is on the "
            f"reference objects. Reading that number off the curve above turns "
            f"an accuracy figure into a route cost, which is the only unit that "
            f"matters to the planner. Values are interpolated on the log-ratio "
            f"axis between the rungs that were actually run.\n\n")
        parts.append(_table(
            ["estimator accuracy", "as a Gap ratio", "dC mean", "dC worst",
             "gates changed", "extra risky pushes"], rows))

    # 4.3 风险等级误差。
    if risk_ladder:
        parts.append("\n### 4.3 The same sweep on risk levels\n")
        parts.append(
            "A risk level is not a ratio, so this ladder is in levels: `K` "
            "steps from the reference level, clipped at `low` and `extreme`. "
            "Under-calling is the direction that matters, because it is the one "
            "that lets the planner push something it should have walked "
            "around.\n\n")
        body = []
        for rung in risk_ladder:
            for direction in DOORS_DIRECTIONS:
                stat = _doors_stats(_doors_sample(
                    runs, "risk", direction, rung["cost_ratio"],
                    rung["risk_shift"]))
                if not stat.get("n"):
                    continue
                body.append([
                    f"+/-{rung['risk_shift']}", DIRECTION_LABEL[direction],
                    stat["n"], f"{stat['reached']}/{stat['n']}",
                    f"{stat.get('mean', 0):+.1f}%", f"{stat.get('worst', 0):+.1f}%",
                    f"{stat['mean_changed']:.1f}",
                    f"{stat['mean_extra_risky']:+.1f}"])
        parts.append(_table(
            ["Gap K", "arm", "runs", "reached goal", "dC mean", "dC worst",
             "gates changed", "extra risky pushes"], body))

    if joint_ladder:
        parts.append("\n### 4.4 Both at once\n")
        parts.append("Cost and risk perturbed together, so the two "
                     "contributions can be compared against their sum.\n\n")
        body = []
        for rung in joint_ladder:
            for direction in DOORS_DIRECTIONS:
                stat = _doors_stats(_doors_sample(
                    runs, "joint", direction, rung["cost_ratio"],
                    rung["risk_shift"]))
                if not stat.get("n"):
                    continue
                alone = []
                for family, key in (("cost", "cost_ratio"),
                                    ("risk", "risk_shift")):
                    part = _doors_stats(_doors_sample(
                        runs, family, direction,
                        rung["cost_ratio"] if family == "cost" else 1.0,
                        rung["risk_shift"] if family == "risk" else 0))
                    alone.append(part.get("mean"))
                total = sum(v for v in alone if v is not None)
                body.append([
                    f"{rung['cost_ratio']:g}x / +/-{rung['risk_shift']}",
                    DIRECTION_LABEL[direction], stat["n"],
                    f"{stat.get('mean', 0):+.1f}%",
                    _fmt_pct(alone[0]), _fmt_pct(alone[1]), f"{total:+.1f}%",
                    f"{stat['mean_changed']:.1f}"])
        parts.append(_table(
            ["Gap F / K", "arm", "runs", "dC together", "dC cost alone",
             "dC risk alone", "sum of the two", "gates changed"], body))

    # 4.5 逐门：哪一道门在多大的 Gap 上先改主意。
    parts.append("\n### 4.5 Which decisions move first\n")
    parts.append(
        "A gate flips when the error exceeds its own margin, so the ladder "
        "rung at which each gate first changes its mind is a direct reading of "
        "how much slack that decision had. `A/B ratio` is the true difficulty "
        "ratio between the two doors: a random-direction error has to exceed "
        "roughly that ratio before door A and door B swap places.\n\n")
    body = []
    for g in doors["gate_summary"]:
        gate = str(g["gate"])
        cost_flip = _first_flip(doors, gate, "cost")
        risk_flip = _first_flip(doors, gate, "risk")
        flips = sum(1 for r in runs[1:] if r["choices"][gate]
                    != base["choices"][gate])
        body.append([
            gate, g["kind"], f"{g['detour_m']:g} m", f"{g['ab_ratio']:g}x",
            f"{g['A']['difficulty']:.0f} N {g['A']['risk']}",
            f"{g['B']['difficulty']:.0f} N {g['B']['risk']}",
            base["choices"][gate],
            f"{cost_flip['cost_ratio']:g}x" if cost_flip else "never",
            f"+/-{risk_flip['risk_shift']}" if risk_flip else "never",
            f"{flips}/{len(runs) - 1}"])
    parts.append(_table(
        ["gate", "kind", "detour", "A/B ratio", "door A", "door B",
         "exact chose", "first cost flip", "first risk flip",
         "runs that chose otherwise"], body))

    calibration = _load("doors_calibration.json")
    if calibration:
        parts.append("\n### 4.6 Measured margins per gate\n")
        parts.append(
            "The `doors-calibrate` stage prices each option on its own: the "
            "third opening is walled up and one door at a time is made too "
            "heavy to move, so the planner has exactly one way past that gate. "
            "Only the `difficulty x distance` term of a push moves with the "
            "belief, so the factor at which each decision flips follows in "
            "closed form, and the `first cost flip` column above is the "
            "measurement it predicts.\n\n")
        body = []
        for row in calibration["gates"]:
            costs, even = row["C"], row.get("break_even_difficulty") or {}
            body.append([
                row["gate"], row["kind"],
                *[f"{costs[k]:,.0f}" if costs.get(k) else "-"
                  for k in ("A", "B", "detour")],
                row["best"] or "-",
                f"{row['margin_J']:,.0f}" if row["margin_J"] else "-",
                f"{row['flip_over']:g}x" if row["flip_over"] else "never",
                (f"{row['flip_under']:g}x -> {row['flip_under_option']}"
                 if row["flip_under"] else "never"),
                " / ".join(f"{side} {value:,.0f}" if value
                           else f"{side} not by weight"
                           for side, value in sorted(even.items())) or "-"])
        parts.append(_table(
            ["gate", "kind", "C if A", "C if B", "C if detour", "cheapest",
             "margin J", "flips when over-estimated",
             "flips when under-estimated", "break-even difficulty N"], body))
        parts.append(
            "\nThe last column is the map's tuning knob. A door heavier than "
            "its break-even value is walked around and a lighter one is pushed, "
            "so the ten difficulties in `scenarios/ten_doors.py` have to "
            "straddle these numbers. If they all sit on one side, every gate "
            "makes the same choice and no amount of estimate error can move "
            "anything: the sweep would measure nothing. `not by weight` marks "
            "a door that loses to the detour even at zero weight, so its risk "
            "surcharge or its approach travel decides it and no difficulty "
            "setting will change that.\n")

    shots = doors.get("screenshots") or {}
    if shots:
        parts.append("\n### 4.7 Route screenshots\n")
        parts.append(_table(["run", "image"],
                            [[label, f"`{name}`"]
                             for label, name in sorted(shots.items())]))
    parts.append(
        f"\nThe machine-readable version of section 4.1 is "
        f"`doors_gap_vs_cost.csv`; `doors_gap.png` plots it and "
        f"`doors_gates.png` breaks it down per gate.\n")
    return parts


def stage_report() -> str:
    acc = _load("accuracy.json")
    risk = _load("risk.json")
    doors = _load("doors.json")
    # 风险阶段全部失败时视为无可用数据。
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

    # 标题摘要。
    overall = _accuracy_stats(rows)
    snapshot = _anchor_snapping(rows)
    head = [
        f"- The estimator is off by a typical factor of "
        f"**{overall['median_abs_factor']:.1f}x**; only "
        f"**{overall['within_2x']:.0%}** of objects land within 2x of the "
        f"reference, and it over-estimates "
        f"({overall['median_ratio']:.1f}x median bias).",
    ]
    # 仅在实际观测到时报告锚点集中度。
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
        head.extend(_doors_headline(doors, acc))
    heur = _accuracy_stats(rows, key="heuristic")
    if heur.get("n") and heur["median_abs_factor"] < overall["median_abs_factor"]:
        head.append(
            f"- The offline fallback is **better than the LLM it replaced**: "
            f"`material_mu_rho` scores {heur['median_abs_factor']:.1f}x "
            f"({heur['within_2x']:.0%} within 2x) on the same items, against the "
            f"model's {overall['median_abs_factor']:.1f}x. In the shipped "
            f"configuration, calling the API makes the estimate worse.")
    parts.append("\n## Headline\n\n" + "\n".join(head) + "\n")

    # 准确率部分。
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

    # 风险部分。
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

    # 尺寸独立性部分。
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

    # 锚点顺序部分。
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

    # 路线影响部分。
    if doors:
        parts.extend(_report_doors(doors, acc, risk))

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
        _write_doors_csv(doors)
        _chart_doors_gap(doors, acc, risk,
                         os.path.join(OUT_DIR, "doors_gap.png"))
        _chart_doors_gates(doors, os.path.join(OUT_DIR, "doors_gates.png"))
    return path


STAGES = ("accuracy", "risk", "size", "order", "doors", "doors-calibrate",
          "report", "all")
# `all` 不包含 doors-calibrate：它只在改过地图之后需要重跑一次。
ALL_STAGES = ("accuracy", "risk", "size", "order", "doors", "report")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--repeats", type=int, default=2,
                    help="LLM calls per item in the accuracy and risk stages")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    ap.add_argument("--doors-seeds", type=int, default=DOORS_SEEDS,
                    help="random-direction repeats per rung of the Gap ladder")
    ap.add_argument("--doors-workers", type=int,
                    default=max(1, min(6, (os.cpu_count() or 2) - 1)),
                    help="parallel planner runs in the offline doors stages; "
                         "runs are independent and seeded, so this does not "
                         "change any result")
    ap.add_argument("--doors-quick", action="store_true",
                    help="short Gap ladder for a smoke test")
    ap.add_argument("--no-doors-shots", action="store_true",
                    help="skip the route screenshots the doors stage renders")
    args = ap.parse_args()

    cfg = Config()
    cfg.verbose = False
    # 两个估计器只在对应策略位打开时才持有密钥，本实验两个都要测。
    cfg.strategy = validate_strategy("llm-cost-risk")
    wanted = ALL_STAGES if args.stage == "all" else (args.stage,)

    if "accuracy" in wanted:
        stage_accuracy(cfg, args.repeats, args.workers)
    if "risk" in wanted:
        stage_risk(cfg, args.repeats, args.workers)
    if "size" in wanted:
        stage_size(cfg, args.workers)
    if "order" in wanted:
        stage_order(cfg, args.workers)
    if "doors-calibrate" in wanted:
        stage_doors_calibrate(args.doors_workers)
    if "doors" in wanted:
        stage_doors(args.doors_seeds, args.doors_workers, args.doors_quick,
                    not args.no_doors_shots)
    if "report" in wanted:
        stage_report()


if __name__ == "__main__":
    sys.exit(main())
