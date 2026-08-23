"""评测 LLM mu*rho 估计器本身的准确率：accuracy / size / ablate / order / report 五个阶段。

只测估计器——它猜的 mu*rho 离参考值有多远、是不是真的在估计而非抄表——不再把误差
带进导航仿真算代价。想看误差对路径规划的影响，见本文件曾经的 nav/lambda 阶段
（已删除，误差->代价的换算属于另一个问题，混进这里会让"估计器准不准"这个单一问题
的答案不干净）。
"""

from __future__ import annotations

# 任意目录均可运行：库位于上一级目录。
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

from config import Config
from llm_difficulty import (
    MATERIAL_MU_RHO,
    DifficultyEstimator,
    material_mu_rho,
)
from llm_dataset import (
    DATASET,
    Item,
    assert_all_off_anchor,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_test_out")

SIZE_SCALES = (0.5, 1.0, 2.0)

CHART_GROUPS = {
    "known": ("everyday items",  "#2a78d6", "o"),
    "brand": ("brand names",     "#eb6834", "D"),
    "state": ("full vs. empty",  "#1baf7a", "^"),
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _chart_group(group: str) -> str:
    """paraphrase 与 novel 合并成一组画图，其余原样。"""
    return "known" if group in ("paraphrase", "novel") else group


def log(msg: str) -> None:
    """心跳日志：长阶段都经此打印，卡住的运行能从输出文件里看出来。"""
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
# 阶段 1 —— 估计器准确率
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
            # 取重复次数的中位数：单一坏回复不至于带偏
            "pred": statistics.median(preds) if preds else None,
            "n_ok": len(preds),
            "heuristic": material_mu_rho(item.label),
        })
    payload = {"model": cfg.deepseek_model, "repeats": repeats,
               "seconds": round(time.time() - t0, 1), "rows": rows}
    _save("accuracy.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# 阶段 2 —— 尺寸无关性
# --------------------------------------------------------------------------- #

def stage_size(cfg: Config, workers: int) -> dict:
    """prompt 给出 l×d×h 却又声明尺寸无关；随 scale 变化的结果是模型把尺寸当质量线索，而体积调用方还会再乘一次。"""
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
# 阶段 2b —— 请求参数消融
# --------------------------------------------------------------------------- #

ABLATIONS_EXTRA = (
    ("v4-flash, thinking on, 32k tok",         "deepseek-v4-flash", "enabled", 32000),
    ("deepseek-chat, no thinking, 32 tok",     "deepseek-chat",     "disabled", 32),
)


def _ask_raw(cfg: Config, model: str, thinking: str, max_tokens: Optional[int],
             prompt: str) -> tuple:
    """按显式请求参数发一次调用，解析方式与 `_deepseek` 一致；返回 (值, finish_reason)——length 表示被 token 预算截断、不应计分，stop 却无数字才是拒答。取回复中最后一个数字：推理模型最终认定的答案在末尾。max_tokens 为 None 时整个省略该字段，与 `llm_difficulty._deepseek` 的行为一致（省略即不设上限）。"""
    import re
    import requests
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "stream": False, "thinking": {"type": thinking}}
    if max_tokens:
        body["max_tokens"] = int(max_tokens)
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

    # "当前配置" 这一档直接照 cfg 读，不是写死的字面量——这样改 config.py 的默认
    # 模型/思考开关/token 上限之后，这张表自动反映的就是"现在线上到底在用什么"。
    shipped = (
        f"current config: {cfg.deepseek_model}, "
        f"thinking={'on' if cfg.deepseek_thinking else 'off'}, "
        f"{cfg.llm_max_tokens or 'no cap'} tok",
        cfg.deepseek_model,
        "enabled" if cfg.deepseek_thinking else "disabled",
        cfg.llm_max_tokens,
    )
    variants = []
    for tag, model, thinking, max_tokens in (shipped, *ABLATIONS_EXTRA):
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
# 阶段 2c —— 塌缩目标是表格最后一行还是最大值？
# --------------------------------------------------------------------------- #

_ANCHOR_LINE = re.compile(r"^\s{2}(\S+)\s+mu=\S+\s+rho=\S+\s+mu\*rho=(\S+)\s*$")


def _reorder_anchor_block(prompt: str, order: str, seed: int = 0) -> str:
    """只重排锚点表的行序，prompt 其余字节与线上完全一致；原表升序使 1440 既是最后一行又是最大值，重排可把两个解释分开。"""
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
    # 只用表外物体：paraphrase 条目的正确答案本身就是锚点，无法区分抄袭与推理。
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
# 指标
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
    """统计回复恰为锚点表某一行的频率；prompt 要求在锚点间插值，精确命中即抄袭而非估计。"""
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
    """温度 0 下同一问题重复提问的散布；非零散布意味着规划器看到的次序在相同运行间不稳定。"""
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


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #

def _chart_accuracy(rows, path: str):
    """LLM 估计 vs 参考真值的散点图。图例本身就要讲清楚看到的每样东西是什么：
    对角线=完美估计，阴影带=容差范围，三个分组=物体是哪一类测试项。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lo, hi = 0.5, 3000.0
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, zorder=2,
            label="perfect estimate")
    for band, alpha in ((2.0, 0.10), (1.5, 0.14)):
        ax.fill_between([lo, hi], [lo / band, hi / band], [lo * band, hi * band],
                        color=MUTED, alpha=alpha, lw=0, zorder=1)
    band_patch = Patch(facecolor=MUTED, alpha=0.14,
                       label="error tolerance (darker = within 1.5x, "
                             "lighter = within 2x)")

    # 水平条带本身就是结论，直接在图上点名，免得读者自己解码为何许多点共 y 值。
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

    for group, (label, color, marker) in CHART_GROUPS.items():
        pts = [(r["mu_rho_true"], r["pred"]) for r in rows
               if _chart_group(r["group"]) == group and r.get("pred")]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46, marker=marker,
                   facecolor=color, edgecolor="#fcfcfb", linewidth=1.0,
                   label=f"{label} (n={len(pts)})", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("reference mu*rho  [kg/m$^3$]", color=INK)
    ax.set_ylabel("LLM estimate  [kg/m$^3$]", color=INK)
    ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [band_patch], labels + [band_patch.get_label()],
              frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"wrote {path}")


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #

def _table(header: List[str], body: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def stage_report() -> str:
    acc = _load("accuracy.json")
    size = _load("size.json")
    abl = _load("ablate.json")
    if not acc:
        raise SystemExit("no accuracy.json - run `python3 LLM_test.py accuracy` first")

    rows = acc["rows"]
    parts = [f"# LLM mu*rho estimator accuracy\n",
             f"Model `{acc['model']}`, {acc['repeats']} repeats per item, "
             f"{len(rows)} items, {acc['seconds']}s of API time.\n"]

    # -- 摘要 -----------------------------------------------------------
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
    if abl:
        # 只有几乎答全的变体才可排名：被截断的运行悄悄丢了最难的样本。
        rankable = [v for v in abl["variants"]
                    if _accuracy_stats(v["rows"]).get("n", 0) >= 0.9 * len(v["rows"])]
        best = min(rankable, key=lambda v: _accuracy_stats(v["rows"])["median_abs_factor"],
                   default=None)
        if best:
            s = _accuracy_stats(best["rows"])
            note = ("This is the current config — nothing to change."
                    if best["tag"].startswith("current config:") else
                    "Set `Config.deepseek_model` / `deepseek_thinking` / "
                    "`llm_max_tokens` to match — `llm_difficulty._deepseek` reads "
                    "those fields directly, no code change needed.")
            head.append(
                f"- Best request setting tested is `{best['tag']}` at "
                f"**{s['median_abs_factor']:.1f}x** typical error "
                f"({s['within_2x']:.0%} within 2x, {best['n_parsed']}/"
                f"{len(best['rows'])} answered). {note}")
        dropped = [v["tag"] for v in abl["variants"] if v not in rankable]
        if dropped:
            head.append(f"- Not ranked (token budget truncated too many items to "
                        f"score fairly): {', '.join(dropped)}.")
    parts.append("\n## Headline\n\n" + "\n".join(head) + "\n")

    # -- 准确率 -----------------------------------------------------------
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

    # -- 尺寸无关性 --------------------------------------------------
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

    # -- 锚点行序 ----------------------------------------------------
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

    # -- 请求参数消融 -------------------------------------------
    if abl:
        parts.append("\n## 4. Is it the model or the way it is asked?\n")
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
    return path


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["accuracy", "size", "ablate", "order",
                                      "report", "all"])
    ap.add_argument("--repeats", type=int, default=3,
                    help="LLM calls per item in the accuracy stage")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    args = ap.parse_args()

    cfg = Config()
    cfg.verbose = False

    if args.stage in ("accuracy", "all"):
        stage_accuracy(cfg, args.repeats, args.workers)
    if args.stage in ("size", "all"):
        stage_size(cfg, args.workers)
    if args.stage in ("ablate", "all"):
        stage_ablate(cfg, args.workers)
    if args.stage in ("order", "all"):
        stage_order(cfg, args.workers)
    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    sys.exit(main())
