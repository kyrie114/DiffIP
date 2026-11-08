"""实验一：7B 上把表示压到几维。

规则在出结果之前写死，对应 `试验方案_维数层数与无关模型.md`。
判定算术直接调用 `exp_d_prompt_calib.one_direction`（种子 0 定阈值，
种子 1 和 2 留出；强正例与微调簇召回 ≥ 0.90，误报 ≤ 0.10）。
中文衍生只随方向结果留下，不参与通过，也不参与选维。

间隔：两个参照方向里，更差的那个方向上，fine_tune 在种子 1 和 2
的间隔中位数。稳：这两个方向里，同一批间隔的标准差（ddof=0）的较大者。

选维：
- 全都不通过：留 8，原因 none_passed。
- 8 通过：另一维只有在间隔至少高 0.02，且标准差不高过 8 维 0.01 时才替换。
  0.02 是实验 B 的 seed 稳定线，也是实验 C/E 的 0.020 带。
  多个维都跨过这条线时，取间隔更大者；间隔相同取标准差更小者，再取更小的维。
  没有维跨过这条线则留 8，原因 gap_not_clear。
- 8 不通过但有别的维通过：在标准差不比最稳的通过维高过 0.01 的那些里，
  取间隔最大者；间隔相同取标准差更小者，再取更小的维。
  若这样一个都不剩，就在全部通过维里按同一顺序取。原因 best_passing。

N=128。特征用已经采好的均匀 4 层，不覆盖 `scores/`。
若重打的 8 维、20 维和已存分数的最大绝对差超过 1e-3，不改默认维。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import score  # noqa: E402
from exp_d_prompt_calib import margins_for, one_direction  # noqa: E402

N = 128
DIMS = [8, 10, 15, 20, 30, 40]
NAMES = [
    "fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff",
    "derivative_cn",
    "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm",
    "neg_qwen2_5_7b", "neg_llama3_8b",
]
NEG_PAIRS = (
    ("neg_qwen2_5_7b", "neg_llama3_8b"),
    ("neg_llama3_8b", "neg_qwen2_5_7b"),
)
MARGIN_GAIN = 0.02
STD_SLACK = 0.01
STORED = H / "scores"
OUT_DIR = H / "scores_dim"
RESULT = H / "exp_dim_result.json"
IDENT_TOL = 1e-3


def choose(records):
    """Return (dims, reason). `records` items have dims, passed, margin, std."""
    passing = [r for r in records if r["passed"]]
    if not passing:
        return 8, "none_passed"
    base = next((r for r in records if r["dims"] == 8), None)
    if base is not None and base["passed"]:
        better = [
            r for r in passing
            if r["dims"] != 8
            and r["margin"] >= base["margin"] + MARGIN_GAIN
            and r["std"] <= base["std"] + STD_SLACK
        ]
        if not better:
            return 8, "gap_not_clear"
        better.sort(key=lambda r: (-r["margin"], r["std"], r["dims"]))
        return better[0]["dims"], "clear_gain"
    best_std = min(r["std"] for r in passing)
    stable = [r for r in passing if r["std"] <= best_std + STD_SLACK]
    pool = stable or passing
    pool.sort(key=lambda r: (-r["margin"], r["std"], r["dims"]))
    return pool[0]["dims"], "best_passing"


def _self_test():
    def rec(dims, passed, margin, std):
        return {"dims": dims, "passed": passed, "margin": margin, "std": std}

    cases = [
        ([rec(8, False, 0.01, 0.01), rec(20, False, 0.02, 0.01)],
         (8, "none_passed")),
        ([rec(8, True, 0.10, 0.02), rec(20, True, 0.11, 0.02)],
         (8, "gap_not_clear")),
        ([rec(8, True, 0.10, 0.02), rec(20, True, 0.119, 0.02)],
         (8, "gap_not_clear")),
        ([rec(8, True, 0.10, 0.02), rec(40, True, 0.125, 0.02), rec(20, True, 0.13, 0.04)],
         (40, "clear_gain")),
        ([rec(8, True, 0.10, 0.005), rec(40, True, 0.13, 0.02)],
         (8, "gap_not_clear")),
        ([rec(8, False, 0.01, 0.05), rec(10, True, 0.05, 0.02), rec(30, True, 0.08, 0.04)],
         (10, "best_passing")),
        ([rec(8, False, 0.01, 0.2), rec(30, True, 0.08, 0.04)],
         (30, "best_passing")),
        ([rec(8, False, 0.0, 0.2), rec(20, True, 0.05, 0.01), rec(15, True, 0.05, 0.02)],
         (20, "best_passing")),
    ]
    for rows, expect in cases:
        got = choose(rows)
        if got != expect:
            raise SystemExit(f"choose self-test failed: {got} != {expect} for {rows}")


def _job(job):
    name, pi, ki, d = job
    sus = score.load_sus("main", name)
    vic = score._ld_cached("main", "victim")
    dm, sim, used = score.als(sus[pi, ki, :, :N], vic[pi, ki, :, :N], d)
    return dict(pair=name, prompt=pi, seed_set=ki, N=N, dims=d,
                distance_mean=dm, similarity=sim, n_components=used)


def _stored_map(name, dims):
    rows = json.loads((STORED / f"main_{name}.json").read_text())
    return {
        (r["prompt"], r["seed_set"]): r["similarity"]
        for r in rows if r["N"] == N and r["dims"] == dims
    }


def ident_max(per_pair):
    worst = 0.0
    detail = {}
    for name, rows in per_pair.items():
        for dims in (8, 20):
            stored = _stored_map(name, dims)
            diffs = []
            for r in rows:
                if r["dims"] != dims:
                    continue
                diffs.append(abs(r["similarity"] - stored[(r["prompt"], r["seed_set"])]))
            m = float(max(diffs)) if diffs else None
            detail[f"{name}/d{dims}"] = m
            if m is not None:
                worst = max(worst, m)
    return worst, detail


def record_for(table, dims):
    directions = []
    meds, stds = [], []
    passed = True
    for anchor, test in NEG_PAIRS:
        raw = one_direction(table, anchor, test, N, dims)
        op = raw["operating_points"]["fpr0"]
        held = [op["by_seed"]["1"], op["by_seed"]["2"]]
        m = margins_for(table, ["fine_tune"], anchor, N, dims, {1, 2})
        med = float(np.median(m))
        sd = float(np.std(m))
        meds.append(med)
        stds.append(sd)
        passed = passed and bool(op["heldout_pass"])
        directions.append({
            "anchor": anchor,
            "test": test,
            "t": op["t"],
            "auc_deploy": raw["auc_deploy"],
            "auc_sft": raw["auc_sft"],
            "auc_hard": raw["auc_hard"],
            "held_strong_tpr": [h["strong"]["tpr"] for h in held],
            "held_sft_tpr": [h["sft"]["tpr"] for h in held],
            "held_fpr": [h["deploy"]["fpr"] for h in held],
            "held_hard_tpr": [h["hard"]["tpr"] for h in held],
            "ft_held_median": med,
            "ft_held_std": sd,
            "pass": bool(op["heldout_pass"]),
        })
    return {
        "dims": dims,
        "passed": passed,
        "margin": float(min(meds)),
        "std": float(max(stds)),
        "directions": directions,
    }


def main():
    _self_test()
    missing = [n for n in NAMES if n not in ("rep_perm", "scale")
               and not (H / "feats" / f"main_{n}.npz").exists()]
    if missing:
        raise SystemExit(f"missing feats: {missing}")
    if not (H / "feats" / "main_fine_tune.npz").exists():
        raise SystemExit("missing fine_tune feats")
    if not (H / "feats" / "main_victim.npz").exists():
        raise SystemExit("missing victim feats")

    jobs = [(n, p, k, d) for n in NAMES for p in range(32) for k in range(3) for d in DIMS]
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "8"))
    print(f"jobs={len(jobs)} workers={nworkers}", flush=True)
    per_pair = defaultdict(list)
    with Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            per_pair[row["pair"]].append(row)
            if (i + 1) % 400 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in per_pair.items():
        (OUT_DIR / f"main_{name}.json").write_text(json.dumps(rows))
    worst, detail = ident_max(per_pair)
    print(f"ident max |dsim| vs stored d8/d20 = {worst:.3e}", flush=True)
    table = {name: rows for name, rows in per_pair.items()}
    records = [record_for(table, d) for d in DIMS]
    stat_dims, stat_reason = choose(records)
    chosen, reason = stat_dims, stat_reason
    if worst > IDENT_TOL:
        chosen, reason = 8, "rescore_mismatch"
    result = {
        "N": N,
        "dims_compared": DIMS,
        "margin_gain": MARGIN_GAIN,
        "std_slack": STD_SLACK,
        "ident_tol": IDENT_TOL,
        "ident_max_abs": worst,
        "ident_worst_cells": {k: v for k, v in detail.items() if v is not None and v > 1e-5},
        "statistical_choice": {"dims": stat_dims, "reason": stat_reason},
        "chosen_dims": chosen,
        "reason": reason,
        "records": [{k: v for k, v in r.items()} for r in records],
    }
    RESULT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE chosen={chosen} reason={reason} ident={worst:.3e}", flush=True)


if __name__ == "__main__":
    main()
