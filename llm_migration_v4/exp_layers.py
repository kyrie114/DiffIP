"""实验二：层数。规则在出结果之前写死。

维数读 `exp_dim_result.json` 的 chosen_dims，不在这里重选。
特征在 feats_all/（采集时 V4_ALL_LAYERS=1）。不覆盖 feats/ 和 scores/。

小盘只有：victim、fine_tune、neg_pruned_sparsegpt、neg_qwen2_5_7b、neg_llama3_8b。
rep_perm 和 scale 从这份 fine_tune 特征推导，变换和 score.py 相同。
weight_perm、prune_dpoff、wanda、gblm、derivative_cn 不在这张盘上，不外推它们的召回。

留出判定和实验 D 同一条算术，正例改成盘上实际有的：
强正例 = sparsegpt；微调簇 = fine_tune、rep_perm、scale。
通过：两个参照方向上，种子 1 和 2 的强正例召回、微调簇召回都 ≥ 0.90，误报 ≤ 0.10。

四种抽法：
- control：各模型已存 feats/ 的 layers4。32 层模型是 0,10,21,31。
  Qwen2.5-7B 只有 28 层，已存的是 0,9,18,27。两侧都是 4 层，按存储顺序配对。
- uniform8：32 层用方案写明的 0,4,9,13,18,22,26,31。
  28 层用 even_layers(28, 8)。方案里的 26 不是 even_layers(32, 8) 算出的 27。
- last4：各模型最后 4 层。32 层即 28..31，28 层即 24..27。
- all：32 层模型用全部 32 层，彼此按层号配对。
  Qwen 只有 28 层。一次 ALS 要求两侧层数相同，所以 Qwen 对 victim 时
  Qwen 用自己的 28 层，victim 用 even_layers(32, 28)（0 和 31 都在，不重复）。
  同为 32 层的模型之间仍然是 32 对 32。

复现门：control 在选定维、N=128 上，和实验一用旧 4 层特征打出的分数比。
最大绝对差 > 1e-3 则整组不信，默认留在均匀 4 层，原因 reproduce_failed。

选层的间隔、稳、0.02 / 0.01 替换线和实验一相同，基线是 control。
只有 uniform8 或 all 可以替换默认。last4 只报告，不能成为默认。
control 不通过时，若 uniform8 或 all 通过，按实验一的 best_passing 在这两者里选。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import score  # noqa: E402
from common import even_layers  # noqa: E402
from exp_d_prompt_calib import auc  # noqa: E402
from exp_dim import MARGIN_GAIN, STD_SLACK  # noqa: E402

N = 128
FEATS = H / "feats_all"
OLD = H / "feats"
OUT_DIR = H / "scores_layers"
RESULT = H / "exp_layers_result.json"
IDENT_TOL = 1e-3
COLLECTED = [
    "fine_tune", "neg_pruned_sparsegpt", "neg_qwen2_5_7b", "neg_llama3_8b",
]
DERIVED = ["rep_perm", "scale"]
STRONG = ["neg_pruned_sparsegpt"]
SFT = ["fine_tune", "rep_perm", "scale"]
NEG_PAIRS = (
    ("neg_qwen2_5_7b", "neg_llama3_8b"),
    ("neg_llama3_8b", "neg_qwen2_5_7b"),
)
UNIFORM8_32 = [0, 4, 9, 13, 18, 22, 26, 31]
CAN_REPLACE = ("uniform8", "all")
SUBSETS = ("control", "uniform8", "all", "last4")

# Filled in the parent before the pool is forked. Workers only read.
ARRAYS: dict[str, np.ndarray] = {}


def _derive(base, name):
    if name == "rep_perm":
        from llm_migration_v2.transforms import make_permutation
        perm = make_permutation(base.shape[-1], seed=1).argmax(0).numpy()
        return base[..., perm]
    from llm_migration_v2.transforms import make_anisotropic_scale
    scale = np.diag(make_anisotropic_scale(base.shape[-1], seed=0).numpy())
    return base * scale


def _load_pair(name):
    old = np.load(OLD / f"main_{name}.npz")
    new = np.load(FEATS / f"main_{name}.npz")
    control = [int(x) for x in old["layers4"]]
    layers = [int(x) for x in new["layers"]]
    total = int(new["total"])
    if layers != list(range(total)):
        raise SystemExit(f"{name} feats_all layers are not 0..{total - 1}")
    if old["pert"].shape[-1] != new["pert"].shape[-1]:
        raise SystemExit(f"{name} hidden size changed")
    sliced = new["pert"][:, :, control]
    if sliced.shape != old["pert"].shape:
        raise SystemExit(f"{name} control slice {sliced.shape} != old {old['pert'].shape}")
    diff = float(np.max(np.abs(sliced.astype(np.float64) - old["pert"].astype(np.float64))))
    return new["pert"], total, control, diff


def subset_index(total, control, subset, peer_total):
    """Layer indices for one model. `peer_total` is the other side's depth.

    Lengths match across the pair. See the module docstring for the Qwen case.
    """
    if subset == "control":
        idx = list(control)
    elif subset == "uniform8":
        idx = list(UNIFORM8_32) if total == 32 else even_layers(total, 8)
    elif subset == "last4":
        idx = list(range(total - 4, total))
    elif subset == "all":
        if total == peer_total:
            idx = list(range(total))
        elif total < peer_total:
            idx = list(range(total))
        else:
            idx = even_layers(total, peer_total)
    else:
        raise SystemExit(subset)
    if any(i < 0 or i >= total for i in idx):
        raise SystemExit(f"{subset} {idx} outside 0..{total - 1}")
    return idx


def _job(job):
    name, pi, ki, dims, subset, s_idx, v_idx = job
    sus = ARRAYS[name]
    vic = ARRAYS["victim"]
    dm, sim, used = score.als(sus[pi, ki, s_idx, :N], vic[pi, ki, v_idx, :N], dims)
    return dict(pair=name, prompt=pi, seed_set=ki, N=N, dims=dims, subset=subset,
                distance_mean=dm, similarity=sim, n_components=used)


def _margins(table, names, anchor, subset, seeds):
    ag = {(r["prompt"], r["seed_set"]): r["similarity"]
          for r in table[anchor] if r["subset"] == subset}
    vals = []
    for name in names:
        g = {(r["prompt"], r["seed_set"]): r["similarity"]
             for r in table[name] if r["subset"] == subset}
        for key, sim in g.items():
            if key[1] in seeds:
                vals.append(sim - ag[key])
    return np.asarray(vals, dtype=np.float64)


def _rates(pos, neg, t):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    return float((pos >= t).mean()), float((neg >= t).mean())


def direction(table, anchor, test, subset):
    t = float(_margins(table, [test], anchor, subset, {0}).max())
    held = {}
    passed = True
    for seed in (1, 2):
        strong_tpr, _ = _rates(_margins(table, STRONG, anchor, subset, {seed}),
                               _margins(table, [test], anchor, subset, {seed}), t)
        sft_tpr, _ = _rates(_margins(table, SFT, anchor, subset, {seed}),
                            _margins(table, [test], anchor, subset, {seed}), t)
        _, fpr = _rates(_margins(table, STRONG + SFT, anchor, subset, {seed}),
                        _margins(table, [test], anchor, subset, {seed}), t)
        held[str(seed)] = {"strong_tpr": strong_tpr, "sft_tpr": sft_tpr, "fpr": fpr}
        if not (strong_tpr >= 0.90 and sft_tpr >= 0.90 and fpr <= 0.10):
            passed = False
    ft = _margins(table, ["fine_tune"], anchor, subset, {1, 2})
    pos = _margins(table, STRONG + SFT, anchor, subset, {0, 1, 2})
    neg = _margins(table, [test], anchor, subset, {0, 1, 2})
    return {
        "anchor": anchor, "test": test, "t": t, "pass": passed, "held": held,
        "auc_deploy": auc(pos, neg),
        "ft_held_median": float(np.median(ft)),
        "ft_held_std": float(np.std(ft)),
    }


def choose_subset(records):
    """Baseline is control. Only uniform8 and all may become the default."""
    by = {r["subset"]: r for r in records}
    base = by["control"]
    passing = [r for r in records if r["passed"]]
    if not base["passed"] and not any(r["subset"] in CAN_REPLACE for r in passing):
        return "control", "none_passed"
    if base["passed"]:
        better = [
            r for r in passing
            if r["subset"] in CAN_REPLACE
            and r["margin"] >= base["margin"] + MARGIN_GAIN
            and r["std"] <= base["std"] + STD_SLACK
        ]
        if not better:
            last = by["last4"]
            if (last["passed"]
                    and last["margin"] >= base["margin"] + MARGIN_GAIN
                    and last["std"] <= base["std"] + STD_SLACK):
                return "control", "last4_not_a_default"
            return "control", "gap_not_clear"
        better.sort(key=lambda r: (-r["margin"], r["std"], r["subset"]))
        return better[0]["subset"], "clear_gain"
    eligible = [r for r in passing if r["subset"] in CAN_REPLACE]
    best_std = min(r["std"] for r in eligible)
    stable = [r for r in eligible if r["std"] <= best_std + STD_SLACK]
    pool = stable or eligible
    pool.sort(key=lambda r: (-r["margin"], r["std"], r["subset"]))
    return pool[0]["subset"], "best_passing"


def _self_test():
    def rec(subset, passed, margin, std):
        return {"subset": subset, "passed": passed, "margin": margin, "std": std}

    rows = [rec("control", True, 0.10, 0.02), rec("uniform8", True, 0.11, 0.02),
            rec("all", True, 0.13, 0.02), rec("last4", True, 0.20, 0.01)]
    if choose_subset(rows) != ("all", "clear_gain"):
        raise SystemExit(f"self-test 1 {choose_subset(rows)}")
    rows[2]["margin"] = 0.115
    if choose_subset(rows) != ("control", "last4_not_a_default"):
        raise SystemExit(f"self-test 2 {choose_subset(rows)}")
    rows = [rec("control", False, 0.01, 0.05), rec("uniform8", True, 0.08, 0.02),
            rec("all", False, 0.02, 0.02), rec("last4", True, 0.30, 0.01)]
    if choose_subset(rows) != ("uniform8", "best_passing"):
        raise SystemExit(f"self-test 3 {choose_subset(rows)}")


def main():
    _self_test()
    dim_path = H / "exp_dim_result.json"
    if not dim_path.exists():
        raise SystemExit("missing exp_dim_result.json — run exp_dim.py first")
    dims = int(json.loads(dim_path.read_text())["chosen_dims"])
    meta = {}
    for name in ["victim", *COLLECTED]:
        pert, total, control, diff = _load_pair(name)
        ARRAYS[name] = pert
        meta[name] = {"total": total, "control": control, "feat_max_abs": diff}
        print(f"feat {name} control max|Δ|={diff:.3e} layers={control}", flush=True)
    ARRAYS["rep_perm"] = _derive(ARRAYS["fine_tune"], "rep_perm")
    ARRAYS["scale"] = _derive(ARRAYS["fine_tune"], "scale")

    jobs = []
    layer_log = {s: {} for s in SUBSETS}
    for subset in SUBSETS:
        for name in COLLECTED + DERIVED:
            src = "fine_tune" if name in DERIVED else name
            s_idx = subset_index(meta[src]["total"], meta[src]["control"], subset,
                                 meta["victim"]["total"])
            v_idx = subset_index(meta["victim"]["total"], meta["victim"]["control"], subset,
                                 meta[src]["total"])
            if len(s_idx) != len(v_idx):
                raise SystemExit(f"{subset} {name}: {len(s_idx)} vs {len(v_idx)}")
            layer_log[subset][name] = {"suspect": s_idx, "victim": v_idx}
            for p in range(32):
                for k in range(3):
                    jobs.append((name, p, k, dims, subset, s_idx, v_idx))
    print(f"jobs={len(jobs)} dims={dims}", flush=True)
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "4"))
    per_pair = defaultdict(list)
    # Fork so workers share the read-only feature arrays instead of pickling them.
    with get_context("fork").Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            per_pair[row["pair"]].append(row)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in per_pair.items():
        (OUT_DIR / f"main_{name}.json").write_text(json.dumps(rows))

    src = H / "scores_dim"
    repro = {}
    worst = 0.0
    for name in COLLECTED:
        old_rows = json.loads((src / f"main_{name}.json").read_text())
        old_map = {(r["prompt"], r["seed_set"]): r["similarity"]
                   for r in old_rows if r["N"] == N and r["dims"] == dims}
        if len(old_map) != 96:
            raise SystemExit(f"exp1 rows for {name} d={dims} are {len(old_map)}")
        fresh = {(r["prompt"], r["seed_set"]): r["similarity"]
                 for r in per_pair[name] if r["subset"] == "control"}
        m = float(max(abs(fresh[k] - old_map[k]) for k in fresh))
        repro[name] = m
        worst = max(worst, m)
        print(f"repro {name} max|Δsim|={m:.3e}", flush=True)

    records = []
    for subset in SUBSETS:
        dirs = [direction(per_pair, a, b, subset) for a, b in NEG_PAIRS]
        records.append({
            "subset": subset,
            "passed": all(d["pass"] for d in dirs),
            "margin": float(min(d["ft_held_median"] for d in dirs)),
            "std": float(max(d["ft_held_std"] for d in dirs)),
            "directions": dirs,
            "layers": layer_log[subset],
        })
    stat_subset, stat_reason = choose_subset(records)
    chosen, reason = stat_subset, stat_reason
    if worst > IDENT_TOL:
        chosen, reason = "control", "reproduce_failed"
    result = {
        "dims": dims,
        "ident_tol": IDENT_TOL,
        "feat_control_max_abs": {k: v["feat_max_abs"] for k, v in meta.items()},
        "repro_max_abs_sim": repro,
        "repro_worst": worst,
        "statistical_choice": {"subset": stat_subset, "reason": stat_reason},
        "chosen_subset": chosen,
        "reason": reason,
        "records": records,
    }
    RESULT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE subset={chosen} reason={reason} repro={worst:.3e}", flush=True)


if __name__ == "__main__":
    main()
