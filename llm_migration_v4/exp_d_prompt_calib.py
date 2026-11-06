"""实验 D：逐 prompt 相对分数。规则见 实验DE设计.md，不随结果改。"""
import json
import os
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SUB = os.environ.get("V4_LAYER_SUBSET", "all")
S = H / ("scores" if SUB in ("", "all") else f"scores_{SUB}")
OUT = H / f"exp_d_calib_{'even4' if SUB in ('', 'all') else SUB}.json"

STRONG = ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]
SFT = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff"]
HARD = ["derivative_cn"]
NEG = ["neg_qwen2_5_7b", "neg_llama3_8b"]


def load_main():
    out = {}
    for f in sorted(S.glob("main_*.json")):
        name = f.stem[len("main_"):]
        if name.startswith("summary"):
            continue
        out[name] = json.loads(f.read_text())
    return out


def grid(rows, n, dims):
    return {(r["prompt"], r["seed_set"]): r["similarity"]
            for r in rows if r["N"] == n and r["dims"] == dims}


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rates(pos, neg, t):
    return dict(tpr=float((np.asarray(pos) >= t).mean()),
                fpr=float((np.asarray(neg) >= t).mean()),
                n_pos=int(len(pos)), n_neg=int(len(neg)))


def margins_for(table, names, anchor, n, dims, seeds):
    ag = grid(table[anchor], n, dims)
    vals = []
    for name in names:
        g = grid(table[name], n, dims)
        for key, s in g.items():
            if key[1] in seeds:
                vals.append(s - ag[key])
    return np.asarray(vals, dtype=np.float64)


def one_direction(table, anchor, test, n, dims):
    cal_neg = margins_for(table, [test], anchor, n, dims, {0})
    t0 = float(cal_neg.max())
    t05 = float(np.quantile(cal_neg, 0.95))
    out = {"anchor": anchor, "test_negative": test, "N": n, "dims": dims,
           "thresholds": {"fpr0": t0, "fpr05": t05}, "operating_points": {}}
    pos_all = margins_for(table, STRONG + SFT, anchor, n, dims, {0, 1, 2})
    neg_all = margins_for(table, [test], anchor, n, dims, {0, 1, 2})
    out["auc_deploy"] = auc(pos_all, neg_all)
    out["auc_sft"] = auc(margins_for(table, SFT, anchor, n, dims, {0, 1, 2}), neg_all)
    out["auc_strong"] = auc(margins_for(table, STRONG, anchor, n, dims, {0, 1, 2}), neg_all)
    out["auc_hard"] = auc(margins_for(table, HARD, anchor, n, dims, {0, 1, 2}), neg_all)
    for label, t in (("fpr0", t0), ("fpr05", t05)):
        ev = {}
        for seed in (0, 1, 2):
            ev[str(seed)] = {
                "strong": rates(margins_for(table, STRONG, anchor, n, dims, {seed}),
                                margins_for(table, [test], anchor, n, dims, {seed}), t),
                "sft": rates(margins_for(table, SFT, anchor, n, dims, {seed}),
                             margins_for(table, [test], anchor, n, dims, {seed}), t),
                "hard": rates(margins_for(table, HARD, anchor, n, dims, {seed}),
                              margins_for(table, [test], anchor, n, dims, {seed}), t),
                "deploy": rates(margins_for(table, STRONG + SFT, anchor, n, dims, {seed}),
                                margins_for(table, [test], anchor, n, dims, {seed}), t),
            }
        held = [ev["1"], ev["2"]]
        passed = all(h["strong"]["tpr"] >= 0.90 and h["sft"]["tpr"] >= 0.90
                     and h["deploy"]["fpr"] <= 0.10 for h in held)
        out["operating_points"][label] = {"t": t, "by_seed": ev, "heldout_pass": bool(passed)}
    return out


def prompt_corr(table, n, dims):
    def pmean(name):
        g = grid(table[name], n, dims)
        acc = {p: [] for p in range(32)}
        for (p, _), s in g.items():
            acc[p].append(s)
        return np.array([np.mean(acc[p]) for p in range(32)])
    ft = pmean("fine_tune")
    return {neg: pearson(ft, pmean(neg)) for neg in NEG}


def main():
    table = load_main()
    for k in STRONG + SFT + HARD + NEG:
        if k not in table:
            raise SystemExit(f"missing {k} in {S}")
    report = {
        "scores_dir": str(S),
        "layer_subset": "even4" if SUB in ("", "all") else SUB,
        "rule": "m = sim(model)-sim(anchor); seed0 calibrate; see 实验DE设计.md",
        "prompt_pearson_ft_vs_neg": {str(n): prompt_corr(table, n, 8) for n in (128, 20)},
        "directions": [],
    }
    for n in (128, 20):
        for anchor, test in (NEG, NEG[::-1]):
            report["directions"].append(one_direction(table, anchor, test, n, 8))
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    brief = []
    for d in report["directions"]:
        if d["N"] != 128:
            continue
        ops = {k: {"t": v["t"], "pass": v["heldout_pass"],
                   "held_sft_tpr": [v["by_seed"]["1"]["sft"]["tpr"], v["by_seed"]["2"]["sft"]["tpr"]],
                   "held_fpr": [v["by_seed"]["1"]["deploy"]["fpr"], v["by_seed"]["2"]["deploy"]["fpr"]],
                   "held_hard_tpr": [v["by_seed"]["1"]["hard"]["tpr"], v["by_seed"]["2"]["hard"]["tpr"]]}
               for k, v in d["operating_points"].items()}
        brief.append({"anchor": d["anchor"], "test": d["test_negative"],
                      "auc_deploy": d["auc_deploy"], "auc_sft": d["auc_sft"],
                      "auc_hard": d["auc_hard"], "ops": ops})
    print(json.dumps({"out": str(OUT), "subset": report["layer_subset"],
                      "pearson_n128": report["prompt_pearson_ft_vs_neg"]["128"],
                      "directions_n128": brief}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
