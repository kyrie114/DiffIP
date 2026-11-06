"""实验 F：检验「一个正类、逐 prompt 相减才是谱系」这个前提。

规则在跑之前写死，不随结果改。

前提：实验 D 能分开，是因为减去了同一条 prompt 上的跨家族参照，
剩下的是谱系。如果换成下面两种减法仍然通过同一条留出规则，
这个前提不成立。

1. misaligned：减去参照在另一条 prompt 上的分数。prompt 置换用
   numpy Generator(42)，固定，不挑置换。
2. constant：减去该 seed 上参照对 32 条 prompt 的均值，不随 prompt 变。

通过线和实验 D 相同：seed 0 标定，seed 1 和 2 评估；强正例与 SFT 的
留出 TPR >= 0.90 且 FPR <= 0.10。derivative 单独报。

另外解释 0.956 这个相关：它是不是 prompt 长度造成的。长度相关和
去掉长度之后的偏相关只作描述，不参与通过与否。
"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
S = H / "scores"
OUT = H / "exp_f_premise.json"

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


def grid(rows):
    return {(r["prompt"], r["seed_set"]): r["similarity"]
            for r in rows if r["N"] == 128 and r["dims"] == 8}


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


def make_baseline(mode, anchor_g, perm):
    """Return a function (prompt, seed) -> baseline."""
    if mode == "aligned":
        return lambda p, s: anchor_g[(p, s)]
    if mode == "misaligned":
        return lambda p, s: anchor_g[(int(perm[p]), s)]
    if mode == "constant":
        means = {}
        for s in (0, 1, 2):
            means[s] = float(np.mean([anchor_g[(p, s)] for p in range(32)]))
        return lambda p, s: means[s]
    raise KeyError(mode)


def collect(table, names, baseline, seeds):
    vals = []
    for name in names:
        g = grid(table[name])
        for (p, s), sim in g.items():
            if s in seeds:
                vals.append(sim - baseline(p, s))
    return np.asarray(vals, dtype=np.float64)


def evaluate(table, mode, anchor, test, perm):
    base = make_baseline(mode, grid(table[anchor]), perm)
    cal = collect(table, [test], base, {0})
    t = float(cal.max())
    ev = {}
    for seed in (0, 1, 2):
        ev[str(seed)] = {
            "strong": rates(collect(table, STRONG, base, {seed}),
                            collect(table, [test], base, {seed}), t),
            "sft": rates(collect(table, SFT, base, {seed}),
                         collect(table, [test], base, {seed}), t),
            "hard": rates(collect(table, HARD, base, {seed}),
                          collect(table, [test], base, {seed}), t),
            "deploy": rates(collect(table, STRONG + SFT, base, {seed}),
                            collect(table, [test], base, {seed}), t),
        }
    held = [ev["1"], ev["2"]]
    passed = all(h["strong"]["tpr"] >= 0.90 and h["sft"]["tpr"] >= 0.90
                 and h["deploy"]["fpr"] <= 0.10 for h in held)
    pos = collect(table, STRONG + SFT, base, {0, 1, 2})
    neg = collect(table, [test], base, {0, 1, 2})
    hard = collect(table, HARD, base, {0, 1, 2})
    return {
        "mode": mode, "anchor": anchor, "test": test, "t_fpr0": t,
        "heldout_pass": bool(passed),
        "auc_deploy": auc(pos, neg),
        "auc_hard": auc(hard, neg),
        "held_sft_tpr": [ev["1"]["sft"]["tpr"], ev["2"]["sft"]["tpr"]],
        "held_strong_tpr": [ev["1"]["strong"]["tpr"], ev["2"]["strong"]["tpr"]],
        "held_fpr": [ev["1"]["deploy"]["fpr"], ev["2"]["deploy"]["fpr"]],
        "held_hard_tpr": [ev["1"]["hard"]["tpr"], ev["2"]["hard"]["tpr"]],
        "sft_margin_median": float(np.median(collect(table, SFT, base, {0, 1, 2}))),
        "hard_margin_median": float(np.median(hard)),
        "neg_margin_median": float(np.median(neg)),
    }


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def length_check(table):
    import sys
    sys.path.insert(0, str(H))
    from common import PROMPTS
    length = np.array([len(p) for p in PROMPTS], dtype=np.float64)

    def pmean(name):
        g = grid(table[name])
        return np.array([np.mean([g[(p, s)] for s in (0, 1, 2)]) for p in range(32)])

    ft, qwen, llama = pmean("fine_tune"), pmean(NEG[0]), pmean(NEG[1])

    def resid(y, x):
        X = np.c_[np.ones(len(x)), x]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ beta

    return {
        "n_prompts": len(PROMPTS),
        "length_vs_ft": pearson(length, ft),
        "length_vs_qwen": pearson(length, qwen),
        "length_vs_llama": pearson(length, llama),
        "ft_vs_qwen": pearson(ft, qwen),
        "ft_vs_qwen_after_length": pearson(resid(ft, length), resid(qwen, length)),
        "ft_vs_llama_after_length": pearson(resid(ft, length), resid(llama, length)),
    }


def main():
    table = load_main()
    perm = np.random.default_rng(42).permutation(32)
    report = {
        "prompt_perm_seed": 42,
        "fixed_points": int(np.sum(perm == np.arange(32))),
        "perm": perm.tolist(),
        "length": length_check(table),
        "rows": [],
    }
    for mode in ("aligned", "misaligned", "constant"):
        for anchor, test in (NEG, list(reversed(NEG))):
            report["rows"].append(evaluate(table, mode, anchor, test, perm))
    OUT.write_text(json.dumps(report, indent=1))
    brief = [{k: r[k] for k in
              ("mode", "anchor", "test", "heldout_pass", "auc_deploy", "auc_hard",
               "t_fpr0", "held_sft_tpr", "held_fpr", "held_hard_tpr",
               "sft_margin_median", "hard_margin_median")}
             for r in report["rows"]]
    print(json.dumps({"fixed_points": report["fixed_points"], "length": report["length"],
                      "rows": brief}, indent=1))


if __name__ == "__main__":
    main()
