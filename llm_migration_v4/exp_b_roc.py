"""实验 B：家族内 ROC 标定（设计书 7.2）。只读已有分数，不改采集与评分。

预注册规则（本文件写定后不再按结果回改）：

标签（主线）
  正例-强  sparsegpt / wanda / gblm
  正例-SFT fine_tune / weight_perm / rep_perm / scale / prune_dpoff
  正例-难  derivative_cn（不进入部署 ROC 的正类，单独报告落在阈值哪一侧）
  负例    neg_qwen2_5_7b / neg_llama3_8b

样本单位：固定 (N, dims) 下，一个 (模型, prompt, seed_set) 的 similarity。
主配置 N=128、d=8；同时报告 N=20、d=8。even4 与 mid2 由 V4_LAYER_SUBSET 分开，
不把两套层的分数放进同一条 ROC。

操作点
  在 seed_set=0 上标定，在 seed_set=1 与 2 上评估。
  预测规则：similarity >= t 判为正。
  t(FPR=0)：t = 标定负例的最大值；标定集上 FPR=0（无并列时）。
  t(FPR=0.05)：t = 标定负例的 95% 分位数（numpy quantile, linear）。
  部署通过（H-B1）：seed 1 与 seed 2 上，强正例与 SFT 的 TPR 都 >= 0.90，
  且两侧 FPR 都 <= 0.10。用 FPR=0 与 FPR=0.05 两个预注册目标各判一次。
  阈值跨 seed 稳定：三个 seed 各自的「负例最大 similarity」极差 <= 0.02。

命题 T（只对 even4；v3 用 pert_*，即 4 均匀层，不用 pert_middle2_*）
  从 v3、N=128、d=8、全部 3 个 seed、fine_tune vs 5 个负例导出两条绝对阈值，
  然后原样搬到 v4 even4：
    t_mean  = (fine_tune 全体样本均值 + 负例全体样本均值) / 2
    t_youden = 使 Youden J = TPR-FPR 最大的阈值（候选为两类分数的中点）
  主陈述用 t_mean（对应设计书里 0.760 vs 0.24 的均值间隔）。
  v3 逐 prompt 分布若重叠，必须写明：0.5B 上也不存在 TPR=1 且 FPR=0 的全局阈值。
  T 失败：v4 的 SFT 组在 t_mean 下被大量误拒，或跨家族负例被大量误接。

跨线：把 even4、seed 0、FPR=0 的 t 原样切蒸馏线。预期整条 1B 线尺度不同，切不开。
蒸馏误用：student_distill 为正、student_control 为负做 ROC。预期不可分。
"""
import json
import os
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SUB = os.environ.get("V4_LAYER_SUBSET", "all")
S = H / ("scores" if SUB in ("", "all") else f"scores_{SUB}")
V3 = H.parent / "llm_migration_v3" / "scores"
OUT = H / f"exp_b_roc_{'even4' if SUB in ('', 'all') else SUB}.json"

STRONG = ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]
SFT = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff"]
HARD = ["derivative_cn"]
NEG = ["neg_qwen2_5_7b", "neg_llama3_8b"]


def load_line(scores_dir, line):
    out = {}
    for f in sorted(scores_dir.glob(f"{line}_*.json")):
        if f.name.startswith(f"{line}_summary") or f.stem.endswith("summary"):
            continue
        name = f.stem[len(line) + 1:]
        if name.startswith("summary"):
            continue
        out[name] = json.loads(f.read_text())
    return out


def take(rows, n, dims, seeds):
    return np.asarray([r["similarity"] for r in rows
                       if r["N"] == n and r["dims"] == dims and r["seed_set"] in seeds],
                      dtype=np.float64)


def pool(table, names, n, dims, seeds):
    parts = [take(table[k], n, dims, seeds) for k in names]
    return np.concatenate(parts) if parts else np.asarray([])


def roc_curve(pos, neg):
    """Return thresholds (descending score), tpr, fpr. Positive if score >= t."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    # evaluate at the end of each tied score
    distinct = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tpr = tps[distinct] / len(pos)
    fpr = fps[distinct] / len(neg)
    thr = scores[distinct]
    return thr, tpr, fpr


def auc(pos, neg):
    """Tie-averaged Mann-Whitney AUC. 0.5 = 不可分。"""
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
        # ranks are 1-based average
        avg = 0.5 * (i + j) + 1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rates(pos, neg, t):
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    return dict(tpr=float((pos >= t).mean()) if len(pos) else None,
                fpr=float((neg >= t).mean()) if len(neg) else None,
                n_pos=int(len(pos)), n_neg=int(len(neg)))


def youden_threshold(pos, neg):
    thr, tpr, fpr = roc_curve(pos, neg)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(fpr[k]), float(j[k])


def summarize_setting(table, n, dims):
    pos = pool(table, STRONG + SFT, n, dims, {0, 1, 2})
    neg = pool(table, NEG, n, dims, {0, 1, 2})
    hard = pool(table, HARD, n, dims, {0, 1, 2})
    strong = pool(table, STRONG, n, dims, {0, 1, 2})
    sft = pool(table, SFT, n, dims, {0, 1, 2})
    cal_neg = pool(table, NEG, n, dims, {0})
    t0 = float(cal_neg.max())
    t05 = float(np.quantile(cal_neg, 0.95))
    seed_max = {str(s): float(pool(table, NEG, n, dims, {s}).max()) for s in (0, 1, 2)}
    out = {
        "N": n, "dims": dims,
        "auc_strong_sft_vs_neg": auc(pos, neg),
        "auc_strong_only": auc(strong, neg),
        "auc_sft_only": auc(sft, neg),
        "auc_hard_only": auc(hard, neg),
        "n_pos_deploy": int(len(pos)), "n_neg": int(len(neg)),
        "seed0_thresholds": {"fpr0": t0, "fpr05": t05},
        "per_seed_neg_max": seed_max,
        "neg_max_range": float(max(seed_max.values()) - min(seed_max.values())),
        "stable_across_seeds": bool(max(seed_max.values()) - min(seed_max.values()) <= 0.02),
        "operating_points": {},
        "hard_vs_t": {},
        "per_model_mean": {},
    }
    for label, t in (("fpr0", t0), ("fpr05", t05)):
        ev = {}
        for seed in (0, 1, 2):
            ev[str(seed)] = {
                "strong": rates(pool(table, STRONG, n, dims, {seed}),
                                pool(table, NEG, n, dims, {seed}), t),
                "sft": rates(pool(table, SFT, n, dims, {seed}),
                             pool(table, NEG, n, dims, {seed}), t),
                "hard": rates(pool(table, HARD, n, dims, {seed}),
                              pool(table, NEG, n, dims, {seed}), t),
                "deploy": rates(pool(table, STRONG + SFT, n, dims, {seed}),
                                pool(table, NEG, n, dims, {seed}), t),
            }
        held = [ev["1"], ev["2"]]
        passed = all(h["strong"]["tpr"] >= 0.90 and h["sft"]["tpr"] >= 0.90
                     and h["deploy"]["fpr"] <= 0.10 for h in held)
        out["operating_points"][label] = {"t": t, "by_seed": ev, "heldout_pass": bool(passed)}
        out["hard_vs_t"][label] = {
            "heldout_tpr": [ev["1"]["hard"]["tpr"], ev["2"]["hard"]["tpr"]],
        }
    for name, rows in table.items():
        if name == "victim":
            continue
        v = take(rows, n, dims, {0, 1, 2})
        if len(v) == 0:
            continue
        out["per_model_mean"][name] = {
            "mean": float(v.mean()),
            "above_t_fpr0": float((v >= t0).mean()),
            "above_t_fpr05": float((v >= t05).mean()),
        }
    return out


def load_v3():
    """v3 默认 pert（4 均匀层）。文件名里的点保留。"""
    out = {}
    for f in sorted(V3.glob("pert_*.json")):
        if f.name.startswith("pert_middle2"):
            continue
        name = f.stem[len("pert_"):]
        out[name] = json.loads(f.read_text())
    return out


def proposition_t(v4):
    v3 = load_v3()
    if "fine_tune" not in v3:
        return {"error": "missing v3 pert_fine_tune.json"}
    neg_names = [k for k in v3 if k.startswith("neg_")]
    ft = pool(v3, ["fine_tune"], 128, 8, {0, 1, 2})
    neg = pool(v3, neg_names, 128, 8, {0, 1, 2})
    t_mean = float((ft.mean() + neg.mean()) / 2)
    t_y, tpr_y, fpr_y, j = youden_threshold(ft, neg)
    # overlap on v3 itself
    overlap = bool(ft.min() <= neg.max())
    def apply(t):
        sft = pool(v4, SFT, 128, 8, {0, 1, 2})
        strong = pool(v4, STRONG, 128, 8, {0, 1, 2})
        hard = pool(v4, HARD, 128, 8, {0, 1, 2})
        neg4 = pool(v4, NEG, 128, 8, {0, 1, 2})
        return {
            "t": t,
            "v4_sft_tpr": float((sft >= t).mean()),
            "v4_strong_tpr": float((strong >= t).mean()),
            "v4_hard_tpr": float((hard >= t).mean()),
            "v4_neg_fpr": float((neg4 >= t).mean()),
            "v4_sft_mean": float(sft.mean()),
            "v4_neg_mean": float(neg4.mean()),
        }
    return {
        "v3_files_neg": neg_names,
        "v3_ft_mean": float(ft.mean()),
        "v3_neg_mean": float(neg.mean()),
        "v3_ft_min": float(ft.min()),
        "v3_neg_max": float(neg.max()),
        "v3_per_prompt_overlap": overlap,
        "v3_auc_ft_vs_neg": auc(ft, neg),
        "t_mean": apply(t_mean),
        "t_youden": {**apply(t_y), "v3_tpr": tpr_y, "v3_fpr": fpr_y, "v3_youden_j": j},
        "note": ("t_mean 是两类均值的中点，对应设计书的均值间隔叙述。"
                 "若 v3_per_prompt_overlap 为真，则 0.5B 上也不存在同时 TPR=1 与 FPR=0 的全局绝对阈值；"
                 "v3 的 32/32 来自逐 prompt 与负例比大小，不是一条全局阈值。"),
    }


def distill_checks(main_table, t_from_main):
    D = load_line(S, "distill")
    if "student_distill" not in D or "student_control" not in D:
        return {"error": "missing distill scores in " + str(S)}
    out = {"applied_t": t_from_main, "models": {}}
    for name, rows in D.items():
        if name == "victim":
            continue
        v = take(rows, 128, 8, {0, 1, 2})
        out["models"][name] = {
            "mean": float(v.mean()),
            "frac_above_main_t": float((v >= t_from_main).mean()),
        }
    pos = take(D["student_distill"], 128, 8, {0, 1, 2})
    neg = take(D["student_control"], 128, 8, {0, 1, 2})
    # paired by (prompt, seed)
    gpos = {(r["prompt"], r["seed_set"]): r["similarity"] for r in D["student_distill"]
            if r["N"] == 128 and r["dims"] == 8}
    gneg = {(r["prompt"], r["seed_set"]): r["similarity"] for r in D["student_control"]
            if r["N"] == 128 and r["dims"] == 8}
    keys = sorted(set(gpos) & set(gneg))
    delta = np.asarray([gpos[k] - gneg[k] for k in keys])
    out["distill_vs_control"] = {
        "auc": auc(pos, neg),
        "mean_distill": float(pos.mean()),
        "mean_control": float(neg.mean()),
        "paired_mean_delta": float(delta.mean()),
        "paired_distill_higher": int((delta > 0).sum()),
        "n_pairs": int(len(delta)),
    }
    return out


def main():
    if not S.is_dir():
        raise SystemExit(f"missing {S}")
    table = load_line(S, "main")
    for k in STRONG + SFT + HARD + NEG:
        if k not in table:
            raise SystemExit(f"missing main scores for {k} in {S}")
    report = {
        "scores_dir": str(S),
        "layer_subset": "even4" if SUB in ("", "all") else SUB,
        "rule": "seed0 calibrate; positive if similarity>=t; see module docstring",
        "settings": [summarize_setting(table, n, 8) for n in (128, 20)],
    }
    t0 = report["settings"][0]["seed0_thresholds"]["fpr0"]
    if report["layer_subset"] == "even4":
        report["proposition_T"] = proposition_t(table)
    else:
        report["proposition_T"] = {
            "skipped": True,
            "reason": "命题 T 只在 even4 上检验；mid2 与 v3 全层中间 2 层不是同一组层，不搬绝对阈值。",
        }
    report["cross_line_and_distill_misuse"] = distill_checks(table, t0)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({
        "out": str(OUT),
        "subset": report["layer_subset"],
        "N128": {k: report["settings"][0][k] for k in
                 ("auc_strong_sft_vs_neg", "auc_sft_only", "auc_hard_only",
                  "seed0_thresholds", "neg_max_range", "stable_across_seeds")},
        "ops": {k: {"t": v["t"], "heldout_pass": v["heldout_pass"],
                    "held_sft_tpr": [v["by_seed"]["1"]["sft"]["tpr"], v["by_seed"]["2"]["sft"]["tpr"]],
                    "held_fpr": [v["by_seed"]["1"]["deploy"]["fpr"], v["by_seed"]["2"]["deploy"]["fpr"]],
                    "held_hard_tpr": v["by_seed"]["1"]["hard"]["tpr"]}
                for k, v in report["settings"][0]["operating_points"].items()},
        "T": report["proposition_T"] if report["layer_subset"] != "even4"
             else {k: report["proposition_T"][k] for k in
                   ("v3_ft_mean", "v3_neg_mean", "v3_per_prompt_overlap", "t_mean", "t_youden")},
        "distill_misuse": report["cross_line_and_distill_misuse"].get("distill_vs_control"),
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
