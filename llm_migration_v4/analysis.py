"""Post-hoc analysis for the v4 report (reads scores/*.json).

Extra views that score.py's summarize does not print:
- wins vs *easy* (cross-family) negatives only, and vs hard negatives only
- per-prompt paired hard-neg vs fine_tune order
- distill vs control paired deltas (mean diff, prompt sign test)
- pooled ROC AUC same-origin vs cross-family at the largest setting
"""
import collections
import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
S = H / "scores"

MAIN_POS = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff", "derivative_cn"]
MAIN_HARD = ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]
MAIN_EASY = ["neg_qwen2_5_7b", "neg_llama3_8b"]
DISTILL_NAMES = ["student_distill", "student_control", "neg_llama3_2_3b",
                 "neg_sheared_1_3b", "neg_qwen2_5_7b", "neg_llama2_7b"]


def load(line):
    out = {}
    for f in sorted(S.glob(f"{line}_*.json")):
        name = f.stem[len(line) + 1:]
        out[name] = json.loads(f.read_text())
    return out


def sim_grid(rows):
    """[(N, dims, prompt, seed)] -> similarity for fine-grained lookup."""
    return {(r["N"], r["dims"], r["prompt"], r["seed_set"]): r["similarity"]
            for r in rows}


def wins(lines, names, negs, n, dims):
    g = {k: sim_grid(lines[k]) for k in names + negs}
    out = {}
    for k in names:
        if k.startswith("neg_"):
            continue
        w = sum(g[k][(n, dims, p, 0)] > max(g[e][(n, dims, p, 0)] for e in negs)
                for p in range(32))
        out[k] = w
    return out


def auc(pos, neg):
    pos = np.asarray(sorted(pos))
    neg = np.asarray(sorted(neg))
    # tie-aware Mann-Whitney AUC
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    M = load("main")
    D = load("distill")
    print("=" * 20, "MAIN")

    for n in (20, 128):
        for d in (8, 20):
            we = wins(M, MAIN_POS, MAIN_EASY, n, d)
            wh = wins(M, MAIN_POS, MAIN_HARD, n, d)
            print(f"[wins vs easy-only]  N={n} d={d}: " +
                  " ".join(f"{k}={v}/32" for k, v in we.items()))
            print(f"[wins vs hard-only]  N={n} d={d}: " +
                  " ".join(f"{k}={v}/32" for k, v in wh.items()))
            # per-prompt best hard neg vs fine_tune
            ft = sim_grid(M["fine_tune"])
            hng = {k: sim_grid(M[k]) for k in MAIN_HARD}
            hn = None
            hng = {k: sim_grid(M[k]) for k in MAIN_HARD}
            beats = sum(max(v[(n, d, p, 0)] for v in hng.values()) > ft[(n, d, p, 0)]
                        for p in range(32))
            print(f"[hard-negs beat fine_tune on prompts] N={n} d={d}: {beats}/32")

    # distance table at N=128
    for d in (8, 20):
        vals = {k: np.mean([r["distance_mean"] for r in v
                            if r["N"] == 128 and r["dims"] == d])
                for k, v in M.items()}
        print(f"[distance_mean N=128 d={d}] " +
              " ".join(f"{k}={vals[k]:.3f}" for k in sorted(vals)))

    # pooled AUC: same-origin vs cross-family (N=128, d=8)
    for d in (8, 20):
        pos = [r["similarity"] for k in MAIN_POS for r in M[k]
               if r["N"] == 128 and r["dims"] == d]
        neg = [r["similarity"] for k in MAIN_EASY for r in M[k]
               if r["N"] == 128 and r["dims"] == d]
        print(f"[AUC same-origin vs cross-family] d={d}: {auc(pos, neg):.3f} "
              f"({len(pos)} vs {len(neg)} rows)")

    print("=" * 20, "DISTILL")
    for n in (20, 128):
        for d in (8, 20):
            sd = sim_grid(D["student_distill"])
            sc = sim_grid(D["student_control"])
            diff = [sd[(n, d, p, 0)] - sc[(n, d, p, 0)] for p in range(32)]
            print(f"[distill-control] N={n} d={d}: mean={np.mean(diff):+.4f} "
                  f"min={min(diff):+.4f} max={max(diff):+.4f} "
                  f"prompts distill>control={sum(x > 0 for x in diff)}/32")
    # students vs negatives margins (N=128)
    for d in (8, 20):
        for stud in ("student_distill", "student_control"):
            sg = {r["prompt"]: r["similarity"] for r in D[stud]
                  if r["N"] == 128 and r["dims"] == d and r["seed_set"] == 0}
            for negk in ("neg_llama3_2_3b", "neg_sheared_1_3b", "neg_llama2_7b"):
                ng = {r["prompt"]: r["similarity"] for r in D[negk]
                      if r["N"] == 128 and r["dims"] == d and r["seed_set"] == 0}
                w = sum(sg[p] > ng[p] for p in sg)
                print(f"[{stud}>{negk}] d={d}: {w}/32 "
                      f"margin={np.mean([sg[p]-ng[p] for p in sg]):+.4f}")


if __name__ == "__main__":
    main()
