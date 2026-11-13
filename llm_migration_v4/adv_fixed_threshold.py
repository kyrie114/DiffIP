"""优点 6：固定阈值协议。阈值只在种子组 0 上定一次，在种子组 1、2 上留出检验。

判定量：同一条提示上 Δ_p(M) = sim_p(M, 受害) − sim_p(R, 受害)，R 是参照无关模型。
阈值 = 种子组 0 上「留出无关模型」32 条提示 Δ 的最大值（误报 0 的点）。
每个参照方向（参照 R、留出无关模型 T 的有序对）分别报告每个正例在种子组 1、2 上的
召回，以及 T 的误报。通过：召回 ≥ 0.90 且 误报 ≤ 0.10，两个种子组都满足。

Usage: python adv_fixed_threshold.py --neg A B [C ...] --pos X Y [Z ...] [--n 128 --dims 8]
需要 scores/main_{name}.json（score.py main 产出）。输出 adv_scores/fixed_threshold.json"""
import argparse, itertools, json
import numpy as np
from common import H

S = H / "scores"; OUT = H / "adv_scores"; OUT.mkdir(exist_ok=True)
# scores/ is the original roster. The three expanded negatives live in
# scores_unrel/, and this round's new names live in scores_panel/.
# The margin formula below is unchanged. lineage.py requires the scores/
# stems to match its roster exactly, so new rows are not copied in.
_SCORE_DIRS = (S, H / "scores_unrel", H / "scores_panel")


def score_file(name):
    for d in _SCORE_DIRS:
        p = d / f"main_{name}.json"
        if p.is_file():
            return p
    raise SystemExit(f"missing main_{name}.json in scores, scores_unrel, or scores_panel")


def grid(name, n, d):
    rows = json.loads(score_file(name).read_text())
    return {(r["prompt"], r["seed_set"]): r["similarity"] for r in rows if r["N"] == n and r["dims"] == d}


def margins(g, a, seed):
    return np.array([g[k] - a[k] for k in sorted(g) if k[1] == seed])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", nargs="+", required=True)
    ap.add_argument("--pos", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--dims", type=int, default=8)
    a = ap.parse_args()
    G = {k: grid(k, a.n, a.dims) for k in a.neg + a.pos}
    out = []
    for ref, test in itertools.permutations(a.neg, 2):
        t = float(margins(G[test], G[ref], 0).max())
        row = dict(reference=ref, heldout_negative=test, threshold=t, seeds={})
        ok = True
        for seed in (1, 2):
            fpr = float((margins(G[test], G[ref], seed) >= t).mean())
            tpr = {p: float((margins(G[p], G[ref], seed) >= t).mean()) for p in a.pos}
            row["seeds"][seed] = dict(fpr=fpr, tpr=tpr)
            ok &= fpr <= 0.10 and all(v >= 0.90 for v in tpr.values())
        row["pass"] = bool(ok); out.append(row)
        print(f"参照 {ref:<18} 留出 {test:<18} 阈值 {t:+.4f} 误报 "
              f"{row['seeds'][1]['fpr']:.3f}/{row['seeds'][2]['fpr']:.3f} 通过 {ok}")
        for p in a.pos:
            print(f"    {p:<28} 召回 {row['seeds'][1]['tpr'][p]:.3f}/{row['seeds'][2]['tpr'][p]:.3f}")
    (OUT / "fixed_threshold.json").write_text(json.dumps(dict(args=vars(a), directions=out), indent=1, ensure_ascii=False))
