"""Aggregate per-job score files into tables (markdown) + summary json."""
import json, sys, numpy as np
from collections import defaultdict
from common import H, PROMPTS, CATEGORY
MODE = sys.argv[1] if len(sys.argv) > 1 else "pert"
ORDER = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff", "neg_smollm2_135m", "neg_smollm2_360m",
         "neg_gpt2", "neg_pythia_160m", "neg_qwen2_0.5b"]
rows = []
for n in ORDER:
    f = H / "scores" / f"{MODE}_{n}.json"
    if f.exists(): rows += json.loads(f.read_text())
pairs = [n for n in ORDER if any(r["pair"] == n for r in rows)]
Ns = sorted({r["N"] for r in rows}); DIMS = [8, 20]
# per (pair,N,dims,seed_set): mean over prompts; then mean/std over seed sets
acc = defaultdict(list)
for r in rows: acc[(r["pair"], r["N"], r["dims"], r["seed_set"])].append((r["similarity"], r["distance_mean"]))
summ = {}
for (p, n, d, k), v in acc.items():
    summ.setdefault((p, n, d), []).append(np.mean(v, 0))
out = {"mode": MODE, "convergence": [], "per_prompt": []}
lines = []
for d in DIMS:
    lines.append(f"\n**{MODE}, dims={d}: 跨 prompt 均值的 similarity / distance，3 个 seed set 的 mean ± std**\n")
    lines.append("| pair | " + " | ".join(f"N={n}" for n in Ns) + " |"); lines.append("|---|" + "---|" * len(Ns))
    for p in pairs:
        cells = []
        for n in Ns:
            a = np.array(summ[(p, n, d)]); m, s = a.mean(0), a.std(0, ddof=1)
            out["convergence"].append(dict(pair=p, N=n, dims=d, sim_mean=m[0], sim_std_seedsets=s[0], dist_mean=m[1], dist_std_seedsets=s[1], n_seed_sets=len(a)))
            cells.append(f"{m[0]:.3f}±{s[0]:.3f} / {m[1]:.3f}±{s[1]:.3f}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
# per prompt summary, seed set 0
NP = max(r["prompt"] for r in rows) + 1
for n in [20, 128] if MODE == "pert" else [20, 64]:
    for d in DIMS:
        lines.append(f"\n**{MODE}, 逐 prompt（seed set 0），N={n}, dims={d}：{NP} 个 prompt 的 mean / std / min / max similarity，及 distance 均值**\n")
        lines.append("| pair | sim mean | sim std | sim min | sim max | dist mean | 高于该 prompt 全部 5 负例的 prompt 数 |"); lines.append("|---|---|---|---|---|---|---|")
        tab = {(r["pair"], r["prompt"]): r for r in rows if r["N"] == n and r["dims"] == d and r["seed_set"] == 0}
        for p in pairs:
            sims = np.array([tab[(p, i)]["similarity"] for i in range(NP)]); dist = np.array([tab[(p, i)]["distance_mean"] for i in range(NP)])
            negmax = np.array([max(tab[(q, i)]["similarity"] for q in pairs if q.startswith("neg_")) for i in range(NP)])
            wins = "—" if p.startswith("neg_") else f"{int((sims > negmax).sum())}/{NP}"
            out["per_prompt"].append(dict(pair=p, N=n, dims=d, sim_mean=sims.mean(), sim_std=sims.std(ddof=1), sim_min=sims.min(), sim_max=sims.max(), dist_mean=dist.mean(), wins=wins))
            lines.append(f"| {p} | {sims.mean():.3f} | {sims.std(ddof=1):.3f} | {sims.min():.3f} | {sims.max():.3f} | {dist.mean():.3f} | {wins} |")
(H / f"summary_{MODE}.json").write_text(json.dumps(out, indent=1, default=float))
(H / f"tables_{MODE}.md").write_text("\n".join(lines)); print("\n".join(lines))
