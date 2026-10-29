"""Re-score v3 perturbation panel with middle2 layers [1,2] of the L=4 even stack.
No recollect: reuse feats/pert_*.npz. Protocol = score.py (PCA + ALS, DP off) with L'=2.
Rank guard: dims must satisfy d <= 2*N - 1 (was 4*N-1 for all4).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, itertools, math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent))
from common import N_GRID, SEED_SETS, NEGS, make_permutation, make_anisotropic_scale
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment

MIDDLE2 = [1, 2]  # same definition as improve/exp2_prune_layers.py layer_sets["middle2"]
DIMS = [8, 20]
GRID = sorted(set(N_GRID + [20]))
F = H / "feats"
OUT_SCORES = H / "scores"
LAYER_TAG = "middle2"


def ld(n):
    return np.load(F / f"pert_{n}.npz")["pert"].astype(np.float64)  # [P,K,4,Nmax,D]


victim = ld("victim")
ins = ld("instruct")
D = ins.shape[-1]
perm = make_permutation(D, seed=1).argmax(0).numpy()
s = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float64)
SUS = {
    "fine_tune": ins,
    "rep_perm": ins[..., perm],
    "scale": ins * s,
    "prune_dpoff": ld("instruct_prune"),
}
if (F / "pert_instruct_wperm.npz").exists():
    SUS["weight_perm"] = ld("instruct_wperm")
for k in NEGS:
    SUS[k] = ld(k)


def als(sus, vic, d):
    """sus/vic: [L', N, D] with L'=2."""
    pr = project_sequences(sus, vic, d)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(sus.shape[0])]
    return float(np.mean(dl)), math.exp(-sum(dl) / len(dl)), pr.n_components


def job(a):
    name, p, k, n, d = a
    assert d <= 2 * n - 1, f"dims={d} exceeds L'*N-1=2*{n}-1"
    sus = SUS[name][p, k, MIDDLE2, :n]  # [2, N, D]
    vic = victim[p, k, MIDDLE2, :n]
    dm, sim, used = als(sus, vic, d)
    assert used == d, f"used={used} d={d} N={n}"
    return dict(
        pair=name, prompt=p, seed_set=k, N=n, dims=d,
        layers=LAYER_TAG, layer_idx=MIDDLE2,
        distance_mean=dm, similarity=sim,
    )


def valid_grid():
    out = []
    for n, d in itertools.product(GRID, DIMS):
        if d <= 2 * n - 1:
            out.append((n, d))
    return out


def score_all():
    OUT_SCORES.mkdir(exist_ok=True)
    P = victim.shape[0]
    vg = valid_grid()
    print("valid (N,dims):", vg, flush=True)
    with Pool(8) as pool:
        for name in SUS:
            f = OUT_SCORES / f"pert_{LAYER_TAG}_{name}.json"
            if f.exists():
                print("skip", name, flush=True)
                continue
            jobs = [
                (name, p, k, n, d)
                for p in range(P)
                for k in range(len(SEED_SETS))
                for n, d in vg
            ]
            rows = pool.map(job, jobs, chunksize=16)
            f.write_text(json.dumps(rows))
            print(LAYER_TAG, name, len(rows), flush=True)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    c = 0.0
    for p in pos:
        c += (neg < p).sum() + 0.5 * (neg == p).sum()
    return float(c / (len(pos) * len(neg))) if len(pos) and len(neg) else float("nan")


def aggregate():
    ORDER = [
        "fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff",
        "neg_smollm2_135m", "neg_smollm2_360m", "neg_gpt2", "neg_pythia_160m", "neg_qwen2_0.5b",
    ]
    rows = []
    for n in ORDER:
        f = OUT_SCORES / f"pert_{LAYER_TAG}_{n}.json"
        if f.exists():
            rows += json.loads(f.read_text())
    pairs = [n for n in ORDER if any(r["pair"] == n for r in rows)]
    Ns = sorted({r["N"] for r in rows})
    dims_seen = sorted({r["dims"] for r in rows})
    acc = defaultdict(list)
    for r in rows:
        acc[(r["pair"], r["N"], r["dims"], r["seed_set"])].append(
            (r["similarity"], r["distance_mean"])
        )
    summ = {}
    for (p, n, d, k), v in acc.items():
        summ.setdefault((p, n, d), []).append(np.mean(v, 0))

    out = {
        "mode": "pert",
        "layers": LAYER_TAG,
        "layer_idx": MIDDLE2,
        "note": "Re-scored from existing feats; no recollect. Rank: d<=2*N-1.",
        "convergence": [],
        "per_prompt": [],
        "compare_all4": [],
    }
    lines = [
        f"# v3 perturbation panel — layers={LAYER_TAG} {MIDDLE2}",
        "",
        "复用 `feats/pert_*.npz`，不重新采集。协议同 `score.py`（PCA + 逐层 ALS，DP off），",
        f"仅取 4 均匀层中的中间两层 `{MIDDLE2}`（与 `improve/exp2_prune_layers.py` 的 middle2 一致）。",
        "秩约束改为 `dims ≤ 2·N − 1`（原 all4 为 `4·N − 1`），故 N=8 时不下 dims=20。",
        "",
    ]
    for d in dims_seen:
        lines.append(f"\n**pert/{LAYER_TAG}, dims={d}: 跨 prompt 均值的 similarity / distance，3 个 seed set 的 mean ± std**\n")
        ns_d = [n for n in Ns if any(r["N"] == n and r["dims"] == d for r in rows)]
        lines.append("| pair | " + " | ".join(f"N={n}" for n in ns_d) + " |")
        lines.append("|---|" + "---|" * len(ns_d))
        for p in pairs:
            cells = []
            for n in ns_d:
                a = np.array(summ[(p, n, d)])
                m, s = a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros(2)
                out["convergence"].append(dict(
                    pair=p, N=n, dims=d, layers=LAYER_TAG,
                    sim_mean=float(m[0]), sim_std_seedsets=float(s[0]),
                    dist_mean=float(m[1]), dist_std_seedsets=float(s[1]),
                    n_seed_sets=len(a),
                ))
                cells.append(f"{m[0]:.3f}±{s[0]:.3f} / {m[1]:.3f}±{s[1]:.3f}")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")

    NP = max(r["prompt"] for r in rows) + 1
    for n in [x for x in [20, 128] if x in Ns]:
        for d in dims_seen:
            if not any(r["N"] == n and r["dims"] == d for r in rows):
                continue
            lines.append(
                f"\n**pert/{LAYER_TAG}, 逐 prompt（seed set 0），N={n}, dims={d}："
                f"{NP} 个 prompt 的 mean / std / min / max similarity，及 distance 均值**\n"
            )
            lines.append(
                "| pair | sim mean | sim std | sim min | sim max | dist mean | "
                "高于该 prompt 全部 5 负例的 prompt 数 |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            tab = {
                (r["pair"], r["prompt"]): r
                for r in rows
                if r["N"] == n and r["dims"] == d and r["seed_set"] == 0
            }
            for p in pairs:
                sims = np.array([tab[(p, i)]["similarity"] for i in range(NP)])
                dist = np.array([tab[(p, i)]["distance_mean"] for i in range(NP)])
                negmax = np.array([
                    max(tab[(q, i)]["similarity"] for q in pairs if q.startswith("neg_"))
                    for i in range(NP)
                ])
                wins = "—" if p.startswith("neg_") else f"{int((sims > negmax).sum())}/{NP}"
                out["per_prompt"].append(dict(
                    pair=p, N=n, dims=d, layers=LAYER_TAG,
                    sim_mean=float(sims.mean()), sim_std=float(sims.std(ddof=1)),
                    sim_min=float(sims.min()), sim_max=float(sims.max()),
                    dist_mean=float(dist.mean()), wins=wins,
                ))
                lines.append(
                    f"| {p} | {sims.mean():.3f} | {sims.std(ddof=1):.3f} | "
                    f"{sims.min():.3f} | {sims.max():.3f} | {dist.mean():.3f} | {wins} |"
                )

    # Compare all4 (from summary_pert.json) vs middle2 at N=128, dims=8
    all4_path = H / "summary_pert.json"
    compare_lines = [
        "",
        "## all4 vs middle2 对照（N=128, dims=8；convergence = 3 seed-set 均值）",
        "",
        "all4 数字来自既有 `summary_pert.json` / `tables_pert.md`；middle2 为本脚本重算。",
        "",
        "| pair | all4 sim | middle2 sim | Δ sim |",
        "|---|---:|---:|---:|",
    ]
    if all4_path.exists():
        all4 = json.loads(all4_path.read_text())
        a4 = {
            c["pair"]: c
            for c in all4["convergence"]
            if c["N"] == 128 and c["dims"] == 8
        }
        m2 = {
            c["pair"]: c
            for c in out["convergence"]
            if c["N"] == 128 and c["dims"] == 8
        }
        for p in pairs:
            if p not in a4 or p not in m2:
                continue
            da, dm = a4[p]["sim_mean"], m2[p]["sim_mean"]
            compare_lines.append(f"| {p} | {da:.3f} | {dm:.3f} | {dm - da:+.3f} |")
            out["compare_all4"].append(dict(
                pair=p, N=128, dims=8,
                all4_sim=da, middle2_sim=dm, delta=dm - da,
            ))

        # gap / AUC / win-rate on seed0 per-prompt at N=128 dims=8
        derived = [p for p in pairs if not p.startswith("neg_")]
        negs = [p for p in pairs if p.startswith("neg_")]
        tab_m2 = {
            (r["pair"], r["prompt"]): r["similarity"]
            for r in rows
            if r["N"] == 128 and r["dims"] == 8 and r["seed_set"] == 0
        }
        # all4 per-prompt from scores/pert_*.json
        rows_a4 = []
        for n in ORDER:
            f = OUT_SCORES / f"pert_{n}.json"
            if f.exists():
                rows_a4 += [
                    r for r in json.loads(f.read_text())
                    if r["N"] == 128 and r["dims"] == 8 and r["seed_set"] == 0
                ]
        tab_a4 = {(r["pair"], r["prompt"]): r["similarity"] for r in rows_a4}

        def metrics(tab):
            means = {p: float(np.mean([tab[(p, i)] for i in range(NP)])) for p in pairs}
            gap = min(means[p] for p in derived) - max(means[p] for p in negs)
            pos = np.concatenate([[tab[(p, i)] for i in range(NP)] for p in derived])
            neg = np.concatenate([[tab[(p, i)] for i in range(NP)] for p in negs])
            wins = {
                p: int(sum(
                    tab[(p, i)] > max(tab[(q, i)] for q in negs) for i in range(NP)
                ))
                for p in derived
            }
            return means, float(gap), float(auc(pos, neg)), wins

        m_a4, g_a4, auc_a4, w_a4 = metrics(tab_a4)
        m_m2, g_m2, auc_m2, w_m2 = metrics(tab_m2)
        out["detection"] = {
            "N": 128, "dims": 8, "seed_set": 0, "n_prompts": NP,
            "all4": {"means": m_a4, "gap": g_a4, "auc_pooled": auc_a4, "wins": w_a4},
            "middle2": {"means": m_m2, "gap": g_m2, "auc_pooled": auc_m2, "wins": w_m2},
        }
        compare_lines += [
            "",
            f"**检测口径（seed set 0，N=128，dims=8，{NP} prompts）**",
            "",
            "| 设定 | gap = min(derived)−max(neg) | 逐 prompt 合并 AUC | wins（各衍生 vs 5 负例） |",
            "|---|---:|---:|---|",
            f"| all4 | {g_a4:.3f} | {auc_a4:.3f} | "
            + ", ".join(f"{k}={v}/{NP}" for k, v in w_a4.items()) + " |",
            f"| **middle2** | **{g_m2:.3f}** | **{auc_m2:.3f}** | "
            + ", ".join(f"{k}={v}/{NP}" for k, v in w_m2.items()) + " |",
            "",
            "逐对 mean sim（同设定）：",
            "",
            "| pair | all4 | middle2 |",
            "|---|---:|---:|",
        ]
        for p in pairs:
            compare_lines.append(f"| {p} | {m_a4[p]:.3f} | {m_m2[p]:.3f} |")

    lines.extend(compare_lines)
    (H / f"summary_pert_{LAYER_TAG}.json").write_text(
        json.dumps(out, indent=1, default=float)
    )
    (H / f"tables_pert_{LAYER_TAG}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", H / f"tables_pert_{LAYER_TAG}.md", H / f"summary_pert_{LAYER_TAG}.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate()
    else:
        score_all()
        aggregate()
