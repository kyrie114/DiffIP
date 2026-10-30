"""Score + summarize a line under the v3 protocol (PCA + per-layer ALS, no DP).

Usage:
    python score.py main [names...]
    python score.py distill <names...>

For the main line, rep_perm / scale suspects are derived from fine_tune
features (feature-space disguises, no extra collection).
Output: scores/{line}_{name}.json rows + summary_{line}.json + tables_{line}.md
"""
import itertools
import json
import math
import os
import sys
import collections
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from common import (H, MAIN, DISTILL, DIMS, N_GRID, PROMPTS, SEED_SETS)
sys.path.insert(0, str(H.parent))
from diffip.preprocessing import project_sequences  # noqa: E402
from diffip.reversion import fit_step_alignment  # noqa: E402

F = H / "feats"


def ld(line, n):
    return np.load(F / f"{line}_{n}.npz")["pert"].astype(np.float64)


def als(sus, vic, d):
    pr = project_sequences(sus, vic, d)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance
          for i in range(sus.shape[0])]
    return float(np.mean(dl)), math.exp(-sum(dl) / len(dl)), pr.n_components


def _job(a):
    line, name, pi, ki, n, d = a
    assert d <= 4 * n - 1
    sus = ld(line, name)
    vic = ld(line, "victim")
    dm, sim, _ = als(sus[pi, ki, :, :n], vic[pi, ki, :, :n], d)
    return dict(pair=name, prompt=pi, seed_set=ki, N=n, dims=d,
                distance_mean=dm, similarity=sim)


def suspects_for(line, names):
    """Return {name: feat array}, deriving rep_perm/scale from fine_tune."""
    SUS = {}
    for name in names:
        f = F / f"{line}_{name}.npz"
        if f.exists():
            SUS[name] = ld(line, name)
            continue
        if line == "main" and name == "rep_perm" and "fine_tune" in SUS:
            from llm_migration_v2.transforms import make_permutation
            perm = make_permutation(SUS["fine_tune"].shape[-1], seed=1).argmax(0).numpy()
            SUS[name] = SUS["fine_tune"][..., perm]
        elif line == "main" and name == "scale" and "fine_tune" in SUS:
            from llm_migration_v2.transforms import make_anisotropic_scale
            s = np.diag(make_anisotropic_scale(SUS["fine_tune"].shape[-1], seed=0).numpy())
            SUS[name] = SUS["fine_tune"] * s
    return SUS


def summarize(rows, names, groups=None):
    agg = collections.defaultdict(dict)
    for r in rows:
        key = (r["pair"], r["N"], r["dims"])
        agg[key].setdefault(r["seed_set"], []).append(r["similarity"])
    out = {"by_nd": {}}
    print("\n== 跨 32 prompt 均值 similarity（3 个 seed set 的 mean ± std）==")
    for dims in DIMS:
        print(f"\n-- dims={dims} --")
        print(f"{'pair':<22}" + "".join(f"{n:>13}" for n in N_GRID))
        for name in names:
            line = f"{name:<22}"
            for n in N_GRID:
                per_seed = [np.mean(v) for (p, nn, dd), v in agg.items()
                            if p == name and nn == n and dd == dims]
                if per_seed:
                    line += f"{np.mean(per_seed):>9.3f}±{np.std(per_seed):<4.3f}"
                    out["by_nd"].setdefault(str(dims), {}).setdefault(
                        str(n), {})[name] = [round(float(x), 5) for x in per_seed]
                else:
                    line += f"{'--':>13}"
            print(line)
    # per-prompt win counts vs all negatives
    for n in (20, max(N_GRID)):
        for dims in DIMS:
            print(f"\n== 逐 prompt 胜负（N={n}, dims={dims}, seed set 0）："
                  f"高于全部负例的 prompt 数 / 32 ==")
            by = collections.defaultdict(lambda: np.zeros(len(PROMPTS)))
            for r in rows:
                if r["seed_set"] == 0 and r["N"] == n and r["dims"] == dims:
                    by[r["pair"]][r["prompt"]] = r["similarity"]
            negs = [k for k in by if k.startswith("neg_")]
            if negs:
                best = np.max(np.stack([by[k] for k in negs]), axis=0)
                for pair in names:
                    if pair.startswith("neg_"):
                        continue
                    wins = int((by[pair] > best).sum())
                    print(f"  {pair}: {wins}/32   "
                          f"sim mean={by[pair].mean():.3f} min={by[pair].min():.3f}")
    if groups:
        for gname, gmembers in groups.items():
            present = [k for k in gmembers if k in {r['pair'] for r in rows}]
            if present:
                sims = [r["similarity"] for r in rows
                        if r["pair"] in present and r["N"] == max(N_GRID)
                        and r["dims"] == DIMS[0]]
                print(f"\n[{gname}] N={max(N_GRID)} d={DIMS[0]} "
                      f"mean sim = {np.mean(sims):.4f}  ({len(present)} models)")
    return out


def score(line, names):
    from common import MAIN as M, DISTILL as D
    if not (F / f"{line}_victim.npz").exists():
        raise SystemExit(f"missing feats/{line}_victim.npz — run collect.py first")
    # ensure derived suspects see fine_tune first
    ordered = (["fine_tune"] if line == "main" else []) + \
              [n for n in names if n != "fine_tune"]
    SUS = suspects_for(line, ordered)
    names = list(SUS)
    jobs = list(itertools.product([line], [n for n in names], range(len(PROMPTS)),
                                  range(len(SEED_SETS)), N_GRID, DIMS))
    scores_dir = H / "scores"
    scores_dir.mkdir(exist_ok=True)
    per_pair = collections.defaultdict(list)
    with Pool(8) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=16)):
            per_pair[row["pair"]].append(row)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    rows = []
    for name in names:
        f = scores_dir / f"{line}_{name}.json"
        f.write_text(json.dumps(per_pair[name]))
        rows.extend(per_pair[name])
        print("scored", name, len(per_pair[name]), flush=True)
    groups = None
    if line == "main":
        groups = {"same-base pruned (hard negs)":
                  [n for n in names if n.startswith("neg_pruned")],
                  "cross-family (easy negs)":
                  [n for n in names if n.startswith("neg_") and not n.startswith("neg_pruned")]}
    summary = summarize(rows, names, groups)
    (H / f"summary_{line}.json").write_text(json.dumps(summary, indent=1))
    print("done", line, len(rows), "rows", flush=True)


if __name__ == "__main__":
    line = sys.argv[1]
    reg = list(MAIN) if line == "main" else list(DISTILL)
    if line == "main":
        reg = (["victim", "fine_tune", "derivative_cn", "weight_perm",
                "prune_dpoff", "rep_perm", "scale"]
               + [k for k in reg if k.startswith("neg_")])
    names = sys.argv[2:] or reg
    score(line, names)
