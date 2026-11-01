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

# The 7B feature arrays are 0.8-1.6GB each. Repeatedly allocating/freeing
# 16MB PCA temporaries made glibc mmap/munmap them, which under 16 concurrent
# workers drove kcompactd to 100% (compact_stall +450k, sys time ~46%): the
# whole box stalled for 10+ min with zero jobs finished. Keep malloc on the
# heap (no mmap/trim churn) and disable THP for this process.
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallopt(-3, 1 << 30)   # M_MMAP_THRESHOLD: 1 GiB
    _libc.mallopt(-1, 1 << 30)   # M_TRIM_THRESHOLD: 1 GiB
    _libc.mallopt(-2, 1 << 28)   # M_TOP_PAD: 256 MiB
    _libc.prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE
except Exception:
    pass

import numpy as np

from common import (H, MAIN, DISTILL, DIMS, N_GRID, PROMPTS, SEED_SETS)
sys.path.insert(0, str(H.parent))
from diffip.preprocessing import project_sequences  # noqa: E402
from diffip.reversion import fit_step_alignment  # noqa: E402

F = H / "feats"


def ld(line, n):
    # Keep the cache in float32 (half the RSS); diffip's validators convert
    # the per-job slice to float64 anyway.
    return np.load(F / f"{line}_{n}.npz")["pert"]


_CACHE = collections.OrderedDict()
_CACHE_MAX = 3


def _ld_cached(line, n):
    """Per-worker LRU cache for full npz arrays.

    _job runs 1152 (prompt, seed, N, d) jobs per suspect; without this each
    job np.loads the whole 400-800MB file and re-converts it to float64
    (~20s/job under memory-bandwidth contention). With it, each file is
    loaded once per worker."""
    key = (line, n)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    _CACHE[key] = ld(line, n)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return _CACHE[key]


_DERIVED = {}


def _derived(line, name):
    """Feature-space disguises (rep_perm / scale) derived from fine_tune features.

    Computed once per worker process (fork inherits sys.path); the derived
    models have no npz on disk, so _job must not call ld() for them."""
    if (line, name) in _DERIVED:
        return _DERIVED[(line, name)]
    base = _ld_cached(line, "fine_tune")
    if name == "rep_perm":
        from llm_migration_v2.transforms import make_permutation
        perm = make_permutation(base.shape[-1], seed=1).argmax(0).numpy()
        arr = base[..., perm]
    elif name == "scale":
        from llm_migration_v2.transforms import make_anisotropic_scale
        s = np.diag(make_anisotropic_scale(base.shape[-1], seed=0).numpy())
        arr = base * s
    else:
        raise KeyError(name)
    _DERIVED[(line, name)] = arr
    return arr


def load_sus(line, name):
    f = F / f"{line}_{name}.npz"
    if f.exists():
        return _ld_cached(line, name)
    return _derived(line, name)


def als(sus, vic, d):
    pr = project_sequences(sus, vic, d)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance
          for i in range(sus.shape[0])]
    return float(np.mean(dl)), math.exp(-sum(dl) / len(dl)), pr.n_components


def _job(a):
    line, name, pi, ki, n, d = a
    assert d <= 4 * n - 1
    sus = load_sus(line, name)
    vic = _ld_cached(line, "victim")
    dm, sim, _ = als(sus[pi, ki, :, :n], vic[pi, ki, :, :n], d)
    return dict(pair=name, prompt=pi, seed_set=ki, N=n, dims=d,
                distance_mean=dm, similarity=sim)


def suspects_for(line, names):
    """Validate names without loading arrays (the parent process keeps no
    feature memory; workers load/derive on demand). Real npz files must
    exist; rep_perm/scale need only fine_tune (derived in the workers)."""
    SUS = {}
    for name in names:
        f = F / f"{line}_{name}.npz"
        if f.exists():
            SUS[name] = True
            continue
        if line == "main" and name in ("rep_perm", "scale") and \
                (F / f"{line}_fine_tune.npz").exists():
            SUS[name] = True
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
                entry = agg.get((name, n, dims))
                per_seed = ([np.mean(v) for v in entry.values()]
                            if entry else [])
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
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "8"))
    print(f"score workers={nworkers}", flush=True)
    with Pool(nworkers) as pool:
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
    (scores_dir / f"summary_{line}.json").write_text(json.dumps(summary, indent=1))
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
