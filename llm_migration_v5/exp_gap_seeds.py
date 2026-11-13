"""新种子上的三条阈值。规则见 试验方案.md 第三节。看到新分数之前不许改 0.002。"""
from __future__ import annotations

import itertools
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

H = Path(__file__).resolve().parent
V4 = H.parent / "llm_migration_v4"
sys.path.insert(0, str(H.parent))
sys.path.insert(0, str(V4))
import score  # noqa: E402
from exp_d_prompt_calib import SFT, STRONG, auc  # noqa: E402

N = 128
DIMS = 8
BAND = 0.002
NEG5 = ["neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
COLLECTED = ["fine_tune", "weight_perm", "prune_dpoff", *STRONG, *NEG5]
FEATS = H / "feats_newseed"
OUT = H / "exp_gap_seeds_result.json"
ARRAYS = {}
VICTIM = None


def _derive(base, name):
    from llm_migration_v2.transforms import make_anisotropic_scale, make_permutation
    if name == "rep_perm":
        perm = make_permutation(base.shape[-1], seed=1).argmax(0).numpy()
        return base[..., perm]
    scale = np.diag(make_anisotropic_scale(base.shape[-1], seed=0).numpy())
    return base * scale


def _job(job):
    name, pi, ki = job
    dm, sim, _ = score.als(ARRAYS[name][pi, ki, :, :N], VICTIM[pi, ki, :, :N], DIMS)
    return dict(pair=name, prompt=pi, seed_set=ki, N=N, dims=DIMS,
                distance_mean=dm, similarity=sim)


def _margins(table, names, anchor, seeds):
    ag = {(r["prompt"], r["seed_set"]): r["similarity"] for r in table[anchor]}
    vals = []
    for name in names:
        g = {(r["prompt"], r["seed_set"]): r["similarity"] for r in table[name]}
        for key, sim in g.items():
            if key[1] in seeds:
                vals.append(sim - ag[key])
    return np.asarray(vals, dtype=np.float64)


def _held(table, anchor, test, t, seeds):
    out = []
    ok = True
    for seed in seeds:
        strong = _margins(table, STRONG, anchor, {seed})
        sft = _margins(table, SFT, anchor, {seed})
        neg = _margins(table, [test], anchor, {seed})
        pos = _margins(table, STRONG + SFT, anchor, {seed})
        st = float((strong >= t).mean())
        sft_t = float((sft >= t).mean())
        fpr = float((neg >= t).mean())
        out.append({"strong_tpr": st, "sft_tpr": sft_t, "fpr": fpr})
        if not (st >= 0.90 and sft_t >= 0.90 and fpr <= 0.10):
            ok = False
    return out, ok


def rule_dirs(table, t_of, eval_seeds):
    failed = []
    dirs = []
    for anchor, test in itertools.permutations(NEG5, 2):
        t = float(t_of(table, anchor, test))
        held, ok = _held(table, anchor, test, t, eval_seeds)
        pos = _margins(table, STRONG + SFT, anchor, set(eval_seeds) | {0})
        neg = _margins(table, [test], anchor, set(eval_seeds) | {0})
        dirs.append({"anchor": anchor, "test": test, "t": t, "pass": ok,
                     "held": held, "auc_deploy": auc(pos, neg)})
        if not ok:
            failed.append(f"{anchor}->{test}")
    return {"n_failed": len(failed), "failed": failed, "claim_holds": len(failed) == 0,
            "directions": dirs}


def main():
    missing = [n for n in ["victim", *COLLECTED] if not (FEATS / f"main_{n}.npz").exists()]
    if missing:
        raise SystemExit(f"missing feats_newseed: {missing}")
    global VICTIM
    VICTIM = np.load(FEATS / "main_victim.npz")["pert"]
    for name in COLLECTED:
        ARRAYS[name] = np.load(FEATS / f"main_{name}.npz")["pert"]
    ARRAYS["rep_perm"] = _derive(ARRAYS["fine_tune"], "rep_perm")
    ARRAYS["scale"] = _derive(ARRAYS["fine_tune"], "scale")
    jobs = [(n, p, k) for n in ARRAYS for p in range(32) for k in range(3)]
    from collections import defaultdict
    table = defaultdict(list)
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "4"))
    with get_context("fork").Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            table[row["pair"]].append(row)
            if (i + 1) % 400 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}", flush=True)

    def tmax(seeds):
        def _t(table, anchor, test):
            return _margins(table, [test], anchor, set(seeds)).max()
        return _t

    result = {
        "band": BAND,
        "max1": rule_dirs(table, tmax({0}), (1, 2)),
        "band_rule": rule_dirs(table, lambda tb, a, t: tmax({0})(tb, a, t) + BAND, (1, 2)),
        "max2": rule_dirs(table, tmax({0, 1}), (2,)),
    }
    result["band_confirmed"] = bool(result["band_rule"]["claim_holds"])
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE max1={result['max1']['claim_holds']} band={result['band_confirmed']} "
          f"max2={result['max2']['claim_holds']}", flush=True)


if __name__ == "__main__":
    main()
