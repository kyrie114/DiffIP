"""第二个底座：Qwen3-8B-Base 对 Qwen3-8B。规则见 试验方案.md 第四节。"""
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
sys.path.insert(0, str(H))
import score  # noqa: E402
from exp_d_prompt_calib import auc  # noqa: E402
from exp_gap_cka import load_even4  # noqa: E402

N = 128
DIMS = 8
NEG4 = ["llama2", "neg_llama3_8b", "neg_baichuan_7b", "neg_mpt_7b"]
FEATS = H / "feats_qwen3"
OUT = H / "exp_gap_qwen3_result.json"
ARRAYS = {}
VICTIM = None


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


def main():
    for name in ("qwen3_victim", "qwen3_instruct", "qwen3_hard"):
        if not (FEATS / f"main_{name}.npz").exists():
            raise SystemExit(f"missing {name}")
    global VICTIM
    VICTIM = np.load(FEATS / "main_qwen3_victim.npz")["pert"]
    ARRAYS["qwen3_instruct"] = np.load(FEATS / "main_qwen3_instruct.npz")["pert"]
    ARRAYS["qwen3_hard"] = np.load(FEATS / "main_qwen3_hard.npz")["pert"]
    ARRAYS["llama2"] = load_even4("victim")
    for name in ("neg_llama3_8b", "neg_baichuan_7b", "neg_mpt_7b"):
        ARRAYS[name] = load_even4(name)
    if any(a.shape[2] != 4 or a.shape[3] < N for a in ARRAYS.values()):
        raise SystemExit("negative features are not even4 with N>=128")
    jobs = [(n, p, k) for n in ARRAYS for p in range(32) for k in range(3)]
    from collections import defaultdict
    table = defaultdict(list)
    with get_context("fork").Pool(int(os.environ.get("V4_SCORE_WORKERS", "4"))) as pool:
        for row in pool.imap_unordered(_job, jobs, chunksize=8):
            table[row["pair"]].append(row)
    dirs = []
    failed = []
    for anchor, test in itertools.permutations(NEG4, 2):
        t = float(_margins(table, [test], anchor, {0}).max())
        held = []
        ok = True
        for seed in (1, 2):
            pos = _margins(table, ["qwen3_instruct"], anchor, {seed})
            neg = _margins(table, [test], anchor, {seed})
            tpr, fpr = float((pos >= t).mean()), float((neg >= t).mean())
            held.append({"tpr": tpr, "fpr": fpr})
            if not (tpr >= 0.90 and fpr <= 0.10):
                ok = False
        hard = _margins(table, ["qwen3_hard"], anchor, {1, 2})
        pos_all = _margins(table, ["qwen3_instruct"], anchor, {0, 1, 2})
        neg_all = _margins(table, [test], anchor, {0, 1, 2})
        item = {"anchor": anchor, "test": test, "t": t, "pass": ok, "held": held,
                "hard_tpr": float((hard >= t).mean()), "auc": auc(pos_all, neg_all)}
        dirs.append(item)
        if not ok:
            failed.append(f"{anchor}->{test}")
    means = {n: float(np.mean([r["similarity"] for r in table[n]])) for n in table}
    result = {"means": means, "n_failed": len(failed), "failed": failed,
              "claim_holds": len(failed) == 0, "directions": dirs}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE qwen3_claim={result['claim_holds']} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
