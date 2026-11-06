"""实验 E：打乱配对、掺入另一模型、秩高斯化。规则见 实验DE设计.md。"""
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.pop("V4_LAYER_SUBSET", None)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.special import ndtri

import score as score_mod
from score import als

H = Path(__file__).resolve().parent
F = H / "feats"
EVEN4 = H / "scores"
OUT = H / "scores_break"
FT_ANCHOR = 0.363
FLOOR = 0.254
HIDE_CEILING = FLOOR + 0.020
ROBUST_BAND = 0.018
IDENT_TOL = 1e-4

NAMES = [
    "ident", "shuffle", "rankgauss",
    "mix_llama_025", "mix_llama_050", "mix_llama_075",
    "mix_deriv_050",
]
_CACHE = {}


def _load():
    if "ft" not in _CACHE:
        _CACHE["ft"] = np.load(F / "main_fine_tune.npz")["pert"]
        _CACHE["vic"] = np.load(F / "main_victim.npz")["pert"]
        _CACHE["llama"] = np.load(F / "main_neg_llama3_8b.npz")["pert"]
        _CACHE["deriv"] = np.load(F / "main_derivative_cn.npz")["pert"]
        if _CACHE["ft"].shape != _CACHE["llama"].shape or _CACHE["ft"].shape != _CACHE["deriv"].shape:
            raise RuntimeError("mix requires identical feature shapes")
    return _CACHE["ft"], _CACHE["vic"], _CACHE["llama"], _CACHE["deriv"]


def rankgauss(x):
    """x is [L, N, D]. Rank within the N samples, then standard-normal quantile."""
    ranks = np.argsort(np.argsort(x, axis=1, kind="mergesort"), axis=1).astype(np.float64)
    n = x.shape[1]
    return ndtri((ranks + 0.5) / n).astype(np.float32)


def suspect(name, ft, llama, deriv, pi, ki, n):
    s = ft[pi, ki, :, :n]
    if name == "ident":
        return s
    if name == "shuffle":
        perm = np.random.default_rng(1000 + pi * 10 + ki).permutation(n)
        return s[:, perm, :]
    if name == "rankgauss":
        return rankgauss(s)
    if name == "mix_deriv_050":
        return 0.5 * s + 0.5 * deriv[pi, ki, :, :n]
    alpha = {"mix_llama_025": 0.25, "mix_llama_050": 0.50, "mix_llama_075": 0.75}[name]
    return (1.0 - alpha) * s + alpha * llama[pi, ki, :, :n]


def _job(a):
    name, pi, ki, n, d = a
    ft, vic, llama, deriv = _load()
    sus = suspect(name, ft, llama, deriv, pi, ki, n)
    if not np.isfinite(sus).all():
        raise RuntimeError(f"non-finite {name}")
    dm, sim, _ = als(sus, vic[pi, ki, :, :n], d)
    return dict(pair=name, prompt=pi, seed_set=ki, N=n, dims=d,
                distance_mean=dm, similarity=sim)


def _mean_std(rows, n, dims):
    by = {}
    for r in rows:
        if r["N"] == n and r["dims"] == dims:
            by.setdefault(r["seed_set"], []).append(r["similarity"])
    per = [float(np.mean(v)) for v in by.values()]
    return float(np.mean(per)), float(np.std(per))


def _wins(rows, neg_rows, n, dims):
    pos = {r["prompt"]: r["similarity"] for r in rows
           if r["N"] == n and r["dims"] == dims and r["seed_set"] == 0}
    groups = [{r["prompt"]: r["similarity"] for r in nr
               if r["N"] == n and r["dims"] == dims and r["seed_set"] == 0}
              for nr in neg_rows]
    return sum(s > max(g[p] for g in groups) for p, s in pos.items())


def judge_rankgauss(mean, wins):
    if abs(mean - FT_ANCHOR) <= ROBUST_BAND and wins >= 30:
        return "协议稳健"
    if mean <= HIDE_CEILING or wins <= 20:
        return "攻击成功"
    return "部分有效"


def judge_hide(mean, wins):
    if mean <= HIDE_CEILING or wins <= 20:
        return "藏住"
    if wins >= 30 and mean >= FT_ANCHOR - ROBUST_BAND:
        return "没藏住"
    return "部分有效"


def main():
    if score_mod.LAYER_IDX is not None:
        raise SystemExit("even4 required")
    OUT.mkdir(exist_ok=True)
    jobs = list(itertools.product(
        NAMES, range(len(score_mod.PROMPTS)), range(len(score_mod.SEED_SETS)),
        score_mod.N_GRID, score_mod.DIMS))
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "8"))
    print(f"jobs={len(jobs)} workers={nworkers}", flush=True)
    per = {k: [] for k in NAMES}
    with Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            per[row["pair"]].append(row)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    stored = json.loads((EVEN4 / "main_fine_tune.json").read_text())
    ident = {(r["prompt"], r["seed_set"], r["N"], r["dims"]): r["similarity"] for r in per["ident"]}
    diffs = [abs(ident[(r["prompt"], r["seed_set"], r["N"], r["dims"])] - r["similarity"])
             for r in stored if r["N"] == 128 and r["dims"] == 8]
    ident_max = float(max(diffs))
    print(f"ident N=128 d=8 max |dsim|={ident_max:.3e}", flush=True)
    qwen = json.loads((EVEN4 / "main_neg_qwen2_5_7b.json").read_text())
    llama = json.loads((EVEN4 / "main_neg_llama3_8b.json").read_text())
    summary = {"ident_ok": ident_max <= IDENT_TOL, "ident_max_abs_dsim": ident_max, "variants": []}
    if not summary["ident_ok"]:
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
        raise SystemExit("ident check failed")
    for name in NAMES:
        rows = per[name]
        (OUT / f"{name}.json").write_text(json.dumps(rows))
        mean, std = _mean_std(rows, 128, 8)
        if name.startswith("mix_llama"):
            wins = _wins(rows, [qwen], 128, 8)
            wins_label = f"{wins}/32 vs qwen"
            verdict = "对照" if name == "ident" else judge_hide(mean, wins)
        else:
            wins = _wins(rows, [qwen, llama], 128, 8)
            wins_label = f"{wins}/32 vs both"
            if name == "ident":
                verdict = "对照"
            elif name == "shuffle":
                if wins <= 20 or mean <= HIDE_CEILING:
                    verdict = "配对被打掉"
                elif wins >= 30 and abs(mean - FT_ANCHOR) <= ROBUST_BAND:
                    verdict = "配对没起作用"
                else:
                    verdict = "部分有效"
            elif name == "rankgauss":
                verdict = judge_rankgauss(mean, wins)
            else:
                verdict = judge_hide(mean, wins)
        rec = dict(name=name, sim_n128_d8=round(mean, 4), seed_std=round(std, 4),
                   delta_vs_ft=round(mean - FT_ANCHOR, 4),
                   gap_vs_floor=round(mean - FLOOR, 4),
                   wins=wins_label, verdict=verdict)
        m20, s20 = _mean_std(rows, 20, 8)
        if name.startswith("mix_llama"):
            w20 = _wins(rows, [qwen], 20, 8)
        else:
            w20 = _wins(rows, [qwen, llama], 20, 8)
        rec["n20_d8"] = {"sim": round(m20, 4), "seed_std": round(s20, 4), "wins": w20}
        summary["variants"].append(rec)
        print(f"{name:16} sim={mean:.4f}±{std:.4f} {wins_label} {verdict}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("wrote", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
