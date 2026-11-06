"""实验 G：解释「掺入 75% Llama-3 仍有 29/32」。

可能测到的不是鲁棒性，而是两边激活幅度不同，0.75 的系数并不等于
0.75 的能量。先量 RMS。再在各自除以 RMS 之后按同样的 α 重混，
只评 N=128、d=8。胜负只对 Qwen。

判定与实验 E 的 mix 相同：藏住 = 胜负 <= 20/32 或均值 <= 0.274。
norm_ident（只把 fine_tune 除以自己的 RMS）用来看全局缩放会不会被 ALS 吃掉。
它不参与藏住判定。若它和已存 fine_tune 差超过 0.018，全局缩放没有被吃掉。
"""
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.pop("V4_LAYER_SUBSET", None)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import score as score_mod
from score import als

H = Path(__file__).resolve().parent
F = H / "feats"
EVEN4 = H / "scores"
OUT = H / "scores_rms"
FLOOR = 0.254
HIDE = FLOOR + 0.020
FT_ANCHOR = 0.363

NAMES = ["norm_ident", "rms_llama_025", "rms_llama_050", "rms_llama_075"]
_CACHE = {}


def load():
    if "ft" not in _CACHE:
        _CACHE["ft"] = np.load(F / "main_fine_tune.npz")["pert"]
        _CACHE["vic"] = np.load(F / "main_victim.npz")["pert"]
        _CACHE["llama"] = np.load(F / "main_neg_llama3_8b.npz")["pert"]
    return _CACHE["ft"], _CACHE["vic"], _CACHE["llama"]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _job(a):
    name, pi, ki, rms_ft, rms_ll = a
    ft, vic, llama = load()
    s = ft[pi, ki, :, :128] / rms_ft
    if name == "norm_ident":
        sus = s
    else:
        alpha = {"rms_llama_025": 0.25, "rms_llama_050": 0.50, "rms_llama_075": 0.75}[name]
        sus = (1.0 - alpha) * s + alpha * (llama[pi, ki, :, :128] / rms_ll)
    dm, sim, _ = als(sus, vic[pi, ki, :, :128], 8)
    return dict(pair=name, prompt=pi, seed_set=ki, N=128, dims=8,
                distance_mean=dm, similarity=sim)


def mean_std(rows):
    by = {}
    for r in rows:
        by.setdefault(r["seed_set"], []).append(r["similarity"])
    per = [float(np.mean(v)) for v in by.values()]
    return float(np.mean(per)), float(np.std(per))


def wins_vs(rows, neg_rows):
    pos = {r["prompt"]: r["similarity"] for r in rows if r["seed_set"] == 0}
    groups = [{r["prompt"]: r["similarity"] for r in nr
               if r["N"] == 128 and r["dims"] == 8 and r["seed_set"] == 0}
              for nr in neg_rows]
    return sum(s > max(g[p] for g in groups) for p, s in pos.items())


def main():
    if score_mod.LAYER_IDX is not None:
        raise SystemExit("even4 required")
    ft, _, llama = load()
    rms_ft, rms_ll = rms(ft), rms(llama)
    # energy share of the raw coefficient mix, assuming the two clouds are uncorrelated
    raw_share = {}
    for a in (0.25, 0.50, 0.75):
        num = (1 - a) ** 2 * rms_ft ** 2
        den = num + a ** 2 * rms_ll ** 2
        raw_share[str(a)] = num / den
    step = 64
    c = np.corrcoef(ft.ravel()[::step], llama.ravel()[::step])[0, 1]
    print(f"rms ft={rms_ft:.4f} llama={rms_ll:.4f} ratio={rms_ft/rms_ll:.3f} "
          f"corr_stride{step}={c:.4f}", flush=True)
    print("raw-mix energy share of ft", raw_share, flush=True)
    OUT.mkdir(exist_ok=True)
    jobs = [(name, pi, ki, rms_ft, rms_ll)
            for name, pi, ki in itertools.product(NAMES, range(32), range(3))]
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "8"))
    per = {k: [] for k in NAMES}
    with Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            per[row["pair"]].append(row)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(jobs)}", flush=True)
    qwen = json.loads((EVEN4 / "main_neg_qwen2_5_7b.json").read_text())
    stored = json.loads((EVEN4 / "main_fine_tune.json").read_text())
    stored_i = {(r["prompt"], r["seed_set"]): r["similarity"] for r in stored
                if r["N"] == 128 and r["dims"] == 8}
    ident = {(r["prompt"], r["seed_set"]): r["similarity"] for r in per["norm_ident"]}
    scale_gap = float(np.max([abs(ident[k] - stored_i[k]) for k in ident]))
    summary = {
        "rms_ft": rms_ft, "rms_llama": rms_ll,
        "rms_ratio_ft_over_llama": rms_ft / rms_ll,
        "corr_stride": {"step": step, "pearson": float(c)},
        "raw_mix_ft_energy_share_if_uncorrelated": raw_share,
        "norm_ident_max_abs_vs_stored": scale_gap,
        "global_scale_absorbed": scale_gap <= 0.018,
        "variants": [],
    }
    for name in NAMES:
        rows = per[name]
        (OUT / f"{name}.json").write_text(json.dumps(rows))
        mean, std = mean_std(rows)
        w = wins_vs(rows, [qwen])
        if name == "norm_ident":
            verdict = "缩放被吃掉" if scale_gap <= 0.018 else "缩放没有被吃掉"
        elif w <= 20 or mean <= HIDE:
            verdict = "藏住"
        elif w >= 30 and mean >= FT_ANCHOR - 0.018:
            verdict = "没藏住"
        else:
            verdict = "部分有效"
        rec = dict(name=name, sim=round(mean, 4), seed_std=round(std, 4),
                   wins_vs_qwen=f"{w}/32", verdict=verdict)
        summary["variants"].append(rec)
        print(f"{name:16} sim={mean:.4f}±{std:.4f} wins={w}/32 {verdict}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
