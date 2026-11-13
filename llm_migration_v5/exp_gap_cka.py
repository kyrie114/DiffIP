"""同一朵扰动云上的线性 CKA 和同维余弦。规则见 试验方案.md 第一节。

CKA 公式与 v3 score_pert_cka.py 一致。判定调用 exp_d_prompt_calib.one_direction。
历史均匀 4 层特征仍在 llm_migration_v4 的 feats/ 与 feats_all/。
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
V4 = H.parent / "llm_migration_v4"
sys.path.insert(0, str(H.parent))
sys.path.insert(0, str(V4))
from exp_d_prompt_calib import HARD, SFT, STRONG, one_direction  # noqa: E402

N = 128
DIMS = 8
NEG5 = ["neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
SAME_D = ["neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
POS = STRONG + SFT + HARD
NEED = ["victim", *POS, *NEG5]
CTRL = [0, 10, 21, 31]
OUT = H / "exp_gap_cka_result.json"


def _std(X):
    X = X - X.mean(0)
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1
    return X / sd


def lin_cka(X, Y):
    n = X.shape[0]
    Hm = np.eye(n) - 1.0 / n
    a, b = Hm @ (X @ X.T) @ Hm, Hm @ (Y @ Y.T) @ Hm
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def paired_cosine(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    xn = np.linalg.norm(X, axis=1)
    yn = np.linalg.norm(Y, axis=1)
    return float(np.mean((X * Y).sum(1) / (xn * yn + 1e-12)))


def load_even4(name):
    path = V4 / "feats" / f"main_{name}.npz"
    if path.exists():
        print(f"load {name} from feats/", flush=True)
        return np.load(path)["pert"]
    cache = V4 / "feats_even4" / f"main_{name}.npy"
    if cache.exists():
        print(f"load {name} from even4 cache", flush=True)
        return np.load(cache)
    src = V4 / "feats_all" / f"main_{name}.npz"
    print(f"slice {name} from full layers", flush=True)
    z = np.load(src, mmap_mode="r")
    if z["pert"].shape[2] <= max(CTRL):
        raise SystemExit(f"{name} does not have layers {CTRL}")
    sliced = np.array(z["pert"][:, :, CTRL], dtype=np.float32)
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, sliced)
    del z
    return sliced


def derive(base):
    from llm_migration_v2.transforms import make_anisotropic_scale, make_permutation
    perm = make_permutation(base.shape[-1], seed=1).argmax(0).numpy()
    scale = np.diag(make_anisotropic_scale(base.shape[-1], seed=0).numpy()).astype(np.float32)
    return {"rep_perm": base[..., perm], "scale": base * scale}


def score_pair(vic, sus, fn):
    rows = []
    for p in range(32):
        for k in range(3):
            vals = [fn(vic[p, k, l, :N], sus[p, k, l, :N]) for l in range(4)]
            rows.append(dict(pair="", prompt=p, seed_set=k, N=N, dims=DIMS,
                             similarity=float(np.mean(vals))))
    return rows


def directions(table, pairs):
    out = []
    failed = []
    for anchor, test in pairs:
        raw = one_direction(table, anchor, test, N, DIMS)
        op = raw["operating_points"]["fpr0"]
        held = [op["by_seed"]["1"], op["by_seed"]["2"]]
        item = {
            "anchor": anchor, "test": test, "t": op["t"],
            "pass": bool(op["heldout_pass"]), "auc_deploy": raw["auc_deploy"],
            "held_strong_tpr": [h["strong"]["tpr"] for h in held],
            "held_sft_tpr": [h["sft"]["tpr"] for h in held],
            "held_fpr": [h["deploy"]["fpr"] for h in held],
            "held_hard_tpr": [h["hard"]["tpr"] for h in held],
        }
        out.append(item)
        if not item["pass"]:
            failed.append(f"{anchor}->{test}")
    return out, failed


def means(table, names):
    acc = {}
    for name in names:
        acc[name] = float(np.mean([r["similarity"] for r in table[name]]))
    return acc


def main():
    collected = [n for n in NEED if n not in ("rep_perm", "scale")]
    arrays = {n: load_even4(n) for n in collected}
    arrays.update(derive(arrays["fine_tune"]))
    vic = arrays["victim"]
    cka_table = {}
    cos_table = {}
    for name, arr in arrays.items():
        if name == "victim":
            continue
        rows = score_pair(vic, arr, lambda X, Y: lin_cka(_std(X), _std(Y)))
        for r in rows:
            r["pair"] = name
        cka_table[name] = rows
        if arr.shape[-1] == vic.shape[-1]:
            crows = score_pair(vic, arr, paired_cosine)
            for r in crows:
                r["pair"] = name
            cos_table[name] = crows
        print(name, "cka", round(means(cka_table, [name])[name], 4), flush=True)
    cka_dirs, cka_fail = directions(cka_table, itertools.permutations(NEG5, 2))
    cos_dirs, cos_fail = directions(cos_table, itertools.permutations(SAME_D, 2))
    result = {
        "cka": {
            "n_directions": len(cka_dirs),
            "n_failed": len(cka_fail),
            "failed": cka_fail,
            "claim_holds": len(cka_fail) == 0,
            "means": means(cka_table, [n for n in arrays if n != "victim"]),
            "directions": cka_dirs,
        },
        "cosine_same_d": {
            "negatives": SAME_D,
            "excluded": "neg_qwen2_5_7b",
            "n_directions": len(cos_dirs),
            "n_failed": len(cos_fail),
            "failed": cos_fail,
            "claim_holds": len(cos_fail) == 0,
            "means": means(cos_table, list(cos_table)),
            "directions": cos_dirs,
        },
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE cka_claim={result['cka']['claim_holds']} fail={cka_fail} "
          f"cos_claim={result['cosine_same_d']['claim_holds']} cos_fail={cos_fail}",
          flush=True)


if __name__ == "__main__":
    main()
