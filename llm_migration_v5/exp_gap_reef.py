"""官方 REEF 式聚合，外加同维最后 token 余弦。规则见 试验方案.md 第二节。"""
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
from exp_d_prompt_calib import SFT, STRONG  # noqa: E402

ACTS = H / "acts_reef"
OUT = H / "exp_gap_reef_result.json"
NEG5 = ["neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
SAME_D = ["neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
POS = STRONG + SFT
# rep_perm and scale are derived from fine_tune. They have no checkpoint.
FILES = ["victim", *STRONG, "fine_tune", "weight_perm", "prune_dpoff", "derivative_cn", *NEG5]


def derive_sft(base):
    from llm_migration_v2.transforms import make_anisotropic_scale, make_permutation
    perm = make_permutation(base.shape[-1], seed=1).argmax(0).numpy()
    scale = np.diag(make_anisotropic_scale(base.shape[-1], seed=0).numpy()).astype(np.float32)
    return {"rep_perm": base[..., perm], "scale": base * scale}


def _std(X):
    X = X - X.mean(0)
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1
    return X / sd


def lin_cka(X, Y):
    X, Y = _std(X), _std(Y)
    n = X.shape[0]
    Hm = np.eye(n) - 1.0 / n
    a, b = Hm @ (X @ X.T) @ Hm, Hm @ (Y @ Y.T) @ Hm
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def headline(total):
    return int(round(18 / 31 * (total - 1)))


def gap(scores, prompts):
    pos = min(scores[n] for n in POS)
    neg = max(scores[n] for n in NEG5)
    # scores here are already restricted to `prompts` by the caller
    return float(pos - neg), float(pos), float(neg)


def cka_on(acts, prompts):
    vic = acts["victim"]
    hv = headline(vic.shape[1])
    out = {}
    X = vic[prompts, hv]
    for name, arr in acts.items():
        if name == "victim":
            continue
        hs = headline(arr.shape[1])
        out[name] = lin_cka(X, arr[prompts, hs])
    return out


def cosine_rows(acts):
    """Per-prompt cosine at the headline layer. Same hidden size only."""
    vic = acts["victim"]
    hv = headline(vic.shape[1])
    table = {}
    for name, arr in acts.items():
        if name == "victim" or arr.shape[-1] != vic.shape[-1]:
            continue
        hs = headline(arr.shape[1])
        rows = []
        for p in range(32):
            a = vic[p, hv].astype(np.float64)
            b = arr[p, hs].astype(np.float64)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            # One clean forward: seed_set 0 only. Even prompts calibrate.
            rows.append(dict(pair=name, prompt=p, seed_set=0, N=128, dims=8, similarity=sim))
        table[name] = rows
    return table


def cosine_direction(table, anchor, test):
    def marg(names, prompts):
        ag = {r["prompt"]: r["similarity"] for r in table[anchor]}
        vals = []
        for name in names:
            g = {r["prompt"]: r["similarity"] for r in table[name]}
            for p in prompts:
                vals.append(g[p] - ag[p])
        return np.asarray(vals, dtype=np.float64)

    cal = list(range(0, 32, 2))
    ev = list(range(1, 32, 2))
    t = float(marg([test], cal).max())
    strong = marg(STRONG, ev)
    sft = marg(SFT, ev)
    neg = marg([test], ev)
    pos = marg(STRONG + SFT, ev)
    st = float((strong >= t).mean())
    sft_t = float((sft >= t).mean())
    fpr = float((neg >= t).mean())
    return {"anchor": anchor, "test": test, "t": t,
            "strong_tpr": st, "sft_tpr": sft_t, "fpr": fpr,
            "pass": bool(st >= 0.90 and sft_t >= 0.90 and fpr <= 0.10)}


def main():
    missing = [n for n in FILES if not (ACTS / f"main_{n}.npz").exists()]
    if missing:
        raise SystemExit(f"missing acts_reef: {missing}")
    acts = {n: np.load(ACTS / f"main_{n}.npz")["last"] for n in FILES}
    acts.update(derive_sft(acts["fine_tune"]))
    full = cka_on(acts, np.arange(32))
    even = cka_on(acts, np.arange(0, 32, 2))
    odd = cka_on(acts, np.arange(1, 32, 2))
    g_full, pos_f, neg_f = gap(full, None)
    g_even, _, _ = gap(even, None)
    g_odd, _, _ = gap(odd, None)
    cos = cosine_rows(acts)
    cos_dirs = [cosine_direction(cos, a, b) for a, b in itertools.permutations(SAME_D, 2)]
    cos_fail = [f"{d['anchor']}->{d['test']}" for d in cos_dirs if not d["pass"]]
    result = {
        "headline_layer_victim": headline(acts["victim"].shape[1]),
        "cka32": full,
        "gap32": g_full,
        "pos_min": pos_f,
        "neg_max": neg_f,
        "gap_positive": bool(g_full > 0),
        "gap_even": g_even,
        "gap_odd": g_odd,
        "sign_stable": bool((g_full > 0) == (g_even > 0) == (g_odd > 0)),
        "clean_cosine_same_d": {
            "n_failed": len(cos_fail),
            "failed": cos_fail,
            "claim_holds": len(cos_fail) == 0,
            "directions": cos_dirs,
        },
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE reef_gap={g_full:.4f} positive={g_full > 0} stable={result['sign_stable']} "
          f"clean_cos={result['clean_cosine_same_d']['claim_holds']}", flush=True)


if __name__ == "__main__":
    main()
