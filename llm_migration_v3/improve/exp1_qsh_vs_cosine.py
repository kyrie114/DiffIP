"""Fair re-test: constrained Q/S/h pull-back vs direct cosine, leave-out AUC.
Reproduces / refutes the 10/19 claim (cosine AUC≈1.0, full-dim QSh leave-out AUC≈0.538)
on v3 model pairs, with larger calibration sample counts.

Protocol (matches 10/19 实验报告.md method B, not unconstrained linear):
  per layer: center -> orthogonal Procrustes R -> per-dim scale s -> + victim mean
  Fit ONLY on calibration prompts; score cosine on held-out test prompts.
  Direct cosine = cos(suspect, victim) on the same test prompts (no pull-back).

Spaces:
  - acts_last / acts_mean: reef_baseline/acts prompts32_*  (1 vector per prompt)
  - pert_cloud: v3 feats/pert_*.npz; each prompt contributes N sample vectors
    (this is the 'larger calibration samples' lever)

Same-dim pairs only (D=896): fine_tune, weight_perm, rep_perm, scale, prune,
neg_qwen2_0.5b. Cross-dim negatives cannot do cosine/QSh without a projector.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent))
from common import NEGS, make_permutation, make_anisotropic_scale  # noqa

OUT = Path(__file__).resolve().parent
FEATS = H / "feats"
ACTS = H / "reef_baseline" / "acts"
RNG = np.random.default_rng(0)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney / ROC AUC
    c = 0.0
    for p in pos:
        c += (neg < p).sum() + 0.5 * (neg == p).sum()
    return float(c / (len(pos) * len(neg)))


def fit_qsh(X, Y):
    """X,Y [M,D] -> R [D,D] orthogonal, s [D], mx, my. Maps X -> (X-mx)@R*s + my."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    R, _ = orthogonal_procrustes(Xc, Yc)  # Xc @ R ≈ Yc
    # per-dim scale: minimize ||Xc@R * s - Yc|| ; closed form per column after rotation
    XR = Xc @ R
    # s_j = <XR_j, Yc_j> / ||XR_j||^2
    num = (XR * Yc).sum(0)
    den = (XR * XR).sum(0)
    s = np.ones(X.shape[1], np.float64)
    mask = den > 1e-12
    s[mask] = num[mask] / den[mask]
    return R, s, mx, my


def apply_qsh(X, R, s, mx, my):
    X = np.asarray(X, np.float64)
    return (X - mx) @ R * s + my


def fit_qsh_pca(X, Y, k):
    """PCA on stacked calib (shared basis from Y), then QSh in k dims; return projectors + params."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    my = Y.mean(0); Yc = Y - my
    # PCA basis from victim calib
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    V = Vt[:k].T  # [D,k]
    Xk, Yk = (X - X.mean(0)) @ V, Yc @ V
    R, s, mxk, myk = fit_qsh(Xk, Yk)
    return dict(V=V, R=R, s=s, mx_full=X.mean(0), my_full=my, mxk=mxk, myk=myk, k=k)


def apply_qsh_pca(X, p):
    Xk = (np.asarray(X, np.float64) - p["mx_full"]) @ p["V"]
    return apply_qsh(Xk, p["R"], p["s"], p["mxk"], p["myk"]), (np.asarray(X, np.float64) - p["my_full"]) @ p["V"]
    # wait: victim side should use Y projected. Caller handles.


def cos_rows(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    an = np.linalg.norm(A, axis=1); bn = np.linalg.norm(B, axis=1)
    an[an == 0] = 1; bn[bn == 0] = 1
    return (A * B).sum(1) / (an * bn)


def load_acts_same_dim():
    """Return dict name -> [L, P, D] float64 for D=896 models. prune has L=20."""
    def ld(n, key):
        return np.load(ACTS / f"{n}.npz")[key].astype(np.float64)
    victim = ld("victim", "prompts32_last")  # [24,32,896]
    ins = ld("instruct", "prompts32_last")
    D = ins.shape[-1]
    perm = make_permutation(D, seed=1).argmax(0).numpy()
    s = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float64)
    out = {
        "fine_tune": ins,
        "weight_perm": ld("instruct_wperm", "prompts32_last"),
        "rep_perm": ins[..., perm],
        "scale": ins * s,
        "prune": ld("instruct_prune", "prompts32_last"),  # [20,32,896]
        "neg_qwen2_0.5b": ld("neg_qwen2_0.5b", "prompts32_last"),
    }
    mean_pool = {
        "fine_tune": ld("instruct", "prompts32_mean"),
        "weight_perm": ld("instruct_wperm", "prompts32_mean"),
        "rep_perm": ld("instruct", "prompts32_mean")[..., perm],
        "scale": ld("instruct", "prompts32_mean") * s,
        "prune": ld("instruct_prune", "prompts32_mean"),
        "neg_qwen2_0.5b": ld("neg_qwen2_0.5b", "prompts32_mean"),
    }
    return victim, out, mean_pool


def relative_layer_indices(n_sus, n_vic, n_pick=7):
    """Even relative-depth indices into suspect and victim layer axes."""
    fracs = np.linspace(0, 1, n_pick)
    return [(int(round(f * (n_sus - 1))), int(round(f * (n_vic - 1)))) for f in fracs]


def score_acts_split(victim, suspects, calib_idx, test_idx, mode="qsh", pca_k=None, n_layers=7):
    """mode in {cosine, qsh, qsh_pca}. Returns per-suspect list of per-test-prompt scores (mean over layers)."""
    Lv = victim.shape[0]
    scores = {name: [] for name in suspects}
    # precompute layer pairs per suspect (prune has fewer layers)
    for name, S in suspects.items():
        Ls = S.shape[0]
        pairs = relative_layer_indices(Ls, Lv, n_layers)
        per_prompt = []
        for ti in test_idx:
            layer_cos = []
            for ls, lv in pairs:
                # fit on calib prompts at this layer
                Xc = S[ls, calib_idx]  # [Mc, D]
                Yc = victim[lv, calib_idx]
                Xt = S[ls, ti:ti+1]
                Yt = victim[lv, ti:ti+1]
                if mode == "cosine":
                    layer_cos.append(float(cos_rows(Xt, Yt)[0]))
                elif mode == "qsh":
                    R, s, mx, my = fit_qsh(Xc, Yc)
                    Zt = apply_qsh(Xt, R, s, mx, my)
                    layer_cos.append(float(cos_rows(Zt, Yt)[0]))
                elif mode == "qsh_pca":
                    p = fit_qsh_pca(Xc, Yc, pca_k)
                    Zt = apply_qsh((Xt - p["mx_full"]) @ p["V"], p["R"], p["s"], p["mxk"], p["myk"])
                    Ytp = (Yt - p["my_full"]) @ p["V"]
                    layer_cos.append(float(cos_rows(Zt, Ytp)[0]))
                else:
                    raise ValueError(mode)
            per_prompt.append(float(np.mean(layer_cos)))
        scores[name] = per_prompt
    return scores


def eval_auc_table(scores, derived, negatives):
    """Pooled per-prompt AUC: all derived test scores vs all neg test scores; also gap."""
    pos = np.concatenate([np.asarray(scores[n], float) for n in derived])
    neg = np.concatenate([np.asarray(scores[n], float) for n in negatives])
    gap = float(min(np.mean(scores[n]) for n in derived) - max(np.mean(scores[n]) for n in negatives))
    return dict(
        auc_pooled=auc(pos, neg),
        gap_mean=gap,
        derived_means={n: float(np.mean(scores[n])) for n in derived},
        neg_means={n: float(np.mean(scores[n])) for n in negatives},
        # per derived vs hard neg only
        auc_vs_qwen2={n: auc(scores[n], scores["neg_qwen2_0.5b"]) for n in derived if "neg_qwen2_0.5b" in scores},
        # fine_tune-only (matches 10/19 derived set without perm/scale attacks)
        auc_ft_only=auc(scores["fine_tune"], scores["neg_qwen2_0.5b"]) if "fine_tune" in scores and "neg_qwen2_0.5b" in scores else None,
        gap_ft_only=(float(np.mean(scores["fine_tune"]) - np.mean(scores["neg_qwen2_0.5b"]))
                     if "fine_tune" in scores and "neg_qwen2_0.5b" in scores else None),
    )


def run_acts_protocol(pool="last"):
    victim_last, sus_last, sus_mean = load_acts_same_dim()
    victim = victim_last if pool == "last" else np.load(ACTS / "victim.npz")["prompts32_mean"].astype(np.float64)
    suspects = sus_last if pool == "last" else sus_mean
    derived = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune"]
    negatives = ["neg_qwen2_0.5b"]
    P = victim.shape[1]
    assert P == 32
    results = []
    # calib sizes; remainder = test. Also fixed 10/19-style 16/16 and 24/8.
    for n_cal in (8, 16, 24):
        # 4 random splits
        splits = []
        for s in range(4):
            idx = RNG.permutation(P)
            splits.append((idx[:n_cal].tolist(), idx[n_cal:].tolist()))
        # also one contiguous split like 10/19 (first n_cal / rest)
        splits.append((list(range(n_cal)), list(range(n_cal, P))))
        # cosine + qsh_pca: all splits; full-dim qsh: contiguous split only (896-d SVD is costly)
        for mode, pca_k in [("cosine", None), ("qsh_pca", 20), ("qsh_pca", 32)]:
            split_rows = []
            for calib_idx, test_idx in splits:
                sc = score_acts_split(victim, suspects, calib_idx, test_idx, mode=mode, pca_k=pca_k)
                ev = eval_auc_table(sc, derived, negatives)
                split_rows.append(ev)
            results.append(dict(
                space=f"acts_{pool}", n_cal=n_cal, n_test=P - n_cal, mode=mode,
                pca_k=pca_k, n_splits=len(splits),
                auc_mean=float(np.mean([r["auc_pooled"] for r in split_rows])),
                auc_std=float(np.std([r["auc_pooled"] for r in split_rows])),
                gap_mean=float(np.mean([r["gap_mean"] for r in split_rows])),
                gap_std=float(np.std([r["gap_mean"] for r in split_rows])),
                contiguous=split_rows[-1],
            ))
            print(f"[acts_{pool}] cal={n_cal} mode={mode} k={pca_k} "
                  f"AUC={results[-1]['auc_mean']:.3f}±{results[-1]['auc_std']:.3f} "
                  f"gap={results[-1]['gap_mean']:.3f}", flush=True)
        # full-dim QSh once (contiguous)
        calib_idx, test_idx = splits[-1]
        sc = score_acts_split(victim, suspects, calib_idx, test_idx, mode="qsh", pca_k=None)
        ev = eval_auc_table(sc, derived, negatives)
        results.append(dict(
            space=f"acts_{pool}", n_cal=n_cal, n_test=P - n_cal, mode="qsh",
            pca_k=None, n_splits=1, split="contiguous",
            auc_mean=ev["auc_pooled"], auc_std=0.0,
            gap_mean=ev["gap_mean"], gap_std=0.0, contiguous=ev,
        ))
        print(f"[acts_{pool}] cal={n_cal} mode=qsh k=None (contiguous) "
              f"AUC={ev['auc_pooled']:.3f} gap={ev['gap_mean']:.3f}", flush=True)
    return results


def load_pert_cloud(N=64):
    """Return victim [P,K,L,N,D], suspects dict same shape. Only seed_set 0 to save RAM/time optional use all 3."""
    def ld(n):
        return np.load(FEATS / f"pert_{n}.npz")["pert"][:, :, :, :N].astype(np.float64)
    # load one at a time carefully
    victim = ld("victim")  # [32,3,4,N,896]
    ins = ld("instruct")
    D = ins.shape[-1]
    perm = make_permutation(D, seed=1).argmax(0).numpy()
    sdiag = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float64)
    sus = {
        "fine_tune": ins,
        "weight_perm": ld("instruct_wperm"),
        "rep_perm": ins[..., perm],
        "scale": ins * sdiag,
        "prune": ld("instruct_prune"),
        "neg_qwen2_0.5b": ld("neg_qwen2_0.5b"),
    }
    return victim, sus


def score_pert_split(victim, suspects, calib_idx, test_idx, mode="qsh", pca_k=None, seed_set=0):
    """Fit on stacked (prompt, sample) from calib; score mean cos over (test prompt, layer, sample).
    Each prompt score = mean over layers of cos(pulled X[n], Y[n]) averaged over n in N.
    """
    # victim: [P,K,L,N,D]
    L = victim.shape[2]
    scores = {name: [] for name in suspects}
    for name, S in suspects.items():
        # fit per layer on calib
        fits = []
        for l in range(L):
            Xc = S[calib_idx, seed_set, l].reshape(-1, S.shape[-1])  # [Mc*N, D]
            Yc = victim[calib_idx, seed_set, l].reshape(-1, victim.shape[-1])
            if mode == "cosine":
                fits.append(None)
            elif mode == "qsh":
                fits.append(fit_qsh(Xc, Yc))
            elif mode == "qsh_pca":
                fits.append(fit_qsh_pca(Xc, Yc, pca_k))
        per_prompt = []
        for ti in test_idx:
            layer_scores = []
            for l in range(L):
                Xt = S[ti, seed_set, l]  # [N,D]
                Yt = victim[ti, seed_set, l]
                if mode == "cosine":
                    layer_scores.append(float(cos_rows(Xt, Yt).mean()))
                elif mode == "qsh":
                    R, s, mx, my = fits[l]
                    Zt = apply_qsh(Xt, R, s, mx, my)
                    layer_scores.append(float(cos_rows(Zt, Yt).mean()))
                elif mode == "qsh_pca":
                    p = fits[l]
                    Zt = apply_qsh((Xt - p["mx_full"]) @ p["V"], p["R"], p["s"], p["mxk"], p["myk"])
                    Ytp = (Yt - p["my_full"]) @ p["V"]
                    layer_scores.append(float(cos_rows(Zt, Ytp).mean()))
            per_prompt.append(float(np.mean(layer_scores)))
        scores[name] = per_prompt
    return scores


def run_pert_protocol(N=64):
    print(f"loading pert clouds N={N} ...", flush=True)
    victim, suspects = load_pert_cloud(N)
    derived = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune"]
    negatives = ["neg_qwen2_0.5b"]
    P = victim.shape[0]
    results = []
    for n_cal in (8, 16, 24):
        splits = []
        for s in range(2):
            idx = RNG.permutation(P)
            splits.append((idx[:n_cal].tolist(), idx[n_cal:].tolist()))
        splits.append((list(range(n_cal)), list(range(n_cal, P))))
        # qsh_pca first (fast), cosine, then full qsh contiguous-only handled below
        for mode, pca_k in [("cosine", None), ("qsh_pca", 20)]:
            split_rows = []
            for calib_idx, test_idx in splits:
                sc = score_pert_split(victim, suspects, calib_idx, test_idx, mode=mode, pca_k=pca_k)
                ev = eval_auc_table(sc, derived, negatives)
                split_rows.append(ev)
            results.append(dict(
                space=f"pert_N{N}", n_cal=n_cal, n_test=P - n_cal, mode=mode, pca_k=pca_k,
                n_calib_vectors=n_cal * N,
                n_splits=len(splits),
                auc_mean=float(np.mean([r["auc_pooled"] for r in split_rows])),
                auc_std=float(np.std([r["auc_pooled"] for r in split_rows])),
                gap_mean=float(np.mean([r["gap_mean"] for r in split_rows])),
                gap_std=float(np.std([r["gap_mean"] for r in split_rows])),
                contiguous=split_rows[-1],
            ))
            print(f"[pert N={N}] cal={n_cal} (M={n_cal*N}) mode={mode} k={pca_k} "
                  f"AUC={results[-1]['auc_mean']:.3f}±{results[-1]['auc_std']:.3f} "
                  f"gap={results[-1]['gap_mean']:.3f}", flush=True)
        # full-dim QSh with large M: contiguous only
        calib_idx, test_idx = splits[-1]
        sc = score_pert_split(victim, suspects, calib_idx, test_idx, mode="qsh", pca_k=None)
        ev = eval_auc_table(sc, derived, negatives)
        results.append(dict(
            space=f"pert_N{N}", n_cal=n_cal, n_test=P - n_cal, mode="qsh", pca_k=None,
            n_calib_vectors=n_cal * N, n_splits=1, split="contiguous",
            auc_mean=ev["auc_pooled"], auc_std=0.0,
            gap_mean=ev["gap_mean"], gap_std=0.0, contiguous=ev,
        ))
        print(f"[pert N={N}] cal={n_cal} (M={n_cal*N}) mode=qsh k=None (contiguous) "
              f"AUC={ev['auc_pooled']:.3f} gap={ev['gap_mean']:.3f}", flush=True)
    del victim, suspects
    return results


def reproduce_1019_numbers():
    """Sanity: reload 10/19 cached feats and recompute A/B AUC to confirm claim numbers."""
    root = Path("/workspace/20261019_DiffIP迁移LLM_线性回拉")
    if not (root / "feats/source_qwen25_05b.npy").exists():
        return {"available": False}
    src = np.load(root / "feats/source_qwen25_05b.npy")  # [72,25,896]
    ins = np.load(root / "feats/derived_instruct.npy")
    lora = np.load(root / "feats/derived_lora.npy")
    neg = np.load(root / "feats/independent_qwen2_05b.npy")
    # 10/19 used layers 6,12,18,24 (1-index?) — check run_experiment
    # From meta: primary_layers — read results.json
    meta = json.loads((root / "results.json").read_text())
    layers = meta.get("meta", {}).get("primary_layers") or meta.get("primary_layers")
    if layers is None:
        # fallback from report: 6/12/18/24 — in 0-index with embedding as 0, that's indices
        layers = [6, 12, 18, 24]
        if src.shape[1] == 25:
            pass
        else:
            layers = [min(l, src.shape[1] - 1) for l in layers]
    print("1019 layers", layers, "shape", src.shape, flush=True)
    cal, test = list(range(36)), list(range(36, 72))

    def score_model(S, mode):
        per = []
        for ti in test:
            lc = []
            for l in layers:
                if l >= S.shape[1]:
                    continue
                Xc, Yc = S[cal, l], src[cal, l]
                Xt, Yt = S[ti:ti+1, l], src[ti:ti+1, l]
                if mode == "cosine":
                    lc.append(float(cos_rows(Xt, Yt)[0]))
                else:
                    R, s, mx, my = fit_qsh(Xc, Yc)
                    lc.append(float(cos_rows(apply_qsh(Xt, R, s, mx, my), Yt)[0]))
            per.append(float(np.mean(lc)))
        return per

    out = {"available": True, "layers": layers}
    for mode in ("cosine", "qsh"):
        pos_i = score_model(ins, mode)
        pos_l = score_model(lora, mode)
        neg_s = score_model(neg, mode)
        out[mode] = dict(
            instruct_mean=float(np.mean(pos_i)),
            lora_mean=float(np.mean(pos_l)),
            neg_mean=float(np.mean(neg_s)),
            auc_pooled=auc(pos_i + pos_l, neg_s),
            auc_instruct=auc(pos_i, neg_s),
            auc_lora=auc(pos_l, neg_s),
            gap=float(min(np.mean(pos_i), np.mean(pos_l)) - np.mean(neg_s)),
        )
        print(f"[1019 reproduce] {mode}: AUC={out[mode]['auc_pooled']:.3f} gap={out[mode]['gap']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    all_out = {"claim_source": "20261019_DiffIP迁移LLM_线性回拉/实验报告.md",
               "claim": "direct cosine AUC≈1.0; full-dim QSh leave-out AUC≈0.538",
               "note": "QSh = orthogonal R + per-dim scale + translation ONLY (already what 10/19 used; not unconstrained linear)",
               "reproduce_1019": reproduce_1019_numbers(),
               "acts_last": run_acts_protocol("last"),
               "acts_mean": run_acts_protocol("mean"),
               "pert_N64": run_pert_protocol(64),
               }
    # also N=128 if memory allows — try
    try:
        all_out["pert_N128"] = run_pert_protocol(128)
    except Exception as e:
        all_out["pert_N128_error"] = str(e)
        print("pert N=128 failed:", e, flush=True)
    (OUT / "results_exp1.json").write_text(json.dumps(all_out, indent=1))
    print("wrote", OUT / "results_exp1.json", flush=True)
