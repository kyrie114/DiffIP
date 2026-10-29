"""Distillation / extraction robustness (feasible proxy on CPU).

Victim panels:
  A) teacher = Qwen2.5-0.5B-Instruct  (cross-arch sequence KD source)
  B) base    = SmolLM2-135M           (student init; same-arch SFT-on-teacher-text)

Suspects: student_distill, student_control (existing ckpts), plus negatives.
Metrics:
  (1) DiffIP middle2 + ALS mean-L2 on paired pert (cross-dim OK via PCA)
  (2) large-M constrained QSh holdout when dims match
  (3) cosine control when dims match

No new distill training on this box (CPU-only). Reuses
  /workspace/papers/exp_distill_v2/ckpt/student_{distill,control}
"""
from __future__ import annotations
import os, json, math, gc, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
H = OUT.parent
ROOT = H.parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(ROOT))
from common import (  # noqa
    PROMPTS, SCALE, even_layers, load, perturb_batch, forward_pool,
)
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment

CKPT = Path("/workspace/papers/exp_distill_v2/ckpt")
FEATS = OUT / "feats_distill"
FEATS.mkdir(exist_ok=True)

TEACHER = "Qwen/Qwen2.5-0.5B-Instruct"
BASE = "HuggingFaceTB/SmolLM2-135M"
MODELS = {
    "teacher": TEACHER,
    "base_smollm2_135m": BASE,
    "student_distill": str(CKPT / "student_distill"),
    "student_control": str(CKPT / "student_control"),
    "neg_gpt2": "openai-community/gpt2",
    "neg_qwen2_0.5b": "Qwen/Qwen2-0.5B",
    "neg_smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
}
# Feasible CPU budget: 1 seed set, N=64, 4 even layers, 32 prompts
SEED = 123
N_MAX = 64
N_LAYERS = 4
MIDDLE2 = [1, 2]
DIMS = 8
CHUNK = 16  # smaller chunks for 0.5B on 16GB


def collect_one(name: str) -> Path:
    out = FEATS / f"{name}.npz"
    if out.exists():
        print("skip collect", name, flush=True)
        return out
    path = MODELS[name]
    print(f"[{time.strftime('%H:%M:%S')}] loading {name} from {path}", flush=True)
    model, tok = load(path)
    Ltot = model.config.num_hidden_layers
    layers = even_layers(Ltot, N_LAYERS)
    D = model.config.hidden_size
    feats = np.zeros((len(PROMPTS), N_LAYERS, N_MAX, D), dtype=np.float32)
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        # truncate very long prompts to keep CPU sane
        if ids.shape[1] > 64:
            ids = ids[:, :64]
        base_e = model.get_input_embeddings()(ids)
        embeds = perturb_batch(base_e, N_MAX, SEED)  # [N,T,D]
        pooled = forward_pool(model, embeds, layers, chunk=CHUNK)  # [L,N,D]
        feats[pi] = pooled
        if (pi + 1) % 4 == 0 or pi == 0:
            print(f"  {name} prompt {pi+1}/{len(PROMPTS)}", flush=True)
    np.savez_compressed(out, pert=feats, layers=np.array(layers), total=Ltot,
                        seed=SEED, N=N_MAX, scale=SCALE)
    del model
    gc.collect()
    print("saved", out, feats.shape, flush=True)
    return out


def lin_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    n = X.shape[0]
    Hm = np.eye(n) - 1.0 / n
    a = Hm @ (X @ X.T) @ Hm
    b = Hm @ (Y @ Y.T) @ Hm
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def als_sim_middle2(S, V, n=N_MAX, dims=DIMS, layers=MIDDLE2):
    """S,V: [P,L,N,D] -> list of sim per prompt."""
    out = []
    for p in range(S.shape[0]):
        dists = []
        for l in layers:
            sus = S[p:p+1, l, :n]
            vic = V[p:p+1, l, :n]
            pr = project_sequences(sus, vic, dims)
            dists.append(fit_step_alignment(pr.suspect[0], pr.victim[0]).distance)
        out.append(math.exp(-sum(dists) / len(dists)))
    return out


def cosine_per_prompt(S, V, n=N_MAX, layers=MIDDLE2):
    """Mean over layers of cosine between mean-pooled clouds (same D only)."""
    if S.shape[-1] != V.shape[-1]:
        return [float("nan")] * S.shape[0]
    out = []
    for p in range(S.shape[0]):
        cs = []
        for l in layers:
            a = S[p, l, :n].mean(0); b = V[p, l, :n].mean(0)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cs.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
        out.append(float(np.mean(cs)))
    return out


def qsh_holdout(S, V, n_cal_prompts=16, n=N_MAX, layers=MIDDLE2, pca_k=None):
    """Constrained Q/S/h on concatenated cloud; fit on first n_cal prompts, eval rest.
    Same-dim only. Returns list of residual-sim on holdout prompts (one score each),
    or empty if dims differ / too few.
    """
    if S.shape[-1] != V.shape[-1]:
        return dict(ok=False, reason="dim_mismatch")
    P = S.shape[0]
    cal, te = list(range(n_cal_prompts)), list(range(n_cal_prompts, P))
    # stack cal samples per layer: [M, D]
    from numpy.linalg import svd

    def fit_qsh(X, Y):
        # X,Y [M,D]; solve min ||X R diag(s) + 1 h^T - Y|| ; R orthogonal via SVD on centered
        Xc = X - X.mean(0); Yc = Y - Y.mean(0)
        if pca_k is not None and pca_k < X.shape[1]:
            # joint PCA on concat
            C = np.concatenate([Xc, Yc], 0)
            _, _, Vt = svd(C, full_matrices=False)
            W = Vt[:pca_k].T  # D x k
            Xc, Yc = Xc @ W, Yc @ W
            X, Y = X @ W, Y @ W
        # orthogonal Procrustes on centered
        U, _, Vt = svd(Xc.T @ Yc, full_matrices=False)
        R = U @ Vt
        Xr = Xc @ R
        # diagonal scale
        num = (Xr * Yc).sum(0); den = (Xr * Xr).sum(0); den[den == 0] = 1.0
        s = num / den
        # translation in original (possibly PCA) space
        h = Y.mean(0) - (X.mean(0) @ R) * s
        return R, s, h, (W if pca_k is not None and pca_k < X.shape[1] else None)

    def apply(X, R, s, h, W):
        if W is not None:
            X = X @ W
        return (X @ R) * s + h

    scores = []
    # fit one QSh per layer on cal cloud, score each test prompt
    fits = []
    for l in layers:
        X = np.concatenate([S[p, l, :n] for p in cal], 0)
        Y = np.concatenate([V[p, l, :n] for p in cal], 0)
        fits.append(fit_qsh(X, Y))
    for p in te:
        dists = []
        for li, l in enumerate(layers):
            R, s, h, W = fits[li]
            Xp = S[p, l, :n]; Yp = V[p, l, :n]
            if W is not None:
                Yp = Yp @ W
            pred = apply(Xp, R, s, h, W)
            dists.append(float(np.mean(np.linalg.norm(pred - Yp, axis=1))))
        scores.append(math.exp(-sum(dists) / len(dists)))
    M = n_cal_prompts * n
    return dict(ok=True, n_cal=n_cal_prompts, M_per_layer=M, pca_k=pca_k,
                holdout_sims=scores, mean=float(np.mean(scores)),
                std=float(np.std(scores)))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    c = 0.0
    for p in pos:
        c += (neg < p).sum() + 0.5 * (neg == p).sum()
    return float(c / (len(pos) * len(neg)))


def ld(name):
    return np.load(FEATS / f"{name}.npz")["pert"].astype(np.float64)


def eval_panel(victim_name: str, suspects: list[str], label: str):
    V = ld(victim_name)
    panel = {"victim": victim_name, "label": label, "N": N_MAX, "dims": DIMS,
             "layers": "middle2", "layer_idx": MIDDLE2, "seed": SEED, "pairs": {}}
    for name in suspects:
        S = ld(name)
        diff = als_sim_middle2(S, V)
        cos = cosine_per_prompt(S, V)
        cka = []
        for p in range(S.shape[0]):
            cs = []
            for l in MIDDLE2:
                X = S[p, l].copy(); Y = V[p, l].copy()
                # pad/project not needed for CKA if we use linear CKA on N×D independently —
                # cross-dim CKA: use PCA to min(D) via project side? skip cross-dim CKA
                if X.shape[1] != Y.shape[1]:
                    cs.append(float("nan"))
                else:
                    X = (X - X.mean(0)); Y = (Y - Y.mean(0))
                    xs = X.std(0, ddof=1); ys = Y.std(0, ddof=1)
                    xs[xs == 0] = 1; ys[ys == 0] = 1
                    cs.append(lin_cka(X / xs, Y / ys))
            cka.append(float(np.nanmean(cs)))
        qsh = qsh_holdout(S, V, n_cal_prompts=16, n=N_MAX, layers=MIDDLE2, pca_k=None)
        qsh_pca = qsh_holdout(S, V, n_cal_prompts=16, n=N_MAX, layers=MIDDLE2, pca_k=32)
        panel["pairs"][name] = dict(
            D_sus=int(S.shape[-1]), D_vic=int(V.shape[-1]),
            diffip_middle2=dict(mean=float(np.mean(diff)), std=float(np.std(diff)),
                                min=float(np.min(diff)), max=float(np.max(diff)),
                                per_prompt=diff),
            cosine=dict(mean=float(np.nanmean(cos)), per_prompt=cos),
            cka=dict(mean=float(np.nanmean(cka)), per_prompt=cka),
            qsh_full=qsh,
            qsh_pca32=qsh_pca,
        )
        print(f"[{label}] {name}: DiffIP={np.mean(diff):.3f} cos={np.nanmean(cos):.3f} "
              f"cka={np.nanmean(cka):.3f} qsh={qsh.get('mean', float('nan'))}", flush=True)

    # detection: student_distill vs (student_control + negs)
    pos = panel["pairs"]["student_distill"]["diffip_middle2"]["per_prompt"]
    neg_names = [n for n in suspects if n != "student_distill"]
    neg = np.concatenate([panel["pairs"][n]["diffip_middle2"]["per_prompt"] for n in neg_names])
    panel["detection_diffip"] = dict(
        pos_mean=float(np.mean(pos)),
        max_neg_mean=float(max(np.mean(panel["pairs"][n]["diffip_middle2"]["per_prompt"])
                               for n in neg_names)),
        gap=float(np.mean(pos) - max(np.mean(panel["pairs"][n]["diffip_middle2"]["per_prompt"])
                                     for n in neg_names)),
        auc_vs_all_negs=auc(pos, neg),
        wins_vs_control=int(sum(
            pos[i] > panel["pairs"]["student_control"]["diffip_middle2"]["per_prompt"][i]
            for i in range(len(pos))
        )),
        neg_names=neg_names,
    )
    return panel


def main():
    names = list(MODELS)
    for n in names:
        collect_one(n)

    results = {
        "protocol": dict(
            scale=SCALE, seed=SEED, N=N_MAX, n_layers_collected=N_LAYERS,
            score_layers=MIDDLE2, dims=DIMS, n_prompts=len(PROMPTS),
            note="CPU proxy; no new KD training; ckpts from exp_distill_v2",
        ),
        "panels": {},
    }
    # Panel A: victim=teacher (cross-arch distill source)
    results["panels"]["vs_teacher"] = eval_panel(
        "teacher",
        ["student_distill", "student_control", "base_smollm2_135m",
         "neg_gpt2", "neg_qwen2_0.5b", "neg_smollm2_360m"],
        "vs_teacher",
    )
    # Panel B: victim=base (same-arch; distill ≈ SFT on teacher text)
    results["panels"]["vs_base"] = eval_panel(
        "base_smollm2_135m",
        ["student_distill", "student_control",
         "neg_gpt2", "neg_smollm2_360m"],
        "vs_base",
    )
    out = OUT / "results_exp3_distill.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
