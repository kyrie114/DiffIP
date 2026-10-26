"""Linear / RBF CKA exactly as official REEF src/compute_cka.py (center + per-dim std scale as
load_acts(center=True, scale=True); linear_CKA is the default, kernel_CKA w/ median-heuristic sigma
is the commented alternative). Suspect transforms rep_perm / scale applied to instruct features
exactly as v3 score.py. Outputs results.json."""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from common import H, NEGS, even_layers, make_permutation, make_anisotropic_scale
A = H / "reef_baseline" / "acts"
def std(X):
    X = X - X.mean(0); s = X.std(0, ddof=1); s[s == 0] = 1; return X / s
def cent(K):
    n = K.shape[0]; Hm = np.eye(n) - 1.0 / n; return Hm @ K @ Hm
def lin_cka(X, Y):
    a, b = cent(X @ X.T), cent(Y @ Y.T); return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
def rbf(X):
    G = X @ X.T; K = np.diag(G) - G + (np.diag(G) - G).T; sig2 = np.median(K[K != 0])
    return np.exp(-0.5 * K / sig2)
def rbf_cka(X, Y):
    a, b = cent(rbf(X)), cent(rbf(Y)); return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
def ld(n): return dict(np.load(A / f"{n}.npz"))
V = ld("victim"); ins = ld("instruct"); D = ins["tqa200_last"].shape[-1]
perm = make_permutation(D, seed=1).argmax(0).numpy()
s = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float32)
SUS = {"fine_tune": ins, "weight_perm": ld("instruct_wperm"),
       "rep_perm": {k: (v[..., perm] if k != "total" else v) for k, v in ins.items()},
       "scale": {k: (v * s if k != "total" else v) for k, v in ins.items()},
       "prune": ld("instruct_prune")}
for k in NEGS: SUS[k] = ld(k)
REL = 18 / 31  # REEF default --base_layers 18 for 32-layer LLaMA-2-7B -> relative depth
def at(total, r): return round(r * (total - 1))
out = {}
Tv = int(V["total"])
for name, S in SUS.items():
    Ts = int(S["total"]); r = {"layers_suspect": Ts}
    for feat in ["tqa200_last", "tqa200_mean", "prompts32_last", "prompts32_mean"]:
        X = {l: std(V[feat][l].astype(np.float64)) for l in range(Tv)}
        Y = {l: std(S[feat][l].astype(np.float64)) for l in range(Ts)}
        M = np.array([[lin_cka(X[i], Y[j]) for j in range(Ts)] for i in range(Tv)])
        rel = [lin_cka(X[at(Tv, q)], Y[at(Ts, q)]) for q in np.linspace(0, 1, 9)]
        e4 = [lin_cka(X[a], Y[b]) for a, b in zip(even_layers(Tv, 4), even_layers(Ts, 4))]
        hv, hs = at(Tv, REL), at(Ts, REL)
        r[feat] = dict(headline_linear=lin_cka(X[hv], Y[hs]), headline_rbf=rbf_cka(X[hv], Y[hs]),
                       headline_layers=[hv, hs], last_layer=lin_cka(X[Tv-1], Y[Ts-1]),
                       even4_layers=e4, rel_depth_curve=rel, matrix_max=float(M.max()),
                       matrix_diag_mean_rel=float(np.mean(rel)), matrix=M.round(4).tolist())
        if feat == "tqa200_last":  # 10 disjoint chunks of 20 inputs -> spread for AUC
            r[feat]["chunks20"] = [lin_cka(X[hv][c*20:(c+1)*20], Y[hs][c*20:(c+1)*20]) for c in range(10)]
    out[name] = r; print(name, round(r["tqa200_last"]["headline_linear"], 4), flush=True)
(H / "reef_baseline" / "results.json").write_text(json.dumps(out, indent=1))
