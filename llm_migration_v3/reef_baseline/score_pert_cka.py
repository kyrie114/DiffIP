"""Apples-to-apples: linear CKA (REEF formula) on the very same v3 perturbation clouds
feats/pert_*.npz [32 prompts, 3 seed sets, 4 layers, 128, D], N=128 and N=20, same 4 layers.
Each (prompt, seed set, layer): X,Y = [N, D] paired samples. Report mean over layers, prompts, sets."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
import numpy as np
from common import H, NEGS, make_permutation, make_anisotropic_scale
def std(X):
    X = X - X.mean(0); sd = X.std(0, ddof=1); sd[sd == 0] = 1; return X / sd
def lin_cka(X, Y):
    n = X.shape[0]; Hm = np.eye(n) - 1.0 / n
    a, b = Hm @ (X @ X.T) @ Hm, Hm @ (Y @ Y.T) @ Hm
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
F = H / "feats"
def ld(n): return np.load(F / f"pert_{n}.npz")["pert"]
V = ld("victim"); ins = ld("instruct"); D = ins.shape[-1]
perm = make_permutation(D, seed=1).argmax(0).numpy()
s = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float32)
SUS = {"fine_tune": lambda: ins, "weight_perm": lambda: ld("instruct_wperm"), "rep_perm": lambda: ins[..., perm],
       "scale": lambda: ins * s, "prune": lambda: ld("instruct_prune"), **{k: (lambda k=k: ld(k)) for k in NEGS}}
out = {}
for name, f in SUS.items():
    S = f(); r = {}
    for N in (20, 128):
        a = np.array([[[lin_cka(std(V[p, k, l, :N].astype(np.float64)), std(S[p, k, l, :N].astype(np.float64)))
                        for l in range(4)] for k in range(3)] for p in range(32)])
        r[N] = dict(mean=float(a.mean()), per_prompt=a.mean((1, 2)).tolist(), per_layer=a.mean((0, 1)).tolist())
    out[name] = r; print(name, round(r[128]["mean"], 4), flush=True)
(H / "reef_baseline" / "results_pert_cka.json").write_text(json.dumps(out, indent=1))
