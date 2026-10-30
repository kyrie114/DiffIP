"""Score dyn_collect features with the lit_protocol metric (N=128, PCA 8, ALS Q/S/h,
sim=exp(-mean d)); middle2 & all4 on seed set 0 vs the 5 base v3 negatives (../feats), and a
per-layer curve (single layer, all decoder layers, seed set 0) vs the victim's all-layer feats."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, math
from multiprocessing import Pool
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; V3 = H.parent
sys.path.insert(0, str(V3)); sys.path.insert(0, str(V3.parent))
from common import NEGS
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
LS = {"all4": [0, 1, 2, 3], "middle2": [1, 2]}
def sim(x, y):
    pr = project_sequences(x.astype(np.float64), y.astype(np.float64), 8)
    return math.exp(-np.mean([fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(len(x))]))
_c = {}
def ld(p, key="pert"):
    if (p, key) not in _c:
        if len(_c) > 3: _c.clear()
        _c[(p, key)] = np.load(p)[key]
    return _c[(p, key)]
VP = V3 / "feats/pert_victim.npz"; VA = H / "feats/victim.npz"
def job(a):
    kind, path, p, t = a
    if kind == "L": return sim(ld(path, "all0")[p][[t]], ld(VA, "all0")[p][[t]])
    return sim(ld(path)[p, 0][LS[t]], ld(VP)[p, 0][LS[t]])
if __name__ == "__main__":
    vars_ = [f.stem for f in sorted((H / "feats").glob("*.npz")) if f.stem != "victim"]
    negs = {k: V3 / f"feats/pert_{k}.npz" for k in NEGS}
    out = {"neg5": {}, "variants": {}}
    with Pool(6, maxtasksperchild=100) as pool:
        for k, f in negs.items():
            out["neg5"][k] = {t: pool.map(job, [("P", f, p, t) for p in range(32)]) for t in LS}
        neg = {t: np.concatenate([v[t] for v in out["neg5"].values()]) for t in LS}
        mx = {t: max(np.mean(v[t]) for v in out["neg5"].values()) for t in LS}
        T = int(np.load(VA)["total"])
        for k in vars_:
            f = H / f"feats/{k}.npz"; r = {}
            for t in LS:
                s = np.array(pool.map(job, [("P", f, p, t) for p in range(32)]))
                r[t] = dict(sim=float(s.mean()), gap=float(s.mean() - mx[t]), auc=float(((s[:, None] > neg[t]) + .5 * (s[:, None] == neg[t])).mean()), per_prompt=s.tolist())
            r["layer_curve"] = [float(np.mean(pool.map(job, [("L", f, p, l) for p in range(32)]))) for l in range(T)]
            out["variants"][k] = r; print(k, {t: (round(r[t]["sim"], 3), round(r[t]["gap"], 3), round(r[t]["auc"], 3)) for t in LS}, np.round(r["layer_curve"], 3).tolist(), flush=True)
    out["neg5_max"] = mx
    (H / "results_dynamic.json").write_text(json.dumps(out, indent=1))
