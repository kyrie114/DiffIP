"""Score every (suspect vs victim) pair per prompt / seed set / N / dims with the
official PCA (preprocessing.project_sequences) + per-layer ALS (reversion.fit_step_alignment),
DP off: distance_mean = mean_l d_l, similarity = exp(-sum_l d_l / L)."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import json, sys, itertools, math
from multiprocessing import Pool
import numpy as np
from common import H, N_GRID, SEED_SETS, NEGS, PROMPTS, CATEGORY, make_permutation, make_anisotropic_scale
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
MODE = sys.argv[1] if len(sys.argv) > 1 else "pert"
DIMS = [8, 20]
GRID = sorted(set(N_GRID + [20])) if MODE == "pert" else [8, 16, 20, 32, 64]
F = H / "feats"
def ld(n): return np.load(F / f"{MODE}_{n}.npz")["pert"].astype(np.float64)
victim = ld("victim"); ins = ld("instruct"); D = ins.shape[-1]
perm = make_permutation(D, seed=1).argmax(0).numpy()
s = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float64)
SUS = {"fine_tune": ins, "rep_perm": ins[..., perm], "scale": ins * s,
       "prune_dpoff": ld("instruct_prune")}
if (F / f"{MODE}_instruct_wperm.npz").exists(): SUS["weight_perm"] = ld("instruct_wperm")
for k in NEGS: SUS[k] = ld(k)
def als(sus, vic, d):
    pr = project_sequences(sus, vic, d)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(sus.shape[0])]
    return float(np.mean(dl)), math.exp(-sum(dl) / len(dl)), pr.n_components
def job(a):
    name, p, k, n, d = a
    assert d <= 4 * n - 1, "dims must stay below L*N-1"
    dm, sim, used = als(SUS[name][p, k, :, :n], victim[p, k, :, :n], d)
    assert used == d
    return dict(pair=name, prompt=p, seed_set=k, N=n, dims=d, distance_mean=dm, similarity=sim)
if __name__ == "__main__":
    P = victim.shape[0]; out = H / "scores"; out.mkdir(exist_ok=True)
    with Pool(8) as pool:
        for name in SUS:
            f = out / f"{MODE}_{name}.json"
            if f.exists(): print("skip", name, flush=True); continue
            jobs = list(itertools.product([name], range(P), range(len(SEED_SETS)), GRID, DIMS))
            rows = pool.map(job, jobs, chunksize=16)
            f.write_text(json.dumps(rows)); print(MODE, name, len(rows), flush=True)
