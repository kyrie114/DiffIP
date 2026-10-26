"""Pruning with DiffIP's DP temporal alignment ON (official DiffIPMetric.compare):
all decoder layers as the step axis (pruned Instruct 20 layers vs victim 24), N=20
(paper's N), seed set 0, every prompt, dims 8 and 20. Fine-tune and the 5 negatives
are scored the same way as references. DP-off counterpart on the same all-layer data:
matched even subset (victim 24 -> 20 layers) + 1:1 ALS."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import json, itertools, math
from multiprocessing import Pool
import numpy as np
from common import H, NEGS
from diffip.metric import DiffIPMetric
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
F = H / "feats"
def ld(n): return np.load(F / f"pert_{n}.npz")["dp_all"].astype(np.float64)
V = ld("victim"); SUS = {"prune": ld("instruct_prune"), "fine_tune": ld("instruct")}
for k in NEGS: SUS[k] = ld(k)
def sub(x, n): return x[[round(i * (x.shape[0] - 1) / (n - 1)) for i in range(n)]]
def job(a):
    name, p, d = a; s, v = SUS[name][p], V[p]
    on = DiffIPMetric(n_components=d).compare(s, v)
    n = min(s.shape[0], v.shape[0]); s2, v2 = sub(s, n), sub(v, n)
    pr = project_sequences(s2, v2, d)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(n)]
    return dict(pair=name, prompt=p, dims=d, Ls=s.shape[0], Lv=v.shape[0],
                dp_on_similarity=on.similarity, dp_on_distance=on.distance,
                dp_on_distance_per_step=on.distance / math.sqrt(s.shape[0] * v.shape[0]),
                dp_off_similarity=math.exp(-sum(dl) / n), dp_off_distance_mean=float(np.mean(dl)))
if __name__ == "__main__":
    out = H / "scores"; out.mkdir(exist_ok=True)
    with Pool(8) as pool:
        for name in SUS:
            f = out / f"dp_{name}.json"
            if f.exists(): print("skip", name, flush=True); continue
            rows = pool.map(job, list(itertools.product([name], range(V.shape[0]), [8, 20])), chunksize=2)
            f.write_text(json.dumps(rows)); print("dp", name, len(rows), flush=True)
