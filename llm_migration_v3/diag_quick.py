"""Quick collapse diagnosis from cached pert feats (no new collection). 4 prompts x 3 seed sets.
paired vs shuffled (sample order of suspect permuted -> pairing broken); rel_resid = d / victim within-layer var."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
import json, math, numpy as np
from common import H
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
rng = np.random.default_rng(0)
def ld(n): return np.load(H/"feats"/f"pert_{n}.npz")["pert"][:4].astype(np.float64)
V = ld("victim"); S = {"fine_tune": ld("instruct"), "neg_gpt2": ld("neg_gpt2")}
def sc(s, v, d):
    pr = project_sequences(s, v, d); ds, rel = [], []
    for l in range(s.shape[0]):
        a = fit_step_alignment(pr.suspect[l], pr.victim[l]); yc = pr.victim[l]-pr.victim[l].mean(0)
        ds.append(a.distance); rel.append(a.distance/((yc**2).sum()/yc.shape[0]))
    return math.exp(-np.mean(ds)), np.mean(ds), np.mean(rel)
out = []
for name, s in S.items():
    for n in [8, 16, 32, 64, 128]:
        for d in [8, 20]:
            pr = [sc(s[p,k,:,:n], V[p,k,:,:n], d) for p in range(4) for k in range(3)]
            sh = [sc(s[p,k][:, rng.permutation(n)], V[p,k,:,:n], d) for p in range(4) for k in range(3)]
            m = lambda x,i: float(np.mean([y[i] for y in x]))
            out.append(dict(pair=name,N=n,dims=d,sim=m(pr,0),dist=m(pr,1),rel_resid=m(pr,2),shuf_sim=m(sh,0),shuf_dist=m(sh,1),shuf_rel_resid=m(sh,2)))
            print(f"{name:10} N={n:3} d={d:2} paired sim {out[-1]['sim']:.3f} d {out[-1]['dist']:.3f} rel {out[-1]['rel_resid']:.3f} | shuffled sim {out[-1]['shuf_sim']:.3f} d {out[-1]['shuf_dist']:.3f} rel {out[-1]['shuf_rel_resid']:.3f}", flush=True)
(H/"scores"/"diag_quick.json").write_text(json.dumps(out, indent=1))
