"""Why do positives 'collapse' as N grows? Diagnostics on 4 prompts, seed set 0.
D1 unpaired control: shuffle the suspect's sample order (breaks seed pairing). If a
   shuffled pair scores as well as the true pair, the score is not using pairing (ALS overfit).
D2 relative residual: d / ((1/N)||Y_c||^2) = unexplained fraction of victim within-layer variance.
D3 within-layer share of the PCA-retained variance (signal ALS actually sees after centering).
D4 perturbation-scale sweep (fresh collection, victim/instruct/gpt2, prompt 0): scales 5e-4..2e-2."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import json, math, numpy as np
from common import *
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
F = H / "feats"; rng = np.random.default_rng(0)
def ld(n): return np.load(F / f"pert_{n}.npz")["pert"][:4, 0].astype(np.float64)  # [4 prompts, L, N, D]
V = ld("victim"); S = {"fine_tune": ld("instruct"), "neg_gpt2": ld("neg_gpt2"), "neg_pythia_160m": ld("neg_pythia_160m")}
def score(s, v, d, shuffle=False):
    if shuffle: s = s[:, rng.permutation(s.shape[1])]
    pr = project_sequences(s, v, d); ds, rel, ws = [], [], []
    for l in range(s.shape[0]):
        a = fit_step_alignment(pr.suspect[l], pr.victim[l]); ds.append(a.distance)
        yc = pr.victim[l] - pr.victim[l].mean(0); rel.append(a.distance / ((yc**2).sum() / yc.shape[0]))
    allv = pr.victim.reshape(-1, pr.victim.shape[-1]); tot = ((allv - allv.mean(0))**2).sum()
    within = sum(((pr.victim[l] - pr.victim[l].mean(0))**2).sum() for l in range(s.shape[0]))
    return math.exp(-sum(ds)/len(ds)), float(np.mean(ds)), float(np.mean(rel)), float(within/tot)
out = {"D1_D2_D3": []}
for name, s in S.items():
    for n in [8, 16, 32, 64, 128]:
        for d in [8, 20]:
            r = [score(s[p, :, :n], V[p, :, :n], d) for p in range(4)]
            sh = [score(s[p, :, :n], V[p, :, :n], d, True) for p in range(4)]
            m = lambda xs, i: float(np.mean([x[i] for x in xs]))
            out["D1_D2_D3"].append(dict(pair=name, N=n, dims=d, sim=m(r,0), dist=m(r,1), rel_resid=m(r,2),
                within_share=m(r,3), shuffled_sim=m(sh,0), shuffled_dist=m(sh,1), shuffled_rel_resid=m(sh,2)))
            print(out["D1_D2_D3"][-1], flush=True)
# D4
os.environ["OMP_NUM_THREADS"] = "8"; torch.set_num_threads(8)
reps = {}
for name in ["victim", "instruct", "neg_gpt2"]:
    m, tok, emb, _ = build_model(name); L = even_layers(len(decoder_layers(m)), 4)
    base = emb(tok(PROMPTS[0], return_tensors="pt")["input_ids"]).detach()
    reps[name] = {sc: None for sc in [5e-4, 2e-3, 8e-3, 2e-2]}
    for sc in reps[name]:
        g = torch.Generator().manual_seed(123); xs = [base] + [base + sc*torch.randn(base.shape, generator=g) for _ in range(127)]
        reps[name][sc] = forward_pool(m, torch.cat(xs), L).astype(np.float64)
    del m; gc.collect()
out["D4_scale"] = []
for sc in [5e-4, 2e-3, 8e-3, 2e-2]:
    for n in [8, 128]:
        for d in [8, 20]:
            v = reps["victim"][sc][:, :n]
            ft = score(reps["instruct"][sc][:, :n], v, d); ng = score(reps["neg_gpt2"][sc][:, :n], v, d)
            out["D4_scale"].append(dict(scale=sc, N=n, dims=d, fine_tune_sim=ft[0], fine_tune_dist=ft[1], fine_tune_rel=ft[2],
                                        gpt2_sim=ng[0], gpt2_dist=ng[1], gpt2_rel=ng[2])); print(out["D4_scale"][-1], flush=True)
(H / "scores" / "diagnose.json").write_text(json.dumps(out, indent=1))
