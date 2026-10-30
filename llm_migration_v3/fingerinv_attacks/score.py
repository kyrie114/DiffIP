"""Score attacked models vs victim at the v3 main-table setting (N=128, PCA dims=8),
all 3 seed sets, with both all4 layers (= ../score.py) and middle2 [1,2] (= ../score_middle2.py).
Negatives / reference rows are read from the existing ../scores/*.json (not recomputed).
Metrics on seed set 0, 32 prompts (same as score_middle2.aggregate): mean sim, gap =
sim(attack) - max_neg mean sim, pooled AUC (attack prompts vs all neg prompts), wins
(prompts where attack > all 5 negs), plus a one-sided permutation-test p-value
(mean of 32 attack sims vs 160 neg sims, 20000 label shuffles)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, math
from multiprocessing import Pool
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; V3 = H.parent
sys.path.insert(0, str(V3)); sys.path.insert(0, str(V3.parent))
from common import SEED_SETS, NEGS
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
N, D_ = 128, 8
LS = {"all4": [0, 1, 2, 3], "middle2": [1, 2]}
victim = np.load(V3 / "feats/pert_victim.npz")["pert"].astype(np.float64)

def job(a):
    name, p, k, tag = a
    x = np.load(H / f"feats/pert_{name}.npz")["pert"][p, k][LS[tag], :N].astype(np.float64)
    pr = project_sequences(x, victim[p, k, LS[tag], :N], D_)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(len(LS[tag]))]
    return dict(pair=name, prompt=p, seed_set=k, layers=tag, N=N, dims=D_, distance_mean=float(np.mean(dl)), similarity=math.exp(-sum(dl) / len(dl)))

def ref_rows(tag, names):
    out = {}
    for n in names:
        f = V3 / "scores" / (f"pert_{n}.json" if tag == "all4" else f"pert_middle2_{n}.json")
        for r in json.loads(f.read_text()):
            if r["N"] == N and r["dims"] == D_: out[(n, r["prompt"], r["seed_set"])] = r["similarity"]
    return out

def perm_p(pos, neg, rng, B=20000):
    allv = np.concatenate([pos, neg]); obs = pos.mean() - neg.mean(); c = 0
    for _ in range(B):
        rng.shuffle(allv); c += (allv[:len(pos)].mean() - allv[len(pos):].mean()) >= obs
    return (c + 1) / (B + 1)

if __name__ == "__main__":
    names = sys.argv[1:]
    (H / "scores").mkdir(exist_ok=True)
    with Pool(8) as pool:
        for n in names:
            f = H / f"scores/{n}.json"
            if f.exists(): continue
            rows = pool.map(job, [(n, p, k, t) for t in LS for p in range(32) for k in range(3)])
            f.write_text(json.dumps(rows)); print("scored", n, flush=True)
    refs = ["fine_tune", "prune_dpoff"]; negs = list(NEGS)
    res = {}
    rng = np.random.default_rng(0)
    for tag in LS:
        R = ref_rows(tag, refs + negs)
        for n in sorted(H.glob("scores/*.json")):
            for r in json.loads(n.read_text()):
                if r["layers"] == tag: R[(r["pair"], r["prompt"], r["seed_set"])] = r["similarity"]
        att = refs + [p.stem for p in sorted(H.glob("scores/*.json"))]
        neg0 = {q: np.array([R[(q, i, 0)] for i in range(32)]) for q in negs}
        negmax_mean = max(v.mean() for v in neg0.values())
        negpool = np.concatenate(list(neg0.values())); negmax_pp = np.max(np.stack(list(neg0.values())), 0)
        res[tag] = {"neg_means": {q: float(v.mean()) for q, v in neg0.items()}, "max_neg_mean": float(negmax_mean), "attacks": {}}
        for a in att:
            s = np.array([R[(a, i, 0)] for i in range(32)])
            ss = [np.mean([R[(a, i, k)] for i in range(32)]) for k in range(3)]
            res[tag]["attacks"][a] = dict(sim_seed0=float(s.mean()), sim_3set_mean=float(np.mean(ss)), sim_3set_std=float(np.std(ss, ddof=1)),
                gap=float(s.mean() - negmax_mean), auc=float(((s[:, None] > negpool[None]) + 0.5 * (s[:, None] == negpool[None])).mean()),
                wins=int((s > negmax_pp).sum()), perm_p=float(perm_p(s.copy(), negpool.copy(), rng)))
    (H / "results.json").write_text(json.dumps(res, indent=1))
    for tag in LS:
        print(f"\n## {tag}  max neg mean={res[tag]['max_neg_mean']:.3f}")
        print("| attack | sim(seed0) | sim 3-set mean±std | gap | AUC | wins | perm p |")
        for a, v in res[tag]["attacks"].items():
            print(f"| {a} | {v['sim_seed0']:.3f} | {v['sim_3set_mean']:.3f}±{v['sim_3set_std']:.3f} | {v['gap']:+.3f} | {v['auc']:.3f} | {v['wins']}/32 | {v['perm_p']:.1e} |")
