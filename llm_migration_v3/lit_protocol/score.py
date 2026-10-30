"""Score the literature-protocol panel. Same metric code as ../fingerinv_attacks/score.py
(N=128, PCA 8, ALS Q/S/h, sim = exp(-mean d), layers all4 / middle2, seed set 0 for
gap/AUC/wins/perm-p, 3 seed sets for mean±std).
Two decision panels:
  base5 = the 5 original v3 negatives (from ../scores, not recomputed)
  ext7  = base5 + the 2 new hard negatives (Qwen2-0.5B-Instruct, Qwen3-0.6B-Base)
Extra (HuRef NeurIPS'24 §5.1.3 style base attribution): every model is also aligned to two
alternative candidate bases, Qwen2-0.5B and SmolLM2-360M (their v3 pert feats), and we
report which candidate base gives the highest sim."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, math
from multiprocessing import Pool
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; V3 = H.parent
sys.path.insert(0, str(V3)); sys.path.insert(0, str(V3.parent)); sys.path.insert(0, str(H))
from common import NEGS
from lit_models import MODELS
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
N, D_ = 128, 8
LS = {"all4": [0, 1, 2, 3], "middle2": [1, 2]}
BASES = {"qwen2.5_0.5b": V3 / "feats/pert_victim.npz", "qwen2_0.5b": V3 / "feats/pert_neg_qwen2_0.5b.npz",
         "smollm2_360m": V3 / "feats/pert_neg_smollm2_360m.npz"}
def feat(name):
    f = H / f"feats/pert_{name}.npz"
    return f if f.exists() else V3 / f"feats/pert_{name}.npz"
_c = {}
def ld(p):
    if p not in _c:
        if len(_c) >= 3: _c.pop(next(k for k in _c if k not in BASES.values()), None) if any(k not in BASES.values() for k in _c) else _c.clear()
        _c[p] = np.load(p, mmap_mode="r")["pert"] if False else np.load(p)["pert"]
    return _c[p]

def job(a):
    name, base, p, k, tag = a
    x = ld(feat(name))[p, k][LS[tag], :N].astype(np.float64); y = ld(BASES[base])[p, k][LS[tag], :N].astype(np.float64)
    pr = project_sequences(x, y, D_)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(len(LS[tag]))]
    return dict(pair=name, base=base, prompt=p, seed_set=k, layers=tag, similarity=math.exp(-sum(dl) / len(dl)))

def perm_p(pos, neg, rng, B=20000):
    allv = np.concatenate([pos, neg]); obs = pos.mean() - neg.mean(); c = 0
    for _ in range(B):
        rng.shuffle(allv); c += (allv[:len(pos)].mean() - allv[len(pos):].mean()) >= obs
    return (c + 1) / (B + 1)
def auc(s, neg): return float(((s[:, None] > neg[None]) + 0.5 * (s[:, None] == neg[None])).mean())

if __name__ == "__main__":
    have = [m for m in MODELS if (H / f"feats/pert_{m}.npz").exists()]
    (H / "scores").mkdir(exist_ok=True)
    with Pool(6, maxtasksperchild=200) as pool:
        for n in have:
            f = H / f"scores/{n}.json"
            if f.exists(): continue
            # main victim: all 3 seed sets; alt bases: seed set 0 only (attribution)
            jobs = [(n, "qwen2.5_0.5b", p, k, t) for t in LS for p in range(32) for k in range(3)]
            jobs += [(n, b, p, 0, t) for b in ("qwen2_0.5b", "smollm2_360m") for t in LS for p in range(32)]
            f.write_text(json.dumps(pool.map(job, jobs))); print("scored", n, flush=True)
        # attribution rows for reference models (instruct = positive control) and the 5 old negs
        for n in ["instruct"] + list(NEGS):
            f = H / f"scores/attr_{n}.json"
            if f.exists(): continue
            jobs = [(n, b, p, 0, t) for b in BASES for t in LS for p in range(32) if not BASES[b] == feat(n)]
            f.write_text(json.dumps(pool.map(job, jobs))); print("attr", n, flush=True)
    rng = np.random.default_rng(0); res = {"models": {k: dict(hf=v[0], label=v[1], what=v[2]) for k, v in MODELS.items()},
        "missing": [m for m in MODELS if m not in have]}
    for tag in LS:
        R = {}
        for n in list(NEGS) + ["fine_tune"]:
            fn = V3 / "scores" / (f"pert_{n}.json" if tag == "all4" else f"pert_middle2_{n}.json")
            for r in json.loads(fn.read_text()):
                if r["N"] == N and r["dims"] == D_: R[(n, "qwen2.5_0.5b", r["prompt"], r["seed_set"])] = r["similarity"]
        for fn in H.glob("scores/*.json"):
            for r in json.loads(fn.read_text()):
                if r["layers"] == tag: R[(r["pair"], r["base"], r["prompt"], r["seed_set"])] = r["similarity"]
        V = "qwen2.5_0.5b"; s0 = lambda a, b=V: np.array([R[(a, b, i, 0)] for i in range(32)])
        hard = [m for m in have if MODELS[m][1] == "neg"]
        pan = {"base5": list(NEGS), "ext7": list(NEGS) + hard}
        out = {"panels": {}, "rows": {}, "attribution": {}}
        for pn, negs in pan.items():
            nv = {q: s0(q) for q in negs}; pool_ = np.concatenate(list(nv.values()))
            out["panels"][pn] = dict(negs=negs, neg_means={q: float(v.mean()) for q, v in nv.items()}, max_neg_mean=float(max(v.mean() for v in nv.values())))
            nmax = max(v.mean() for v in nv.values()); npp = np.max(np.stack(list(nv.values())), 0)
            for a in ["fine_tune"] + have:
                if a in negs: continue
                s = s0(a); ss = [np.mean([R[(a, V, i, k)] for i in range(32)]) for k in range(3)]
                out["rows"].setdefault(a, {})[pn] = dict(sim=float(s.mean()), sim3=float(np.mean(ss)), sim3_std=float(np.std(ss, ddof=1)),
                    gap=float(s.mean() - nmax), auc=auc(s, pool_), wins=int((s > npp).sum()), perm_p=float(perm_p(s.copy(), pool_.copy(), rng)))
        # panel-level ROC: all positives vs all negatives (old + hard), prompt-level
        posv = np.concatenate([s0(a) for a in have if MODELS[a][1] == "pos"])
        negv = np.concatenate([s0(q) for q in pan["ext7"]])
        out["pooled_auc_pos_vs_ext7"] = auc(posv, negv) if len(posv) else None
        thr = max(out["panels"]["base5"]["max_neg_mean"], 0)
        out["mean_level_tpr_at_base5_max"] = float(np.mean([s0(a).mean() > thr for a in have if MODELS[a][1] == "pos"])) if len(posv) else None
        for a in ["instruct"] + list(NEGS) + have:
            c = {b: float(np.mean([R[(a, b, i, 0)] for i in range(32)])) for b in BASES if (a, b, 0, 0) in R}
            if a in NEGS and (a, V, 0, 0) in R: c[V] = float(s0(a).mean())
            if c: out["attribution"][a] = dict(sims=c, argmax=max(c, key=c.get))
        res[tag] = out
    (H / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for tag in LS:
        o = res[tag]; print(f"\n## {tag}  base5 max neg={o['panels']['base5']['max_neg_mean']:.3f}  ext7 max neg={o['panels']['ext7']['max_neg_mean']:.3f}")
        print("| key | HF id | label | sim | sim 3set | gap(b5) | AUC(b5) | wins(b5) | p(b5) | gap(e7) | AUC(e7) | p(e7) |")
        for a, v in o["rows"].items():
            b, e = v["base5"], v.get("ext7", {})
            hf = MODELS[a][0] if a in MODELS else "Qwen/Qwen2.5-0.5B-Instruct"; lab = MODELS[a][1] if a in MODELS else "pos"
            es = f"{e['gap']:+.3f} | {e['auc']:.3f} | {e['perm_p']:.1e}" if e else "— | — | —"
            print(f"| {a} | {hf} | {lab} | {b['sim']:.3f} | {b['sim3']:.3f}±{b['sim3_std']:.3f} | {b['gap']:+.3f} | {b['auc']:.3f} | {b['wins']}/32 | {b['perm_p']:.1e} | {es} |")
        print("pooled AUC pos vs ext7:", o["pooled_auc_pos_vs_ext7"], " TPR@base5max:", o["mean_level_tpr_at_base5_max"])
        print("| model | " + " | ".join(BASES) + " | argmax |")
        for a, v in o["attribution"].items():
            print(f"| {a} | " + " | ".join(f"{v['sims'].get(b, float('nan')):.3f}" for b in BASES) + f" | {v['argmax']} |")
