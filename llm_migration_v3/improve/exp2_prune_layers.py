"""Prune specialization: layer choice (relative depth) + last-token vs mean-pool.
Harder negative = same-family Qwen2-0.5B (already in v3). Also report vs all 5 v3 negs
where DiffIP ALS on pert clouds can handle cross-dim (reuse existing score json) for reference.

Uses reef_baseline/acts (last & mean) and v3 feats dp_all (mean-pool, N=20, all layers).
Metrics: DiffIP-style residual sim is heavy; here we report:
  (1) direct cosine (same-dim only)
  (2) linear CKA on pert cloud at selected layers (from feats/pert, 4 even layers + dp_all)
  (3) DiffIP ALS sim from existing scores / recomputed on selected layer subsets of pert
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent))
from common import make_permutation, make_anisotropic_scale, NEGS  # noqa
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment

OUT = Path(__file__).resolve().parent
ACTS = H / "reef_baseline" / "acts"
FEATS = H / "feats"


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    c = 0.0
    for p in pos:
        c += (neg < p).sum() + 0.5 * (neg == p).sum()
    return float(c / (len(pos) * len(neg)))


def cos_vec(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def lin_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    n = X.shape[0]
    Hm = np.eye(n) - 1.0 / n
    a = Hm @ (X @ X.T) @ Hm
    b = Hm @ (Y @ Y.T) @ Hm
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def rel_idx(n, frac):
    return int(round(frac * (n - 1)))


def sweep_acts_cosine():
    """Per relative-depth cosine: prune & fine_tune & neg_qwen2 vs victim; last vs mean."""
    Vlast = np.load(ACTS / "victim.npz")["prompts32_last"].astype(np.float64)
    Vmean = np.load(ACTS / "victim.npz")["prompts32_mean"].astype(np.float64)
    models = {
        "fine_tune": ("instruct", None),
        "prune": ("instruct_prune", None),
        "neg_qwen2_0.5b": ("neg_qwen2_0.5b", None),
    }
    fracs = [i / 10 for i in range(11)]
    rows = []
    for pool, V in [("last", Vlast), ("mean", Vmean)]:
        Lv = V.shape[0]
        loaded = {}
        for name, (fname, _) in models.items():
            key = f"prompts32_{pool}"
            loaded[name] = np.load(ACTS / f"{fname}.npz")[key].astype(np.float64)
        for f in fracs:
            lv = rel_idx(Lv, f)
            scores = {}
            for name, S in loaded.items():
                ls = rel_idx(S.shape[0], f)
                scores[name] = [cos_vec(S[ls, p], V[lv, p]) for p in range(32)]
            gap_ft = float(np.mean(scores["fine_tune"]) - np.mean(scores["neg_qwen2_0.5b"]))
            gap_pr = float(np.mean(scores["prune"]) - np.mean(scores["neg_qwen2_0.5b"]))
            rows.append(dict(
                pool=pool, frac=f, victim_layer=lv,
                fine_tune=float(np.mean(scores["fine_tune"])),
                prune=float(np.mean(scores["prune"])),
                neg_qwen2=float(np.mean(scores["neg_qwen2_0.5b"])),
                gap_fine_tune=gap_ft, gap_prune=gap_pr,
                auc_prune=auc(scores["prune"], scores["neg_qwen2_0.5b"]),
                auc_fine_tune=auc(scores["fine_tune"], scores["neg_qwen2_0.5b"]),
                wins_prune=int(sum(scores["prune"][p] > scores["neg_qwen2_0.5b"][p] for p in range(32))),
            ))
            print(f"[acts {pool}] frac={f:.1f} L={lv} prune={rows[-1]['prune']:.3f} "
                  f"neg={rows[-1]['neg_qwen2']:.3f} gap={gap_pr:.3f} auc={rows[-1]['auc_prune']:.3f} "
                  f"wins={rows[-1]['wins_prune']}/32", flush=True)
    return rows


def sweep_dp_all_diffip(dims=8, N=20):
    """DiffIP ALS on dp_all all-layer features (seed set 0, N<=20), per relative-depth single layer
    and selected middle-layer bundles. Same-dim + cross-dim via PCA project_sequences."""
    def ld(n):
        d = np.load(FEATS / f"pert_{n}.npz")
        return d["dp_all"].astype(np.float64), int(d["total"])  # [P,L,20,D]

    V, Lv = ld("victim")
    ins, _ = ld("instruct")
    pr, Lp = ld("instruct_prune")
    D = ins.shape[-1]
    perm = make_permutation(D, seed=1).argmax(0).numpy()
    sdiag = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float64)
    sus = {
        "fine_tune": ins,
        "rep_perm": ins[..., perm],
        "scale": ins * sdiag,
        "prune": pr,
    }
    for k in NEGS:
        sus[k], _ = ld(k)

    def als_sim(S_layer, V_layer, d):
        # S_layer, V_layer: [P, N, D] — score each prompt, return list
        out = []
        for p in range(S_layer.shape[0]):
            # shapes for project_sequences: [T=1, N, D]
            sus = S_layer[p:p+1, :N]
            vic = V_layer[p:p+1, :N]
            # if D differs, project_sequences still works (independent PCA)
            try:
                prj = project_sequences(sus, vic, d)
                dist = fit_step_alignment(prj.suspect[0], prj.victim[0]).distance
                used = prj.n_components
                if used < d:
                    # rank-deficient; still ok
                    pass
                out.append(math.exp(-dist))
            except Exception as e:
                out.append(float("nan"))
        return out

    fracs = [i / 10 for i in range(11)]
    rows = []
    # single relative-depth layer
    for f in fracs:
        lv = rel_idx(Lv, f)
        scores = {}
        for name, S in sus.items():
            ls = rel_idx(S.shape[1], f)
            scores[name] = als_sim(S[:, ls], V[:, lv], dims)
        derived = ["fine_tune", "rep_perm", "scale", "prune"]
        negs = list(NEGS)
        pos = np.concatenate([scores[n] for n in derived])
        neg = np.concatenate([scores[n] for n in negs])
        # filter nan
        def clean(a):
            a = np.asarray(a, float); return a[~np.isnan(a)]
        gap = float(min(np.nanmean(scores[n]) for n in derived) - max(np.nanmean(scores[n]) for n in negs))
        rows.append(dict(
            kind="single_frac", frac=f, victim_layer=lv, dims=dims, N=N,
            means={n: float(np.nanmean(scores[n])) for n in list(scores)},
            gap=gap,
            auc_pooled=auc(clean(pos), clean(neg)),
            auc_prune_vs_all=auc(clean(scores["prune"]), clean(neg)),
            auc_prune_vs_qwen2=auc(clean(scores["prune"]), clean(scores["neg_qwen2_0.5b"])),
            wins_prune_vs_qwen2=int(sum(
                scores["prune"][p] > scores["neg_qwen2_0.5b"][p]
                for p in range(32) if not (math.isnan(scores["prune"][p]) or math.isnan(scores["neg_qwen2_0.5b"][p]))
            )),
        ))
        print(f"[dp DiffIP d={dims}] frac={f:.1f} prune={rows[-1]['means']['prune']:.3f} "
              f"maxneg={max(rows[-1]['means'][k] for k in negs):.3f} gap={gap:.3f} "
              f"auc_pr={rows[-1]['auc_prune_vs_all']:.3f}", flush=True)

    # middle-bundle: relative depths 0.3,0.5,0.7 (3 layers) and 0.25..0.75
    bundles = {
        "middle3": [0.3, 0.5, 0.7],
        "mid5": [0.25, 0.375, 0.5, 0.625, 0.75],
        "late3": [0.7, 0.85, 1.0],
        "early3": [0.0, 0.15, 0.3],
        "even4": [0.0, 1/3, 2/3, 1.0],  # like v3 L4
    }
    for bname, frs in bundles.items():
        scores = {n: [] for n in sus}
        for name, S in sus.items():
            per_p = []
            for p in range(32):
                dists = []
                for f in frs:
                    ls = rel_idx(S.shape[1], f); lv = rel_idx(Lv, f)
                    sus1 = S[p:p+1, ls, :N]; vic1 = V[p:p+1, lv, :N]
                    try:
                        prj = project_sequences(sus1, vic1, dims)
                        dists.append(fit_step_alignment(prj.suspect[0], prj.victim[0]).distance)
                    except Exception:
                        pass
                if dists:
                    per_p.append(math.exp(-sum(dists) / len(dists)))
                else:
                    per_p.append(float("nan"))
            scores[name] = per_p
        derived = ["fine_tune", "rep_perm", "scale", "prune"]
        negs = list(NEGS)
        pos = np.concatenate([scores[n] for n in derived])
        neg = np.concatenate([scores[n] for n in negs])
        def clean(a):
            a = np.asarray(a, float); return a[~np.isnan(a)]
        gap = float(min(np.nanmean(scores[n]) for n in derived) - max(np.nanmean(scores[n]) for n in negs))
        rows.append(dict(
            kind="bundle", bundle=bname, fracs=frs, dims=dims, N=N,
            means={n: float(np.nanmean(scores[n])) for n in scores},
            gap=gap,
            auc_pooled=auc(clean(pos), clean(neg)),
            auc_prune_vs_all=auc(clean(scores["prune"]), clean(neg)),
            auc_prune_vs_qwen2=auc(clean(scores["prune"]), clean(scores["neg_qwen2_0.5b"])),
            wins_prune_vs_qwen2=int(sum(
                scores["prune"][p] > scores["neg_qwen2_0.5b"][p]
                for p in range(32) if not (math.isnan(scores["prune"][p]) or math.isnan(scores["neg_qwen2_0.5b"][p]))
            )),
        ))
        print(f"[dp DiffIP bundle {bname}] prune={rows[-1]['means']['prune']:.3f} gap={gap:.3f} "
              f"auc_pr={rows[-1]['auc_prune_vs_all']:.3f}", flush=True)
    return rows


def sweep_pert4_cka_and_diffip(N=128, dims=8):
    """On the existing 4 even layers, try subsets (drop layer 0 / keep middle2 / last only)."""
    def ld(n):
        return np.load(FEATS / f"pert_{n}.npz")["pert"][:, 0, :, :N].astype(np.float64)  # [P,4,N,D] seed0

    V = ld("victim")
    ins = ld("instruct")
    D = ins.shape[-1]
    perm = make_permutation(D, seed=1).argmax(0).numpy()
    sdiag = np.diag(make_anisotropic_scale(D, seed=0).numpy()).astype(np.float64)
    sus = {
        "fine_tune": ins,
        "rep_perm": ins[..., perm],
        "scale": ins * sdiag,
        "prune": ld("instruct_prune"),
        **{k: ld(k) for k in NEGS},
    }
    layer_sets = {
        "all4": [0, 1, 2, 3],
        "drop_embed": [1, 2, 3],
        "middle2": [1, 2],
        "late2": [2, 3],
        "last1": [3],
        "early2": [0, 1],
    }
    rows = []
    for lname, layers in layer_sets.items():
        # DiffIP
        diff = {}
        cka = {}
        for name, S in sus.items():
            d_scores, c_scores = [], []
            for p in range(32):
                dists = []
                ckas = []
                for l in layers:
                    sus1 = S[p:p+1, l]; vic1 = V[p:p+1, l]
                    prj = project_sequences(sus1, vic1, dims)
                    dists.append(fit_step_alignment(prj.suspect[0], prj.victim[0]).distance)
                    # CKA on raw N×D (centered+std like REEF)
                    X = S[p, l].copy(); Y = V[p, l].copy()
                    X = (X - X.mean(0)); Y = (Y - Y.mean(0))
                    xs = X.std(0, ddof=1); ys = Y.std(0, ddof=1)
                    xs[xs == 0] = 1; ys[ys == 0] = 1
                    ckas.append(lin_cka(X / xs, Y / ys))
                d_scores.append(math.exp(-sum(dists) / len(dists)))
                c_scores.append(float(np.mean(ckas)))
            diff[name] = d_scores
            cka[name] = c_scores
        derived = ["fine_tune", "rep_perm", "scale", "prune"]
        negs = list(NEGS)
        for metric, sc in [("diffip", diff), ("cka", cka)]:
            pos = np.concatenate([sc[n] for n in derived])
            neg = np.concatenate([sc[n] for n in negs])
            gap = float(min(np.mean(sc[n]) for n in derived) - max(np.mean(sc[n]) for n in negs))
            rows.append(dict(
                kind="pert4_subset", layers=lname, layer_idx=layers, metric=metric, N=N, dims=dims,
                means={n: float(np.mean(sc[n])) for n in sc},
                gap=gap,
                auc_pooled=auc(pos, neg),
                auc_prune_vs_all=auc(sc["prune"], neg),
                auc_prune_vs_qwen2=auc(sc["prune"], sc["neg_qwen2_0.5b"]),
                wins_prune=int(sum(sc["prune"][p] > max(sc[k][p] for k in negs) for p in range(32))),
            ))
            print(f"[pert4 {lname} {metric}] prune={rows[-1]['means']['prune']:.3f} gap={gap:.3f} "
                  f"auc={rows[-1]['auc_pooled']:.3f} wins_pr={rows[-1]['wins_prune']}/32", flush=True)
    return rows


if __name__ == "__main__":
    out = {
        "acts_cosine_depth_sweep": sweep_acts_cosine(),
        "dp_all_diffip_depth": sweep_dp_all_diffip(dims=8, N=20),
        "pert4_subsets": sweep_pert4_cka_and_diffip(N=128, dims=8),
    }
    (OUT / "results_exp2.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "results_exp2.json", flush=True)
