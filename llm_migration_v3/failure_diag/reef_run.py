"""REEF on the literature panel, reusing ../reef_baseline code paths verbatim:
 (a) official setting: same activation collection as reef_baseline/collect.py (TruthfulQA200,
     random.Random(2024), Question+' '+Best Answer, no padding, all decoder layers, last token),
     same scoring as reef_baseline/score_reef.py (center + per-dim std, linear & RBF CKA at
     relative depth 18/31, rel-depth curve over 9 points, 10x20 chunks).
 (b) CKA-on-perturbation: reef_baseline/score_pert_cka.py formula on ../lit_protocol/feats.
Negatives for gap: the 5 base v3 negatives (reef_baseline/acts, ../feats). Usage:
  reef_run.py collect   then   reef_run.py score"""
import sys, csv, random, time, json, math
from pathlib import Path
H0 = Path(__file__).resolve().parent; sys.path.insert(0, str(H0.parent))
from common import *
RB = H0.parent / "reef_baseline"; sys.argv_ = sys.argv
P7 = {"coder_base": "Qwen/Qwen2.5-Coder-0.5B", "coder_inst": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
      "r1_draft": "alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B", "vikhr_sft": "Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct",
      "ita_sft": "ReDiX/Qwen2.5-0.5B-Instruct-ITA", "qwen2_inst": "Qwen/Qwen2-0.5B-Instruct", "qwen3_base": "Qwen/Qwen3-0.6B-Base"}
LAB = {"coder_base": "pos", "coder_inst": "pos", "r1_draft": "pos", "vikhr_sft": "pos", "ita_sft": "pos", "qwen2_inst": "neg", "qwen3_base": "neg"}
ACT = H0 / "acts"; ACT.mkdir(exist_ok=True)
def collect():
    rows = list(csv.DictReader(open(RB / "TruthfulQA.csv"))); random.Random(2024).shuffle(rows)
    TQA = [r["Question"].strip() + " " + r["Best Answer"].strip() for r in rows[:200]]
    for name, mid in P7.items():
        f = ACT / f"{name}.npz"
        if f.exists(): continue
        t0 = time.time(); m, tok = load(mid); E = m.get_input_embeddings(); dec = decoder_layers(m); allL = list(range(len(dec))); hid = {}
        hooks = [dec[i].register_forward_hook((lambda i: lambda _m, _in, o: hid.__setitem__(i, o[0] if isinstance(o, tuple) else o))(i)) for i in allL]
        last = []
        with torch.no_grad():
            for t in TQA:
                e = E(tok(t, return_tensors="pt")["input_ids"])
                m(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long), use_cache=False)
                last.append(torch.stack([hid[i][0, -1] for i in allL]).float())
        for h in hooks: h.remove()
        np.savez(f, tqa200_last=torch.stack(last, 1).numpy(), total=len(allL)); print(name, f"{time.time()-t0:.0f}s", flush=True); del m; gc.collect()
def score():
    sys.argv = ["x"]
    import importlib.util
    def std(X): X = X - X.mean(0); s = X.std(0, ddof=1); s[s == 0] = 1; return X / s
    def cent(K): n = K.shape[0]; Hm = np.eye(n) - 1.0 / n; return Hm @ K @ Hm
    def lin_cka(X, Y): a, b = cent(X @ X.T), cent(Y @ Y.T); return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    def rbf(X): G = X @ X.T; K = np.diag(G) - G + (np.diag(G) - G).T; return np.exp(-0.5 * K / np.median(K[K != 0]))
    def rbf_cka(X, Y): a, b = cent(rbf(X)), cent(rbf(Y)); return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    at = lambda T, r: round(r * (T - 1)); REL = 18 / 31
    V = np.load(RB / "acts/victim.npz"); Tv = int(V["total"])
    X = {l: std(V["tqa200_last"][l].astype(np.float64)) for l in range(Tv)}
    def reef(S):
        Ts = int(S["total"]); Y = {l: std(S["tqa200_last"][l].astype(np.float64)) for l in range(Ts)}; hv, hs = at(Tv, REL), at(Ts, REL)
        return dict(headline_linear=lin_cka(X[hv], Y[hs]), headline_rbf=rbf_cka(X[hv], Y[hs]), layers=[hv, hs],
                    rel_depth_curve=[lin_cka(X[at(Tv, q)], Y[at(Ts, q)]) for q in np.linspace(0, 1, 9)],
                    chunks20=[lin_cka(X[hv][c*20:(c+1)*20], Y[hs][c*20:(c+1)*20]) for c in range(10)])
    def pstd(X): return std(X)
    Vp = np.load(H0.parent / "feats/pert_victim.npz")["pert"]
    def pert(path):
        S = np.load(path)["pert"]; r = {}
        for N in (20, 128):
            a = np.array([[[lin_cka(std(Vp[p, k, l, :N].astype(np.float64)), std(S[p, k, l, :N].astype(np.float64))) for l in range(4)] for k in range(3)] for p in range(32)])
            r[N] = dict(mean=float(a.mean()), per_prompt=a.mean((1, 2)).tolist()); 
        return r
    out = {"neg5": {}, "panel": {}}
    for k in NEGS:
        out["neg5"][k] = dict(official=reef(np.load(RB / f"acts/{k}.npz")), pert=pert(H0.parent / f"feats/pert_{k}.npz")); print(k, flush=True)
    for k in P7:
        out["panel"][k] = dict(label=LAB[k], official=reef(np.load(ACT / f"{k}.npz")), pert=pert(H0.parent / f"lit_protocol/feats/pert_{k}.npz")); print(k, flush=True)
    mo = max(v["official"]["headline_linear"] for v in out["neg5"].values()); mp = max(v["pert"][128]["mean"] for v in out["neg5"].values())
    negc = np.concatenate([v["official"]["chunks20"] for v in out["neg5"].values()]); negp = np.concatenate([v["pert"][128]["per_prompt"] for v in out["neg5"].values()])
    for v in out["panel"].values():
        v["official_gap"] = v["official"]["headline_linear"] - mo; v["pert_gap"] = v["pert"][128]["mean"] - mp
        c = np.array(v["official"]["chunks20"]); v["official_auc"] = float(((c[:, None] > negc) + .5 * (c[:, None] == negc)).mean())
        c = np.array(v["pert"][128]["per_prompt"]); v["pert_auc"] = float(((c[:, None] > negp) + .5 * (c[:, None] == negp)).mean())
    out["neg5_max"] = dict(official=mo, pert=mp)
    (H0 / "results_reef.json").write_text(json.dumps(out, indent=1))
    for k, v in out["panel"].items(): print(k, v["label"], *(round(x, 3) for x in [v["official"]["headline_linear"], v["official_gap"], v["official_auc"], v["pert"][128]["mean"], v["pert_gap"], v["pert_auc"]]))
{"collect": collect, "score": score}[sys.argv[1]]()
