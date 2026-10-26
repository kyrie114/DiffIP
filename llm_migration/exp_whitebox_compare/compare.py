"""白盒指纹对比：REEF(线性CKA) vs 直接余弦 vs DiffIP-20，同一批缓存特征、同一批提示子集。"""
import sys, json, time, collections
import numpy as np
sys.path.insert(0, "/workspace/papers/exp_methodA_diffip20")
import run as A                       # 复用模型清单与子集构造（同 seed）
OUT = "/workspace/papers/exp_whitebox_compare"

def lin_cka(X, Y):                    # 原样复用 /workspace/reef_traj/reef_traj.py:54-58（用户的 REEF 复现口径：中心化+逐特征标准化）
    X = X - X.mean(0); Y = Y - Y.mean(0)
    X /= X.std(0) + 1e-6; Y /= Y.std(0) + 1e-6
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))

def subsets():
    rng = np.random.default_rng(0)
    s1 = {"calib36": np.arange(36), "test36": np.arange(36, 72)}
    for r in range(8): s1[f"rand36_{r}"] = np.sort(rng.choice(72, 36, replace=False))
    s2 = {f"rand36_{r}": np.sort(rng.choice(180, 36, replace=False)) for r in range(5)}; s2["all180"] = np.arange(180)
    return {"S1": s1, "S2": s2}

def scores(HV, HS):
    LV, LS = HV.shape[1], HS.shape[1]; out = {}; tm = {}
    t = time.perf_counter(); out["REEF"] = float(lin_cka(HV[:, int(LV*0.75)].astype(np.float64), HS[:, int(LS*0.75)].astype(np.float64))); tm["REEF"] = time.perf_counter()-t
    t = time.perf_counter()
    out["REEF_avg4"] = float(np.mean([lin_cka(HV[:, int(LV*f)].astype(np.float64), HS[:, int(LS*f)].astype(np.float64)) for f in (0.25, 0.5, 0.75, 1.0 - 1e-9)]))
    tm["REEF_avg4"] = time.perf_counter()-t
    if HV.shape[2] == HS.shape[2]:
        t = time.perf_counter(); L = [6, 12, 18, 24]
        a, b = HV[:, L].astype(np.float64), HS[:, L].astype(np.float64)
        out["cos"] = float(((a*b).sum(-1)/np.linalg.norm(a, axis=-1)/np.linalg.norm(b, axis=-1)).mean()); tm["cos"] = time.perf_counter()-t
    return out, tm

raw = json.load(open("/workspace/papers/exp_methodA_diffip20/raw_scores.json"))
dif = {(r["setting"], r["subset"], r["model"]): (r["score"], r["sec"]) for r in raw if r["method"] == "pca20"}
rows = []; times = collections.defaultdict(list)
for st, cfg in (("S1", A.S1), ("S2", A.S2)):
    HVall = np.load(cfg["victim"])
    for sn, sub in subsets()[st].items():
        for m, (p, y) in cfg["models"].items():
            o, tm = scores(HVall[sub], np.load(p)[sub])
            o["DiffIP20"], tm["DiffIP20"] = dif[(st, sn, m)]
            for k, v in tm.items(): times[k].append(v)
            rows.append(dict(setting=st, subset=sn, model=m, label=y, **o))

def auc(P, N): return float(np.mean([1. if a > b else .5 if a == b else 0. for a in P for b in N]))
table = []
for st, variants in (("S1", ["all"]), ("S2", ["no_student", "with_student"])):
    for v in variants:
        for meth in ["cos", "REEF", "REEF_avg4", "DiffIP20"]:
            recs = []
            for sn in subsets()[st]:
                R = [r for r in rows if r["setting"] == st and r["subset"] == sn and not (v == "no_student" and r["model"] == "student_gpt2")]
                P = [r[meth] for r in R if r["label"] == 1 and meth in r]; N = [r[meth] for r in R if r["label"] == 0 and meth in r]
                if not P or not N: continue
                rng_ = max(P + N) - min(P + N)
                recs.append(dict(AUC=auc(P, N), margin=min(P)-max(N), rel=(min(P)-max(N))/rng_ if rng_ > 0 else None,
                                 mean_pos=float(np.mean(P)), max_neg=max(N)))
            if not recs: table.append(dict(setting=st, variant=v, method=meth, note="不适用（负模型维度不同）")); continue
            g = lambda k: [r[k] for r in recs]
            table.append(dict(setting=st, variant=v, method=meth, n_subsets=len(recs), AUC_mean=float(np.mean(g("AUC"))),
                              margin_mean=float(np.mean(g("margin"))), margin_min=float(min(g("margin"))), margin_max=float(max(g("margin"))),
                              rel_margin_mean=float(np.mean(g("rel"))), mean_pos=float(np.mean(g("mean_pos"))), max_neg=float(np.mean(g("max_neg"))),
                              separable=f"{sum(m > 0 for m in g('margin'))}/{len(recs)}"))
cost = {k: dict(mean_sec=float(np.mean(v)), median_sec=float(np.median(v)), n=len(v)) for k, v in times.items()}
json.dump(dict(table=table, cost_per_comparison=cost, per_subset_scores=rows), open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)
for t in table: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
print({k: {a: round(b, 5) if isinstance(b, float) else b for a, b in v.items()} for k, v in cost.items()})
