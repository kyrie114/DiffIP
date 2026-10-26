import json, numpy as np, collections
OUT = "/workspace/papers/exp_methodA_diffip20"
rows = json.load(open(f"{OUT}/raw_scores.json"))
def auc(p, n): return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in p for b in n]))
G = collections.defaultdict(dict)
for r in rows: G[(r["setting"], r["method"], r["subset"])][r["model"]] = (r["score"], r["label"])
summary = collections.defaultdict(list); per = []
for (st, meth, sub), d in sorted(G.items()):
    for variant in (["all"] if st == "S1" else ["no_student", "with_student"]):
        dd = {m: v for m, v in d.items() if not (variant == "no_student" and m == "student_gpt2")}
        P = [s for s, y in dd.values() if y == 1]; N = [s for s, y in dd.values() if y == 0]
        if not P or not N: continue
        rec = dict(setting=st, variant=variant, method=meth, subset=sub, AUC=auc(P, N), margin=min(P) - max(N),
                   scores={m: round(s, 6) for m, (s, y) in dd.items()})
        per.append(rec); summary[(st, variant, meth)].append(rec)
table = []
for (st, v, meth), recs in sorted(summary.items()):
    A = np.array([r["AUC"] for r in recs]); M = np.array([r["margin"] for r in recs])
    table.append(dict(setting=st, variant=v, method=meth, n_subsets=len(recs), AUC_mean=float(A.mean()), AUC_min=float(A.min()),
                      margin_mean=float(M.mean()), margin_min=float(M.min()), margin_max=float(M.max()),
                      frac_subsets_separable=float((M > 0).mean())))
json.dump(dict(table=table, per_subset=per), open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)
for t in table: print(t)
