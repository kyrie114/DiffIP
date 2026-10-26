"""Build tables.md + comparison.json from results.json, results_pert_cka.json and v3 scores (N=128, dims=8)."""
import json, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; V3 = H.parent
R = json.loads((H/"results.json").read_text()); P = json.loads((H/"results_pert_cka.json").read_text())
DER = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune"]
NEG = ["neg_smollm2_135m", "neg_smollm2_360m", "neg_gpt2", "neg_pythia_160m", "neg_qwen2_0.5b"]
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())
def diff(name, N=128, d=8):
    f = "prune_dpoff" if name == "prune" else name
    rows = [r for r in json.loads((V3/"scores"/f"pert_{f}.json").read_text()) if r["N"] == N and r["dims"] == d]
    pp = np.zeros((32, 3))
    for r in rows: pp[r["prompt"], r["seed_set"]] = r["similarity"]
    return pp
D = {n: diff(n) for n in DER + NEG}
cols = {
 "REEF 官方(TQA200,末token,相对深度18/31,linear)": lambda n: R[n]["tqa200_last"]["headline_linear"],
 "REEF 同设置 RBF": lambda n: R[n]["tqa200_last"]["headline_rbf"],
 "REEF 末层 linear": lambda n: R[n]["tqa200_last"]["last_layer"],
 "REEF 9个相对深度均值": lambda n: R[n]["tqa200_last"]["matrix_diag_mean_rel"],
 "REEF TQA200 mean-pool": lambda n: R[n]["tqa200_mean"]["headline_linear"],
 "REEF 32prompt 末token": lambda n: R[n]["prompts32_last"]["headline_linear"],
 "REEF 32prompt mean-pool": lambda n: R[n]["prompts32_mean"]["headline_linear"],
 "CKA@v3扰动云 N=128(4层)": lambda n: P[n]["128"]["mean"],
 "CKA@v3扰动云 N=20": lambda n: P[n]["20"]["mean"],
 "DiffIP v3 sim N=128 d=8": lambda n: float(D[n].mean()),
}
L = ["# REEF 基线 vs DiffIP v3（自动生成：make_tables.py）\n",
     "| 方法/设置 | " + " | ".join(DER + NEG) + " | 衍生最小 − 负例最大 | pair级AUC(5 vs 5) |", "|" + "---|" * (len(DER + NEG) + 3)]
comp = {}
for c, f in cols.items():
    v = {n: f(n) for n in DER + NEG}; comp[c] = v
    gap = min(v[n] for n in DER) - max(v[n] for n in NEG)
    L.append(f"| {c} | " + " | ".join(f"{v[n]:.3f}" for n in DER + NEG) + f" | {gap:+.3f} | {auc([v[n] for n in DER],[v[n] for n in NEG]):.3f} |")
# finer-grained AUC
a_reef = auc(sum([R[n]["tqa200_last"]["chunks20"] for n in DER], []), sum([R[n]["tqa200_last"]["chunks20"] for n in NEG], []))
a_pcka = auc(sum([P[n]["128"]["per_prompt"] for n in DER], []), sum([P[n]["128"]["per_prompt"] for n in NEG], []))
a_diff = auc(np.concatenate([D[n].mean(1) for n in DER]), np.concatenate([D[n].mean(1) for n in NEG]))
a_diff_ft = auc(D["fine_tune"].mean(1), np.concatenate([D[n].mean(1) for n in NEG]))
L += ["", "**细粒度 AUC（衍生 vs 负例，所有样本混合）**", "",
      "| 方法 | 样本单位 | AUC |", "|---|---|---|",
      f"| REEF 官方设置 | 200 条 TQA 切成 10 块×20 条，每块一个 CKA（5×10 vs 5×10） | {a_reef:.3f} |",
      f"| CKA@v3扰动云 N=128 | 每个 prompt 一个值（3 seed set×4 层平均；5×32 vs 5×32） | {a_pcka:.3f} |",
      f"| DiffIP v3 N=128 d=8 | 每个 prompt 一个值（3 seed set 平均；5×32 vs 5×32） | {a_diff:.3f} |",
      f"| DiffIP v3（仅 fine_tune vs 负例） | 32 vs 160 | {a_diff_ft:.3f} |"]
L += ["", "**REEF 官方设置（TQA200 末 token, linear）沿相对深度 0,1/8,…,1 的 CKA**", "",
      "| pair | " + " | ".join(f"{q:.3f}" for q in np.linspace(0,1,9)) + " |", "|" + "---|"*10]
for n in DER + NEG: L.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in R[n]["tqa200_last"]["rel_depth_curve"]) + " |")
L += ["", "**CKA@v3扰动云 N=128 逐层（v3 的 4 个均匀层）**", "", "| pair | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|"]
for n in DER + NEG: L.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in P[n]["128"]["per_layer"]) + " |")
L += ["", "**REEF 全层 CKA 矩阵最大值（victim 任意层 × suspect 任意层）**", "", "| pair | TQA末token | TQA mean | 层数 |", "|---|---|---|---|"]
for n in DER + NEG: L.append(f"| {n} | {R[n]['tqa200_last']['matrix_max']:.3f} | {R[n]['tqa200_mean']['matrix_max']:.3f} | {R[n]['layers_suspect']} |")
(H/"tables.md").write_text("\n".join(L) + "\n")
(H/"comparison.json").write_text(json.dumps(dict(columns=comp, auc_fine=dict(reef=a_reef, pert_cka=a_pcka, diffip=a_diff, diffip_ft=a_diff_ft)), indent=1))
print("\n".join(L))
