"""步骤5：打分。受害者=教师 Qwen2.5-0.5B-Instruct。正样本=蒸馏学生；负样本=未蒸馏基座、对照学生、无关模型。
DiffIP20 用与方法A完全相同的 steps()（7 个等距层、去均值、缩放）；CKA 取相对深度 0.75 的层（与旋转实验同口径），
另报 7 个等距层逐对 CKA 的平均；余弦只在维度相同时可算（否则 N/A）。"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/workspace/papers/DiffIP"); sys.path.insert(0, "/workspace/papers/exp_methodA_diffip20")
import numpy as np
from multiprocessing import Pool
from diffip import DiffIPMetric
from run import steps
F = "/workspace/papers/exp_distill_v2/feats"
SUS = {"student_distill": 1, "student_control": 0, "smollm135_base": 0, "smollm360": 0, "gpt2": 0, "pythia160": 0, "qwen2_05b": 0}
SUBS = {"calib36": np.arange(36), "test36": np.arange(36, 72), "all72": np.arange(72)}
def _c(X, std):
    X = X - X.mean(0)
    return X / (X.std(0) + 1e-6) if std else X
def cka(X, Y, std):
    X, Y = _c(X, std), _c(Y, std)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
def lay(H, f): return H[:, int(round((H.shape[1] - 1) * f))]
def job(a):
    sub, m = a; V = np.load(f"{F}/teacher.npy").astype(np.float64)[SUBS[sub]]; S = np.load(f"{F}/{m}.npy").astype(np.float64)[SUBS[sub]]
    r = dict(subset=sub, model=m, label=SUS[m], dim=S.shape[2], layers=S.shape[1] - 1)
    if S.shape[2] == V.shape[2]:
        r["cos"] = float(np.mean([(a * b).sum(-1).mean() / 1 for a, b in [(lay(V, f) / np.linalg.norm(lay(V, f), axis=-1, keepdims=True), lay(S, f) / np.linalg.norm(lay(S, f), axis=-1, keepdims=True)) for f in (.25, .5, .75, 1.0)]]))
    else: r["cos"] = None
    r["CKA_std"] = cka(lay(V, .75), lay(S, .75), True); r["CKA_plain"] = cka(lay(V, .75), lay(S, .75), False)
    fs = np.linspace(0, 1, 7)
    r["CKA_plain_7avg"] = float(np.mean([cka(lay(V, f), lay(S, f), False) for f in fs]))
    t = time.time(); r["DiffIP20"] = float(DiffIPMetric(n_components=20).compare(steps(S), steps(V)).similarity); r["diffip_sec"] = time.time() - t
    return r
if __name__ == "__main__":
    with Pool(7) as p: rows = p.map(job, [(s, m) for s in SUBS for m in SUS])
    meths = ["cos", "CKA_std", "CKA_plain", "CKA_plain_7avg", "DiffIP20"]; summ = {}
    for s in SUBS:
        R = [r for r in rows if r["subset"] == s]
        for me in meths:
            if any(r[me] is None for r in R if r["label"]): summ[f"{s}/{me}"] = "N/A（学生维度576≠教师896）"; continue
            P = [r[me] for r in R if r["label"]]; N = [r[me] for r in R if not r["label"] and r[me] is not None]
            Nh = [r[me] for r in R if r["model"] in ("student_control", "smollm135_base")]
            summ[f"{s}/{me}"] = dict(margin=min(P) - max(N), margin_vs_same_base=min(P) - max(Nh), AUC=float(np.mean([a > b for a in P for b in N])),
                                    top_negative=max((r for r in R if not r["label"] and r[me] is not None), key=lambda r: r[me])["model"])
    json.dump(dict(rows=rows, summary=summ), open("results.json", "w"), indent=1, ensure_ascii=False)
    for r in rows:
        if r["subset"] == "all72": print(r["model"], {k: (None if r[k] is None else round(r[k], 4)) for k in meths})
    for k, v in summ.items(): print(k, v)
