import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/workspace/papers/DiffIP"); sys.path.insert(0, "/workspace/papers/exp_methodA_diffip20")
import numpy as np
from multiprocessing import Pool
from diffip import DiffIPMetric
from run import steps                                   # 与方法A重跑完全相同的层选取/归一化
OUT = "/workspace/papers/exp_rotation_attack"; F = f"{OUT}/feats"
SUS = {"derived": 1, "derived_perm": 1, "derived_rot": 1, "derived_foldonly": 1, "independent": 0}
SUBS = {"calib36": np.arange(36), "test36": np.arange(36, 72), "all72": np.arange(72)}

def cka_std(X, Y):        # 用户 REEF 复现口径（reef_traj.py:54-58）：中心化 + 逐特征标准化
    X = X - X.mean(0); Y = Y - Y.mean(0); X /= X.std(0) + 1e-6; Y /= Y.std(0) + 1e-6
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
def cka_plain(X, Y):      # 教科书线性 CKA（只中心化），正交不变——作对照
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
def cos(A, B, L=(6, 12, 18, 24)):
    a, b = A[:, list(L)], B[:, list(L)]
    return float(((a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)).mean())

def job(args):
    sub, m = args; V = np.load(f"{F}/source.npy").astype(np.float64)[SUBS[sub]]; S = np.load(f"{F}/{m}.npy").astype(np.float64)[SUBS[sub]]
    L = V.shape[1]; l = int(L * 0.75); r = dict(subset=sub, model=m, label=SUS[m])
    r["cos"] = cos(V, S); r["cos_L6_18"] = cos(V, S, (6, 12, 18))
    r["REEF_cka_std"] = cka_std(V[:, l], S[:, l]); r["REEF_cka_plain"] = cka_plain(V[:, l], S[:, l])
    t = time.time(); r["DiffIP20"] = float(DiffIPMetric(n_components=20).compare(steps(S), steps(V)).similarity); r["diffip_sec"] = time.time() - t
    return r

if __name__ == "__main__":
    with Pool(8) as p: rows = p.map(job, [(s, m) for s in SUBS for m in SUS])
    meths = ["cos", "cos_L6_18", "REEF_cka_std", "REEF_cka_plain", "DiffIP20"]
    summ = {}
    for s in SUBS:
        R = [r for r in rows if r["subset"] == s]
        for meth in meths:
            P = [r[meth] for r in R if r["label"]]; N = [r[meth] for r in R if not r["label"]]
            summ[f"{s}/{meth}"] = dict(margin=min(P) - max(N), AUC=float(np.mean([a > b for a in P for b in N])))
    # 健全性：本次重抽的 source 与 10/19 缓存特征是否一致
    old = np.load("/workspace/20261019_DiffIP迁移LLM_线性回拉/feats/source_qwen25_05b.npy"); new = np.load(f"{F}/source.npy")
    sanity = dict(source_vs_cached_maxabs=float(np.abs(old - new).max()), source_absmax=float(np.abs(old).max()))
    D, Pm = np.load(f"{F}/derived.npy"), np.load(f"{F}/derived_perm.npy")
    sanity["perm_hidden_is_column_permutation_of_derived"] = bool(np.allclose(np.sort(D[:, 6], -1), np.sort(Pm[:, 6], -1), atol=1e-3))
    json.dump(dict(rows=rows, summary=summ, sanity=sanity, verify=json.load(open(f"{OUT}/verify.json"))), open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)
    for r in rows:
        if r["subset"] == "all72": print(r["model"], {k: round(r[k], 4) for k in meths})
    for k, v in summ.items(): print(k, {a: round(b, 4) for a, b in v.items()})
    print(sanity, np.mean([r["diffip_sec"] for r in rows]))
