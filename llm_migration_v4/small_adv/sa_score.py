"""打分与汇总：本方法（扰动 128 次、主成分 8 维、逐层对齐，sim = exp(−平均残差)）、
REEF（TruthfulQA 200 条末 token、相对深度 18/31 层、线性中心化核对齐，同 v3 score_reef）、
直接权重余弦（meta.json）。另做固定阈值留出检验与逐层残差图。输出 results.json、figs/。"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import itertools, json, math, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
SA = Path(__file__).resolve().parent; V3 = SA.parent.parent / "llm_migration_v3"
sys.path.insert(0, str(SA.parent.parent)); sys.path.insert(0, str(SA.parent.parent / "llm_migration_v2"))
from diffip.preprocessing import project_sequences  # noqa
from diffip.reversion import fit_step_alignment  # noqa
from transforms import make_permutation, make_anisotropic_scale  # noqa

NEGS = ["neg_smollm2_135m", "neg_smollm2_360m", "neg_gpt2", "neg_pythia_160m", "neg_qwen2_0.5b"]
CORE = ["instruct", "instruct_wperm", "rep_perm", "scale", "instruct_prune"]   # v3 已有正例
NEW = ["drop2", "drop6", "drop8", "drop12", "width75", "width50", "width75h", "width50h", "vocab_ext", "vocab_swap"]
NEW = [n for n in NEW if (SA / "feats" / f"pert_{n}.npz").exists()]
ALL = CORE + NEW + NEGS
N, D = 128, 8


def feats(name):
    if name in ("rep_perm", "scale"):
        b = feats("instruct"); Dm = b.shape[-1]
        if name == "rep_perm": return b[..., make_permutation(Dm, seed=1).argmax(0).numpy()]
        return b * np.diag(make_anisotropic_scale(Dm, seed=0).numpy()).astype(np.float32)
    f = SA / "feats" / f"pert_{name}.npz"
    if not f.exists(): f = V3 / "feats" / f"pert_{name}.npz"
    return np.load(f)["pert"]


def job(a):
    s, v = a   # 一条提示、一个种子组的 [4, N, D] 切片，由父进程传入，工作进程不缓存整份特征（省内存）
    pr = project_sequences(s.astype(np.float64), v.astype(np.float64), D)
    return [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(pr.suspect.shape[0])]


def distances(name, pool, Vf):
    """逐模型算，结果存 dist/{name}.npy，中断后跳过已完成的模型。"""
    f = SA / "dist" / f"{name}.npy"; f.parent.mkdir(exist_ok=True)
    if f.exists(): return np.load(f)
    S = feats(name)
    d = np.array(pool.map(job, [(S[p, k, :, :N], Vf[p, k, :, :N]) for p in range(32) for k in range(3)])).reshape(32, 3, -1)
    np.save(f, d); print("dist", name, flush=True); del S
    return d


def std(X):
    X = X - X.mean(0); s = X.std(0, ddof=1); s[s == 0] = 1; return X / s
def lin_cka(X, Y):
    n = X.shape[0]; Hm = np.eye(n) - 1.0 / n
    a, b = Hm @ (X @ X.T) @ Hm, Hm @ (Y @ Y.T) @ Hm
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
def acts(name):
    n2 = {"instruct": "instruct", "instruct_wperm": "instruct_wperm", "instruct_prune": "instruct_prune"}.get(name, name)
    f = SA / "acts" / f"{n2}.npz"
    if not f.exists(): f = V3 / "reef_baseline" / "acts" / f"{n2}.npz"
    return np.load(f)["tqa200_last"].astype(np.float64)
REL = 18 / 31
def at(t): return round(REL * (t - 1))


def reef(name):
    V = acts("victim"); hv = at(V.shape[0])
    if name in ("rep_perm", "scale"):
        S = acts("instruct"); Dm = S.shape[-1]
        S = S[..., make_permutation(Dm, seed=1).argmax(0).numpy()] if name == "rep_perm" else S * np.diag(make_anisotropic_scale(Dm, seed=0).numpy())
    else:
        S = acts(name)
    hs = at(S.shape[0]); X, Y = std(V[hv]), std(S[hs])
    return dict(cka=lin_cka(X, Y), layers=[hv, hs],
                chunks20=[lin_cka(std(V[hv][c*20:(c+1)*20]), std(S[hs][c*20:(c+1)*20])) for c in range(10)])


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


if __name__ == "__main__":
    Vf = feats("victim")
    with Pool(6) as pool:
        dist = {n: distances(n, pool, Vf) for n in ALL}
    sim = {n: np.exp(-dist[n].mean(-1)) for n in ALL}           # [32, 3]
    meta = json.loads((SA / "meta.json").read_text())
    R = {"ours": {}, "reef": {}, "weight_cos": {}, "layer_residual": {}, "meta": meta}
    negsim = np.stack([sim[n] for n in NEGS])                    # [5,32,3]
    for n in ALL:
        s = sim[n]
        R["ours"][n] = dict(mean=float(s.mean()), seed_means=s.mean(0).tolist(),
                            per_prompt_auc_vs_negs=None if n in NEGS else auc(s.ravel(), negsim.ravel()),
                            per_prompt_wins_over_all_negs=None if n in NEGS else int((s > negsim.max(0)).sum()))
        R["reef"][n] = reef(n)
        R["layer_residual"][n] = dict(mean=dist[n].mean((0, 1)).tolist(), std=dist[n].std((0, 1)).tolist())
        R["weight_cos"][n] = meta.get(n, {}).get("weight_cos")
    negchunks = np.concatenate([R["reef"][n]["chunks20"] for n in NEGS])
    for n in ALL:
        if n not in NEGS: R["reef"][n]["chunk_auc_vs_negs"] = auc(R["reef"][n]["chunks20"], negchunks)
    R["max_neg"] = dict(ours=max(R["ours"][n]["mean"] for n in NEGS), reef=max(R["reef"][n]["cka"] for n in NEGS))
    # 优点 6：固定阈值。Δ = sim(M) − sim(参照无关模型)，同一提示同一种子组
    POS = [n for n in ALL if n not in NEGS]
    dirs = []
    for ref, test in itertools.permutations(NEGS, 2):
        t = float((sim[test][:, 0] - sim[ref][:, 0]).max())
        row = dict(reference=ref, heldout=test, threshold=t, fpr={}, tpr={})
        for sd in (1, 2):
            row["fpr"][sd] = float(((sim[test][:, sd] - sim[ref][:, sd]) >= t).mean())
            row["tpr"][sd] = {p: float(((sim[p][:, sd] - sim[ref][:, sd]) >= t).mean()) for p in POS}
        row["pass_core"] = all(row["fpr"][sd] <= 0.10 and all(row["tpr"][sd][p] >= 0.90 for p in CORE) for sd in (1, 2))
        dirs.append(row)
    R["fixed_threshold"] = dirs
    (SA / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print("max_neg", R["max_neg"])
    for n in ALL:
        print(f"{n:<18} ours {R['ours'][n]['mean']:.4f}  reef {R['reef'][n]['cka']:.4f}  wc {R['weight_cos'][n]}")
    print("fixed-threshold directions passing (core):", sum(r["pass_core"] for r in dirs), "/", len(dirs))
