"""方法A基线重跑：官方 DiffIPMetric 作用在真实 LLM 逐层 hidden 上（层轴=时间轴，提示=“种子”轴）。
特征全部复用 box 上已缓存的真实特征（10/19 与 10/16 实验产物），不重新抽取。"""
import os, sys, json, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/workspace/papers/DiffIP")
import numpy as np
from multiprocessing import Pool
from diffip import DiffIPMetric

OUT = "/workspace/papers/exp_methodA_diffip20"
F19 = "/workspace/20261019_DiffIP迁移LLM_线性回拉/feats"
F16 = "/workspace/reef_traj/feats"
NSTEP = 7                                  # 层轴下采样到 7 个相对深度 0,1/6,...,1
NCS = [5, 10, 20, 35]

S1 = dict(victim=f"{F19}/source_qwen25_05b.npy",
          models={"instruct": (f"{F19}/derived_instruct.npy", 1), "lora": (f"{F19}/derived_lora.npy", 1),
                  "qwen2_05b": (f"{F19}/independent_qwen2_05b.npy", 0)})
S2 = dict(victim=f"{F16}/qwen05_H.npy",
          models={"instruct": (f"{F16}/qwen05_instruct_H.npy", 1), "int8": (f"{F16}/qwen05_int8_H.npy", 1),
                  "prune30": (f"{F16}/qwen05_prune30_H.npy", 1), "student_gpt2": (f"{F16}/student_gpt2_H.npy", 1),
                  "smollm2_360m": (f"{F16}/smollm2_360m_H.npy", 0), "pythia_410m": (f"{F16}/pythia_410m_H.npy", 0),
                  "gpt2_medium": (f"{F16}/gpt2_medium_H.npy", 0), "gpt2_base": (f"{F16}/gpt2_base_H.npy", 0)})

def steps(H):   # H: [P, L, D] -> [NSTEP, P, D]
    L = H.shape[1]; idx = np.round(np.linspace(0, 1, NSTEP) * (L - 1)).astype(int)
    X = np.transpose(H[:, idx, :], (1, 0, 2)).astype(np.float64)
    # 偏离说明：原始 hidden 范数很大（Qwen 有离群维），DiffIP 距离会让 exp(-d) 下溢为 0。
    # 这里对每个模型、每一步做去均值 + 缩放到“每样本平均平方范数=1”，只改尺度不改几何。
    X = X - X.mean(axis=1, keepdims=True)
    return X / np.sqrt((X ** 2).sum(-1).mean(axis=1))[:, None, None]

def job(args):
    setting, subset_name, sub, model, path, vpath, method = args
    V = steps(np.load(vpath)[sub]); S = steps(np.load(path)[sub]); t0 = time.time()
    if method == "cos":       # 10/19 对照A：第6/12/18/24层逐提示余弦平均（仅同维）
        Hv, Hs = np.load(vpath)[sub][:, [6, 12, 18, 24]], np.load(path)[sub][:, [6, 12, 18, 24]]
        c = (Hv * Hs).sum(-1) / np.linalg.norm(Hv, axis=-1) / np.linalg.norm(Hs, axis=-1)
        score = float(c.mean())
    else:
        nc = None if method == "full" else int(method[3:])
        score = DiffIPMetric(n_components=nc).compare(S, V).similarity
    return dict(setting=setting, subset=subset_name, model=model, method=method, score=score, sec=round(time.time()-t0, 1))

def build():
    jobs = []; rng = np.random.default_rng(0)
    # S1：原 10/19 划分（前36=校准, 后36=测试）各作为一个 N=36 子集，另加 8 个随机 36 子集
    subs1 = {"calib36": np.arange(36), "test36": np.arange(36, 72)}
    for r in range(8): subs1[f"rand36_{r}"] = np.sort(rng.choice(72, 36, replace=False))
    for sn, sub in subs1.items():
        for m, (p, y) in S1["models"].items():
            for meth in ["cos"] + [f"pca{n}" for n in NCS] + (["full"] if sn in ("calib36", "test36") else []):
                jobs.append(("S1", sn, sub, m, p, S1["victim"], meth))
    # S2：180 样本中抽 N=36（5 个随机子集）+ 全部 180
    subs2 = {f"rand36_{r}": np.sort(rng.choice(180, 36, replace=False)) for r in range(5)}; subs2["all180"] = np.arange(180)
    D = {m: np.load(p, mmap_mode="r").shape[2] for m, (p, y) in S2["models"].items()}
    for sn, sub in subs2.items():
        for m, (p, y) in S2["models"].items():
            meths = [f"pca{n}" for n in NCS] + (["cos"] if D[m] == 896 else [])
            for meth in meths: jobs.append(("S2", sn, sub, m, p, S2["victim"], meth))
    return jobs

if __name__ == "__main__":
    jobs = build(); jobs.sort(key=lambda j: j[-1] != "full")     # 慢任务先发
    print("jobs", len(jobs), flush=True); t = time.time(); rows = []
    with Pool(7) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            if r["method"] == "full" or len(rows) % 50 == 0: print(len(rows), r, round(time.time()-t), flush=True)
    labels = {**{("S1", m): y for m, (p, y) in S1["models"].items()}, **{("S2", m): y for m, (p, y) in S2["models"].items()}}
    for r in rows: r["label"] = labels[(r["setting"], r["model"])]
    json.dump(rows, open(f"{OUT}/raw_scores.json", "w"), indent=1)
    print("done", round(time.time()-t), flush=True)
