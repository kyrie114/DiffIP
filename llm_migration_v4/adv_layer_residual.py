"""优点 7：逐层残差图。对每个模型、每条提示、每个种子组，在扰动 128 次、主成分 8 维
下逐层做对齐，记录每层残差 d_l（相似度 = exp(−各层 d_l 的平均)）。画出各模型在
4 个取样层上的平均残差（误差线为 32 条提示 × 3 个种子组的标准差）。

Usage: python adv_layer_residual.py <name> [name...]   # 需要 feats/main_{name}.npz
输出 adv_scores/layer_residual.json 与 figs_adv/layer_residual.png"""
import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from multiprocessing import Pool
import numpy as np
from common import H
sys.path.insert(0, str(H.parent))
from diffip.preprocessing import project_sequences  # noqa: E402
from diffip.reversion import fit_step_alignment  # noqa: E402

N, D = 128, 8
F = H / "feats"; OUT = H / "adv_scores"; FIG = H / "figs_adv"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
_V = {}


def arr(name):
    if name not in _V:
        _V[name] = np.load(F / f"main_{name}.npz")["pert"]
    return _V[name]


def job(a):
    name, p, k = a
    pr = project_sequences(arr(name)[p, k, :, :N], arr("victim")[p, k, :, :N], D)
    return [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(pr.suspect.shape[0])]


if __name__ == "__main__":
    names = sys.argv[1:]
    res = {}
    with Pool(int(os.environ.get("V4_THREADS", "8"))) as pool:
        for n in names:
            d = np.array(pool.map(job, [(n, p, k) for p in range(32) for k in range(3)]))  # [96, 4]
            res[n] = dict(mean=d.mean(0).tolist(), std=d.std(0).tolist(),
                          layers=np.load(F / f"main_{n}.npz")["layers4"].tolist())
            print(n, np.round(d.mean(0), 4), flush=True)
    (OUT / "layer_residual.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        ax.errorbar(range(1, 5), res[n]["mean"], yerr=res[n]["std"], marker="o", capsize=3, label=n)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["layer 1/4", "layer 2/4", "layer 3/4", "layer 4/4"])
    ax.set_ylabel("mean residual after alignment"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "layer_residual.png", dpi=200); print("saved", FIG / "layer_residual.png")
