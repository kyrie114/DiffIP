"""实验 C：特征空间非线性伪装（设计书 7.3 族 1–3）。

只改嫌疑特征的变换。冻结 even4、N 网格、d∈{8,20}、32 prompt、3 个 seed、PCA+ALS。
不重新采集。不写入 scores/ 或 scores_mid2/。对照锚是 fine_tune，不是 victim。
权空间攻击（族 4）本脚本不跑：设计书规定族 1–3 仍 ≡ fine_tune 且导师要求加压时才启动。

预注册变换（种子写死，不随分数改）：
  ident            fine_tune 原特征。用来核对与 scores/main_fine_tune.json 逐项一致。
  族1 逐维单调
    f1_pow05       sign(x)|x|^0.5
    f1_pow2        sign(x)|x|^2
    f1_tanh        tanh(x)
    f1_perm_pow05  与 rep_perm 相同的维置换（seed=1）之后再 pow05
    f1_scale_tanh  与 scale 相同的各向异性缩放（seed=0）之后再 tanh
  族2 混合
    f2_block64_pow05  块大小 64 的块正交旋转（seed=2）之后再 pow05
    f2_lowrank     x + tanh(x A) A^T，A 为 16 列、列单位化，seed=3
  族3 层间不一致（layers4 的 4 个存储层）
    f3_layerpow    四层幂次 0.5 / 1 / 2 / 0.5
    f3_layermix    前两层 tanh，后两层 pow05

预注册判定（只看 N=128, d=8；锚是设计书已核实的 even4 数字）：
  fine_tune 锚 0.363，其 seed std 锚 0.006；跨家族地板锚 0.254。
  协议稳健：|mean − 0.363| <= 0.018 且对跨家族逐 prompt 胜负 >= 30/32。
  攻击成功：mean − 0.254 <= 0.020，或胜负 <= 20/32。
  部分有效：低于稳健带，但未到攻击成功。
  无效：出现非有限值，或 ident 与已入库 even4 分数最大绝对差 > 1e-4。
两种结果都保留：稳健是检测阳性，打到地板是攻击阳性。
"""
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

# 必须在 import score 之前锁定 even4。score 在导入时读取层子集。
os.environ.pop("V4_LAYER_SUBSET", None)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import score as score_mod
from score import als

H = Path(__file__).resolve().parent
F = H / "feats"
OUT = H / "scores_nl"
EVEN4 = H / "scores"
FT_ANCHOR = 0.363
FLOOR_ANCHOR = 0.254
ROBUST_BAND = 0.018          # 3 * 0.006
ATTACK_FLOOR_BAND = 0.020
# 判定配置 N=128、d=8 上，重跑与已存 even4 逐项差 < 1e-4（实测约 1e-6）。
# 其它 N/d 上 sklearn PCA(svd_solver=auto) 可能走随机 SVD，单点差可达 ~0.02，
# 不作为恒等检验失败，只在辅助格子里注明。
IDENT_TOL = 1e-4

NAMES = [
    "ident",
    "f1_pow05", "f1_pow2", "f1_tanh", "f1_perm_pow05", "f1_scale_tanh",
    "f2_block64_pow05", "f2_lowrank",
    "f3_layerpow", "f3_layermix",
]
FAMILY = {
    "ident": "sanity",
    "f1_pow05": "1", "f1_pow2": "1", "f1_tanh": "1",
    "f1_perm_pow05": "1", "f1_scale_tanh": "1",
    "f2_block64_pow05": "2", "f2_lowrank": "2",
    "f3_layerpow": "3", "f3_layermix": "3",
}

_CACHE = {}


def _base():
    if "ft" not in _CACHE:
        _CACHE["ft"] = np.load(F / "main_fine_tune.npz")["pert"]
        _CACHE["vic"] = np.load(F / "main_victim.npz")["pert"]
    return _CACHE["ft"], _CACHE["vic"]


def _pow(x, p):
    return np.sign(x) * np.abs(x) ** p


def _block_q(dim, block, seed):
    rng = np.random.default_rng(seed)
    q = np.eye(dim, dtype=np.float32)
    for i in range(0, dim - dim % block, block):
        a = rng.standard_normal((block, block)).astype(np.float64)
        qq, _ = np.linalg.qr(a)
        q[i:i + block, i:i + block] = qq.astype(np.float32)
    return q


def _apply_q(x, q):
    flat = np.matmul(x.reshape(-1, x.shape[-1]), q)
    return flat.reshape(x.shape)


def transform(name, x):
    if name == "ident":
        return x
    if name == "f1_pow05":
        return _pow(x, 0.5)
    if name == "f1_pow2":
        return _pow(x, 2.0)
    if name == "f1_tanh":
        return np.tanh(x)
    if name == "f1_perm_pow05":
        from llm_migration_v2.transforms import make_permutation
        perm = make_permutation(x.shape[-1], seed=1).argmax(0).numpy()
        return _pow(x[..., perm], 0.5)
    if name == "f1_scale_tanh":
        from llm_migration_v2.transforms import make_anisotropic_scale
        s = np.diag(make_anisotropic_scale(x.shape[-1], seed=0).numpy()).astype(np.float32)
        return np.tanh(x * s)
    if name == "f2_block64_pow05":
        q = _block_q(x.shape[-1], 64, seed=2)
        return _pow(_apply_q(x, q), 0.5)
    if name == "f2_lowrank":
        rng = np.random.default_rng(3)
        a = rng.standard_normal((x.shape[-1], 16)).astype(np.float32)
        a /= np.linalg.norm(a, axis=0, keepdims=True)
        flat = x.reshape(-1, x.shape[-1])
        corr = np.matmul(np.tanh(np.matmul(flat, a)), a.T)
        return x + corr.reshape(x.shape)
    if name == "f3_layerpow":
        out = np.empty_like(x)
        for i, p in enumerate((0.5, 1.0, 2.0, 0.5)):
            out[:, :, i] = _pow(x[:, :, i], p)
        return out
    if name == "f3_layermix":
        out = np.empty_like(x)
        out[:, :, 0] = np.tanh(x[:, :, 0])
        out[:, :, 1] = np.tanh(x[:, :, 1])
        out[:, :, 2] = _pow(x[:, :, 2], 0.5)
        out[:, :, 3] = _pow(x[:, :, 3], 0.5)
        return out
    raise KeyError(name)


def get_pair(name):
    if name not in _CACHE:
        ft, vic = _base()
        arr = transform(name, ft)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite features: {name}")
        _CACHE[name] = arr
        _CACHE["vic"] = vic
    return _CACHE[name], _CACHE["vic"]


def _job(a):
    name, pi, ki, n, d = a
    sus, vic = get_pair(name)
    dm, sim, _ = als(sus[pi, ki, :, :n], vic[pi, ki, :, :n], d)
    return dict(pair=name, prompt=pi, seed_set=ki, N=n, dims=d,
                distance_mean=dm, similarity=sim)


def _mean_std(rows, n, dims):
    by = {}
    for r in rows:
        if r["N"] == n and r["dims"] == dims:
            by.setdefault(r["seed_set"], []).append(r["similarity"])
    per = [float(np.mean(v)) for v in by.values()]
    return float(np.mean(per)), float(np.std(per))


def _wins(rows, neg_rows, n, dims):
    pos = {(r["prompt"]): r["similarity"] for r in rows
           if r["N"] == n and r["dims"] == dims and r["seed_set"] == 0}
    neg = []
    for nr in neg_rows:
        neg.append({r["prompt"]: r["similarity"] for r in nr
                    if r["N"] == n and r["dims"] == dims and r["seed_set"] == 0})
    w = 0
    for p, s in pos.items():
        if s > max(g[p] for g in neg):
            w += 1
    return w


def judge(mean, wins):
    if abs(mean - FT_ANCHOR) <= ROBUST_BAND and wins >= 30:
        return "协议稳健"
    if (mean - FLOOR_ANCHOR) <= ATTACK_FLOOR_BAND or wins <= 20:
        return "攻击成功"
    return "部分有效"


def main():
    if score_mod.LAYER_IDX is not None:
        raise SystemExit(f"even4 required, got layer subset {score_mod.LAYER_SUBSET}")
    OUT.mkdir(exist_ok=True)
    n_grid = score_mod.N_GRID
    dims = score_mod.DIMS
    n_prompts = len(score_mod.PROMPTS)
    n_seeds = len(score_mod.SEED_SETS)
    jobs = list(itertools.product(NAMES, range(n_prompts), range(n_seeds), n_grid, dims))
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "8"))
    print(f"nonlinear suspects={len(NAMES)} jobs={len(jobs)} workers={nworkers}", flush=True)
    per = {k: [] for k in NAMES}
    with Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            per[row["pair"]].append(row)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    stored = json.loads((EVEN4 / "main_fine_tune.json").read_text())
    ident = {(r["prompt"], r["seed_set"], r["N"], r["dims"]): r["similarity"] for r in per["ident"]}
    diffs = [abs(ident[(r["prompt"], r["seed_set"], r["N"], r["dims"])] - r["similarity"])
             for r in stored]
    diffs_judge = [abs(ident[(r["prompt"], r["seed_set"], r["N"], r["dims"])] - r["similarity"])
                   for r in stored if r["N"] == 128 and r["dims"] == 8]
    ident_max = float(max(diffs_judge))
    ident_max_all = float(max(diffs))
    print(f"ident vs even4 N=128 d=8 max |dsim|={ident_max:.3e} "
          f"(all N/d max {ident_max_all:.3e})", flush=True)
    neg = [json.loads((EVEN4 / f"main_{k}.json").read_text())
           for k in ("neg_qwen2_5_7b", "neg_llama3_8b")]
    ft_rows = stored
    summary = {
        "ident_max_abs_dsim_n128_d8": ident_max,
        "ident_max_abs_dsim_all_grid": ident_max_all,
        "ident_ok": ident_max <= IDENT_TOL,
        "anchor": {"fine_tune": FT_ANCHOR, "floor": FLOOR_ANCHOR,
                   "robust_band": ROBUST_BAND, "attack_floor_band": ATTACK_FLOOR_BAND},
        "variants": [],
    }
    if not summary["ident_ok"]:
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
        raise SystemExit("ident check failed — attack rows are not interpretable")
    for name in NAMES:
        rows = per[name]
        (OUT / f"{name}.json").write_text(json.dumps(rows))
        mean, std = _mean_std(rows, 128, 8)
        wins = _wins(rows, neg, 128, 8)
        ft_mean, ft_std = _mean_std(ft_rows, 128, 8)
        verdict = "对照" if name == "ident" else judge(mean, wins)
        rec = dict(name=name, family=FAMILY[name],
                   sim_n128_d8=round(mean, 4), seed_std=round(std, 4),
                   delta_vs_ft_anchor=round(mean - FT_ANCHOR, 4),
                   delta_vs_stored_ft=round(mean - ft_mean, 4),
                   gap_vs_floor=round(mean - FLOOR_ANCHOR, 4),
                   wins_vs_cross_family=f"{wins}/32",
                   verdict=verdict,
                   stored_ft_mean=round(ft_mean, 4), stored_ft_std=round(ft_std, 4))
        # 辅助：N=20 d=8 与 N=128 d=20，不参与判定
        for n, d, tag in ((20, 8, "n20_d8"), (128, 20, "n128_d20")):
            m, s = _mean_std(rows, n, d)
            rec[tag] = {"sim": round(m, 4), "seed_std": round(s, 4),
                        "wins": f"{_wins(rows, neg, n, d)}/32"}
        summary["variants"].append(rec)
        print(f"{name:18} sim={mean:.4f}±{std:.4f} wins={wins}/32 {verdict}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("wrote", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
