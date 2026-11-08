"""实验三：五个无关模型两两做参照。规则在出结果之前写死。

维数和层数读实验一、实验二的 chosen。层数不是 control（已存的均匀 4 层）时，
wanda、gblm、weight_perm、prune_dpoff 还没有对应层的特征，脚本停下来，不拿
4 层分数冒充。

层数是 control 时：
- 已有正例和 Qwen、Llama-3 用实验一在旧 4 层特征上打出的分数。
- 新的三个无关模型从 feats_all/ 里切已存的均匀层。32 层模型是 0,10,21,31。
  和 victim 的旧 4 层按存储顺序配对。
- Qwen 对 Llama-3 这一对用同一张表重算，确认口径没换。

五个无关模型：Qwen2.5-7B、Llama-3-8B、llama-7b-hf、Baichuan-7B、mpt-7b。
两两有序：一个当参照，另一个留出。其余三个不进这一对的误报。
一共 20 对。每一对都要过：种子 1 和 2 上，三种剪枝和官方微调簇的召回都 ≥ 0.90，
误报 ≤ 0.10。有一对新组合不过，就不能说相对分数已经不依赖原来那一对无关模型。

中文衍生只附带召回，不参与通过。
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import score  # noqa: E402
from exp_d_prompt_calib import HARD, NEG, SFT, STRONG, one_direction  # noqa: E402

N = 128
NEW = ["neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
# Order is fixed so the pair list does not depend on directory order.
NEG5 = ["neg_qwen2_5_7b", "neg_llama3_8b", *NEW]
NEED_IF_NOT_CONTROL = ["neg_pruned_wanda", "neg_pruned_gblm", "weight_perm", "prune_dpoff"]
OUT = H / "scores_unrel"
RESULT = H / "exp_unrel_result.json"

ARRAYS = {}
VICTIM = None  # old 4-layer victim, shape [32, 3, 4, 128, D]


def _job(job):
    name, pi, ki, dims, idx = job
    sus = ARRAYS[name]
    dm, sim, used = score.als(sus[pi, ki, idx, :N], VICTIM[pi, ki, :, :N], dims)
    return dict(pair=name, prompt=pi, seed_set=ki, N=N, dims=dims,
                distance_mean=dm, similarity=sim, n_components=used)


def _rows_at(path, dims):
    rows = json.loads(path.read_text())
    kept = [r for r in rows if r["N"] == N and r["dims"] == dims]
    if len(kept) != 96:
        raise SystemExit(f"{path.name} has {len(kept)} rows at N={N} d={dims}")
    return kept


def main():
    dim = json.loads((H / "exp_dim_result.json").read_text())
    lay = json.loads((H / "exp_layers_result.json").read_text())
    dims = int(dim["chosen_dims"])
    subset = lay["chosen_subset"]
    if subset != "control":
        missing = [n for n in NEED_IF_NOT_CONTROL
                   if not (H / "feats_all" / f"main_{n}.npz").exists()]
        raise SystemExit(
            f"chosen subset is {subset}, not control. "
            f"missing all-layer feats: {missing or 'none — rescore before this script'}")
    if dims != 8:
        # Experiment 1 scored every original model at the chosen dim into scores_dim.
        pass
    for name in ["victim", *STRONG, *SFT, *HARD, *NEG]:
        src = H / "scores_dim" / f"main_{name}.json"
        if name == "victim":
            continue
        if not src.exists():
            raise SystemExit(f"missing {src}")
    victim_old = np.load(H / "feats" / "main_victim.npz")
    control = [int(x) for x in victim_old["layers4"]]
    if control != [0, 10, 21, 31]:
        raise SystemExit(f"victim control layers {control} are not 0,10,21,31")
    global VICTIM
    VICTIM = victim_old["pert"]
    for name in NEW:
        path = H / "feats_all" / f"main_{name}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        z = np.load(path)
        layers = [int(x) for x in z["layers"]]
        if any(i not in layers for i in control):
            raise SystemExit(f"{name} does not store {control}")
        # Absolute indices. Slice to the same order as the stored victim layers.
        ARRAYS[name] = z["pert"][:, :, control]
        if ARRAYS[name].shape != VICTIM.shape:
            raise SystemExit(f"{name} slice {ARRAYS[name].shape} != victim {VICTIM.shape}")
    jobs = [(n, p, k, dims, control) for n in NEW for p in range(32) for k in range(3)]
    # idx is not used; the array is already sliced. Keep the argument so the
    # job tuple stays explicit.
    def run_job(job):
        name, pi, ki, d, _idx = job
        dm, sim, used = score.als(ARRAYS[name][pi, ki], VICTIM[pi, ki], d)
        return dict(pair=name, prompt=pi, seed_set=ki, N=N, dims=d,
                    distance_mean=dm, similarity=sim, n_components=used)
    # Local function is not picklable. Use the module-level job with pre-sliced arrays
    # by stuffing the sliced tensor where _job indexes idx. Rebuild as full-axis
    # dummy: _job does sus[pi, ki, idx]. Our arrays are already [32,3,4,128,D].
    # Call als directly in-process if the pool cannot see a closure.
    print(f"jobs={len(jobs)} dims={dims} layers={control}", flush=True)
    rows_new = {n: [] for n in NEW}
    nworkers = int(os.environ.get("V4_SCORE_WORKERS", "8"))
    # Module-level _job expects unsliced arrays and an index list. Adapt.
    for name in NEW:
        # Put a view that ignores idx by wrapping: store the sliced tensor and
        # pass idx that selects all stored rows in order, i.e. [0,1,2,3].
        ARRAYS[name] = ARRAYS[name]
    jobs = [(n, p, k, dims, [0, 1, 2, 3]) for n in NEW for p in range(32) for k in range(3)]
    with get_context("fork").Pool(nworkers) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            rows_new[row["pair"]].append(row)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}", flush=True)
    OUT.mkdir(exist_ok=True)
    for name, rows in rows_new.items():
        (OUT / f"main_{name}.json").write_text(json.dumps(rows))

    table = {}
    for name in [*STRONG, *SFT, *HARD, *NEG]:
        table[name] = _rows_at(H / "scores_dim" / f"main_{name}.json", dims)
    for name, rows in rows_new.items():
        table[name] = rows

    # one_direction closes over exp_d's NEG only inside prompt_corr, not here.
    # It uses STRONG, SFT, HARD from that module, which is the full gate.
    directions = []
    failed = []
    for anchor, test in itertools.permutations(NEG5, 2):
        raw = one_direction(table, anchor, test, N, dims)
        op = raw["operating_points"]["fpr0"]
        held = [op["by_seed"]["1"], op["by_seed"]["2"]]
        item = {
            "anchor": anchor, "test": test, "t": op["t"], "pass": bool(op["heldout_pass"]),
            "auc_deploy": raw["auc_deploy"],
            "held_strong_tpr": [h["strong"]["tpr"] for h in held],
            "held_sft_tpr": [h["sft"]["tpr"] for h in held],
            "held_fpr": [h["deploy"]["fpr"] for h in held],
            "held_hard_tpr": [h["hard"]["tpr"] for h in held],
            "legacy_pair": {anchor, test} == set(NEG),
        }
        directions.append(item)
        if not item["pass"]:
            failed.append(f"{anchor}->{test}")
    legacy = [d for d in directions if d["legacy_pair"]]
    result = {
        "dims": dims,
        "subset": subset,
        "control_layers": control,
        "n_directions": len(directions),
        "n_failed": len(failed),
        "failed": failed,
        "claim_holds": len(failed) == 0,
        "legacy_qwen_llama": legacy,
        "directions": directions,
    }
    RESULT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"DONE claim={result['claim_holds']} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
