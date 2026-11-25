"""4 基座重跑前的数据复用准备（幂等，可重复运行）。

1) 复制 3 个 Ours 特征：feats_qwen3/main_qwen3_*.npz -> feats/sota_baseC_*.npz
   协议一致依据：feats_qwen3 由 v4 collect.py 采集，同一 32 提示 / 种子 123,1123,2123 /
   均匀 4 层 / N=128，npz 键 pert (32,3,4,128,D)+layers4+total，与套件 load_ours_4layers
   的期望完全一致。
2) 转置复制 13 个 REEF TQA-200 激活：v4/acts_reef_tqa/{name}.npz (last: (200,L,D))
   -> reef_acts/sota_{key}.npz (last: (L,200,D))
   协议一致依据：v4 collect_reef_tqa.py 与 suite_sota.load_tqa200 的 200 条陈述构造
   完全相同（TruthfulQA 前 200 行文件序，Question.strip()+" "+Best Answer.strip()），
   层语义相同（hidden_states[i+1] 的每层最后 token）。仅存储轴序不同，转置即一致。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
V4 = H.parent / "llm_migration_v4"
FEATS = H / "feats"
REEF = H / "reef_acts"

OURS_COPY = {
    "sota_baseC_victim": H / "feats_qwen3" / "main_qwen3_victim.npz",
    "sota_baseC_sft_post": H / "feats_qwen3" / "main_qwen3_instruct.npz",
    "sota_baseC_dom_deepseek_r1": H / "feats_qwen3" / "main_qwen3_hard.npz",
}

REEF_TPOSE = {
    "victim": "sota_baseA_victim",
    "neg_pruned_sparsegpt": "sota_baseA_prune_sparsegpt",
    "neg_pruned_wanda": "sota_baseA_prune_wanda",
    "neg_pruned_gblm": "sota_baseA_prune_gblm",
    "fine_tune": "sota_baseA_sft_chat",
    "pos_vicuna": "sota_baseA_sft_vicuna",
    "derivative_cn": "sota_baseA_dom_chinese",
    "cpt_codellama": "sota_baseA_dom_codellama",
    "neg_llama3_8b": "sota_baseB_victim",
    "neg_mpt_7b": "sota_baseD_victim",
    "neg_qwen2_5_7b": "sota_unrel_qwen2_5_7b",
    "neg_llama1_7b": "sota_unrel_llama1_7b",
    "neg_baichuan_7b": "sota_unrel_baichuan_7b",
}


def main():
    FEATS.mkdir(exist_ok=True)
    REEF.mkdir(exist_ok=True)
    for key, src in OURS_COPY.items():
        dst = FEATS / f"{key}.npz"
        if dst.exists():
            print(f"skip {dst.name} (exists)")
            continue
        z = np.load(src)
        pert = z["pert"]
        assert pert.shape == (32, 3, 4, 128, pert.shape[-1]) and np.isfinite(pert).all(), \
            f"bad source {src}"
        shutil.copy(src, dst)
        print(f"copied {src.name} -> {dst.name} {pert.shape}")
    for name, key in REEF_TPOSE.items():
        src = V4 / "acts_reef_tqa" / f"{name}.npz"
        dst = REEF / f"{key}.npz"
        if dst.exists():
            print(f"skip {dst.name} (exists)")
            continue
        z = np.load(src)
        last = z["last"]
        assert last.ndim == 3 and last.shape[0] == 200 and np.isfinite(last).all(), \
            f"bad source {src}: {last.shape}"
        arr = np.ascontiguousarray(last.transpose(1, 0, 2))  # (L, 200, D)
        shape = arr.shape
        np.savez(dst, last=arr)
        del arr
        print(f"tposed {name}.npz -> {dst.name} {shape}")
    print("DONE prep")


if __name__ == "__main__":
    main()
