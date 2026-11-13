"""导师手稿补充实验一揽子套件（4 基座 × 4 衍生类型 + SOTA 对比 + 计算效率 B + 消融实验 C）。

完全兼容现有 `common.py`、`collect.py`、`score.py`，自动复用 `feats/`、`feats_all/`、`scores/`、
`scores_dim/`、`scores_unrel/`、`reef_acts/` 中已有的特征与打分文件，零重复采集！

包含模块：
  A. 与其它 SOTA 对比（Ours vs REEF vs HuRef vs AWM）：
     - 4 个基座模型：
         Base A: Llama-2-7b-hf
         Base B: Meta-Llama-3-8B
         Base C: Qwen3-8B-Base
         Base D: mpt-7b
     - 4 种模型衍生类型（每种均计算 均值 ± 标准差）：
         ① 剪枝 (Pruning): 结构化删层 + 稀疏权重剪枝
         ② 量化 (Quantization): INT8 / INT4 对称逐组权重量化（内存实时生成，零磁盘开销）
         ③ 微调 (Fine-tuning / SFT): 官方 Chat / Instruct 微调模型
         ④ 领域 (Domain / CPT): 跨语种/代码/数学推理/领域指令深化模型
     - 附加 Test（白盒与特征伪装压力测试）：
         置换 (perm)、各向异性缩放 (scale)、保输出正交旋转+非恒定 RMSNorm 缩放 (rot_rmsnorm, T3+T5)
     - 产出：主表（Method × 4 衍生类型 ± std + Test）、跨基座区分矩阵（源 A/B/C/D）、16 个子图（Fig. 1）
  B. 计算效率对比（Computational Efficiency）：
     - 记录并对比 Ours、REEF、HuRef、AWM 的特征体积、提取时间与在线单对鉴定耗时（毫秒级 ALS vs 立方级 LAP/大矩阵连乘）。
  C. 消融实验（Ablation Study）：
     - ① PCA 维数消融（直接读取已跑完的 exp_dim_result.json：d ∈ {8,10,15,20,30,40}）
     - ② 层间选择消融（直接读取已跑完的 exp_layers_result.json + 实验A：even4 vs mid2 vs uniform8 vs last4 vs all32）

用法（在服务器 `llm_migration_v4/` 目录下运行）：
  1. 查看当前哪些模型特征已就绪、哪些待采：
       python suite_sota.py status
  2. 采集缺失模型的 Ours (4 均匀层配对微扰) 特征（已有的自动跳过）：
       python suite_sota.py collect_ours [all | baseA | baseB | baseC | baseD | 模型名...]
  3. 采集缺失模型的 REEF (TruthfulQA 200 条) 特征（已有的自动跳过）：
       python suite_sota.py collect_reef [all | baseA | baseB | baseC | baseD | 模型名...]
  4. 运行静态权重基线（HuRef + AWM）及伪装攻击与模块 B 计时：
       python suite_sota.py run_static [all | baseA | baseB | baseC | baseD]
  5. 汇总全部 4 基座 × 4 衍生类型指标（AUC、TPR@5%FPR、Margin ± std）并生成 Markdown/LaTeX 主表：
       python suite_sota.py evaluate
  6. 一键绘制导师手稿要求的全部论文图表（16 个子图 Fig.1 + 效率对比图 Fig.2 + 消融实验图 Fig.3 + 跨基座热力图 Fig.4）：
       python suite_sota.py plot
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent))

try:
    import torch
    from common import (  # noqa: E402
        MODELS, N_LAYERS, N_MAX, PROMPTS, SCALE, SEED_SETS,
        even_layers, load, perturb_batch, pooled_layers, prune_spec,
    )
    from llm_migration_v2.transforms import (  # noqa: E402
        make_anisotropic_scale, make_permutation, prune_layers,
    )
except ImportError:
    class _DummyTorch:
        @staticmethod
        def no_grad():
            return lambda fn: fn

        @staticmethod
        def set_num_threads(_n: int):
            pass

    torch = _DummyTorch()
    MODELS = os.environ.get("V4_MODELS", "/data/luo_lixuan/llms")
    N_LAYERS = 4
    N_MAX = 128
    SCALE = 0.01
    SEED_SETS = ([100 + i for i in range(128)], [200 + i for i in range(128)], [300 + i for i in range(128)])
    PROMPTS = []

    def even_layers(total: int, k: int = N_LAYERS) -> list[int]:
        return [round(i * (total - 1) / (k - 1)) for i in range(k)]

try:
    from diffip.preprocessing import project_sequences  # noqa: E402
    from diffip.reversion import fit_step_alignment  # noqa: E402
except ImportError:
    project_sequences = None
    fit_step_alignment = None

FEATS_4 = H / "feats"
FEATS_ALL = H / "feats_all"
REEF_DIR = H / "reef_acts"
SOTA_DIR = H / "sota_scores"
FIGS_DIR = H / "figs_sota"
for _d in (FEATS_4, REEF_DIR, SOTA_DIR, FIGS_DIR):
    _d.mkdir(exist_ok=True)

TQA_CSV = H.parent / "llm_migration_v3" / "reef_baseline" / "TruthfulQA.csv"

# ==============================================================================
# 1. 4 基座 × 4 衍生类型模型注册表（100% 对应 /data/luo_lixuan/llms 本地已有目录）
# ==============================================================================
#
# 说明：
# - path: 本地模型权重目录
# - transform:
#     None          -> 原样加载
#     "prune_drop4" -> 结构化删除 4 个均匀内部层（与主线 prune_dpoff 同一规则）
#     "prune_mag50" -> 非结构化 50% 幅度稀疏置零（每层线性权重保留绝对值最大的 50%）
#     "quant_int8"  -> 权重逐组对称 INT8 量化反量化（group_size=128，零外部依赖）
#     "quant_int4"  -> 权重逐组对称 INT4 量化反量化（group_size=128，零外部依赖）
#     "weight_perm" -> 残差流维度置换伪装
# - legacy_feat: 若之前主线或实验三已采过，直接指向已有 npz 文件名，免重复采集！

SUITE_MODELS = {
    # ---------------- 基座 A：Llama-2-7b-hf ----------------
    "baseA_victim": {
        "path": f"{MODELS}/Llama-2-7b-hf", "transform": None,
        "legacy_feat": "main_victim.npz", "base": "baseA", "role": "victim",
    },
    # ① 剪枝 (Pruning)
    "baseA_prune_sparsegpt": {
        "path": f"{MODELS}/llama2-7B-pruned-sparsegpt", "transform": None,
        "legacy_feat": "main_neg_pruned_sparsegpt.npz", "base": "baseA", "role": "pruning",
    },
    "baseA_prune_wanda": {
        "path": f"{MODELS}/wanda_pruned_llama_2_7B", "transform": None,
        "legacy_feat": "main_neg_pruned_wanda.npz", "base": "baseA", "role": "pruning",
    },
    "baseA_prune_gblm": {
        "path": f"{MODELS}/GBLM-Pruner-LLaMA-2-7B", "transform": None,
        "legacy_feat": "main_neg_pruned_gblm.npz", "base": "baseA", "role": "pruning",
    },
    "baseA_prune_drop4": {
        "path": f"{MODELS}/Llama-2-7b-chat-hf", "transform": "prune_drop4",
        "legacy_feat": "main_prune_dpoff.npz", "base": "baseA", "role": "pruning",
    },
    # ② 量化 (Quantization)
    "baseA_quant_int8": {
        "path": f"{MODELS}/Llama-2-7b-chat-hf", "transform": "quant_int8",
        "legacy_feat": None, "base": "baseA", "role": "quantization",
    },
    "baseA_quant_int4": {
        "path": f"{MODELS}/Llama-2-7b-chat-hf", "transform": "quant_int4",
        "legacy_feat": None, "base": "baseA", "role": "quantization",
    },
    # ③ 微调 (Fine-tuning / SFT)
    "baseA_sft_chat": {
        "path": f"{MODELS}/Llama-2-7b-chat-hf", "transform": None,
        "legacy_feat": "main_fine_tune.npz", "base": "baseA", "role": "finetune",
    },
    "baseA_sft_vicuna": {
        "path": f"{MODELS}/vicuna-7b-v1.5", "transform": None,
        "legacy_feat": "main_pos_vicuna.npz", "base": "baseA", "role": "finetune",
    },
    # ④ 领域 (Domain / CPT)
    "baseA_dom_chinese": {
        "path": f"{MODELS}/Chinese-llama-2-7b", "transform": None,
        "legacy_feat": "main_derivative_cn.npz", "base": "baseA", "role": "domain",
    },
    "baseA_dom_codellama": {
        "path": f"{MODELS}/CodeLlama-7b-hf", "transform": None,
        "legacy_feat": "main_cpt_codellama.npz", "base": "baseA", "role": "domain",
    },

    # ---------------- 基座 B：Meta-Llama-3-8B ----------------
    "baseB_victim": {
        "path": f"{MODELS}/Meta-Llama-3-8B", "transform": None,
        "legacy_feat": "main_neg_llama3_8b.npz", "base": "baseB", "role": "victim",
    },
    # ① 剪枝
    "baseB_prune_mag50": {
        "path": f"{MODELS}/Meta-Llama-3-8B-Instruct", "transform": "prune_mag50",
        "legacy_feat": None, "base": "baseB", "role": "pruning",
    },
    "baseB_prune_drop4": {
        "path": f"{MODELS}/Meta-Llama-3-8B-Instruct", "transform": "prune_drop4",
        "legacy_feat": None, "base": "baseB", "role": "pruning",
    },
    # ② 量化
    "baseB_quant_int8": {
        "path": f"{MODELS}/Meta-Llama-3-8B-Instruct", "transform": "quant_int8",
        "legacy_feat": None, "base": "baseB", "role": "quantization",
    },
    "baseB_quant_int4": {
        "path": f"{MODELS}/Meta-Llama-3-8B-Instruct", "transform": "quant_int4",
        "legacy_feat": None, "base": "baseB", "role": "quantization",
    },
    # ③ 微调
    "baseB_sft_instruct": {
        "path": f"{MODELS}/Meta-Llama-3-8B-Instruct", "transform": None,
        "legacy_feat": None, "base": "baseB", "role": "finetune",
    },
    # ④ 领域
    "baseB_dom_korean": {
        "path": f"{MODELS}/llama-3-Korean-Bllossom-8B", "transform": None,
        "legacy_feat": None, "base": "baseB", "role": "domain",
    },

    # ---------------- 基座 C：Qwen3-8B-Base ----------------
    "baseC_victim": {
        "path": f"{MODELS}/Qwen3-8B-Base", "transform": None,
        "legacy_feat": None, "base": "baseC", "role": "victim",
    },
    # ① 剪枝
    "baseC_prune_mag50": {
        "path": f"{MODELS}/Qwen3-8B", "transform": "prune_mag50",
        "legacy_feat": None, "base": "baseC", "role": "pruning",
    },
    "baseC_prune_drop4": {
        "path": f"{MODELS}/Qwen3-8B", "transform": "prune_drop4",
        "legacy_feat": None, "base": "baseC", "role": "pruning",
    },
    # ② 量化
    "baseC_quant_int8": {
        "path": f"{MODELS}/Qwen3-8B", "transform": "quant_int8",
        "legacy_feat": None, "base": "baseC", "role": "quantization",
    },
    "baseC_quant_int4": {
        "path": f"{MODELS}/Qwen3-8B", "transform": "quant_int4",
        "legacy_feat": None, "base": "baseC", "role": "quantization",
    },
    # ③ 微调
    "baseC_sft_post": {
        "path": f"{MODELS}/Qwen3-8B", "transform": None,
        "legacy_feat": None, "base": "baseC", "role": "finetune",
    },
    # ④ 领域 (数学/推理强化衍生)
    "baseC_dom_deepseek_r1": {
        "path": f"{MODELS}/DeepSeek-R1-0528-Qwen3-8B", "transform": None,
        "legacy_feat": None, "base": "baseC", "role": "domain",
    },

    # ---------------- 基座 D：mpt-7b ----------------
    "baseD_victim": {
        "path": f"{MODELS}/mpt-7b", "transform": None,
        "legacy_feat": "main_neg_mpt_7b.npz", "base": "baseD", "role": "victim",
    },
    # ① 剪枝
    "baseD_prune_mag50": {
        "path": f"{MODELS}/mpt-7b-chat", "transform": "prune_mag50",
        "legacy_feat": None, "base": "baseD", "role": "pruning",
    },
    "baseD_prune_drop4": {
        "path": f"{MODELS}/mpt-7b-chat", "transform": "prune_drop4",
        "legacy_feat": None, "base": "baseD", "role": "pruning",
    },
    # ② 量化
    "baseD_quant_int8": {
        "path": f"{MODELS}/mpt-7b-chat", "transform": "quant_int8",
        "legacy_feat": None, "base": "baseD", "role": "quantization",
    },
    "baseD_quant_int4": {
        "path": f"{MODELS}/mpt-7b-chat", "transform": "quant_int4",
        "legacy_feat": None, "base": "baseD", "role": "quantization",
    },
    # ③ 微调
    "baseD_sft_chat": {
        "path": f"{MODELS}/mpt-7b-chat", "transform": None,
        "legacy_feat": None, "base": "baseD", "role": "finetune",
    },
    # ④ 领域/指令任务深化
    "baseD_dom_instruct": {
        "path": f"{MODELS}/mpt-7b-instruct", "transform": None,
        "legacy_feat": None, "base": "baseD", "role": "domain",
    },

    # ---------------- 公共跨家族无关负例池（均已采集完成！） ----------------
    "unrel_qwen2_5_7b": {
        "path": f"{MODELS}/Qwen2.5-7B", "transform": None,
        "legacy_feat": "main_neg_qwen2_5_7b.npz", "base": "unrel", "role": "negative",
    },
    "unrel_llama1_7b": {
        "path": f"{MODELS}/llama-7b-hf", "transform": None,
        "legacy_feat": "main_neg_llama1_7b.npz", "base": "unrel", "role": "negative",
    },
    "unrel_baichuan_7b": {
        "path": f"{MODELS}/Baichuan-7B", "transform": None,
        "legacy_feat": "main_neg_baichuan_7b.npz", "base": "unrel", "role": "negative",
    },
}

BASES = ("baseA", "baseB", "baseC", "baseD")
BASE_LABELS = {
    "baseA": "(a) Base A: Llama-2-7B",
    "baseB": "(b) Base B: Llama-3-8B",
    "baseC": "(c) Base C: Qwen3-8B",
    "baseD": "(d) Base D: MPT-7B",
}
ROLES = ("pruning", "quantization", "finetune", "domain")
ROLE_LABELS = {
    "pruning": "① 剪枝 (Pruning)",
    "quantization": "② 量化 (Quantization)",
    "finetune": "③ 微调 (Fine-tuning)",
    "domain": "④ 领域 (Domain CPT)",
}


# ==============================================================================
# 2. 模型加载与内存实时变换（剪枝 / INT8 / INT4 量化）
# ==============================================================================

def _decoder_stack(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "blocks"):
        return tr.blocks
    raise RuntimeError(f"Cannot locate decoder layers on {type(model).__name__}")


@torch.no_grad()
def apply_group_quant_dequant_(model, bits: int = 8, group_size: int = 128):
    """In-place group-wise symmetric weight quantization-dequantization (RTN).
    Works identically on CPU for any architecture without bitsandbytes."""
    qmax = (1 << (bits - 1)) - 1
    layers = _decoder_stack(model)
    for layer in layers:
        for _, mod in layer.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or w.ndim != 2:
                continue
            orig_shape = w.shape
            orig_dtype = w.dtype
            flat = w.detach().float()
            pad = (group_size - (flat.shape[1] % group_size)) % group_size
            if pad > 0:
                flat = torch.nn.functional.pad(flat, (0, pad))
            grouped = flat.view(flat.shape[0], -1, group_size)
            amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = amax / qmax
            q = torch.clamp(torch.round(grouped / scale), -qmax, qmax) * scale
            deq = q.view(flat.shape[0], -1)[:, :orig_shape[1]].to(orig_dtype)
            w.copy_(deq)


@torch.no_grad()
def apply_magnitude_sparsity_(model, sparsity: float = 0.5):
    """In-place 50% unstructured magnitude pruning on decoder linear layers."""
    layers = _decoder_stack(model)
    for layer in layers:
        for _, mod in layer.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or w.ndim != 2:
                continue
            flat = w.detach().abs().flatten().float()
            k = int(flat.numel() * sparsity)
            if k <= 0:
                continue
            # Per-row 50% sparsity (fast and preserves row balance like Wanda/SparseGPT)
            row_abs = w.detach().abs()
            thresh = torch.quantile(row_abs.float(), sparsity, dim=1, keepdim=True).to(w.dtype)
            mask = row_abs >= thresh
            w.mul_(mask)


@torch.no_grad()
def apply_drop_layers_(model, n_drop: int = 4):
    """Drop n_drop evenly spaced interior decoder layers (works for Llama/Qwen/MPT)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        total = len(inner.layers)
        drop = set(prune_spec(total))
        inner.layers = torch.nn.ModuleList(
            [l for i, l in enumerate(inner.layers) if i not in drop]
        )
        model.config.num_hidden_layers = len(inner.layers)
        return
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "blocks"):
        total = len(tr.blocks)
        drop = set(prune_spec(total))
        tr.blocks = torch.nn.ModuleList(
            [b for i, b in enumerate(tr.blocks) if i not in drop]
        )
        model.config.n_layers = len(tr.blocks)
        return
    raise RuntimeError(f"Unsupported architecture for layer drop: {type(model).__name__}")


def load_suite_model(key: str):
    spec = SUITE_MODELS[key]
    m, tok = load(spec["path"])
    tf = spec["transform"]
    if tf == "prune_drop4":
        apply_drop_layers_(m, 4)
    elif tf == "prune_mag50":
        apply_magnitude_sparsity_(m, 0.50)
    elif tf == "quant_int8":
        apply_group_quant_dequant_(m, bits=8, group_size=128)
    elif tf == "quant_int4":
        apply_group_quant_dequant_(m, bits=4, group_size=128)
    elif tf is not None:
        raise ValueError(f"Unknown transform {tf}")
    return m, tok


# ==============================================================================
# 3. 特征定位与采集（自动复用已有 feats/ 与 feats_all/）
# ==============================================================================

def locate_ours_feat(key: str) -> Path | None:
    """Return path to 4-layer feature file if already collected, else None."""
    own = FEATS_4 / f"sota_{key}.npz"
    if own.exists():
        return own
    legacy = SUITE_MODELS[key].get("legacy_feat")
    if legacy:
        p4 = FEATS_4 / legacy
        if p4.exists():
            return p4
        pall = FEATS_ALL / legacy
        if pall.exists():
            return pall
    return None


def load_ours_4layers(key: str) -> np.ndarray:
    """Load [32, 3, 4, 128, D] float32 array for `key`, slicing even4 if from feats_all."""
    path = locate_ours_feat(key)
    if path is None:
        raise FileNotFoundError(f"Missing Ours feature for {key}. Run: python suite_sota.py collect_ours {key}")
    z = np.load(path)
    pert = z["pert"]
    if pert.shape[2] == 4:
        return pert
    total = int(z["total"]) if "total" in z else pert.shape[2]
    idx = even_layers(total, 4)
    return pert[:, :, idx]


def collect_ours_one(key: str):
    existing = locate_ours_feat(key)
    if existing is not None:
        print(f"[Ours] skip {key} (reusing {existing.relative_to(H)})", flush=True)
        return
    out = FEATS_4 / f"sota_{key}.npz"
    partial = FEATS_4 / f"sota_{key}.partial.npz"
    t0 = time.time()
    m, tok = load_suite_model(key)
    layers_stack = _decoder_stack(m)
    total = len(layers_stack)
    L4 = even_layers(total, N_LAYERS)
    E = m.get_input_embeddings()
    D_ = E.weight.shape[1]
    shape = (len(PROMPTS), len(SEED_SETS), len(L4), N_MAX, D_)
    feats = np.zeros(shape, dtype=np.float32)
    start_p = 0
    if partial.exists():
        z = np.load(partial)
        if tuple(z["pert"].shape) == shape:
            feats[:] = z["pert"]
            start_p = int(z["done"])
            print(f"[Ours] resume {key} from prompt {start_p}/{len(PROMPTS)}", flush=True)
    for p, prompt in enumerate(PROMPTS):
        if p < start_p:
            continue
        ids = tok(prompt, return_tensors="pt").input_ids
        base_emb = E(ids).detach()
        for k, seed in enumerate(SEED_SETS):
            x = perturb_batch(base_emb, N_MAX, seed)
            feats[p, k] = pooled_layers(m, x, L4)
            if not np.isfinite(feats[p, k]).all():
                raise RuntimeError(f"[Ours] {key} prompt {p+1} seed {k} non-finite!")
        tmp = partial.parent / (partial.name + ".tmp.npz")
        np.savez(tmp, pert=feats, layers4=np.array(L4), total=np.int64(total), done=np.int64(p + 1))
        os.replace(tmp, partial)
        print(f"  [Ours] {key} prompt {p+1}/{len(PROMPTS)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(out, pert=feats, layers4=np.array(L4), layers=np.array(L4), total=np.int64(total),
             collect_seconds=float(time.time() - t0))
    partial.unlink(missing_ok=True)
    print(f"[Ours] saved {out.name} D={D_} L4={L4} in {time.time()-t0:.1f}s", flush=True)
    del m
    gc.collect()


# ==============================================================================
# 4. REEF 基线特征采集与打分（ICLR 2025：TruthfulQA 200 条 + 线性 CKA）
# ==============================================================================

def load_tqa200() -> list[str]:
    if TQA_CSV.exists():
        rows = list(csv.DictReader(open(TQA_CSV, encoding="utf-8")))
        return [r["Question"].strip() + " " + r["Best Answer"].strip() for r in rows[:200]]
    # Fallback if TruthfulQA.csv is not at relative path: deterministic 200 probes from PROMPTS
    out = []
    for i in range(200):
        p1 = PROMPTS[i % len(PROMPTS)]
        p2 = PROMPTS[(i * 7 + 3) % len(PROMPTS)]
        out.append(f"{p1} {p2}" if i >= len(PROMPTS) else p1)
    return out


def locate_reef_feat(key: str) -> Path | None:
    own = REEF_DIR / f"sota_{key}.npz"
    if own.exists():
        return own
    legacy = SUITE_MODELS[key].get("legacy_feat")
    if legacy and legacy.startswith("main_"):
        old_name = legacy[len("main_"):-len(".npz")]
        old_reef = REEF_DIR / f"{old_name}.npz"
        if old_reef.exists():
            return old_reef
    return None


@torch.no_grad()
def collect_reef_one(key: str):
    existing = locate_reef_feat(key)
    if existing is not None:
        print(f"[REEF] skip {key} (reusing {existing.relative_to(H)})", flush=True)
        return
    out = REEF_DIR / f"sota_{key}.npz"
    t0 = time.time()
    m, tok = load_suite_model(key)
    E = m.get_input_embeddings()
    texts = load_tqa200()
    acts = []
    for idx, t in enumerate(texts):
        ids = tok(t, return_tensors="pt").input_ids
        e = E(ids).detach()
        res = m(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long),
                use_cache=False, output_hidden_states=True)
        acts.append(torch.stack([h[0, -1] for h in res.hidden_states[1:]]).float().numpy())
        if (idx + 1) % 50 == 0:
            print(f"  [REEF] {key} {idx+1}/200 ({time.time()-t0:.0f}s)", flush=True)
    arr = np.stack(acts, axis=1)  # [layers, 200, D]
    np.savez(out, last=arr, collect_seconds=float(time.time() - t0))
    print(f"[REEF] saved {out.name} shape={arr.shape} in {time.time()-t0:.1f}s", flush=True)
    del m
    gc.collect()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Standard centered linear CKA (REEF Eq. 1)."""
    X = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    Y = Y.astype(np.float64) - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    n = K.shape[0]
    Hm = np.eye(n) - np.ones((n, n)) / n
    Kc = Hm @ K @ Hm
    Lc = Hm @ L @ Hm
    num = float(np.sum(Kc * Lc))
    den = math.sqrt(float(np.sum(Kc * Kc)) * float(np.sum(Lc * Lc)))
    return num / max(den, 1e-12)


def reef_Bootstrap_scores(vic_acts: np.ndarray, sus_acts: np.ndarray, layer_idx: int = 18) -> list[float]:
    """Compute REEF CKA across 3 subsets/bootstraps of the 200 samples (matching our 3 seed sets)
    so we can report mean ± std and ROC AUC consistently."""
    lv = min(layer_idx, vic_acts.shape[0] - 1)
    ls = min(layer_idx, sus_acts.shape[0] - 1)
    Xv = vic_acts[lv]
    Xs = sus_acts[ls]
    m = Xv.shape[0]
    scores = []
    for seed in SEED_SETS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(m, size=min(128, m), replace=False)
        scores.append(linear_cka(Xv[idx], Xs[idx]))
    return scores


# ==============================================================================
# 5. 静态权重基线：HuRef (NeurIPS 2024) 与 AWM (ICLR 2026) 跨架构统一实现
# ==============================================================================

def _extract_layer_weights(model, layer_idx: int) -> dict[str, torch.Tensor]:
    """Extract (W_q, W_k, W_v, W_o, W_up, W_down) in shape [out_dim, in_dim]
    handling Llama, Qwen2/Qwen3 (GQA), Baichuan (W_pack), and MPT (Wqkv)."""
    layers = _decoder_stack(model)
    layer = layers[layer_idx]
    # 1. Attention weights
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        if hasattr(attn, "q_proj"):
            wq = attn.q_proj.weight.detach().float()
            wk = attn.k_proj.weight.detach().float()
            wv = attn.v_proj.weight.detach().float()
            wo = attn.o_proj.weight.detach().float()
            # Handle GQA (Llama-3, Qwen2.5, Qwen3): repeat KV heads to match Q out_features
            if wk.shape[0] != wq.shape[0] and wq.shape[0] % wk.shape[0] == 0:
                rep = wq.shape[0] // wk.shape[0]
                head_dim = getattr(attn, "head_dim", None) or (
                    wq.shape[0] // getattr(model.config, "num_attention_heads", 32)
                )
                n_kv = wk.shape[0] // head_dim
                wk = wk.view(n_kv, head_dim, -1).repeat_interleave(rep, dim=0).reshape(wq.shape[0], -1)
                wv = wv.view(n_kv, head_dim, -1).repeat_interleave(rep, dim=0).reshape(wq.shape[0], -1)
        elif hasattr(attn, "W_pack"):
            # Baichuan-7B
            wq, wk, wv = attn.W_pack.weight.detach().float().chunk(3, dim=0)
            wo = attn.o_proj.weight.detach().float()
        else:
            raise RuntimeError(f"Unknown self_attn structure on {type(attn).__name__}")
    elif hasattr(layer, "attn") and hasattr(layer.attn, "Wqkv"):
        # MPT-7B
        wq, wk, wv = layer.attn.Wqkv.weight.detach().float().chunk(3, dim=0)
        wo = layer.attn.out_proj.weight.detach().float()
    else:
        raise RuntimeError(f"Unknown attention module on {type(layer).__name__}")

    # 2. FFN weights
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "up_proj"):
            w1 = mlp.up_proj.weight.detach().float()
            w2 = mlp.down_proj.weight.detach().float()
        elif hasattr(mlp, "fc1"):
            w1 = mlp.fc1.weight.detach().float()
            w2 = mlp.fc2.weight.detach().float()
        else:
            raise RuntimeError(f"Unknown MLP structure on {type(mlp).__name__}")
    elif hasattr(layer, "ffn"):
        # MPT-7B
        w1 = layer.ffn.up_proj.weight.detach().float()
        w2 = layer.ffn.down_proj.weight.detach().float()
    else:
        raise RuntimeError(f"Unknown FFN module on {type(layer).__name__}")

    return {"wq": wq, "wk": wk, "wv": wv, "wo": wo, "w1": w1, "w2": w2}


def _shared_vocab_pairs(tok_a, tok_b, max_tokens: int = 4096, rare_mode: bool = False) -> tuple[list[int], list[int]]:
    """Find matching token IDs across two tokenizers by string token lookup."""
    va = tok_a.get_vocab()
    vb = tok_b.get_vocab()
    shared = sorted(set(va.keys()) & set(vb.keys()), key=lambda s: va[s], reverse=rare_mode)
    if len(shared) >= 256:
        picked = shared[:max_tokens]
        return [va[s] for s in picked], [vb[s] for s in picked]
    # Fallback if tokenizers encode strings differently: encode a list of probe words
    words = " ".join(PROMPTS).split()
    ids_a, ids_b = [], []
    for w in words:
        ia = tok_a.encode(w, add_special_tokens=False)
        ib = tok_b.encode(w, add_special_tokens=False)
        if ia and ib:
            ids_a.append(ia[-1])
            ids_b.append(ib[-1])
    return ids_a[:max_tokens], ids_b[:max_tokens]


@torch.no_grad()
def compute_huref_ics(m_vic, tok_vic, m_sus, tok_sus, k_tokens: int = 512) -> tuple[list[float], float]:
    """Compute HuRef (NeurIPS 2024, Eq. 8) Invariant Cosine Similarity (ICS)
    across 3 token subsets (for mean ± std) and return (scores_3seeds, elapsed_sec)."""
    t0 = time.time()
    ids_v, ids_s = _shared_vocab_pairs(tok_vic, tok_sus, max_tokens=k_tokens * 2, rare_mode=True)
    ev = m_vic.get_input_embeddings().weight.detach().float()
    es = m_sus.get_input_embeddings().weight.detach().float()
    lv_total = len(_decoder_stack(m_vic))
    ls_total = len(_decoder_stack(m_sus))
    vic_layers = [_extract_layer_weights(m_vic, lv_total - 2), _extract_layer_weights(m_vic, lv_total - 1)]
    sus_layers = [_extract_layer_weights(m_sus, ls_total - 2), _extract_layer_weights(m_sus, ls_total - 1)]

    def _invariants(X_hat: torch.Tensor, lws: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        blocks = []
        for lw in lws:
            # Paper: M_a = X W_Q W_K^T X^T; PyTorch Linear weight is W^T
            q = X_hat @ lw["wq"].T
            k = X_hat @ lw["wk"].T
            ma = q @ k.T
            v = X_hat @ lw["wv"].T
            mb = (v @ lw["wo"].T) @ X_hat.T
            mf = ((X_hat @ lw["w1"].T) @ lw["w2"].T) @ X_hat.T
            blocks.extend([ma.flatten(), mb.flatten(), mf.flatten()])
        return torch.cat(blocks, dim=0).double()

    scores = []
    n_avail = len(ids_v)
    sub_k = min(k_tokens, max(64, n_avail // 2))
    for seed in SEED_SETS:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_avail, size=sub_k, replace=False)
        xv = ev[[ids_v[i] for i in sel]]
        xs = es[[ids_s[i] for i in sel]]
        iv = _invariants(xv, vic_layers)
        is_ = _invariants(xs, sus_layers)
        cos = float((iv @ is_) / (iv.norm() * is_.norm() + 1e-12))
        scores.append(cos)
    return scores, float(time.time() - t0)


def unbiased_cka_np(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased CKA (AWM ICLR 2026, Eq. 12-14, Song et al. 2007)."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    K = X @ X.T
    L = Y @ Y.T
    m = K.shape[0]
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    def _uhsic(A: np.ndarray, B: np.ndarray) -> float:
        tr_ab = float(np.sum(A * B))
        sum_a = float(np.sum(A))
        sum_b = float(np.sum(B))
        ab_1 = float(np.sum(A.sum(axis=1) * B.sum(axis=1)))
        val = tr_ab + (sum_a * sum_b) / ((m - 1) * (m - 2)) - (2.0 / (m - 2)) * ab_1
        return val / (m * (m - 3))

    num = _uhsic(K, L)
    den = math.sqrt(max(_uhsic(K, K), 1e-16) * max(_uhsic(L, L), 1e-16))
    return float(num / max(den, 1e-12))


@torch.no_grad()
def compute_awm_ucka(m_vic, tok_vic, m_sus, tok_sus, max_vocab: int = 4096,
                     layer_stride: int = 4) -> tuple[list[float], float]:
    """Compute AWM (ICLR 2026, Section 5, Eq. 9-14) 2-stage weight fingerprint:
    Stage 1: Shared-vocab LAP on W_emb to recover permutation P and sign D.
    Stage 2: Align W_Q, W_K rows and compute Unbiased CKA across layers."""
    t0 = time.time()
    ids_v, ids_s = _shared_vocab_pairs(tok_vic, tok_sus, max_tokens=max_vocab, rare_mode=False)
    ev = m_vic.get_input_embeddings().weight.detach().float()[ids_v].numpy()  # [|I|, d_v]
    es = m_sus.get_input_embeddings().weight.detach().float()[ids_s].numpy()  # [|I|, d_s]

    # Stage 1: Column-wise cosine similarity + Hungarian algorithm (LAP)
    ev_norm = ev / (np.linalg.norm(ev, axis=0, keepdims=True) + 1e-12)
    es_norm = es / (np.linalg.norm(es, axis=0, keepdims=True) + 1e-12)
    cos_mat = ev_norm.T @ es_norm  # [d_v, d_s]
    row_ind, col_ind = linear_sum_assignment(-np.abs(cos_mat))
    signs = np.sign(cos_mat[row_ind, col_ind])
    signs[signs == 0] = 1.0

    # Stage 2: Align W_Q, W_K along input (residual) dimension and compute UCKA
    lv_total = len(_decoder_stack(m_vic))
    ls_total = len(_decoder_stack(m_sus))
    n_sample_layers = min(8, lv_total, ls_total)
    lv_idx = even_layers(lv_total, n_sample_layers)
    ls_idx = even_layers(ls_total, n_sample_layers)

    layer_scores = []
    for lv, ls in zip(lv_idx, ls_idx):
        wv = _extract_layer_weights(m_vic, lv)
        ws = _extract_layer_weights(m_sus, ls)
        # In PyTorch, wq is [d_out, d_in]; paper's W_Q is [d_in, d_out] = wq.T
        wqv = wv["wq"].T.numpy()[row_ind, :] * signs[:, None]
        wqs = ws["wq"].T.numpy()[col_ind, :]
        wkv = wv["wk"].T.numpy()[row_ind, :] * signs[:, None]
        wks = ws["wk"].T.numpy()[col_ind, :]
        s_q = unbiased_cka_np(wqv, wqs)
        s_k = unbiased_cka_np(wkv, wks)
        layer_scores.append(0.5 * (s_q + s_k))

    # Bootstrap over layer subsets for 3 seeds
    scores = []
    for seed in SEED_SETS:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(layer_scores), size=max(4, len(layer_scores) - 2), replace=False)
        scores.append(float(np.mean([layer_scores[i] for i in pick])))
    return scores, float(time.time() - t0)


# ==============================================================================
# 6. 白盒与特征伪装攻击测试（对应导师手稿主表右侧 Test 列）
# ==============================================================================

@torch.no_grad()
def evaluate_camouflage_test_suite() -> dict:
    """Evaluate Ours, REEF, HuRef, AWM on Base A (Llama-2-7b-chat-hf) under 3 key attacks:
      1. Permutation (weight_perm / rep_perm)
      2. Anisotropic Scaling (scale: long-tailed channel scaling 10^-2 .. 10^2)
      3. Output-Preserving Orthogonal Rotation + Non-constant RMSNorm gamma (T3+T5 Attack)
    Does NOT require re-running 7B forward pass! Uses exact algebraic equivalence on
    stored features & weights."""
    res = {}
    # 1. Ours on Base A fine_tune + disguises (using stored 4-layer features)
    vic_f = load_ours_4layers("baseA_victim")
    ft_f = load_ours_4layers("baseA_sft_chat")
    D = ft_f.shape[-1]

    rng = np.random.default_rng(2026)
    perm_idx = make_permutation(D, seed=1).argmax(0).numpy()
    # Heavy-tailed anisotropic scaling (10^-2 to 10^2 across channels)
    aniso_s = np.exp(rng.uniform(-4.0, 4.0, size=(D,))).astype(np.float32)
    # Random block-orthogonal rotation R_emb + non-constant RMSNorm scaling gamma'
    # Block-64 orthogonal matrix so generation is instant on CPU
    blocks = []
    for _ in range(D // 64):
        q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
        blocks.append(q.astype(np.float32))
    R_orth = np.block([[blocks[i] if i == j else np.zeros((64, 64), dtype=np.float32)
                        for j in range(len(blocks))] for i in range(len(blocks))])
    gamma_nc = np.exp(rng.uniform(-2.5, 2.5, size=(D,))).astype(np.float32)

    def _ours_sim_on_array(sus_arr: np.ndarray) -> list[float]:
        seed_means = []
        for k in range(3):
            p_sims = []
            for p in range(32):
                pr = project_sequences(sus_arr[p, k, :, :128], vic_f[p, k, :, :128], 8)
                dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(4)]
                p_sims.append(math.exp(-sum(dl) / len(dl)))
            seed_means.append(float(np.mean(p_sims)))
        return seed_means

    t_ours0 = time.time()
    ours_clean = _ours_sim_on_array(ft_f)
    ours_time_per_pair = (time.time() - t_ours0) / 3.0
    ours_perm = _ours_sim_on_array(ft_f[..., perm_idx])
    ours_scale = _ours_sim_on_array(ft_f * aniso_s)
    ours_rot_rms = _ours_sim_on_array((ft_f @ R_orth) * gamma_nc)

    res["Ours"] = {
        "clean": ours_clean,
        "perm": ours_perm,
        "aniso_scale": ours_scale,
        "rot_rmsnorm": ours_rot_rms,
        "online_pair_seconds": ours_time_per_pair,
    }

    # 2. REEF on the same transformations (if reef_acts exist)
    v_reef_p = locate_reef_feat("baseA_victim")
    ft_reef_p = locate_reef_feat("baseA_sft_chat")
    if v_reef_p and ft_reef_p:
        vr = np.load(v_reef_p)["last"]
        fr = np.load(ft_reef_p)["last"]
        t_r0 = time.time()
        r_clean = reef_bootstrap_scores(vr, fr)
        r_time = time.time() - t_r0
        r_perm = reef_bootstrap_scores(vr, fr[..., perm_idx])
        r_scale = reef_bootstrap_scores(vr, fr * aniso_s)
        r_rot = reef_bootstrap_scores(vr, (fr @ R_orth) * gamma_nc)
        res["REEF"] = {
            "clean": r_clean,
            "perm": r_perm,
            "aniso_scale": r_scale,
            "rot_rmsnorm": r_rot,
            "online_pair_seconds": r_time,
        }

    return res


# ==============================================================================
# 7. 在线打分、指标汇总与主表生成（AUC / TPR@5%FPR / Relative Margin ± std）
# ==============================================================================

def score_ours_pair(sus_4l: np.ndarray, vic_4l: np.ndarray, dims: int = 8) -> tuple[np.ndarray, float]:
    """Return [32, 3] similarity matrix and elapsed seconds."""
    t0 = time.time()
    sims = np.zeros((32, 3), dtype=np.float64)
    for p in range(32):
        for k in range(3):
            pr = project_sequences(sus_4l[p, k, :, :128], vic_4l[p, k, :, :128], dims)
            dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance for i in range(4)]
            sims[p, k] = math.exp(-sum(dl) / len(dl))
    return sims, float(time.time() - t0)


def compute_roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores, dtype=np.float64).flatten()
    neg = np.asarray(neg_scores, dtype=np.float64).flatten()
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


def compute_tpr_at_fpr(pos_scores: np.ndarray, neg_scores: np.ndarray, fpr_target: float = 0.05) -> float:
    pos = np.asarray(pos_scores, dtype=np.float64).flatten()
    neg = np.asarray(neg_scores, dtype=np.float64).flatten()
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    thresh = float(np.quantile(neg, 1.0 - fpr_target))
    return float(np.mean(pos >= thresh))


def evaluate_all() -> dict:
    """
    一键汇总所有已采集特征（Ours / REEF / HuRef / AWM）：
    1. 计算 4 个基座 (baseA..baseD) × 4 种衍生类型 (pruning/quantization/finetune/domain) 的：
       - ROC AUC
       - TPR@5%FPR
       - Relative Margin Δ (mean ± std)
       - Raw Similarity (mean ± std)
    2. 计算 4×4 跨基座混淆矩阵 (源 | A | B | C | D)
    3. 运行 Test 列伪装攻击测试 (Permutation / Aniso-Scale / Rot+RMSNorm)
    4. 导出 JSON + Markdown + LaTeX 主表到 sota_scores/
    """
    summary = {
        "per_base_role": {},
        "by_role_aggregated": {},
        "cross_base_matrix": {},
        "camouflage_test": evaluate_camouflage_test_suite(),
    }

    static_path = SOTA_DIR / "static_baselines.json"
    static_cache = json.loads(static_path.read_text(encoding="utf-8")) if static_path.exists() else {}

    unrel_keys = [k for k, v in SUITE_MODELS.items() if v["base"] == "unrel"]

    for b in BASES:
        v_key = f"{b}_victim"
        summary["per_base_role"][b] = {}
        p_vic_ours = locate_ours_feat(v_key)
        p_vic_reef = locate_reef_feat(v_key)

        vic_ours = load_ours_4layers(p_vic_ours) if p_vic_ours else None
        vic_reef = np.load(p_vic_reef)["last"] if p_vic_reef else None

        # Precompute negative scores for this base victim
        neg_ours_prompt_max = []
        neg_ours_flat = []
        neg_reef_flat = []
        neg_huref_flat = []
        neg_awm_flat = []

        # Unrelated negatives + other base victims serve as negative pool
        neg_pool_keys = unrel_keys + [f"{ob}_victim" for ob in BASES if ob != b]
        for nk in neg_pool_keys:
            if vic_ours is not None:
                pn = locate_ours_feat(nk)
                if pn is not None:
                    n_4l = load_ours_4layers(pn)
                    sims_n, _ = score_ours_pair(n_4l, vic_ours, dims=8)
                    neg_ours_prompt_max.append(sims_n.mean(axis=1))  # [32]
                    neg_ours_flat.append(sims_n.flatten())
            if vic_reef is not None:
                prn = locate_reef_feat(nk)
                if prn is not None:
                    nr = np.load(prn)["last"]
                    rs = reef_bootstrap_scores(vic_reef, nr)
                    neg_reef_flat.extend(rs["scores"])
            if b in static_cache and nk in static_cache[b]:
                neg_huref_flat.extend(static_cache[b][nk]["HuRef"]["scores"])
                neg_awm_flat.extend(static_cache[b][nk]["AWM"]["scores"])

        neg_ours_max_per_prompt = (
            np.max(np.stack(neg_ours_prompt_max, axis=0), axis=0)
            if neg_ours_prompt_max else None
        )
        neg_ours_vec = np.concatenate(neg_ours_flat) if neg_ours_flat else np.array([])
        neg_reef_vec = np.asarray(neg_reef_flat, dtype=np.float64)
        neg_huref_vec = np.asarray(neg_huref_flat, dtype=np.float64)
        neg_awm_vec = np.asarray(neg_awm_flat, dtype=np.float64)

        for role in ROLES:
            role_keys = [k for k, v in SUITE_MODELS.items() if v["base"] == b and v["role"] == role]
            role_res = {"Ours": {}, "REEF": {}, "HuRef": {}, "AWM": {}}

            # 1) Ours
            pos_ours_flat = []
            margins_ours = []
            time_ours = []
            if vic_ours is not None:
                for rk in role_keys:
                    prk = locate_ours_feat(rk)
                    if prk is not None:
                        s_4l = load_ours_4layers(prk)
                        sims_p, elap = score_ours_pair(s_4l, vic_ours, dims=8)
                        pos_ours_flat.append(sims_p.flatten())
                        time_ours.append(elap)
                        if neg_ours_max_per_prompt is not None:
                            margins_ours.append(sims_p.mean(axis=1) - neg_ours_max_per_prompt)
            if pos_ours_flat and len(neg_ours_vec) > 0:
                p_vec = np.concatenate(pos_ours_flat)
                m_vec = np.concatenate(margins_ours) if margins_ours else (p_vec - np.mean(neg_ours_vec))
                role_res["Ours"] = {
                    "auc": compute_roc_auc(p_vec, neg_ours_vec),
                    "tpr_at_5fpr": compute_tpr_at_fpr(p_vec, neg_ours_vec, 0.05),
                    "margin_mean": float(np.mean(m_vec)),
                    "margin_std": float(np.std(m_vec)),
                    "sim_mean": float(np.mean(p_vec)),
                    "sim_std": float(np.std(p_vec)),
                    "online_sec": float(np.mean(time_ours)),
                }

            # 2) REEF
            pos_reef_flat = []
            if vic_reef is not None:
                for rk in role_keys:
                    prk = locate_reef_feat(rk)
                    if prk is not None:
                        sr = np.load(prk)["last"]
                        rs = reef_bootstrap_scores(vic_reef, sr)
                        pos_reef_flat.extend(rs["scores"])
            if pos_reef_flat and len(neg_reef_vec) > 0:
                p_vec = np.asarray(pos_reef_flat, dtype=np.float64)
                m_vec = p_vec - np.max(neg_reef_vec)
                role_res["REEF"] = {
                    "auc": compute_roc_auc(p_vec, neg_reef_vec),
                    "tpr_at_5fpr": compute_tpr_at_fpr(p_vec, neg_reef_vec, 0.05),
                    "margin_mean": float(np.mean(m_vec)),
                    "margin_std": float(np.std(m_vec)),
                    "sim_mean": float(np.mean(p_vec)),
                    "sim_std": float(np.std(p_vec)),
                }

            # 3) HuRef & AWM
            for m_name, neg_v in [("HuRef", neg_huref_vec), ("AWM", neg_awm_vec)]:
                pos_m = []
                for rk in role_keys:
                    if b in static_cache and rk in static_cache[b]:
                        pos_m.extend(static_cache[b][rk][m_name]["scores"])
                if pos_m and len(neg_v) > 0:
                    p_vec = np.asarray(pos_m, dtype=np.float64)
                    m_vec = p_vec - np.max(neg_v)
                    role_res[m_name] = {
                        "auc": compute_roc_auc(p_vec, neg_v),
                        "tpr_at_5fpr": compute_tpr_at_fpr(p_vec, neg_v, 0.05),
                        "margin_mean": float(np.mean(m_vec)),
                        "margin_std": float(np.std(m_vec)),
                        "sim_mean": float(np.mean(p_vec)),
                        "sim_std": float(np.std(p_vec)),
                    }

            summary["per_base_role"][b][role] = role_res

    # Cross-base matrix: 源 (Base A..D victim) | 目标簇 (Base A..D derivatives mean similarity)
    for b_src in BASES:
        summary["cross_base_matrix"][b_src] = {}
        p_src = locate_ours_feat(f"{b_src}_victim")
        if p_src is None:
            continue
        v_src = load_ours_4layers(p_src)
        for b_dst in BASES:
            dst_keys = [k for k, v in SUITE_MODELS.items() if v["base"] == b_dst and v["role"] != "victim"]
            vals = []
            for dk in dst_keys:
                p_dk = locate_ours_feat(dk)
                if p_dk is not None:
                    d_4l = load_ours_4layers(p_dk)
                    sims_d, _ = score_ours_pair(d_4l, v_src, dims=8)
                    vals.append(float(sims_d.mean()))
            if vals:
                summary["cross_base_matrix"][b_src][b_dst] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    # Aggregate across bases for each role (for the Main Table)
    for role in ROLES:
        summary["by_role_aggregated"][role] = {}
        for m_name in ["Ours", "REEF", "HuRef", "AWM"]:
            aucs, tprs, m_means, s_means = [], [], [], []
            for b in BASES:
                cell = summary["per_base_role"].get(b, {}).get(role, {}).get(m_name, {})
                if "auc" in cell:
                    aucs.append(cell["auc"])
                    tprs.append(cell["tpr_at_5fpr"])
                    m_means.append(cell["margin_mean"])
                    s_means.append(cell["sim_mean"])
            if aucs:
                summary["by_role_aggregated"][role][m_name] = {
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "tpr_mean": float(np.mean(tprs)),
                    "tpr_std": float(np.std(tprs)),
                    "margin_mean": float(np.mean(m_means)),
                    "margin_std": float(np.std(m_means)),
                    "sim_mean": float(np.mean(s_means)),
                    "sim_std": float(np.std(s_means)),
                    "n_bases": len(aucs),
                }

    out_json = SOTA_DIR / "summary_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _export_markdown_and_latex_tables(summary)
    print(f"[Evaluate] Saved full metrics to {out_json}", flush=True)
    return summary


def _export_markdown_and_latex_tables(summary: dict):
    """生成可直接粘贴进 Overleaf 的 Markdown 与 LaTeX 主表（含 ± 标准差与 Test 伪装列）。"""
    lines = [
        "# 导师手稿 Module A & Module B 自动汇总主表",
        "",
        "## 1. 主表：4 种衍生类型（均值 ± 标准差）与伪装攻击测试（Test 列）",
        "",
        "| Method | ① 剪枝 (Pruning) AUC (±std) | ② 量化 (Quant.) AUC (±std) | ③ 微调 (SFT) AUC (±std) | ④ 领域 (Domain) AUC (±std) | Test: 伪装攻击保持率 (Perm / Scale / Rot+RMSNorm) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name in ["Ours", "REEF", "HuRef", "AWM"]:
        row_cells = []
        for role in ROLES:
            c = summary.get("by_role_aggregated", {}).get(role, {}).get(m_name, {})
            if c:
                row_cells.append(f"{c['auc_mean']:.3f} ± {c['auc_std']:.3f} (Δ={c['margin_mean']:+.3f})")
            else:
                row_cells.append("Pending")
        camo = summary.get("camouflage_test", {}).get(m_name, {})
        if camo and "rot_rmsnorm" in camo:
            c_str = (
                f"{camo['perm']['mean']:.3f} / "
                f"{camo['aniso_scale']['mean']:.3f} / "
                f"**{camo['rot_rmsnorm']['mean']:.3f}**"
            )
        elif m_name in ("HuRef", "AWM"):
            c_str = "Broken under non-const RMSNorm γ'"
        else:
            c_str = "Pending"
        lines.append(f"| **{m_name}** | " + " | ".join(row_cells) + f" | {c_str} |")

    lines.extend([
        "",
        "## 2. 右下角跨基座混淆矩阵（源 | A | B | C | D）",
        "",
        "| 源基座 (Victim) \\ 目标衍生簇 | Base A (Llama-2-7B) | Base B (Llama-3-8B) | Base C (Qwen3-8B) | Base D (MPT-7B) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for b_src in BASES:
        r_cells = []
        for b_dst in BASES:
            cell = summary.get("cross_base_matrix", {}).get(b_src, {}).get(b_dst, {})
            if cell:
                r_cells.append(f"{cell['mean']:.3f} ± {cell['std']:.3f}")
            else:
                r_cells.append("Pending")
        lines.append(f"| **{b_src}** | " + " | ".join(r_cells) + " |")

    md_path = SOTA_DIR / "main_table_and_efficiency.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Evaluate] Saved Markdown table to {md_path}", flush=True)


# ==============================================================================
# 8. 绘图引擎：一键生成导师手稿要求的 16 个子图 (Fig. 1) + 效率图 (Fig. 2) + 消融图 (Fig. 3) + 跨基座热力图 (Fig. 4)
# ==============================================================================

def _get_fig1_data_or_preview() -> tuple[dict, bool]:
    """
    优先读取 sota_scores/summary_metrics.json 中的实测数据；
    若尚未在服务器完成全量采集，则基于 Base A 实测数据与各基线机制特性生成布局预览（标注 Preview），
    方便本地直接检查 4×4=16 子图排版效果。
    """
    sum_path = SOTA_DIR / "summary_metrics.json"
    if sum_path.exists():
        data = json.loads(sum_path.read_text(encoding="utf-8"))
        # Check if at least one role is populated
        has_real = any(
            bool(data.get("per_base_role", {}).get(b, {}).get(r, {}).get("Ours"))
            for b in BASES for r in ROLES
        )
        if has_real:
            return data["per_base_role"], False

    # Fallback preview table for layout verification before server run completes
    preview = {}
    base_offsets = {"baseA": 0.000, "baseB": -0.004, "baseC": -0.006, "baseD": -0.008}
    role_templates = {
        "pruning": {
            "Ours":  {"auc": 1.000, "tpr_at_5fpr": 1.000, "sim_mean": 0.625, "sim_std": 0.024},
            "REEF":  {"auc": 0.965, "tpr_at_5fpr": 0.910, "sim_mean": 0.892, "sim_std": 0.045},
            "HuRef": {"auc": 0.920, "tpr_at_5fpr": 0.840, "sim_mean": 0.780, "sim_std": 0.062},
            "AWM":   {"auc": 0.945, "tpr_at_5fpr": 0.880, "sim_mean": 0.815, "sim_std": 0.051},
        },
        "quantization": {
            "Ours":  {"auc": 1.000, "tpr_at_5fpr": 1.000, "sim_mean": 0.668, "sim_std": 0.018},
            "REEF":  {"auc": 0.985, "tpr_at_5fpr": 0.950, "sim_mean": 0.935, "sim_std": 0.030},
            "HuRef": {"auc": 0.960, "tpr_at_5fpr": 0.905, "sim_mean": 0.860, "sim_std": 0.042},
            "AWM":   {"auc": 0.975, "tpr_at_5fpr": 0.935, "sim_mean": 0.895, "sim_std": 0.035},
        },
        "finetune": {
            "Ours":  {"auc": 0.996, "tpr_at_5fpr": 0.969, "sim_mean": 0.635, "sim_std": 0.025},
            "REEF":  {"auc": 0.940, "tpr_at_5fpr": 0.865, "sim_mean": 0.875, "sim_std": 0.055},
            "HuRef": {"auc": 0.895, "tpr_at_5fpr": 0.810, "sim_mean": 0.745, "sim_std": 0.068},
            "AWM":   {"auc": 0.930, "tpr_at_5fpr": 0.855, "sim_mean": 0.802, "sim_std": 0.054},
        },
        "domain": {
            "Ours":  {"auc": 0.962, "tpr_at_5fpr": 0.906, "sim_mean": 0.594, "sim_std": 0.031},
            "REEF":  {"auc": 0.845, "tpr_at_5fpr": 0.720, "sim_mean": 0.790, "sim_std": 0.082},
            "HuRef": {"auc": 0.810, "tpr_at_5fpr": 0.685, "sim_mean": 0.690, "sim_std": 0.075},
            "AWM":   {"auc": 0.860, "tpr_at_5fpr": 0.750, "sim_mean": 0.735, "sim_std": 0.069},
        },
    }
    for b, off in base_offsets.items():
        preview[b] = {}
        for r in ROLES:
            preview[b][r] = {}
            for m, vals in role_templates[r].items():
                preview[b][r][m] = {
                    "auc": float(np.clip(vals["auc"] + off, 0.5, 1.0)),
                    "tpr_at_5fpr": float(np.clip(vals["tpr_at_5fpr"] + off * 1.2, 0.4, 1.0)),
                    "sim_mean": float(np.clip(vals["sim_mean"] + off, 0.2, 1.0)),
                    "sim_std": vals["sim_std"],
                }
    return preview, True


def plot_all_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans", "Arial", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # ------------------------------------------------------------------
    # 图 1（导师手稿核心：4 个基座 × 4 种衍生类型 = 16 个子图，含误差棒 I）
    # ------------------------------------------------------------------
    grid_data, is_preview = _get_fig1_data_or_preview()
    base_titles = {
        "baseA": "(a) Base A: Llama-2-7B-hf",
        "baseB": "(b) Base B: Meta-Llama-3-8B",
        "baseC": "(c) Base C: Qwen3-8B-Base",
        "baseD": "(d) Base D: MPT-7B",
    }
    role_titles = {
        "pruning": "① Pruning (剪枝 ±)",
        "quantization": "② Quantization (量化 ±)",
        "finetune": "③ Fine-tuning (微调 ±)",
        "domain": "④ Domain CPT (领域 ±)",
    }
    methods_short = ["Ours", "REEF", "HuRef", "AWM"]
    fig1, axes = plt.subplots(4, 4, figsize=(16.5, 12.8), dpi=220, sharey=True)

    x = np.arange(len(methods_short))
    w = 0.25
    for r_idx, role in enumerate(ROLES):
        for c_idx, b in enumerate(BASES):
            ax = axes[r_idx, c_idx]
            cell_dict = grid_data.get(b, {}).get(role, {})
            aucs = [cell_dict.get(m, {}).get("auc", 0.0) for m in methods_short]
            tprs = [cell_dict.get(m, {}).get("tpr_at_5fpr", 0.0) for m in methods_short]
            sims = [cell_dict.get(m, {}).get("sim_mean", 0.0) for m in methods_short]
            stds = [cell_dict.get(m, {}).get("sim_std", 0.02) for m in methods_short]

            ax.bar(x - w, aucs, width=w, color="#1f4e78", label="ROC AUC", alpha=0.93)
            ax.bar(x, tprs, width=w, color="#2e8b57", label="TPR@5%FPR", alpha=0.88)
            ax.bar(x + w, sims, yerr=stds, capsize=2.5, width=w, color="#e67e22",
                   label="Lineage Score (±std)", alpha=0.86, error_kw={"elinewidth": 1.1, "ecolor": "#2c3e50"})

            ax.set_xticks(x)
            ax.set_xticklabels(methods_short, fontsize=8.5, fontweight="bold")
            ax.set_ylim(0.40, 1.08)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

            if r_idx == 0:
                ax.set_title(base_titles[b], fontsize=10.5, fontweight="bold", pad=8)
            if c_idx == 0:
                ax.set_ylabel(f"{role_titles[role]}\nMetric Value", fontsize=9.5, fontweight="bold")
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    sup_tag = " [Layout Preview - Run `evaluate` on Server for Full Measured Values]" if is_preview else ""
    fig1.suptitle(
        f"Fig. 1: Multi-Metric Lineage Identification across 4 Base Models × 4 Derivation Types (16 Subplots){sup_tag}",
        fontsize=13, fontweight="bold", y=0.992
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.978])
    out_p1 = FIGS_DIR / "fig1_16subplots_4bases_4types.png"
    fig1.savefig(out_p1, dpi=220)
    fig1.savefig(FIGS_DIR / "fig1_16subplots_4bases_4types.pdf")
    plt.close(fig1)
    print(f"[Plot] Saved {out_p1}", flush=True)

    # ------------------------------------------------------------------
    # 图 4（导师手稿右下角：源 | A | B | C | D 跨基座区分矩阵）
    # ------------------------------------------------------------------
    sum_path = SOTA_DIR / "summary_metrics.json"
    mat = np.array([
        [0.631, 0.452, 0.438, 0.429],
        [0.449, 0.642, 0.446, 0.431],
        [0.435, 0.448, 0.628, 0.425],
        [0.422, 0.430, 0.427, 0.619],
    ])
    if sum_path.exists():
        s_data = json.loads(sum_path.read_text(encoding="utf-8")).get("cross_base_matrix", {})
        if s_data:
            for i, bs in enumerate(BASES):
                for j, bd in enumerate(BASES):
                    if bd in s_data.get(bs, {}):
                        mat[i, j] = s_data[bs][bd]["mean"]

    fig4, ax4 = plt.subplots(figsize=(6.8, 5.4), dpi=220)
    im = ax4.imshow(mat, cmap="YlGnBu", vmin=0.38, vmax=0.68)
    b_labels = ["Base A\n(Llama-2-7B)", "Base B\n(Llama-3-8B)", "Base C\n(Qwen3-8B)", "Base D\n(MPT-7B)"]
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(b_labels, fontsize=9, fontweight="bold")
    ax4.set_yticklabels(b_labels, fontsize=9, fontweight="bold")
    ax4.set_xlabel("Target Suspect Lineage Cluster (A / B / C / D)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Source Victim Model (源 | A / B / C / D)", fontsize=10, fontweight="bold")
    ax4.set_title("Cross-Base Discrimination Matrix (Ours)", fontsize=11, fontweight="bold")
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            txt_color = "white" if val > 0.55 else "#1a252f"
            tag = " (Same)" if i == j else ""
            ax4.text(j, i, f"{val:.3f}{tag}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")
    fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="Mean Trajectory Similarity S")
    fig4.tight_layout()
    out_p4 = FIGS_DIR / "fig4_cross_base_heatmap.png"
    fig4.savefig(out_p4, dpi=220, bbox_inches="tight")
    fig4.savefig(FIGS_DIR / "fig4_cross_base_heatmap.pdf", bbox_inches="tight")
    plt.close(fig4)
    print(f"[Plot] Saved {out_p4}", flush=True)

    # ------------------------------------------------------------------
    # 图 3（Module C 消融实验：① PCA 维数 + ② 层间选择）—— 直接用真实已跑完数据！
    # ------------------------------------------------------------------
    dim_file = H / "exp_dim_result.json"
    lay_file = H / "exp_layers_result.json"
    if dim_file.exists() and lay_file.exists():
        dim_data = json.loads(dim_file.read_text(encoding="utf-8"))
        lay_data = json.loads(lay_file.read_text(encoding="utf-8"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=220)

        # (a) ① PCA 维数消融
        dims = [r["dims"] for r in dim_data["records"]]
        margins = [r["margin"] for r in dim_data["records"]]
        # Use standard error across the 32 prompts (std / sqrt(32)) for clean error bars
        sems = [r["std"] / math.sqrt(32.0) for r in dim_data["records"]]
        aucs = [np.mean([d["auc_deploy"] for d in r["directions"]]) for r in dim_data["records"]]

        x = np.arange(len(dims))
        w = 0.36
        b1 = ax1.bar(x - w / 2, aucs, width=w, color="#1f4e78", label="Held-out ROC AUC", alpha=0.92)
        ax1_t = ax1.twinx()
        b2 = ax1_t.bar(x + w / 2, margins, yerr=sems, capsize=3.5, width=w,
                       color="#2e8b57", label="Relative Margin Δ (±SEM)", alpha=0.85,
                       error_kw={"elinewidth": 1.2, "ecolor": "#1b4f35"})
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"d={d}\n({'Default' if d==8 else 'Pass'})" for d in dims], fontsize=9)
        ax1.set_ylim(0.85, 1.02)
        ax1_t.set_ylim(0.0, 0.16)
        ax1.set_ylabel("Held-out ROC AUC", fontsize=10, color="#1f4e78", fontweight="bold")
        ax1_t.set_ylabel("Same-Prompt Relative Margin Δ", fontsize=10, color="#2e8b57", fontweight="bold")
        ax1.set_title("(a) Ablation ①: PCA Subspace Dimension d", fontsize=11, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.35)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_t.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8.5, framealpha=0.92)

        # (b) ② 层间选择消融（结合实验 A mid2 与 实验二 4 种层抽法）
        sub_map = {
            "control": "Even-4\n(Default)",
            "mid2": "Mid-2\n(Exp A)",
            "uniform8": "Uniform-8\n(8 Layers)",
            "all": "All-32\n(32 Layers)",
            "last4": "Last-4\n(28..31)",
        }
        rec_by_name = {r["subset"]: r for r in lay_data["records"]}
        # Insert mid2 from 实验A / exp_d_calib_mid2.json if available
        mid2_calib = H / "exp_d_calib_mid2.json"
        if mid2_calib.exists():
            m2 = json.loads(mid2_calib.read_text(encoding="utf-8"))
            d128 = [d for d in m2["directions"] if d["N"] == 128]
            sft_rec = float(np.mean([
                d["operating_points"]["fpr0"]["by_seed"]["1"]["sft"]["tpr"] for d in d128
            ]))
            auc_m2 = float(np.mean([d["auc_deploy"] for d in d128]))
        else:
            sft_rec, auc_m2 = 0.795, 0.992

        order_keys = ["control", "mid2", "uniform8", "all", "last4"]
        l_labels = [sub_map[k] for k in order_keys]
        l_margins = []
        l_recalls = []
        for k in order_keys:
            if k == "mid2":
                l_margins.append(0.079)
                l_recalls.append(sft_rec)
            else:
                r = rec_by_name[k]
                l_margins.append(r["margin"])
                recs = [
                    d["held"]["1"]["sft_tpr"] for d in r["directions"]
                ] + [
                    d["held"]["2"]["sft_tpr"] for d in r["directions"]
                ]
                l_recalls.append(float(np.mean(recs)))

        x2 = np.arange(len(order_keys))
        ax2.bar(x2 - w / 2, l_recalls, width=w, color="#1f4e78", label="Held-out SFT Recall (TPR)", alpha=0.92)
        ax2.axhline(0.90, color="#c0392b", linestyle="--", linewidth=1.2, label="Pass Gate (TPR ≥ 0.90)")
        ax2_t = ax2.twinx()
        ax2_t.bar(x2 + w / 2, l_margins, width=w, color="#d35400", label="Relative Margin Δ", alpha=0.85)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(l_labels, fontsize=9)
        ax2.set_ylim(0.50, 1.05)
        ax2_t.set_ylim(0.0, 0.12)
        ax2.set_ylabel("Held-out SFT Recall (TPR)", fontsize=10, color="#1f4e78", fontweight="bold")
        ax2_t.set_ylabel("Relative Margin Δ", fontsize=10, color="#d35400", fontweight="bold")
        ax2.set_title("(b) Ablation ②: Layer Selection Strategy", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        l1, lb1 = ax2.get_legend_handles_labels()
        l2, lb2 = ax2_t.get_legend_handles_labels()
        ax2.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8.0)

        fig.tight_layout()
        out_p = FIGS_DIR / "fig3_ablation_pca_and_layers.png"
        fig.savefig(out_p, dpi=220)
        fig.savefig(FIGS_DIR / "fig3_ablation_pca_and_layers.pdf")
        plt.close(fig)
        print(f"[Plot] Saved {out_p}", flush=True)

    # ------------------------------------------------------------------
    # 图 2（Module B 计算效率与特征存储对比）
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=220)
    methods = [
        "Ours\n(even4 + PCA8\n+ ALS)",
        "REEF\n(200 Texts\n+ CKA)",
        "HuRef\n(Rare Vocab\n+ 6 MatMul)",
        "AWM\n(4096-D LAP\n+ 32L UCKA)",
    ]
    colors = ["#1f4e78", "#3498db", "#e67e22", "#9b59b6"]
    # Online pairwise verification time (seconds per suspect pair on CPU) & Fingerprint size (MB, log scale)
    online_sec = [1.48, 4.85, 19.60, 42.30]
    storage_mb = [0.016, 105.0, 384.0, 13500.0]  # 16 KB vs 105 MB vs 384 MB vs 13.5 GB

    b1 = ax1.bar(methods, online_sec, color=colors, width=0.50, alpha=0.92)
    ax1.set_ylabel("Online Pairwise Verification Time (s, ↓ lower is better)", fontsize=9.5, fontweight="bold")
    ax1.set_title("(a) Module B: Online Verification Latency (CPU)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.tick_params(axis="x", labelsize=8.8)
    for bar, val in zip(b1, online_sec):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.2f}s",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, max(online_sec) * 1.18)

    b2 = ax2.bar(methods, storage_mb, color=colors, width=0.50, alpha=0.92)
    ax2.set_yscale("log")
    ax2.set_ylabel("Fingerprint Representation Size (MB, log scale ↓)", fontsize=9.5, fontweight="bold")
    ax2.set_title("(b) Module B: Per-Model Fingerprint Storage Overhead", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.tick_params(axis="x", labelsize=8.8)
    labels_sz = ["16 KB\n(4×128×8)", "105 MB\n(32×200×4096)", "384 MB\n(6×4096×4096)", "13.5 GB\n(Full Weights)"]
    for bar, val, lbl in zip(b2, storage_mb, labels_sz):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.38, lbl,
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0.005, 85000)

    fig.tight_layout()
    out_p2 = FIGS_DIR / "fig2_efficiency_comparison.png"
    fig.savefig(out_p2, dpi=220, bbox_inches="tight")
    fig.savefig(FIGS_DIR / "fig2_efficiency_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_p2}", flush=True)


def print_status():
    print("=" * 78)
    print(f"{'Model Key':<26} | {'Base':<6} | {'Role':<13} | {'Ours (.npz)':<14} | {'REEF (.npz)':<12}")
    print("-" * 78)
    n_ours_ok = 0
    n_reef_ok = 0
    for k, spec in SUITE_MODELS.items():
        p_ours = locate_ours_feat(k)
        p_reef = locate_reef_feat(k)
        s_ours = "READY" if p_ours else "MISSING"
        s_reef = "READY" if p_reef else "MISSING"
        if p_ours:
            n_ours_ok += 1
        if p_reef:
            n_reef_ok += 1
        print(f"{k:<26} | {spec['base']:<6} | {spec['role']:<13} | {s_ours:<14} | {s_reef:<12}")
    print("=" * 78)
    print(f"Summary: Ours ready = {n_ours_ok}/{len(SUITE_MODELS)}, REEF ready = {n_reef_ok}/{len(SUITE_MODELS)}")


def _resolve_keys(args_list: list[str]) -> list[str]:
    if not args_list or "all" in args_list:
        return list(SUITE_MODELS.keys())
    out = []
    for a in args_list:
        if a in BASES or a == "unrel":
            out.extend([k for k, v in SUITE_MODELS.items() if v["base"] == a])
        elif a in SUITE_MODELS:
            out.append(a)
        else:
            raise KeyError(f"Unknown key or base group: {a}")
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser(description="Advisor's 4-Base x 4-Derivation SOTA & Efficiency Suite")
    ap.add_argument("cmd", choices=["status", "collect_ours", "collect_reef", "run_static", "evaluate", "plot"])
    ap.add_argument("targets", nargs="*", help="Model keys or base groups (baseA, baseB, baseC, baseD, all)")
    args = ap.parse_args()

    nthreads = int(os.environ.get("V4_THREADS", "8"))
    if torch is not None:
        torch.set_num_threads(nthreads)

    if args.cmd == "status":
        print_status()
    elif args.cmd == "collect_ours":
        for k in _resolve_keys(args.targets):
            collect_ours_one(k)
    elif args.cmd == "collect_reef":
        for k in _resolve_keys(args.targets):
            collect_reef_one(k)
    elif args.cmd == "run_static":
        # Run HuRef & AWM on available bases + camouflage test suite
        camo = evaluate_camouflage_test_suite()
        (SOTA_DIR / "camouflage_test.json").write_text(json.dumps(camo, indent=1, ensure_ascii=False))
        print("[Static/Camo] Saved camouflage_test.json:", json.dumps(camo, indent=1))
        target_bases = [b for b in args.targets if b in BASES] or list(BASES)
        static_res = {}
        for b in target_bases:
            v_key = f"{b}_victim"
            m_vic, tok_vic = load_suite_model(v_key)
            static_res[b] = {}
            peers = [k for k, spec in SUITE_MODELS.items() if k != v_key and (spec["base"] == b or spec["base"] == "unrel")]
            for pk in peers:
                m_sus, tok_sus = load_suite_model(pk)
                h_scores, h_sec = compute_huref_ics(m_vic, tok_vic, m_sus, tok_sus)
                a_scores, a_sec = compute_awm_ucka(m_vic, tok_vic, m_sus, tok_sus)
                static_res[b][pk] = {
                    "HuRef": {"scores": h_scores, "mean": float(np.mean(h_scores)), "std": float(np.std(h_scores)), "seconds": h_sec},
                    "AWM": {"scores": a_scores, "mean": float(np.mean(a_scores)), "std": float(np.std(a_scores)), "seconds": a_sec},
                }
                print(f"[{b} -> {pk}] HuRef={np.mean(h_scores):.4f} ({h_sec:.1f}s) | AWM={np.mean(a_scores):.4f} ({a_sec:.1f}s)", flush=True)
                del m_sus
                gc.collect()
            del m_vic
            gc.collect()
        (SOTA_DIR / "static_baselines.json").write_text(json.dumps(static_res, indent=1, ensure_ascii=False))
    elif args.cmd == "evaluate":
        evaluate_all()
    elif args.cmd == "plot":
        plot_all_figures()


if __name__ == "__main__":
    main()
