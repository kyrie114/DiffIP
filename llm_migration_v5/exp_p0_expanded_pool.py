"""P0 扩展验证套件：模型融合/合并衍生 (FuseLLM-7B, Abel-7B-002) 与 16+ 异构大负例池。

完全复现论文与 suite_sota.py 的采集与打分口径（Zero-download，直接读取本地模型目录）：
  1. 复用 baseA_victim (Llama-2-7b-hf) 与原 5 个校准锚点特征；
  2. 支持多架构解码层与隐状态排布兼容（Llama / Qwen / InternLM2 / BLOOM / Pythia / OPT / OLMo2 / ChatGLM2 / Mixtral）；
  3. 对新负例严格应用 sota_scores/matched_rule.json 中冻结的 20 方向 tau_0 阈值评估零泄漏 FPR；
  4. 同步采集 REEF TruthfulQA-200 激活，报告 full200 / bootstrap / 100-100 Even-Odd 留出对照。

用法:
  python exp_p0_expanded_pool.py status [--include-large]
  python exp_p0_expanded_pool.py collect_ours [all | key...] [--include-large]
  python exp_p0_expanded_pool.py collect_reef [all | key...] [--include-large]
  python exp_p0_expanded_pool.py evaluate [--include-large]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

H = Path(__file__).resolve().parent
V4_DIR = H.parent / "llm_migration_v4"
sys.path.insert(0, str(H))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(H.parent))

import suite_sota as S  # noqa: E402

try:
    import torch  # noqa: E402
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
    import common as C  # noqa: E402
    HAS_TORCH = True
except ImportError:
    torch = S.torch
    AutoConfig = None
    AutoModelForCausalLM = None
    C = None
    HAS_TORCH = False

MODELS = S.MODELS
FEATS_DIR = S.FEATS_4
REEF_DIR = S.REEF_DIR
SOTA_DIR = S.SOTA_DIR
OUT_JSON = SOTA_DIR / "p0_expanded_pool_result.json"
OUT_MD = H / "docs" / "表_P0扩池与合并模型.md"

P0_MODELS = {
    # ---------------- P0-A: Llama-2-7B 融合/合并与数学衍生 ----------------
    "pos_fusellm_7b": {
        "path": f"{MODELS}/FuseLLM-7B", "role": "positive", "tier": "core",
        "family": "Model-Merging", "note": "REEF 分布融合正例 (Llama-2-7B)",
    },
    "pos_abel_7b_002": {
        "path": f"{MODELS}/Abel-7B-002", "role": "positive", "tier": "core",
        "family": "Domain-Math", "note": "Llama-2-7B 数学 CPT 衍生",
    },
    # ---------------- P0-B: 10 个新增 7B 级独立无关负例 ----------------
    "neg_openllama_7b_v2": {
        "path": f"{MODELS}/open_llama_7b_v2", "role": "negative", "tier": "core",
        "family": "OpenLLaMA-v2", "note": "同架构独立训练",
    },
    "neg_internlm2_base_7b": {
        "path": f"{MODELS}/internlm2-base-7b", "role": "negative", "tier": "core",
        "family": "InternLM2", "note": "书生 InternLM2 基座",
    },
    "neg_olmo2_7b": {
        "path": f"{MODELS}/OLMo-2-1124-7B-Instruct", "role": "negative", "tier": "core",
        "family": "OLMo-2", "note": "AI2 完全开源预训练族",
    },
    "neg_bloom_7b1": {
        "path": f"{MODELS}/bloom-7b1", "role": "negative", "tier": "core",
        "family": "BLOOM", "note": "ALiBi + LayerNorm 异构家族",
    },
    "neg_pythia_6_9b": {
        "path": f"{MODELS}/pythia-6.9b", "role": "negative", "tier": "core",
        "family": "Pythia", "note": "GPT-NeoX 并行残差族",
    },
    "neg_opt_6_7b": {
        "path": f"{MODELS}/opt-6.7b", "role": "negative", "tier": "core",
        "family": "OPT", "note": "Meta OPT 异构家族",
    },
    "neg_redpajama_7b": {
        "path": f"{MODELS}/RedPajama-INCITE-7B-Base", "role": "negative", "tier": "core",
        "family": "RedPajama", "note": "GPT-NeoX 开源预训练族",
    },
    "neg_chatglm2_6b": {
        "path": f"{MODELS}/chatglm2-6b", "role": "negative", "tier": "core",
        "family": "ChatGLM2", "note": "智谱 GLM 旋转编码族",
    },
    "neg_openchat_3_5": {
        "path": f"{MODELS}/openchat_3.5", "role": "negative", "tier": "core",
        "family": "Mistral-deriv", "note": "Mistral-7B 谱系（对 Llama-2 独立）",
    },
    "neg_starling_7b": {
        "path": f"{MODELS}/Starling-LM-7B-alpha", "role": "negative", "tier": "core",
        "family": "Mistral-deriv", "note": "Mistral-7B RLHF 谱系（对 Llama-2 独立）",
    },
    # ---------------- P0-C: 可选大尺寸 / MoE 独立负例 ----------------
    "neg_mixtral_8x7b": {
        "path": f"{MODELS}/Mixtral-8x7B-Instruct-v0.1", "role": "negative", "tier": "large",
        "family": "Mixtral-MoE", "note": "8x7B 稀疏混合专家模型",
    },
    "neg_internlm2_20b": {
        "path": f"{MODELS}/internlm2-chat-20b-llama-old", "role": "negative", "tier": "large",
        "family": "InternLM2-20B", "note": "20B 跨参数规模负例",
    },
}

BASE_ANCHOR_TO_SUITE = {
    "neg_qwen2_5_7b": "unrel_qwen2_5_7b",
    "neg_llama3_8b": "baseB_victim",
    "neg_llama1_7b": "unrel_llama1_7b",
    "neg_baichuan_7b": "unrel_baichuan_7b",
    "neg_mpt_7b": "baseD_victim",
}
EXISTING_NEG_KEYS = list(BASE_ANCHOR_TO_SUITE.values()) + ["unrel_openllama_7b"]


def active_keys(include_large: bool = False, requested: list[str] | None = None) -> list[str]:
    if requested and requested != ["all"]:
        unknown = [k for k in requested if k not in P0_MODELS]
        if unknown:
            raise SystemExit(f"Unknown P0 model keys: {unknown}")
        return requested
    return [
        k for k, v in P0_MODELS.items()
        if include_large or v["tier"] == "core"
    ]


def locate_p0_ours(key: str) -> Path | None:
    p = FEATS_DIR / f"p0_{key}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p)
        if z["pert"].shape[:4] == (32, 3, 4, 128) and np.isfinite(z["pert"]).all():
            return p
    except Exception:
        return None
    return None


def locate_p0_reef(key: str) -> Path | None:
    p = REEF_DIR / f"p0_{key}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p)
        if z["last"].ndim == 3 and z["last"].shape[1] == 200 and np.isfinite(z["last"]).all():
            return p
    except Exception:
        return None
    return None


def _load_p0_model(mid: str):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch / transformers required for collection.")
    C._shim_removed_llama_rope()
    tok = C.load_tok(mid)
    cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    C._shim_mpt_remote_class(mid, cfg)
    rs = getattr(cfg, "rope_scaling", None)
    if (isinstance(rs, dict) and "type" not in rs
            and rs.get("rope_type", "default") == "default"
            and not type(cfg).__module__.startswith("transformers.")):
        cfg.rope_scaling = None
    name_l = Path(mid).name.lower()
    mtype = str(getattr(cfg, "model_type", "")).lower()
    if "internlm2" in name_l or mtype.startswith("internlm2"):
        cfg.attn_implementation = "sdpa"
    m = AutoModelForCausalLM.from_pretrained(
        mid, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    return m, tok


def _infer_num_layers(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    for path in (
        ("model", "layers"),
        ("transformer", "blocks"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "encoder", "layers"),
    ):
        cur = model
        for part in path:
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None and hasattr(cur, "__len__"):
            return len(cur)
    raise RuntimeError(f"Cannot infer decoder layer count for {type(model).__name__}")


def _orient_hs(tensor, batch_size: int):
    # ChatGLM2 returns [seq_len, batch, hidden]; standardize to [batch, seq_len, hidden].
    if tensor.ndim == 3 and tensor.shape[0] != batch_size and tensor.shape[1] == batch_size:
        return tensor.transpose(0, 1)
    return tensor


def _forward_pooled_4layers(model, embeds, layers: list[int]) -> np.ndarray:
    bs = embeds.shape[0]
    am = torch.ones(embeds.shape[:2], dtype=torch.long)
    try:
        with torch.no_grad():
            out = model(inputs_embeds=embeds, attention_mask=am,
                        use_cache=False, output_hidden_states=True)
        hs = out.hidden_states
        picked = torch.stack([_orient_hs(hs[i + 1], bs) for i in layers], 0)
        return picked.mean(dim=2).to(torch.float32).cpu().numpy()
    except RuntimeError as e:
        print(f"  batched forward failed ({str(e)[:80]}); falling back to batch=1", flush=True)
        res = [[] for _ in layers]
        for s in range(bs):
            with torch.no_grad():
                out = model(inputs_embeds=embeds[s:s + 1], attention_mask=am[s:s + 1],
                            use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for k, i in enumerate(layers):
                h_s = _orient_hs(hs[i + 1], 1)
                res[k].append(h_s[0].mean(dim=0))
        return torch.stack([torch.stack(r, 0) for r in res], 0).to(torch.float32).cpu().numpy()


def _forward_last_all_layers(model, embed_tensor) -> np.ndarray:
    bs = embed_tensor.shape[0]
    am = torch.ones(embed_tensor.shape[:2], dtype=torch.long)
    with torch.no_grad():
        out = model(inputs_embeds=embed_tensor, attention_mask=am,
                    use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    return torch.stack([_orient_hs(h, bs)[0, -1] for h in hs[1:]], 0).to(torch.float32).cpu().numpy()


def collect_ours_one(key: str):
    if locate_p0_ours(key) is not None:
        print(f"[P0-Ours] skip {key} (cached)", flush=True)
        return
    spec = P0_MODELS[key]
    out = FEATS_DIR / f"p0_{key}.npz"
    partial = FEATS_DIR / f"p0_{key}.partial.npz"
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("V4_THREADS", "8")))
    m, tok = _load_p0_model(spec["path"])
    total = _infer_num_layers(m)
    L4 = S.even_layers(total, S.N_LAYERS)
    E = m.get_input_embeddings()
    D_ = int(E.weight.shape[1])
    shape = (len(C.PROMPTS), len(C.SEED_SETS), len(L4), C.N_MAX, D_)
    feats = np.zeros(shape, dtype=np.float32)
    start_p = 0
    if partial.exists():
        z = np.load(partial)
        if tuple(z["pert"].shape) == shape:
            feats[:] = z["pert"]
            start_p = int(z["done"])
            print(f"[P0-Ours] resume {key} from prompt {start_p}/{len(C.PROMPTS)}", flush=True)
    for p, prompt in enumerate(C.PROMPTS):
        if p < start_p:
            continue
        ids = tok(prompt, return_tensors="pt").input_ids
        if ids.shape[1] == 0:
            raise RuntimeError(f"[P0-Ours] {key} prompt {p+1} encoded to 0 tokens")
        base_emb = E(ids).detach()
        for k, seed in enumerate(C.SEED_SETS):
            x = C.perturb_batch(base_emb, C.N_MAX, seed)
            feats[p, k] = _forward_pooled_4layers(m, x, L4)
            if not np.isfinite(feats[p, k]).all():
                raise RuntimeError(f"[P0-Ours] {key} prompt {p+1} seed {k} non-finite!")
        tmp = partial.parent / (partial.name + ".tmp.npz")
        np.savez(tmp, pert=feats, layers4=np.array(L4), total=np.int64(total), done=np.int64(p + 1))
        os.replace(tmp, partial)
        print(f"  [P0-Ours] {key} prompt {p+1}/{len(C.PROMPTS)} ({time.time()-t0:.0f}s)", flush=True)
    tmp_out = out.parent / (out.name + ".tmp.npz")
    np.savez(tmp_out, pert=feats, layers4=np.array(L4), layers=np.array(L4),
             total=np.int64(total), collect_seconds=float(time.time() - t0))
    os.replace(tmp_out, out)
    partial.unlink(missing_ok=True)
    print(f"[P0-Ours] saved {out.name} shape={feats.shape} L4={L4} in {time.time()-t0:.1f}s", flush=True)
    del m
    gc.collect()


def collect_reef_one(key: str):
    if locate_p0_reef(key) is not None:
        print(f"[P0-REEF] skip {key} (cached)", flush=True)
        return
    spec = P0_MODELS[key]
    out = REEF_DIR / f"p0_{key}.npz"
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("V4_THREADS", "8")))
    m, tok = _load_p0_model(spec["path"])
    E = m.get_input_embeddings()
    texts = S.load_tqa200()
    acts = []
    for idx, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] == 0:
            raise RuntimeError(f"[P0-REEF] {key} TQA row {idx+1} encoded to 0 tokens")
        e = E(ids).detach()
        last_vec = _forward_last_all_layers(m, e)
        if not np.isfinite(last_vec).all():
            raise RuntimeError(f"[P0-REEF] {key} TQA row {idx+1} non-finite!")
        acts.append(last_vec)
        if (idx + 1) % 50 == 0:
            print(f"  [P0-REEF] {key} {idx+1}/200 ({time.time()-t0:.0f}s)", flush=True)
    arr = np.stack(acts, axis=1)  # [layers, 200, D]
    tmp_out = out.parent / (out.name + ".tmp.npz")
    np.savez(tmp_out, last=arr, collect_seconds=float(time.time() - t0))
    os.replace(tmp_out, out)
    print(f"[P0-REEF] saved {out.name} shape={arr.shape} in {time.time()-t0:.1f}s", flush=True)
    del m
    gc.collect()


def _std_col(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return X / sd


def _load_reef_layers_first(path: Path) -> np.ndarray:
    z = np.load(path)
    arr = z["last"]
    # Standardize [layers, 200, D] vs [200, layers, D]
    if arr.shape[0] == 200 and arr.shape[1] != 200:
        return np.transpose(arr, (1, 0, 2))
    return arr


def cmd_status(include_large: bool):
    keys = active_keys(include_large=include_large)
    print(f"P0 Expanded Pool Status (include_large={include_large}, n={len(keys)}):")
    n_ours, n_reef = 0, 0
    for k in keys:
        po = locate_p0_ours(k)
        pr = locate_p0_reef(k)
        n_ours += int(po is not None)
        n_reef += int(pr is not None)
        exists_dir = Path(P0_MODELS[k]["path"]).exists()
        print(f"  {k:<24s} [{P0_MODELS[k]['role']:8s}/{P0_MODELS[k]['tier']:5s}] "
              f"dir={'OK' if exists_dir else 'NO':2s} "
              f"Ours={'READY' if po else 'MISSING':7s} "
              f"REEF={'READY' if pr else 'MISSING':7s}")
    print(f"Summary: Ours={n_ours}/{len(keys)}, REEF={n_reef}/{len(keys)}")


def cmd_evaluate(include_large: bool):
    if S.locate_ours_feat("baseA_victim") is None:
        raise SystemExit("Missing baseA_victim Ours feature.")
    vic_4l = S.load_ours_4layers("baseA_victim")

    mr_path = SOTA_DIR / "matched_rule.json"
    if not mr_path.exists():
        raise SystemExit(f"Missing frozen threshold file: {mr_path}")
    matched_rule = json.loads(mr_path.read_text(encoding="utf-8-sig"))
    frozen_dirs = matched_rule["ours"]["directions"]  # 20 ordered (anchor, test, t)

    # Load 5 original anchors + 6th original negative (OpenLLaMA)
    anchor_raw_sims = {}
    for orig_name, suite_key in BASE_ANCHOR_TO_SUITE.items():
        p = S.locate_ours_feat(suite_key)
        if p is None:
            raise SystemExit(f"Missing anchor feature {suite_key} ({orig_name})")
        arr = S.load_ours_4layers(p)
        sims_raw, _ = S.score_ours_pair(arr, vic_4l, dims=8)
        anchor_raw_sims[orig_name] = sims_raw

    existing_neg_raw = {}
    existing_neg_norm = {}
    for sk in EXISTING_NEG_KEYS:
        arr = S.load_ours_4layers(sk)
        sr, _ = S.score_ours_pair(arr, vic_4l, dims=8)
        sn, _, _, _ = S.score_ours_pair_normalized(arr, vic_4l, dims=8)
        existing_neg_raw[sk] = sr
        existing_neg_norm[sk] = sn

    keys = active_keys(include_large=include_large)
    ours_results = {}
    new_neg_raw_pool = []
    new_neg_norm_pool = []

    for k in keys:
        p = locate_p0_ours(k)
        if p is None:
            print(f"[eval] skip Ours {k} (not collected yet)")
            continue
        arr = S.load_ours_4layers(p)
        sr, t_raw = S.score_ours_pair(arr, vic_4l, dims=8)
        sn, _, _, t_norm = S.score_ours_pair_normalized(arr, vic_4l, dims=8)

        # Frozen 20-direction evaluation on held-out seeds {1, 2} (1123, 2123)
        per_dir_rates = []
        for d in frozen_dirs:
            anch = d["anchor"]
            gate = float(d["t"])
            diff = sr[:, 1:] - anchor_raw_sims[anch][:, 1:]  # [32, 2]
            rate = float((diff >= gate).mean())
            per_dir_rates.append({"anchor": anch, "test": d["test"], "t": gate, "hit_rate": rate})

        pass_rate_mean = float(np.mean([x["hit_rate"] for x in per_dir_rates]))
        pass_rate_max = float(np.max([x["hit_rate"] for x in per_dir_rates]))
        pass_rate_min = float(np.min([x["hit_rate"] for x in per_dir_rates]))
        dirs_fpr_le10 = sum(1 for x in per_dir_rates if x["hit_rate"] <= 0.10)
        dirs_tpr_ge90 = sum(1 for x in per_dir_rates if x["hit_rate"] >= 0.90)

        entry = {
            "role": P0_MODELS[k]["role"],
            "tier": P0_MODELS[k]["tier"],
            "family": P0_MODELS[k]["family"],
            "D": int(arr.shape[-1]),
            "raw_mean": float(sr.mean()),
            "raw_std": float(sr.std()),
            "norm_mean": float(sn.mean()),
            "norm_std": float(sn.std()),
            "online_sec_raw": t_raw,
            "online_sec_norm": t_norm,
            "frozen20_mean_hit_rate": pass_rate_mean,
            "frozen20_min_hit_rate": pass_rate_min,
            "frozen20_max_hit_rate": pass_rate_max,
            "frozen20_dirs_fpr_le10": dirs_fpr_le10 if P0_MODELS[k]["role"] == "negative" else None,
            "frozen20_dirs_tpr_ge90": dirs_tpr_ge90 if P0_MODELS[k]["role"] == "positive" else None,
            "seed_means_raw": sr.mean(axis=0).tolist(),
            "seed_means_norm": sn.mean(axis=0).tolist(),
            "raw_matrix": sr.tolist(),
            "norm_matrix": sn.tolist(),
        }
        ours_results[k] = entry
        if P0_MODELS[k]["role"] == "negative":
            new_neg_raw_pool.append(sr.flatten())
            new_neg_norm_pool.append(sn.flatten())
        print(f"[Ours] {k:<24s} raw={entry['raw_mean']:.4f} norm={entry['norm_mean']:.4f} "
              f"frozen20_hit={pass_rate_mean:.4f} (max={pass_rate_max:.4f})", flush=True)

    # Compute AUC / TPR@5%FPR for P0 positives against both 6-orig and expanded negative pools
    orig6_raw = np.concatenate([v.flatten() for v in existing_neg_raw.values()])
    orig6_norm = np.concatenate([v.flatten() for v in existing_neg_norm.values()])
    exp_raw = np.concatenate([orig6_raw] + new_neg_raw_pool) if new_neg_raw_pool else orig6_raw
    exp_norm = np.concatenate([orig6_norm] + new_neg_norm_pool) if new_neg_norm_pool else orig6_norm

    for k, entry in ours_results.items():
        if entry["role"] != "positive":
            continue
        pr_arr = np.asarray(entry["raw_matrix"], dtype=np.float64).flatten()
        pn_arr = np.asarray(entry["norm_matrix"], dtype=np.float64).flatten()
        entry["auc_raw_vs_orig6"] = S.compute_roc_auc(pr_arr, orig6_raw)
        entry["auc_norm_vs_orig6"] = S.compute_roc_auc(pn_arr, orig6_norm)
        entry["auc_raw_vs_expanded"] = S.compute_roc_auc(pr_arr, exp_raw)
        entry["auc_norm_vs_expanded"] = S.compute_roc_auc(pn_arr, exp_norm)
        entry["tpr5_raw_vs_expanded"] = S.compute_tpr_at_fpr(pr_arr, exp_raw, 0.05)
        entry["tpr5_norm_vs_expanded"] = S.compute_tpr_at_fpr(pn_arr, exp_norm, 0.05)

    # Evaluate REEF TruthfulQA-200 Layer-18 if available
    reef_results = {}
    pv_reef = S.locate_reef_feat("baseA_victim")
    acc_res_path = H / "exp_gap_accuracy_result.json"
    tE = 0.9741
    tO = 0.9776
    if acc_res_path.exists():
        acc_res = json.loads(acc_res_path.read_text(encoding="utf-8-sig"))
        tE = float(acc_res["part2"]["t_even_max_neg5"])
        tO = float(acc_res["part2"]["t_odd_max_neg5"])

    if pv_reef is not None:
        v_acts = _load_reef_layers_first(pv_reef)  # [L, 200, D]
        x18_all = _std_col(v_acts[18])
        even_idx = np.arange(0, 200, 2)
        odd_idx = np.arange(1, 200, 2)
        xv_even = _std_col(v_acts[18, even_idx])
        xv_odd = _std_col(v_acts[18, odd_idx])
        for k in keys:
            pr = locate_p0_reef(k)
            if pr is None:
                continue
            s_acts = _load_reef_layers_first(pr)
            if s_acts.shape[0] > 18:
                layer, rule = 18, "absolute_18"
            else:
                sc = [S.linear_cka(x18_all, _std_col(s_acts[j])) for j in range(s_acts.shape[0])]
                layer, rule = int(np.argmax(sc)), "max_vs_victim_18"
            c_full = float(S.linear_cka(x18_all, _std_col(s_acts[layer])))
            c_boot = S.reef_bootstrap_scores(v_acts, s_acts, layer_idx=layer)
            c_even = float(S.linear_cka(xv_even, _std_col(s_acts[layer, even_idx])))
            c_odd = float(S.linear_cka(xv_odd, _std_col(s_acts[layer, odd_idx])))
            reef_results[k] = {
                "role": P0_MODELS[k]["role"],
                "layer": layer,
                "rule": rule,
                "cka_full200": c_full,
                "cka_boot3_mean": c_boot["mean"],
                "cka_even": c_even,
                "cka_odd": c_odd,
                "hit_e2o_odd_ge_tE": bool(c_odd >= tE),
                "hit_o2e_even_ge_tO": bool(c_even >= tO),
            }
            print(f"[REEF] {k:<24s} full={c_full:.4f} even={c_even:.4f} odd={c_odd:.4f} (L{layer})", flush=True)

    summary = {
        "n_orig_negatives": len(EXISTING_NEG_KEYS),
        "n_new_negatives_eval": len(new_neg_raw_pool),
        "total_negatives_eval": len(EXISTING_NEG_KEYS) + len(new_neg_raw_pool),
        "new_neg_raw_range": [
            min(v["raw_mean"] for v in ours_results.values() if v["role"] == "negative"),
            max(v["raw_mean"] for v in ours_results.values() if v["role"] == "negative"),
        ] if new_neg_raw_pool else None,
        "new_neg_norm_range": [
            min(v["norm_mean"] for v in ours_results.values() if v["role"] == "negative"),
            max(v["norm_mean"] for v in ours_results.values() if v["role"] == "negative"),
        ] if new_neg_norm_pool else None,
        "new_neg_frozen20_mean_fpr": float(np.mean([
            v["frozen20_mean_hit_rate"] for v in ours_results.values() if v["role"] == "negative"
        ])) if new_neg_raw_pool else None,
        "reef_thresholds": {"t_even_max_neg5": tE, "t_odd_max_neg5": tO},
    }

    payload = {
        "summary": summary,
        "ours": ours_results,
        "reef": reef_results,
    }
    for k in ours_results:
        payload["ours"][k].pop("raw_matrix", None)
        payload["ours"][k].pop("norm_matrix", None)

    SOTA_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write clean Markdown report
    lines = [
        "# P0 扩展验证结果：模型融合衍生（FuseLLM）与扩展负例池",
        "",
        f"- 原负例数：`{summary['n_orig_negatives']}`；本次新增负例数：`{summary['n_new_negatives_eval']}`；合并负例总数：`{summary['total_negatives_eval']}`",
        f"- 新增负例 Raw 区间：`{summary['new_neg_raw_range']}`；Norm ($1-R_\\ell^2$) 区间：`{summary['new_neg_norm_range']}`",
        f"- 冻结 20 方向门限下新增负例平均 FPR：`{summary['new_neg_frozen20_mean_fpr']}`",
        "",
        "## 1. DiffIP-LLM 实测结果（vs. `Llama-2-7b-hf`）",
        "",
        "| 模型键 | 角色 / 家族 | D | Raw MSE 均值 | Norm ($1-R_\\ell^2$) 均值 | 冻结 20 方向命中率 (min / mean / max) | 扩展池 AUC (Raw / Norm) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, e in ours_results.items():
        auc_str = (
            f"{e['auc_raw_vs_expanded']:.4f} / {e['auc_norm_vs_expanded']:.4f}"
            if e["role"] == "positive" else "N/A (Neg)"
        )
        lines.append(
            f"| `{k}` | {e['role']} ({e['family']}) | {e['D']} | {e['raw_mean']:.4f} ± {e['raw_std']:.4f} | "
            f"{e['norm_mean']:.4f} ± {e['norm_std']:.4f} | "
            f"{e['frozen20_min_hit_rate']:.3f} / **{e['frozen20_mean_hit_rate']:.3f}** / {e['frozen20_max_hit_rate']:.3f} | "
            f"{auc_str} |"
        )
    if reef_results:
        lines.extend([
            "",
            f"## 2. REEF TruthfulQA-200 对照（$t_{{even}}={tE:.4f}$, $t_{{odd}}={tO:.4f}$）",
            "",
            "| 模型键 | 角色 | 层 | CKA_full200 | CKA_boot3 | CKA_even | CKA_odd | E→O 触发 | O→E 触发 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for k, r in reef_results.items():
            lines.append(
                f"| `{k}` | {r['role']} | L{r['layer']} | {r['cka_full200']:.4f} | {r['cka_boot3_mean']:.4f} | "
                f"{r['cka_even']:.4f} | {r['cka_odd']:.4f} | {r['hit_e2o_odd_ge_tE']} | {r['hit_o2e_even_ge_tO']} |"
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[eval] saved {OUT_JSON} and {OUT_MD}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="P0 expanded pool & FuseLLM verification")
    parser.add_argument("cmd", choices=["status", "collect_ours", "collect_reef", "evaluate"])
    parser.add_argument("keys", nargs="*", help="Model keys or 'all'")
    parser.add_argument("--include-large", action="store_true", help="Include 20B/MoE negative models")
    args = parser.parse_args()

    if args.cmd == "status":
        cmd_status(args.include_large)
    elif args.cmd == "collect_ours":
        for k in active_keys(args.include_large, args.keys or ["all"]):
            collect_ours_one(k)
    elif args.cmd == "collect_reef":
        for k in active_keys(args.include_large, args.keys or ["all"]):
            collect_reef_one(k)
    elif args.cmd == "evaluate":
        cmd_evaluate(args.include_large)


if __name__ == "__main__":
    main()
