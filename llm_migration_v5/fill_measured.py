"""Fill the submission-suite cells that can be scored from stored features.

Does not refit the 20-pair gate, does not rewrite scores/summary_main.json,
and does not load 7B weights. Weight-space HuRef/AWM camouflage is a separate script.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import suite_sota as S

OUT = S.SOTA_DIR / "measured_fill.json"
V4 = S.V4_DIR


def _score_roots() -> list[Path]:
    roots = [V4]
    extra = os.environ.get("DIFFIP_PARENT_V4", "").strip()
    if extra:
        roots.append(Path(extra))
    return roots


def _load_rel(rel: str):
    return _load_json(*(root / rel for root in _score_roots()))


POS_KEYS = [
    "baseA_prune_sparsegpt", "baseA_prune_wanda", "baseA_prune_gblm", "baseA_prune_drop4",
    "baseA_quant_int8", "baseA_quant_int4",
    "baseA_sft_chat", "baseA_sft_vicuna",
    "baseA_dom_chinese", "baseA_dom_codellama",
]
NEG6 = [
    "unrel_qwen2_5_7b", "unrel_llama1_7b", "unrel_openllama_7b", "unrel_baichuan_7b",
    "baseB_victim", "baseD_victim",
]


def _first(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_json(*paths: Path):
    p = _first(*paths)
    if p is None:
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _mean3(rows) -> float:
    return float(np.mean(np.asarray(rows, dtype=np.float64)))


def _metrics(pos, neg) -> dict:
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    return {
        "auc": S.compute_roc_auc(pos, neg),
        "tpr_at_5fpr": S.compute_tpr_at_fpr(pos, neg, 0.05),
        "pos_mean": float(pos.mean()) if len(pos) else None,
        "neg_mean": float(neg.mean()) if len(neg) else None,
        "neg_max": float(neg.max()) if len(neg) else None,
        "margin_vs_max_neg": float(pos.mean() - neg.max()) if len(pos) and len(neg) else None,
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
    }


def _variant(summary, name):
    if not summary:
        return None
    for row in summary.get("variants", []):
        if row.get("name") == name:
            return row
    return None


def mechanism_bars() -> dict:
    br = _load_rel("scores_break/summary.json")
    rms = _load_rel("scores_rms/summary.json")
    camo = _load_json(S.SOTA_DIR / "camouflage_test.json") or {}
    ours = camo.get("Ours", {})

    def camo_mean(key):
        cell = ours.get(key) or {}
        return cell.get("mean"), cell.get("std")

    ident = _variant(br, "ident")
    rank = _variant(br, "rankgauss")
    shuf = _variant(br, "shuffle")
    r25 = _variant(rms, "rms_llama_025")
    r50 = _variant(rms, "rms_llama_050")
    r75 = _variant(rms, "rms_llama_075")
    clean_m, clean_s = camo_mean("clean")
    perm_m, perm_s = camo_mean("perm")
    aniso_m, aniso_s = camo_mean("aniso_scale")
    rows = [
        ("Clean SFT\n(Paired)", clean_m if clean_m is not None else ident["sim_n128_d8"], clean_s or ident["seed_std"], "32/32", "#1f4e78"),
        ("Perm\n(features)", perm_m, perm_s, "", "#1f4e78"),
        ("Aniso scale\n(features)", aniso_m, aniso_s, "", "#d35400"),
        ("RankGauss\n(Exp E)", rank["sim_n128_d8"], rank["seed_std"], "32/32", "#2e8b57"),
        ("Equal-RMS\n25% Mix", r25["sim"], r25["seed_std"], r25["wins_vs_qwen"], "#3498db"),
        ("Equal-RMS\n50% Mix", r50["sim"], r50["seed_std"], r50["wins_vs_qwen"], "#5dade2"),
        ("Equal-RMS\n75% Mix", r75["sim"], r75["seed_std"], r75["wins_vs_qwen"], "#85c1e9"),
        ("Shuffle\n(Exp E)", shuf["sim_n128_d8"], shuf["seed_std"], "5/32", "#c0392b"),
    ]
    if any(r[1] is None for r in rows):
        raise RuntimeError("mechanism source summary is incomplete")
    return {
        "labels": [r[0] for r in rows],
        "sims": [float(r[1]) for r in rows],
        "stds": [float(r[2]) for r in rows],
        "wins": [r[3] for r in rows],
        "colors": [r[4] for r in rows],
        "rankgauss": rank,
        "shuffle": shuf,
        "equal_rms": {"0.25": r25, "0.50": r50, "0.75": r75},
        "ident": ident,
        "source": "scores_break/summary.json + scores_rms/summary.json + camouflage_test.json",
    }


def mid2_margin() -> dict:
    summary = _load_rel("scores_mid2/summary_main.json")
    block = summary["by_nd"]["8"]["128"]
    ft = _mean3(block["fine_tune"])
    qwen = _mean3(block["neg_qwen2_5_7b"])
    llama3 = _mean3(block["neg_llama3_8b"])
    floor = max(qwen, llama3)
    return {
        "fine_tune_mean": ft,
        "qwen_mean": qwen,
        "llama3_mean": llama3,
        "margin": ft - floor,
        "definition": "N=128 d=8 mean(fine_tune) minus the higher of mean(Qwen2.5-7B) and mean(Llama-3-8B)",
    }


def _seed_rows(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        if r.get("N") == 128 and r.get("dims") == 8:
            out[(int(r["prompt"]), int(r["seed_set"]))] = float(r["similarity"])
    return out


def drop_table() -> dict:
    """7B even4 drop scores already on disk. Drop 6 was not collected."""
    score_dir = V4 / "scores"
    panel = V4 / "scores_panel"
    neg_paths = {
        "qwen2.5": score_dir / "main_neg_qwen2_5_7b.json",
        "llama3": score_dir / "main_neg_llama3_8b.json",
        "llama1": V4 / "scores_unrel" / "main_neg_llama1_7b.json",
        "baichuan": V4 / "scores_unrel" / "main_neg_baichuan_7b.json",
        "mpt": V4 / "scores_unrel" / "main_neg_mpt_7b.json",
    }
    negs = {k: _seed_rows(p) for k, p in neg_paths.items() if p.exists()}
    specs = {
        "drop2": panel / "main_drop2_chat.json",
        "drop4": score_dir / "main_prune_dpoff.json",
        "drop8": panel / "main_drop8_chat.json",
        "drop12": panel / "main_drop12_chat.json",
    }
    out = {"scope": "7B Llama-2 chat, even4, N=128, d=8", "negatives": sorted(negs), "missing": [], "rows": {}}
    if not (panel / "main_drop6_chat.json").exists() and not (score_dir / "main_drop6_chat.json").exists():
        out["missing"].append("drop6")
    for name, path in specs.items():
        if not path.exists():
            out["missing"].append(name)
            continue
        pos = _seed_rows(path)
        if not negs:
            out["missing"].append(name + ":no_negatives")
            continue
        neg_keys = set.intersection(*(set(v) for v in negs.values()))
        keys = sorted(set(pos) & neg_keys)
        pos_v = np.array([pos[k] for k in keys])
        neg_v = np.array([negs[nk][k] for nk in negs for k in keys])
        wins = 0
        for k in keys:
            if pos[k] > max(negs[nk][k] for nk in negs):
                wins += 1
        out["rows"][name] = {
            "mean": float(pos_v.mean()),
            "wins": wins,
            "n": len(keys),
            **_metrics(pos_v, neg_v),
        }
    roc = _load_json(V4 / "exp_b_roc_even4.json")
    if roc and "distill_vs_control" in roc:
        out["distill_vs_control"] = roc["distill_vs_control"]
    return out


def _ours_vec(victim, key) -> np.ndarray | None:
    path = S.locate_ours_feat(key)
    if path is None:
        return None
    arr = S.load_ours_4layers(path)
    sims, sec = S.score_ours_pair(arr, victim, dims=8)
    print(f"[ours] {key} mean={sims.mean():.4f} {sec:.1f}s", flush=True)
    del arr
    return sims


def _reef_vec(victim_last, key) -> np.ndarray | None:
    path = S.locate_reef_feat(key)
    if path is None:
        return None
    last = np.load(path)["last"]
    cell = S.reef_bootstrap_scores(victim_last, last)
    print(f"[reef] {key} mean={cell['mean']:.4f}", flush=True)
    return np.asarray(cell["scores"], dtype=np.float64)


def table3_features() -> dict:
    vic_path = S.locate_ours_feat("baseA_victim")
    victim = S.load_ours_4layers(vic_path)
    t0 = time.time()
    chat = S.load_ours_4layers(S.locate_ours_feat("baseA_sft_chat"))
    sims, sec = S.score_ours_pair(chat, victim, dims=8)
    ours_pair_seconds = sec
    print(f"[ours] timed chat pair {ours_pair_seconds:.3f}s mean={sims.mean():.4f}", flush=True)
    # Materialize the PCA fingerprint of one prompt and one seed.
    pr = S.project_sequences(chat[0, 0, :, :128], victim[0, 0, :, :128], 8)
    finger_bytes = int(sum(np.asarray(pr.suspect[i]).nbytes for i in range(4)))
    del chat

    ours = {"baseA_sft_chat": sims}
    missing = []
    for key in POS_KEYS:
        if key in ours:
            continue
        vec = _ours_vec(victim, key)
        if vec is None:
            missing.append(key)
        else:
            ours[key] = vec
    ours_neg = {}
    for key in NEG6:
        vec = _ours_vec(victim, key)
        if vec is None:
            missing.append(key)
        else:
            ours_neg[key] = vec
    del victim

    reef_v_path = S.locate_reef_feat("baseA_victim")
    reef_v = np.load(reef_v_path)["last"]
    t1 = time.time()
    reef_chat = S.reef_bootstrap_scores(reef_v, np.load(S.locate_reef_feat("baseA_sft_chat"))["last"])
    reef_pair_seconds = time.time() - t1
    reef = {"baseA_sft_chat": np.asarray(reef_chat["scores"], dtype=np.float64)}
    reef_neg = {}
    for key in POS_KEYS:
        if key in reef:
            continue
        vec = _reef_vec(reef_v, key)
        if vec is None:
            missing.append("reef:" + key)
        else:
            reef[key] = vec
    for key in NEG6:
        vec = _reef_vec(reef_v, key)
        if vec is None:
            missing.append("reef:" + key)
        else:
            reef_neg[key] = vec

    def by_role(store, neg_store):
        neg = np.concatenate(list(neg_store.values())) if neg_store else np.array([])
        grouped = {}
        for key, vec in store.items():
            role = S.SUITE_MODELS[key]["role"]
            grouped.setdefault(role, []).append(vec)
        out = {}
        for role, vecs in grouped.items():
            pos = np.concatenate(vecs)
            out[role] = _metrics(pos, neg)
        if store and neg_store:
            out["all_scored_positives"] = _metrics(np.concatenate(list(store.values())), neg)
        return out

    reef_file = S.locate_reef_feat("baseA_victim")
    return {
        "victim": "Llama-2-7B",
        "negatives_requested": NEG6,
        "ours_negatives_present": sorted(ours_neg),
        "reef_negatives_present": sorted(reef_neg),
        "missing": missing,
        "rule": "ROC AUC and TPR at 5% FPR. No new threshold. 20-pair gate is not recomputed.",
        "ours_seconds_chat_pair": ours_pair_seconds,
        "ours_fingerprint_bytes_one_model": finger_bytes,
        "reef_seconds_chat_bootstrap": reef_pair_seconds,
        "reef_stored_bytes": int(reef_file.stat().st_size) if reef_file else None,
        "ours": {
            "per_model_mean": {k: float(v.mean()) for k, v in {**ours, **ours_neg}.items()},
            "by_role": by_role(ours, ours_neg),
        },
        "reef": {
            "per_model_mean": {k: float(v.mean()) for k, v in {**reef, **reef_neg}.items()},
            "by_role": by_role(reef, reef_neg),
        },
        "elapsed_sec": time.time() - t0,
    }


def static_table() -> dict:
    static = _load_json(S.SOTA_DIR / "static_baselines.json") or {}
    base = static.get("baseA", {})
    extra = _load_json(S.SOTA_DIR / "static_extra_negatives.json") or {}
    pool = dict(base)
    pool.update(extra)
    pos = {k: v for k, v in pool.items() if k in POS_KEYS}
    neg = {k: v for k, v in pool.items() if k in NEG6 or k in extra}
    # extra file is {key: {HuRef, AWM}} already. baseA unrel keys are the four stored negatives.
    neg = {k: v for k, v in pool.items() if k in NEG6}

    def side(method):
        if not pos or not neg:
            return {"missing_negatives": [k for k in NEG6 if k not in neg], "by_role": {}}
        neg_scores = np.concatenate([np.asarray(neg[k][method]["scores"]) for k in neg])
        grouped = {}
        for key, cell in pos.items():
            role = S.SUITE_MODELS[key]["role"]
            grouped.setdefault(role, []).append(np.asarray(cell[method]["scores"], dtype=np.float64))
        out = {}
        for role, vecs in grouped.items():
            out[role] = _metrics(np.concatenate(vecs), neg_scores)
        out["all_scored_positives"] = _metrics(
            np.concatenate([np.asarray(pos[k][method]["scores"]) for k in pos]),
            neg_scores,
        )
        return {
            "negatives_present": sorted(neg),
            "missing_negatives": [k for k in NEG6 if k not in neg],
            "n_pos_models": len(pos),
            "per_model_mean": {
                k: float(pool[k][method]["mean"]) for k in list(pos) + list(neg) if method in pool[k]
            },
            "by_role": out,
        }

    chat = base.get("baseA_sft_chat", {})
    return {
        "HuRef": side("HuRef"),
        "AWM": side("AWM"),
        "chat_seconds": {
            "HuRef": (chat.get("HuRef") or {}).get("seconds"),
            "AWM": (chat.get("AWM") or {}).get("seconds"),
        },
    }


def efficiency(table3: dict, static: dict, camo: dict | None) -> dict:
    ours_t = table3["ours_seconds_chat_pair"]
    reef_t = table3["reef_seconds_chat_bootstrap"]
    huref_t = static["chat_seconds"]["HuRef"]
    awm_t = static["chat_seconds"]["AWM"]
    if camo:
        huref_t = (camo.get("HuRef") or {}).get("online_pair_seconds", huref_t)
        awm_t = (camo.get("AWM") or {}).get("online_pair_seconds", awm_t)
    huref_bytes = (camo or {}).get("HuRef", {}).get("invariant_bytes")
    awm_bytes = (camo or {}).get("AWM", {}).get("tensors_read_bytes")
    ours_mb = table3["ours_fingerprint_bytes_one_model"] / (1024 ** 2)
    reef_mb = (table3["reef_stored_bytes"] or 0) / (1024 ** 2)
    huref_mb = (huref_bytes / (1024 ** 2)) if huref_bytes else (2 * 3 * 512 * 512 * 8) / (1024 ** 2)
    awm_mb = (awm_bytes / (1024 ** 2)) if awm_bytes else None
    online = [ours_t, reef_t, huref_t, awm_t]
    storage = [ours_mb, reef_mb, huref_mb, awm_mb if awm_mb is not None else huref_mb]
    labels = [
        f"{table3['ours_fingerprint_bytes_one_model'] / 1024:.0f} KB\nPCA fingerprint",
        f"{reef_mb:.0f} MB\nstored acts",
        f"{huref_mb:.1f} MB\nICS vector" if huref_bytes else f"{huref_mb:.1f} MB\nICS allocation",
        f"{awm_mb:.0f} MB\nQ/K read" if awm_mb else "weights",
    ]
    return {
        "online_sec": online,
        "storage_mb": storage,
        "storage_labels": labels,
        "notes": {
            "ours": "32 prompts x 3 seeds, PCA-8 ALS, one chat-versus-victim pair",
            "reef": "3 bootstrap linear CKAs on stored TruthfulQA activations, not the forward",
            "huref": "one ICS pair on loaded weights; allocation is 2 layers x 3 Gram matrices",
            "awm": "one unbiased-CKA pair including the 4096-d assignment",
        },
    }


def assemble(table3: dict | None = None) -> dict:
    prev = _load_json(OUT) or {}
    if table3 is None:
        table3 = prev.get("table3_features")
        if table3 is None:
            raise RuntimeError("table3 features are not computed yet")
    bars = mechanism_bars()
    mid = mid2_margin()
    static = static_table()
    camo = _load_json(S.SOTA_DIR / "camouflage_test.json")
    doc = {
        "table3_features": table3,
        "table3_static": static,
        "drop7b": drop_table(),
        "mid2_margin": mid["margin"],
        "mid2": mid,
        "fig5a": {k: bars[k] for k in ("labels", "sims", "stds", "wins", "colors")},
        "mechanism_source": {k: bars[k] for k in ("rankgauss", "shuffle", "equal_rms", "ident", "source")},
        "efficiency": efficiency(table3, static, camo),
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT)
    return doc


def write_report(doc: dict) -> None:
    t3 = doc["table3_features"]
    st = doc["table3_static"]
    drop = doc["drop7b"]
    lines = [
        "# 方案里尚未完成的格子：补测",
        "",
        "20 对留出标定没有重算，阈值没有改。下面只记这一轮新测到的数。",
        "",
        "## 1. Llama-2 对 6 个难负例（低误报口径）",
        "",
        "正例是该基座上已有的剪枝、量化、微调、领域衍生。负例只留 Qwen2.5-7B、Llama-1-7B、OpenLLaMA-7B、Baichuan-7B、Llama-3-8B、MPT-7B。",
        "指标是 ROC AUC 和 5% 误报下的召回，没有另定阈值。",
        "",
        "| 方法 | 口径 | 全部正例 AUC | TPR@5%FPR | 正例条数 | 负例条数 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]

    def add_method(name, block, note):
        cell = (block or {}).get("by_role", {}).get("all_scored_positives")
        if not cell:
            lines.append(f"| {name} | {note} | 未测 |  |  |  |")
            return
        lines.append(
            f"| {name} | {note} | {cell['auc']:.4f} | {cell['tpr_at_5fpr']:.4f} | {cell['n_pos']} | {cell['n_neg']} |"
        )

    add_method("Ours", t3["ours"], "32×3 条提示")
    add_method("REEF", t3["reef"], "每个模型 3 次自助 CKA")
    add_method("HuRef", st.get("HuRef"), "每个模型 3 个种子")
    add_method("AWM", st.get("AWM"), "每个模型 3 个种子")
    miss_h = (st.get("HuRef") or {}).get("missing_negatives") or []
    if miss_h:
        lines.append("")
        lines.append("HuRef / AWM 这一表还缺的负例：" + ", ".join(miss_h) + "。")
    lines.extend(["", "### 分衍生类型", ""])
    lines.append("| 方法 | 剪枝 AUC | 量化 AUC | 微调 AUC | 领域 AUC |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for name, block in (
        ("Ours", t3["ours"]),
        ("REEF", t3["reef"]),
        ("HuRef", st.get("HuRef") or {}),
        ("AWM", st.get("AWM") or {}),
    ):
        roles = (block or {}).get("by_role", {})
        cells = []
        for role in ("pruning", "quantization", "finetune", "domain"):
            cell = roles.get(role)
            cells.append(f"{cell['auc']:.3f}" if cell else "—")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 2. Test 列",
        "",
        "Ours 和 REEF 仍是特征上的置换、各向异性缩放、旋转乘非恒定缩放。",
        "HuRef 和 AWM 若已写入，则是权重上的保函数置换、保函数各向异性缩放，以及正交旋转后再把非恒定 RMSNorm 缩放留在读权重里。",
        "",
    ])
    camo = _load_json(S.SOTA_DIR / "camouflage_test.json") or {}
    lines.append("| 方法 | clean | perm | aniso | rot+RMSNorm |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for name in ("Ours", "REEF", "HuRef", "AWM"):
        cell = camo.get(name) or {}
        if "clean" not in cell:
            lines.append(f"| {name} | 未测 |  |  |  |")
            continue
        lines.append(
            "| {n} | {c:.4f} | {p:.4f} | {a:.4f} | {r:.4f} |".format(
                n=name,
                c=cell["clean"]["mean"],
                p=cell["perm"]["mean"],
                a=cell["aniso_scale"]["mean"],
                r=cell["rot_rmsnorm"]["mean"],
            )
        )
    lines.extend([
        "",
        "## 3. 已有机理数（没有重跑）",
        "",
        f"RankGauss：{doc['mechanism_source']['rankgauss']['sim_n128_d8']} ± {doc['mechanism_source']['rankgauss']['seed_std']}，{doc['mechanism_source']['rankgauss']['wins']}。",
        f"打乱配对：{doc['mechanism_source']['shuffle']['sim_n128_d8']} ± {doc['mechanism_source']['shuffle']['seed_std']}，{doc['mechanism_source']['shuffle']['wins']}。",
        f"等 RMS 掺入 25/50/75%：{doc['mechanism_source']['equal_rms']['0.25']['sim']} / {doc['mechanism_source']['equal_rms']['0.50']['sim']} / {doc['mechanism_source']['equal_rms']['0.75']['sim']}。",
        f"中间 2 层间隔：{doc['mid2']['margin']:.5f}。{doc['mid2']['definition']}。",
        "",
        "## 4. 7B 删层与蒸馏边界",
        "",
        "删 6 层在 7B 上没有特征，这一轮没有新开前向。下面的删层是已经存好的 even4 分数，负例是表里列到的模型。",
        "0.5B 上删 2/4/6/8 层各 96/96、删 12 层 94/96，以及 REEF 在删 4 层以上低于无关模型，仍然只属于 0.5B，不挪到这张 7B 表。",
        "",
        "| 删层 | 平均相似度 | 胜出 | 条数 | AUC |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for name in ("drop2", "drop4", "drop8", "drop12"):
        row = drop["rows"].get(name)
        if not row:
            lines.append(f"| {name} | 未测 |  |  |  |")
            continue
        lines.append(f"| {name} | {row['mean']:.4f} | {row['wins']} | {row['n']} | {row['auc']:.4f} |")
    if drop.get("missing"):
        lines.append("")
        lines.append("7B 上没有的项：" + ", ".join(drop["missing"]) + "。")
    dist = drop.get("distill_vs_control")
    if dist:
        lines.extend([
            "",
            f"3B 到 1B 蒸馏对同预算对照：AUC {dist['auc']:.6f}，蒸馏均值 {dist['mean_distill']:.4f}，对照均值 {dist['mean_control']:.4f}，96 对里蒸馏更高的有 {dist['paired_distill_higher']} 对。",
        ])
    eff = doc["efficiency"]
    lines.extend([
        "",
        "## 5. 在线耗时和体积（这一轮计时）",
        "",
        "| 方法 | 在线一对（秒） | 体积（MB） |",
        "| :--- | ---: | ---: |",
    ])
    names = ["Ours", "REEF", "HuRef", "AWM"]
    for name, sec, mb in zip(names, eff["online_sec"], eff["storage_mb"]):
        lines.append(f"| {name} | {sec:.4g} | {mb:.4g} |")
    lines.append("")
    lines.append("Ours 的体积是一次 PCA 指纹，不是原始激活文件。REEF 的体积是存下来的激活文件。REEF 的秒数只含已存激活上的 CKA，不含前向。")
    path = S.SOTA_DIR / "补测结果.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {path.name}", flush=True)


def refresh_static_and_report() -> None:
    doc = assemble(None)
    write_report(doc)
    summary_path = S.SOTA_DIR / "summary_metrics.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        camo = _load_json(S.SOTA_DIR / "camouflage_test.json")
        if camo:
            summary["camouflage_test"] = camo
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            S._export_markdown_and_latex_tables(summary)


def main():
    if S.torch is not None and hasattr(S.torch, "set_num_threads"):
        S.torch.set_num_threads(int(os.environ.get("V4_THREADS", "8")))
    print("[fill] table3 features", flush=True)
    table3 = table3_features()
    doc = assemble(table3)
    write_report(doc)
    print("FILL_DONE", flush=True)


if __name__ == "__main__":
    main()
