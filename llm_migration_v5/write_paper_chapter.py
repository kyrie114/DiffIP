"""Write the experimental chapter from stored JSON. Does not refit any threshold."""
from __future__ import annotations

import json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pooled(unrel: dict) -> tuple[int, int]:
    tp = fn = fp = tn = 0.0
    for d in unrel["directions"]:
        for s in range(2):
            tp += d["held_strong_tpr"][s] * 96 + d["held_sft_tpr"][s] * 160
            fn += (1 - d["held_strong_tpr"][s]) * 96 + (1 - d["held_sft_tpr"][s]) * 160
            fp += d["held_fpr"][s] * 32
            tn += (1 - d["held_fpr"][s]) * 32
    return int(round(tp + tn)), int(round(tp + fn + fp + tn))


def f4(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    unrel = load(ROOT / "llm_migration_v4" / "exp_unrel_result.json")
    seeds = load(H / "exp_gap_seeds_result.json")
    cka = load(H / "exp_gap_cka_result.json")
    qwen = load(H / "exp_gap_qwen3_result.json")
    four = load(H / "sota_scores" / "summary_metrics.json")
    p0 = load(H / "sota_scores" / "p0_expanded_pool_result.json")
    boundary = load(H / "sota_scores" / "p0_boundary_result.json")
    acc_n, acc_d = pooled(unrel)
    passed = unrel["n_directions"] - unrel["n_failed"]
    max1 = sum(1 for d in seeds["max1"]["directions"] if d["pass"])
    band = sum(1 for d in seeds["band_rule"]["directions"] if d["pass"])
    max2 = sum(1 for d in seeds["max2"]["directions"] if d["pass"])
    cka_pass = cka["cka"]["n_directions"] - cka["cka"]["n_failed"]
    cos_pass = cka["cosine_same_d"]["n_directions"] - cka["cosine_same_d"]["n_failed"]
    q_pass = len(qwen["directions"]) - qwen["n_failed"]
    roles = ("pruning", "quantization", "finetune", "domain")
    means = " / ".join(f4(four["by_role_aggregated"][r]["Ours"]["auc_mean"]) for r in roles)
    cam = four["camouflage_test"]["Ours"]
    fuse = p0["ours"]["pos_fusellm_7b"]
    abel = p0["ours"]["pos_abel_7b_002"]
    fuse_auc = f"{f4(fuse['auc_raw_vs_expanded'])} / {f4(fuse['auc_norm_vs_expanded'])}"
    abel_auc = f"{f4(abel['auc_raw_vs_expanded'])} / {f4(abel['auc_norm_vs_expanded'])}"
    oc = p0["reef"]["neg_openchat_3_5"]["cka_full200"]
    layers = boundary["per_layer"]
    fuse_l = layers["pos_fusellm_7b"]
    abel_l = layers["pos_abel_7b_002"]
    band_l = layers["unrelated_band"]
    worst_ok = 0
    for key, row in p0["ours"].items():
        if row["role"] != "negative":
            continue
        if row["frozen20_max_hit_rate"] <= 0.10:
            worst_ok += 1

    weight_lines = []
    for key in boundary["models"]:
        row = boundary["weights"][key]
        weight_lines.append(
            f"| {key} | {f4(row['HuRef']['mean'])} | {f4(row['AWM']['mean'])} |"
        )

    text = f"""# 实验

三件事分开写。20 对留出、四基座主表、P0 扩池各用各的门限。后一节的高分不改写前一节的失败。

## 1. 冻结的 20 对同题间隔

受害模型是 Llama-2-7B。五个无关模型两两做参照和留出，共 20 个方向。种子 0 用留出负例自己的最大间隔定阈值，种子 1 和 2 才计数。强剪枝每个种子 96 条，微调簇每个种子 160 条。中文衍生不进通过线。

通过 {passed}/20。claim_holds 为假。失败的两对是 neg_qwen2_5_7b 到 neg_mpt_7b，以及 neg_llama3_8b 到 neg_mpt_7b。

种子 1 和 2 合并后，正确数是 {acc_n}/{acc_d}。

同一批事先登记、并且没有通过的线是：同云线性 CKA {cka_pass}/20，同维扰动余弦 {cos_pass}/12，Qwen3 第二底座 {q_pass}/12。DiffIP 新种子 max1 {max1}/20，band {band}/20，band 未确认。max2 {max2}/20。新种子上的同云 CKA 三条规则另计，不把冻结门限改成全通过。

## 2. 四基座归一化主表

四个基座是 Llama-2-7B、Llama-3-8B、Qwen3-8B、MPT-7B。主表分数是逐层方差归一化残差。Ours 四基座均值是 {means}。伪装列是置换 {f4(cam['perm']['mean'])}，各向异性缩放 {f4(cam['aniso_scale']['mean'])}，旋转加 RMSNorm {f4(cam['rot_rmsnorm']['mean'])}。缩放后的分数是 {f4(cam['aniso_scale']['mean'])}，干净分数是 {f4(cam['clean']['mean'])}。

## 3. P0 扩池，以及它到不了的边界

这一节用的仍是原来的 20 组阈值，没有在这 11 个模型上重拟合。

FuseLLM 对扩展负例池的间隔 AUC 是 {fuse_auc}。Abel 是 {abel_auc}。预先写的线是 AUC 至少 0.90。两格都低于这条线。20 个方向里召回达到 0.90 的个数，两个正例都是 0。

9 个新负例的平均误报是 {f4(p0['summary']['new_neg_frozen20_mean_fpr'])}，不超过 0.05。最差方向的误报不超过 0.10 的，只有 {worst_ok} 个，分母是 9。

REEF 的偶/奇阈值是 {f4(p0['summary']['reef_thresholds']['t_even_max_neg5'])} 和 {f4(p0['summary']['reef_thresholds']['t_odd_max_neg5'])}。FuseLLM 全量 CKA {f4(p0['reef']['pos_fusellm_7b']['cka_full200'])}，偶定奇测和奇定偶测都触发。Abel 全量 CKA {f4(p0['reef']['pos_abel_7b_002']['cka_full200'])}，两个方向都不触发。OpenChat 和 Starling 的全量 CKA 都是 {oc:.6f}，逐位相同。

internlm2-base-7b 的前向是非有限值，没有分数。

四基座表上剪枝、量化和微调接近满分，不覆盖这一节。融合模型和数学继续预训练没有被 DiffIP 从无关底噪里分开。

### 3.1 已存四层上的逐层分数

没有新的前向。四个偶数层上，Raw 与方差归一化分数都是该层单独的均值。

FuseLLM Raw { ' '.join(f4(v) for v in fuse_l['raw_mean']) }

FuseLLM 归一化 { ' '.join(f4(v) for v in fuse_l['norm_mean']) }

Abel Raw { ' '.join(f4(v) for v in abel_l['raw_mean']) }

Abel 归一化 { ' '.join(f4(v) for v in abel_l['norm_mean']) }

9 个新负例在这四层上的 Raw 区间是 {' '.join(f4(v) for v in band_l['raw_min'])} 到 {' '.join(f4(v) for v in band_l['raw_max'])}。归一化区间是 {' '.join(f4(v) for v in band_l['norm_min'])} 到 {' '.join(f4(v) for v in band_l['norm_max'])}。

### 3.2 权重侧 HuRef 与 AWM

调用静态基线同一条 HuRef 与 AWM 路径。每个模型一个分数，都是三次种子的均值。

| 模型 | HuRef | AWM |
| --- | ---: | ---: |
{chr(10).join(weight_lines)}
"""
    out = H / "docs" / "实验章节.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.name} chars={len(text)}")


if __name__ == "__main__":
    main()
