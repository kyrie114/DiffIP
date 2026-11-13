"""Fill tables A and B from measured scores. Does not refit the threshold.

Decision cell is N=128, dims=8. Two reference directions are the already
frozen pair: anchor Qwen2.5-7B with Llama-3-8B held out, and the reverse.
The threshold is that held-out negative's maximum seed-0 margin. It is
checked against exp_unrel_result.json and not replaced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
N_PROMPTS = 32
LEGACY = (
    ("neg_qwen2_5_7b", "neg_llama3_8b"),
    ("neg_llama3_8b", "neg_qwen2_5_7b"),
)
NEG5 = [
    "neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b",
]
NEW = [
    "pos_vicuna",
    "pos_sheared_1_3b_pruned", "pos_sheared_1_3b", "pos_sheared_1_3b_sharegpt",
    "pos_sheared_2_7b_pruned", "pos_sheared_2_7b", "pos_sheared_2_7b_sharegpt",
    "cpt_codellama", "cpt_wizardmath", "cpt_llemma",
]
EXISTING = {
    "fine_tune": H / "scores" / "main_fine_tune.json",
    "derivative_cn": H / "scores" / "main_derivative_cn.json",
    "weight_perm": H / "scores" / "main_weight_perm.json",
    "prune_dpoff": H / "scores" / "main_prune_dpoff.json",
    "rep_perm": H / "scores" / "main_rep_perm.json",
    "scale": H / "scores" / "main_scale.json",
    "neg_pruned_sparsegpt": H / "scores" / "main_neg_pruned_sparsegpt.json",
    "neg_pruned_wanda": H / "scores" / "main_neg_pruned_wanda.json",
    "neg_pruned_gblm": H / "scores" / "main_neg_pruned_gblm.json",
    "neg_qwen2_5_7b": H / "scores" / "main_neg_qwen2_5_7b.json",
    "neg_llama3_8b": H / "scores" / "main_neg_llama3_8b.json",
    "neg_llama1_7b": H / "scores_unrel" / "main_neg_llama1_7b.json",
    "neg_baichuan_7b": H / "scores_unrel" / "main_neg_baichuan_7b.json",
    "neg_mpt_7b": H / "scores_unrel" / "main_neg_mpt_7b.json",
}
DIR_OF = {
    "victim": "Llama-2-7b-hf",
    "fine_tune": "Llama-2-7b-chat-hf",
    "derivative_cn": "Chinese-llama-2-7b",
    "pos_vicuna": "vicuna-7b-v1.5",
    "neg_qwen2_5_7b": "Qwen2.5-7B",
    "neg_llama3_8b": "Meta-Llama-3-8B",
    "neg_llama1_7b": "llama-7b-hf",
    "neg_baichuan_7b": "Baichuan-7B",
    "neg_mpt_7b": "mpt-7b",
    "neg_pruned_sparsegpt": "llama2-7B-pruned-sparsegpt",
    "neg_pruned_wanda": "wanda_pruned_llama_2_7B",
    "neg_pruned_gblm": "GBLM-Pruner-LLaMA-2-7B",
    "prune_dpoff": "由 Llama-2-7b-chat-hf 派生，删除 4 层",
    "weight_perm": "打分时由 Llama-2-7b-chat-hf 变出",
    "rep_perm": "由官方微调特征变出",
    "scale": "由官方微调特征变出",
    "pos_sheared_1_3b_pruned": "Sheared-LLaMA-1.3B-Pruned",
    "pos_sheared_1_3b": "Sheared-LLaMA-1.3B",
    "pos_sheared_1_3b_sharegpt": "Sheared-LLaMA-1.3B-ShareGPT",
    "pos_sheared_2_7b_pruned": "Sheared-LLaMA-2.7B-Pruned",
    "pos_sheared_2_7b": "Sheared-LLaMA-2.7B",
    "pos_sheared_2_7b_sharegpt": "Sheared-LLaMA-2.7B-ShareGPT",
    "cpt_codellama": "CodeLlama-7b-hf",
    "cpt_wizardmath": "WizardMath-7B-V1.0",
    "cpt_llemma": "llemma_7b",
}
REEF_NAMES = [
    "victim", "fine_tune", "derivative_cn", "pos_vicuna", *NEG5,
    "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm",
    *NEW[1:],
]


def load_cell(path):
    rows = json.loads(path.read_text())
    cell = {}
    for r in rows:
        if int(r["N"]) != 128 or int(r["dims"]) != 8:
            continue
        cell[(int(r["prompt"]), int(r["seed_set"]))] = float(r["similarity"])
    if len(cell) != N_PROMPTS * 3:
        raise SystemExit(f"{path.name} has {len(cell)} N=128 d=8 cells")
    if any(v != v or v in (float("inf"), float("-inf")) for v in cell.values()):
        raise SystemExit(f"{path.name} has a non-finite similarity")
    return cell


def margin(cells, model, anchor, prompt, seed):
    return cells[model][(prompt, seed)] - cells[anchor][(prompt, seed)]


def threshold(cells, anchor, test):
    return max(margin(cells, test, anchor, p, 0) for p in range(N_PROMPTS))


def held(cells, model, anchor, t):
    out = []
    for seed in (1, 2):
        hits = sum(margin(cells, model, anchor, p, seed) >= t for p in range(N_PROMPTS))
        out.append(hits / N_PROMPTS)
    return out


def mean_sim(cells, model):
    return sum(cells[model].values()) / len(cells[model])


def fmt_sim(x):
    return f"{x:.3f}"


def fmt_dirs(pairs):
    return "；".join(f"{a:.2f}/{b:.2f}" for a, b in pairs)


def layers_of(name):
    path = H / "feats" / f"main_{name}.npz"
    if name in ("rep_perm", "scale"):
        path = H / "feats" / "main_fine_tune.npz"
    if not path.exists():
        raise SystemExit(f"missing feats/main_{name}.npz")
    import numpy as np
    z = np.load(path)
    key = "layers4" if "layers4" in z.files else "layers"
    return [int(x) for x in z[key]]


def frozen_thresholds(cells):
    stored = json.loads((H / "exp_unrel_result.json").read_text())
    stored_t = {}
    for d in stored["directions"]:
        stored_t[(d["anchor"], d["test"])] = float(d["t"])
    if set(stored_t) != {(a, b) for a in NEG5 for b in NEG5 if a != b}:
        raise SystemExit(f"exp_unrel directions {len(stored_t)} != 20")
    for anchor, test in stored_t:
        got = threshold(cells, anchor, test)
        # scores/ and the file that produced exp_unrel differ by about 1e-7
        # on this max. Above that, the frozen threshold is not the same rule.
        if abs(got - stored_t[(anchor, test)]) > 1e-6:
            raise SystemExit(
                f"threshold drift {anchor}->{test}: recomputed {got} stored {stored_t[(anchor, test)]}")
    return stored, stored_t


def main():
    missing_existing = [str(p.name) for p in EXISTING.values() if not p.exists()]
    if missing_existing:
        raise SystemExit("missing " + ", ".join(missing_existing))
    cells = {name: load_cell(path) for name, path in EXISTING.items()}
    stored, stored_t = frozen_thresholds(cells)
    legacy_t = [stored_t[pair] for pair in LEGACY]
    if "--check-threshold" in sys.argv:
        print("threshold_ok", [round(t, 6) for t in legacy_t], flush=True)
        return
    panel = H / "scores_panel"
    missing = []
    for name in NEW:
        if not (panel / f"main_{name}.json").exists():
            missing.append(f"scores_panel/main_{name}.json")
    reef_path = H / "reef_tqa_result.json"
    if not reef_path.exists():
        missing.append("reef_tqa_result.json")
    if missing:
        raise SystemExit("missing " + ", ".join(missing))
    for name in NEW:
        cells[name] = load_cell(panel / f"main_{name}.json")

    reef = json.loads(reef_path.read_text())
    for name in REEF_NAMES:
        if name not in reef["models"]:
            raise SystemExit(f"reef result missing {name}")
    forbidden = [n for n in reef["models"] if n.startswith("neg_sheared") or "internlm" in n]
    if forbidden:
        raise SystemExit(f"forbidden names in reef result: {forbidden}")

    def rates(model):
        return [held(cells, model, anchor, t) for (anchor, _), t in zip(LEGACY, legacy_t)]

    def reef_cell(name):
        row = reef["models"][name]
        text = f"{row['linear_cka']:.4f}"
        if row["rule"] != "absolute_18":
            text += f"（层 {row['suspect_layer']}）"
        return text

    layer_note = {}
    for name in ["neg_qwen2_5_7b", *NEW, "fine_tune"]:
        layer_note[name] = layers_of(name)

    def src_existing_reef():
        return "DiffIP 主实验已有；REEF 本轮新跑"

    rows_a = []
    rows_a.append(("Llama-2-7b-hf", "受害模型", "—", "—", reef_cell("victim"),
                   "REEF 本轮新跑"))
    for name, role, src in (
        ("fine_tune", "官方对话微调", src_existing_reef()),
        ("derivative_cn", "中文衍生，不参与通过判定", src_existing_reef()),
        ("pos_vicuna", "第三方微调", "本轮新跑"),
    ):
        rows_a.append((DIR_OF[name], role, fmt_sim(mean_sim(cells, name)),
                       fmt_dirs(rates(name)), reef_cell(name), src))
    for name in NEG5:
        rows_a.append((DIR_OF[name], "无关模型", fmt_sim(mean_sim(cells, name)),
                       fmt_dirs(rates(name)), reef_cell(name), src_existing_reef()))

    rows_b = []
    b_spec = [
        ("官方对话微调", "fine_tune", src_existing_reef(), True),
        ("同底座稀疏剪枝", "neg_pruned_sparsegpt", src_existing_reef(), True),
        ("同底座稀疏剪枝", "neg_pruned_wanda", src_existing_reef(), True),
        ("同底座稀疏剪枝", "neg_pruned_gblm", src_existing_reef(), True),
        ("对话版上删除 4 层", "prune_dpoff", "DiffIP 主实验已有", False),
        ("权重置换", "weight_perm", "DiffIP 主实验已有", False),
        ("表示置换", "rep_perm", "DiffIP 主实验已有", False),
        ("缩放", "scale", "DiffIP 主实验已有", False),
        ("结构化剪枝", "pos_sheared_1_3b_pruned", "本轮新跑", True),
        ("结构化剪枝后继续训练", "pos_sheared_1_3b", "本轮新跑", True),
        ("结构化剪枝后指令微调", "pos_sheared_1_3b_sharegpt", "本轮新跑", True),
        ("结构化剪枝", "pos_sheared_2_7b_pruned", "本轮新跑", True),
        ("结构化剪枝后继续训练", "pos_sheared_2_7b", "本轮新跑", True),
        ("结构化剪枝后指令微调", "pos_sheared_2_7b_sharegpt", "本轮新跑", True),
        ("继续预训练（代码）", "cpt_codellama", "本轮新跑", True),
        ("继续预训练（数学）", "cpt_wizardmath", "本轮新跑", True),
        ("继续预训练（定理）", "cpt_llemma", "本轮新跑", True),
    ]
    for kind, name, src, has_reef in b_spec:
        rows_b.append((kind, DIR_OF[name], fmt_sim(mean_sim(cells, name)),
                       fmt_dirs(rates(name)), reef_cell(name) if has_reef else "—", src))

    min20 = {}
    for name in ["fine_tune", "derivative_cn", "pos_vicuna", *NEW,
                 "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]:
        worst = 1.0
        for (anchor, test), t in stored_t.items():
            worst = min(worst, min(held(cells, name, anchor, t)))
        min20[name] = worst

    payload = {
        "cell": "N=128 dims=8",
        "directions": [
            {"anchor": "Qwen2.5-7B", "held_out_negative": "Meta-Llama-3-8B", "t": legacy_t[0]},
            {"anchor": "Meta-Llama-3-8B", "held_out_negative": "Qwen2.5-7B", "t": legacy_t[1]},
        ],
        "threshold_source": "exp_unrel_result.json seed-0 max, unchanged",
        "expanded_pairs": stored["n_directions"],
        "expanded_failed": stored["failed"],
        "layers": {DIR_OF[k]: v for k, v in layer_note.items()},
        "min_heldout_rate_over_20_pairs": {DIR_OF[k]: v for k, v in min20.items()},
        "table_a": [dict(zip(
            ["dir", "role", "sim", "held", "reef18", "source"], row)) for row in rows_a],
        "table_b": [dict(zip(
            ["kind", "dir", "sim", "held", "reef18", "source"], row)) for row in rows_b],
        "reef_rule_exceptions": [
            {"name": n, "suspect_layer": reef["models"][n]["suspect_layer"]}
            for n in reef["models"] if reef["models"][n]["rule"] != "absolute_18"
        ],
    }
    (H / "exp_validity_robustness_result.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False))

    def md_table(header, rows):
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join("---" for _ in header) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    qwen_layers = layer_note["neg_qwen2_5_7b"]
    sheared_13 = layer_note["pos_sheared_1_3b"]
    llama_layers = layer_note["fine_tune"]
    vicuna_low = any(x < 0.90 for pair in rates("pos_vicuna") for x in pair)
    vicuna_sentence = ""
    if vicuna_low:
        vicuna_sentence = ("Vicuna 的留出召回低于 0.90，记为该方法在这档微调上的覆盖范围，"
                           "不放宽主实验的通过标准。\n\n")
    text = f"""# 有效性与鲁棒性（Llama-2-7B）

数字来自本轮测量和已经打好的主实验分数。判定是同一条提示上的相对分数：相似度减去一个参照模型的相似度。阈值是种子 0 上留出负例在 32 条提示里的最大差值，在种子 1 和 2 上检验。表内留出一列的前半是参照 Qwen2.5-7B（留出负例 Meta-Llama-3-8B，阈值 {legacy_t[0]:.4f}），后半是参照 Meta-Llama-3-8B（留出负例 Qwen2.5-7B，阈值 {legacy_t[1]:.4f}）；每一半是种子 1/种子 2。正例格是召回，无关模型格是误报。中文衍生版不参与通过判定。相似度是 N=128、8 维、3 组种子、32 条提示的平均。

已有扩盘是 {stored['n_directions']} 对里 {stored['n_directions'] - stored['n_failed']} 对达到召回 ≥0.90 且误报 ≤0.10。没过的是 {', '.join(stored['failed'])}。本轮没有改这些阈值。

## 表 A（有效性）

{md_table(["模型目录", "角色", "DiffIP 平均相似度", "DiffIP 留出召回或误报", "REEF 第 18 层线性中心化核对齐", "来源"], rows_a)}

{vicuna_sentence}DiffIP 层号：Qwen2.5-7B 是 {qwen_layers}。32 层的 Llama 家族是 {llama_layers}。REEF 主表是零基索引第 18 层；层数不足 19 时改记该模型各层对受害模型第 18 层的最大核对齐。

0.5B 上已经做过的官方 REEF 对照不放进这张表。细粒度曲线下面积是本方法 0.956、REEF 0.923。

## 表 B（鲁棒性）

同一受害模型、同一条判定规则。三种稀疏剪枝仍算在已经冻结的通过判定里。Sheared-LLaMA 是谱系正例。

{md_table(["变换类型", "模型目录或派生方式", "DiffIP 平均相似度", "DiffIP 留出召回", "REEF 第 18 层线性中心化核对齐", "来源"], rows_b)}

Sheared-LLaMA-1.3B 三份的 DiffIP 层号是 {sheared_13}。Sheared-LLaMA-2.7B 三份与其余 32 层模型是 {llama_layers}。删除 4 层、权重置换、表示置换和缩放没有单独的权重目录，本轮不为它们单跑 REEF。置换和缩放上本方法分数接近官方微调，这里只记本盘数字。

继续预训练是压力测试。即使留出召回达到 0.90，也不把「对继续预训练稳健」写成已经证明的优势。先前小模型上，约 5.5T token 继续预训练的代码模型是真实漏检，middle2 相似度约 0.154 和 0.148，与负例同一水平。

20 对已冻结阈值上，本轮新模型留出召回的最低值写在 exp_validity_robustness_result.json 的 min_heldout_rate_over_20_pairs。那一项不改通过线，只用来避免只看原来两个参照方向。
"""
    (H / "表_有效性与鲁棒性_20261111.md").write_text(text)
    print("wrote 表_有效性与鲁棒性_20261111.md", flush=True)


if __name__ == "__main__":
    main()
