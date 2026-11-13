"""Write the 2026-11-12 advantage-verification report from measured json only.

Reads adv_scores/fixed_threshold_gated.json, fixed_threshold_adv.json,
weight_cos.json, reef.json, layer_residual.json, plus the score rows those
thresholds were computed from. Does not refit a threshold and does not
treat the ADV aggregate flag as the pre-registered pass line.
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
OUT = H / "adv_scores"
FEATS = H / "feats"

NEG = ["neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b",
       "neg_baichuan_7b", "neg_mpt_7b"]
GATED = ["fine_tune", "neg_pruned_sparsegpt", "neg_pruned_wanda",
         "neg_pruned_gblm", "prune_dpoff", "weight_perm", "rep_perm", "scale"]
ADV = ["pos_vicuna",
       "pos_sheared_1_3b_pruned", "pos_sheared_1_3b", "pos_sheared_1_3b_sharegpt",
       "pos_sheared_2_7b_pruned", "pos_sheared_2_7b", "pos_sheared_2_7b_sharegpt",
       "cpt_codellama", "cpt_wizardmath", "cpt_llemma",
       "drop2_chat", "drop8_chat", "drop12_chat"]
# Weight cosine and the residual plot. rep_perm / scale are feature-space
# disguises with no separate weights and no npz.
STRUCT = ["victim", "fine_tune", "derivative_cn",
          "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm",
          "prune_dpoff", "weight_perm"]


def load(name):
    p = OUT / name
    if not p.is_file():
        raise SystemExit(f"missing adv_scores/{name}")
    return json.loads(p.read_text())


def score_file(name):
    for d in (H / "scores", H / "scores_unrel", H / "scores_panel"):
        p = d / f"main_{name}.json"
        if p.is_file():
            return p
    raise SystemExit(f"missing score rows for {name}")


def mean_sim(name):
    rows = json.loads(score_file(name).read_text())
    vals = [r["similarity"] for r in rows if r["N"] == 128 and r["dims"] == 8]
    if len(vals) != 96:
        raise SystemExit(f"{name}: expected 96 decision cells, got {len(vals)}")
    return sum(vals) / len(vals)


def seeds(row):
    s = row["seeds"]
    return s.get("1", s.get(1)), s.get("2", s.get(2))


def rate(x):
    k = int(round(float(x) * 32))
    return f"{float(x):.3f} ({k}/32)"


def cells(doc, pos):
    out = []
    for row in doc["directions"]:
        a, b = seeds(row)
        out.append((row, a, b))
    return out


def recall_cells(doc, name):
    vals = []
    for row, a, b in cells(doc, None):
        vals.append((row["reference"], row["heldout_negative"], 1, float(a["tpr"][name])))
        vals.append((row["reference"], row["heldout_negative"], 2, float(b["tpr"][name])))
    return vals


def check_args(doc, pos, label):
    if doc["args"]["neg"] != NEG or doc["args"]["pos"] != pos:
        raise SystemExit(f"{label} args do not match the pre-registered lists")
    if doc["args"]["n"] != 128 or doc["args"]["dims"] != 8:
        raise SystemExit(f"{label} is not the N=128, d=8 cell")
    if len(doc["directions"]) != 20:
        raise SystemExit(f"{label} has {len(doc['directions'])} directions, expected 20")


def fail_lines(doc, pos):
    lines = []
    for row, a, b in cells(doc, None):
        for seed, block in ((1, a), (2, b)):
            if float(block["fpr"]) > 0.10:
                lines.append(
                    f"- 参照 {row['reference']}，留出 {row['heldout_negative']}，"
                    f"种子组 {seed}，误报 {rate(block['fpr'])}，阈值 {row['threshold']:.6f}")
            for name in pos:
                if float(block["tpr"][name]) < 0.90:
                    lines.append(
                        f"- 参照 {row['reference']}，留出 {row['heldout_negative']}，"
                        f"种子组 {seed}，{name} 召回 {rate(block['tpr'][name])}")
    return lines


def pos_table(doc, names):
    lines = ["| 模型 | 平均相似度 | 召回最小 | 召回最大 | 达到 0.90 的格子 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        vals = recall_cells(doc, name)
        rec = [v[3] for v in vals]
        hit = sum(v >= 0.90 for v in rec)
        lines.append(
            f"| {name} | {mean_sim(name):.3f} | {min(rec):.3f} | {max(rec):.3f} | {hit}/40 |")
    return lines


def direction_table(doc):
    lines = ["| 参照 | 留出负例 | 阈值 | 误报 种子1 | 误报 种子2 | 方向通过 |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    n_pass = 0
    for row, a, b in cells(doc, None):
        ok = bool(row["pass"])
        n_pass += int(ok)
        lines.append(
            f"| {row['reference']} | {row['heldout_negative']} | {row['threshold']:.6f} | "
            f"{rate(a['fpr'])} | {rate(b['fpr'])} | {'是' if ok else '否'} |")
    return lines, n_pass


def main():
    gated = load("fixed_threshold_gated.json")
    adv = load("fixed_threshold_adv.json")
    cos = load("weight_cos.json")
    reef = load("reef.json")
    residual = load("layer_residual.json")
    check_args(gated, GATED, "gated")
    check_args(adv, ADV, "adv")

    dlines, n_pass = direction_table(gated)
    gfail = fail_lines(gated, GATED)
    afail = fail_lines(adv, ADV)

    text = []
    w = text.append
    w("# 优点验证（2026-11-12）")
    w("")
    w("受害模型是 Llama-2-7B。判定格是扰动 128 次、主成分 8 维。")
    w("同一条提示上的间隔是嫌疑模型与受害模型的相似度减去参照模型与受害模型的相似度。")
    w("每个方向的阈值等于种子组 0 上留出负例 32 条间隔的最大值。")
    w("种子组 1 和 2 上，召回至少 0.90 且误报至多 0.10，该方向才通过。")
    w("阈值在这一轮的脚本里现算。五个无关模型不包含本轮新名字。")
    w("本轮新名字不进入阈值。Sheared-LLaMA 按谱系正例记录。")
    w("")
    w("## 固定阈值：已经登记的通过集")
    w("")
    w("正例是官方对话微调、三种稀疏剪枝，以及登记过的衍生变换（删 4 层、权重置换、表示置换、各向异性缩放）。")
    w(f"20 个方向里，脚本判定通过 {n_pass} 个。")
    w("")
    w("\n".join(dlines))
    w("")
    w("\n".join(pos_table(gated, GATED)))
    w("")
    if gfail:
        w("低于通过线的格子：")
        w("")
        w("\n".join(gfail))
    else:
        w("这 8 个正例在 20 个方向、两个留出种子组上的召回都至少 0.90，五个负例的误报都至多 0.10。")
    w("")
    w("## 固定阈值：本轮谱系与压力测试")
    w("")
    w("这些模型用同一套五个负例和同一条 0.90 / 0.10 线来报告。")
    w("它们不参与定阈值。脚本把「全体正例同时通过」写成方向标志；压力测试里的低召回会让这个标志为否。")
    w("已经登记的通过线以上一节为准。")
    w("")
    n_adv_pass = sum(1 for row in adv["directions"] if row["pass"])
    w(f"13 个模型放在一起时，20 个方向里脚本标志为通过的有 {n_adv_pass} 个。")
    w("")
    w("\n".join(pos_table(adv, ADV)))
    w("")
    fpr_only = [line for line in afail if "误报" in line]
    close = []
    for name in ADV:
        rec = [v[3] for v in recall_cells(adv, name)]
        hit = sum(v >= 0.90 for v in rec)
        if 0 < hit < 40:
            close.extend(line for line in afail if f"，{name} 召回 " in line)
    if fpr_only or close:
        w("误报超过 0.10 的格子，以及召回落在 0 和 40 之间的模型的未达格子：")
        w("")
        w("\n".join(fpr_only + close))
        w("")
    zero_hit = []
    for name in ADV:
        rec = [v[3] for v in recall_cells(adv, name)]
        if sum(v >= 0.90 for v in rec) == 0:
            zero_hit.append(name)
    if zero_hit:
        w("下列模型 40 个格子的召回都低于 0.90，上表已给出最小和最大，这里不逐格展开：")
        w(", ".join(zero_hit) + "。")
        w("")
    w("")
    w("六个 Sheared 与三项继续预训练是压力测试。")
    w("先前小模型上，代码模型在约 5.5T 继续预训练之后中间两层相似度约 0.154 和 0.148，那次漏检仍保留。")
    w("删层模型由 Llama-2-7B-Chat 按均匀内部层删去 2、8、12 层后再采集，层号取删完之后的偶数四层。")
    w("")
    w("| 删层模型 | 删后总层数 | 取样层 |")
    w("| --- | ---: | --- |")
    for name in ("drop2_chat", "drop8_chat", "drop12_chat"):
        z = __import__("numpy").load(FEATS / f"main_{name}.npz")
        layers = [int(x) for x in z["layers4"].tolist()]
        total = int(z["total"])
        w(f"| {name} | {total} | {layers} |")
    w("")
    w("## 权重余弦")
    w("")
    w("按参数名对齐，形状相同且至少二维的参数才进入平均。一个都对不上则记为不可比。")
    w("删层模型先删层，再按删后的新层号和受害模型逐名比较。")
    w("")
    w("| 模型 | 状态 | 平均余弦 | 最小余弦 | 对上 / 受害二维参数 | 词嵌入余弦 |")
    w("| --- | --- | ---: | ---: | ---: | ---: |")
    for name in STRUCT + ADV + NEG:
        row = cos.get(name)
        if row is None:
            raise SystemExit(f"weight cosine missing {name}")
        if row["status"] != "ok":
            w(f"| {name} | {row['status']} |  |  | {row['matched']}/{row['total']} |  |")
            continue
        emb = row["embed_cos"]
        emb_s = f"{emb:.4f}" if emb is not None else ""
        w(f"| {name} | ok | {row['mean_cos']:.4f} | {row['min_cos']:.4f} | "
          f"{row['matched']}/{row['total']} | {emb_s} |")
    w("")
    w("## REEF 第 18 层线性中心化核对齐")
    w("")
    w("TruthfulQA 原文件前 200 条，语句是问题加空格加最佳答案，无聊天模板，无截断。")
    w("每一层取最后一个 token。中心化后按特征标准化，再算线性中心化核对齐。")
    w("主指标是绝对层号 18。层数少于 19 时改为该模型各层对受害模型第 18 层的最大值。")
    w("上一轮已经按这个设定提取过的 21 个模型，本轮把激活转置后用当前脚本重算。")
    w("三个删层模型是本轮新提取。")
    w("这里的对齐数值单独列出。它不套用上面的召回和误报通过线。")
    w("")
    w("| 模型 | 线性 CKA | 规则 | 层数 |")
    w("| --- | ---: | --- | ---: |")
    reef_names = ["victim"] + STRUCT[1:] + ADV + NEG
    # STRUCT[0] is victim, STRUCT[1:] then ADV overlaps vicuna etc. Dedup while keeping order.
    seen = set()
    ordered = []
    for name in reef_names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    for name in ordered:
        row = reef.get(name)
        if row is None:
            raise SystemExit(f"reef missing {name}")
        rule = row["layer_rule"]
        if rule == "max":
            rule = f"max@{row['best_layer']}"
        w(f"| {name} | {row['cka']:.4f} | {rule} | {row['layers']} |")
    ident = reef["victim"]["cka"]
    w("")
    w(f"受害模型第 18 层与自己的线性 CKA 是 {ident:.6f}。")
    w("")
    w("## 逐层残差")
    w("")
    w("每个模型在 4 个取样层上，对 32 条提示和 3 个种子组做扰动 128 次、主成分 8 维的对齐。")
    w("表内是各层残差的平均和标准差。相似度等于这些层残差平均值的指数衰减，所以上面的平均相似度可以对照。")
    w("图在 figs_adv/layer_residual.png。")
    w("")
    w("| 模型 | 取样层 | 残差均值 | 残差标准差 |")
    w("| --- | --- | --- | --- |")
    for name in ["victim"] + [n for n in STRUCT[1:] + ADV + NEG if n != "victim"]:
        if name in ("rep_perm", "scale"):
            continue
        if name not in residual:
            raise SystemExit(f"layer residual missing {name}")
        row = residual[name]
        mean = ", ".join(f"{v:.4f}" for v in row["mean"])
        std = ", ".join(f"{v:.4f}" for v in row["std"])
        w(f"| {name} | {row['layers']} | {mean} | {std} |")
    w("")
    w("## 本轮范围")
    w("")
    w("open_llama_7b、opt-6.7b、pythia-6.9b、bloom-7b1 没有进入这四个脚本，没有采集。")
    w("0.5B 上本方法与 REEF 的对照仍单独引用，不和这张 7B 表放在一起。")
    w("")
    dest = H / "表_优点验证_20261112.md"
    dest.write_text("\n".join(text) + "\n")
    print("wrote", dest.name, flush=True)


if __name__ == "__main__":
    main()
