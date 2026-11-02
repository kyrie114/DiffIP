# -*- coding: utf-8 -*-
"""生成 v4 组会汇报 PPT：数字从 scores/*.json 现算，与实验报告同口径。"""
import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (8, 20, 64, 128)          # PPT 表格展示的 N 列
DIMS = 8                        # 主展示维度

DARK = RGBColor(0x1F, 0x38, 0x64)   # 深蓝
HEAD = RGBColor(0xD9, 0xE2, 0xF3)   # 表头浅蓝
GREY = RGBColor(0xF2, 0xF2, 0xF2)   # victim 行
ORNG = RGBColor(0xFB, 0xE5, 0xD6)   # 同底座剪枝（难负例）
GRN  = RGBColor(0xE2, 0xEF, 0xDA)   # 跨家族（易负例）
YEL  = RGBColor(0xFF, 0xF2, 0xCC)   # SFT/伪装簇
RED  = RGBColor(0xC0, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def agg(line, name, dims, N):
    """跨 32 prompt 均值，3 个 seed set 的 mean/std；distance 取 3 seed 均值。"""
    rows = json.load(open(os.path.join(HERE, "scores", f"{line}_{name}.json")))
    sims, dists = [], []
    for k in range(3):
        rs = sorted((r for r in rows if r["dims"] == dims and r["N"] == N
                     and r["seed_set"] == k), key=lambda r: r["prompt"])
        sims.append(sum(r["similarity"] for r in rs) / len(rs))
        dists.append(sum(r["distance_mean"] for r in rs) / len(rs))
    m = sum(sims) / 3
    sd = (sum((s - m) ** 2 for s in sims) / 3) ** 0.5
    return m, sd, sum(dists) / 3


def cell_fmt(cell, text, size=11, bold=False, color=None, fill=None, align=PP_ALIGN.CENTER):
    cell.text_frame.clear()
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    r = p.add_run(); r.text = text
    f = r.font; f.size = Pt(size); f.bold = bold
    f.color.rgb = color if color else RGBColor(0, 0, 0)
    if fill is not None:
        cell.fill.solid(); cell.fill.fore_color.rgb = fill
    cell.margin_top = cell.margin_bottom = Pt(1)
    cell.vertical_anchor = 3  # MSO_ANCHOR.MIDDLE


def slide_base(prs, title, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.72))
    bar.fill.solid(); bar.fill.fore_color.rgb = DARK; bar.line.fill.background()
    tf = bar.text_frame; tf.clear()
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = title
    r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = WHITE
    if subtitle:
        tb = s.shapes.add_textbox(Inches(0.45), Inches(0.80), Inches(12.4), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        r = p.add_run(); r.text = subtitle
        r.font.size = Pt(13); r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    return s


def bullets(slide, items, left=0.5, top=1.25, width=12.3, height=5.8, size=16):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        lvl, txt = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + txt
        r.font.size = Pt(size if lvl == 0 else size - 2)
        if txt.startswith("【"):
            r.font.bold = True
    return tb


def result_table(slide, line, order, row_fill, left=0.45, top=1.30, width=12.45,
                 height=None, size=11, show_dist=False):
    ncols = 1 + len(NS) * (2 if show_dist else 1)
    tbl = slide.shapes.add_table(len(order) + 1, ncols, Inches(left), Inches(top),
                                 Inches(width), Inches(height or 0.42 * (len(order) + 1))).table
    hdr = ["pair"] + [f"N={n}" for n in NS] * (2 if show_dist else 1)
    if show_dist:
        hdr = ["pair"] + sum(([f"sim N={n}", f"dist"] for n in NS), [])
    for j, h in enumerate(hdr):
        cell_fmt(tbl.cell(0, j), h, size=size, bold=True, fill=HEAD)
    for i, name in enumerate(order, start=1):
        fill = row_fill[name]
        cell_fmt(tbl.cell(i, 0), name, size=size, bold=True, fill=fill, align=PP_ALIGN.LEFT)
        for j, n in enumerate(NS):
            m, sd, d = agg(line, name, DIMS, n)
            if name == "victim":
                cell_fmt(tbl.cell(i, 1 + j * (2 if show_dist else 1)),
                         "1.000" if not show_dist else "1.000 / 0.000",
                         size=size, fill=fill)
            else:
                txt = f"{m:.3f}±{sd:.3f}" if not show_dist else f"{m:.3f}±{sd:.3f} / {d:.3f}"
                cell_fmt(tbl.cell(i, 1 + j * (2 if show_dist else 1)), txt, size=size, fill=fill)
    return tbl


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── 1 封面 ──────────────────────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bar = s.shapes.add_shape(1, 0, Inches(2.1), prs.slide_width, Inches(1.9))
bar.fill.solid(); bar.fill.fore_color.rgb = DARK; bar.line.fill.background()
tf = bar.text_frame; tf.clear()
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "LLM 谱系指纹协议放大到 7B 规模"
r.font.size = Pt(38); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r = p2.add_run(); r.text = "llm_migration_v4 ｜ 主线：Llama-2-7b 真实谱系　·　蒸馏线：Llama-3.2 1B→3B"
r.font.size = Pt(17); r.font.color.rgb = RGBColor(0xCF, 0xDC, 0xF0)
tb = s.shapes.add_textbox(Inches(0), Inches(4.6), prs.slide_width, Inches(0.8))
p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "组会汇报 · 2026-11"
r.font.size = Pt(15); r.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

# ── 2 背景与两个问题 ────────────────────────────────────────────────────
s = slide_base(prs, "背景与要回答的两个问题")
bullets(s, [
    (0, "v3 已在 0.5B（Qwen2.5-0.5B）做出 0.52 margin、32/32 全可分 —— 但留下两个没回答的问题："),
    (1, "victim 太小、负例面板弱（更小模型），\"难负例\"维度缺失"),
    (1, "v2 ROC 发现距离绝对尺度家族依赖 —— 协议能否真正迁移到 7B？"),
    (0, "问题 1：主协议在 7B 上对真实谱系（官方 SFT / 独立衍生 / 同底座剪枝 / 跨家族）的判别力？"),
    (0, "问题 2：序列级蒸馏（1B 学生 ← 3B 教师，同家族同词表）在 hidden 几何里留下的指纹能否被认出？"),
    (0, "方法：v3 协议原封不动 —— 配对种子 embedding 扰动采样 + 逐 prompt 指纹 + PCA→逐层 ALS（DP 关闭）"),
    (1, "主指标 distance_mean（逐层 ALS 平均 L2）；similarity = exp(−Σd/L)"),
], top=1.35, size=17)

# ── 3 实验设计 ──────────────────────────────────────────────────────────
s = slide_base(prs, "实验设计：两条线、19 个模型的全谱系面板")
tb = s.shapes.add_textbox(Inches(0.45), Inches(1.15), Inches(6.1), Inches(5.9))
tf = tb.text_frame; tf.word_wrap = True
tf.paragraphs[0].add_run().text = ""
lines = [
    ("主线（victim = Llama-2-7b-hf，32 层 / D=4096）", True),
    ("· fine_tune：Llama-2-7b-chat（官方 SFT）", False),
    ("· derivative_cn：Chinese-llama-2-7b（独立衍生）", False),
    ("· 三种白盒伪装：weight_perm / rep_perm / scale", False),
    ("· prune_dpoff：chat 剪 4 层（结构变换）", False),
    ("· 难负例：3 个同底座剪枝变体（sparsegpt/wanda/gblm）", False),
    ("· 易负例：Qwen2.5-7B、Llama-3-8B（跨家族）", False),
    ("", False),
    ("蒸馏线（victim = Llama-3.2-1B，16 层 / D=2048）", True),
    ("· student_distill：3B 教师回答上序列级 CE 蒸馏", False),
    ("· student_control：Alpaca 原始回答同预算训练（关键对照）", False),
    ("· 负例：教师 3B 本身 / Sheared-LLaMA-1.3B / Qwen2.5-7B / Llama-2-7b", False),
]
first = True
for txt, bold in lines:
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    first = False
    r = p.add_run(); r.text = txt
    r.font.size = Pt(15); r.font.bold = bold
tb2 = s.shapes.add_textbox(Inches(6.8), Inches(1.15), Inches(6.1), Inches(5.9))
tf2 = tb2.text_frame; tf2.word_wrap = True
lines2 = [
    ("统一协议（与 v3 逐位一致）", True),
    ("· SCALE=2e-3 扰动，3 个配对种子 set × 128 样本", False),
    ("· 32 个探针 prompt（英/中/代码/数学/问答），逐 prompt 独立打分", False),
    ("· 每模型 4 个均匀层，attention-mask 均值池化", False),
    ("· N ∈ {8,16,20,32,64,128}，PCA 降维 d ∈ {8,20}", False),
    ("· DP 关闭（v3 已验证 DP on/off 接近）", False),
    ("", False),
    ("执行环境", True),
    ("· 全程 CPU 48 核（GPU 被 qvh 任务占用），离线本地模型库", False),
    ("· 13+7 个模型特征 11GB，总评分 21888 行", False),
    ("· internlm2-7b 数值发散被隔离（后面单独讲）", False),
]
first = True
for txt, bold in lines2:
    p = tf2.paragraphs[0] if first else tf2.add_paragraph()
    first = False
    r = p.add_run(); r.text = txt
    r.font.size = Pt(15); r.font.bold = bold

# ── 4 主线结果表 ────────────────────────────────────────────────────────
s = slide_base(prs, "主线结果（dims=8）：谱系排序完整保持",
               "格子 = 跨 32 prompt 的 similarity 均值 ± 3 个 seed set 的 std；victim 基线=1.000")
order = ["victim", "fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff",
         "derivative_cn", "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm",
         "neg_qwen2_5_7b", "neg_llama3_8b"]
fillmap = {n: GREY for n in order}
for n in ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff", "derivative_cn"]:
    fillmap[n] = YEL
for n in ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]:
    fillmap[n] = ORNG
for n in ["neg_qwen2_5_7b", "neg_llama3_8b"]:
    fillmap[n] = GRN
result_table(s, "main", order, fillmap, size=11, height=5.4)
tb = s.shapes.add_textbox(Inches(0.45), Inches(6.85), Inches(12.4), Inches(0.5))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "黄=SFT/伪装簇　橙=同底座剪枝（难负例）　绿=跨家族（易负例）｜N=128 时 SFT 簇 vs 跨家族 margin ≈ 0.11"
r.font.size = Pt(12); r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

# ── 5 主线解读 ──────────────────────────────────────────────────────────
s = slide_base(prs, "主线解读：五条结论")
bullets(s, [
    "【1 谱系排序在 7B 保持】 victim(1.0) ≫ 同底座剪枝(0.53–0.56) > SFT 簇(0.357–0.365) > 独立衍生(0.311) > 跨家族(0.254)",
    "【2 三种白盒伪装全部隐形】 weight_perm / rep_perm / scale 与 fine_tune 在噪声内一致（0.363–0.365）—— PCA+ALS 指纹对线性伪装不可见",
    "【3 难负例是最大新维度】 同底座剪枝副本比官方 SFT 更像基座（0.53 vs 0.36）—— 与 v3 排序反转；按\"是否源自基座 X\"判定时应归入同底座一侧",
    "【4 绝对值随规模压缩】 fine_tune 0.76(0.5B)→0.36(7B)，margin 0.52→0.11，跨家族地板不动（0.25）—— 跨规模可迁移排序、不可迁移阈值",
    "【5 收敛行为一致】 N≥64 后 std ≤0.013；N=128 时全部 SFT 类 suspect 32/32 高于跨家族负例（N=20 为 27–29/32）",
], top=1.35, size=17)

# ── 6 v3 对照 ───────────────────────────────────────────────────────────
s = slide_base(prs, "与 v3（0.5B）对照：什么保持、什么变化")
tbl = s.shapes.add_table(8, 3, Inches(0.7), Inches(1.3), Inches(11.9), Inches(3.9)).table
for j, h in enumerate(["指标（N=128, d=8）", "v3（Qwen2.5-0.5B）", "v4（Llama-2-7b）"]):
    cell_fmt(tbl.cell(0, j), h, size=13, bold=True, fill=HEAD)
rows = [
    ("fine_tune（官方 SFT）", "0.760", "0.363"),
    ("三种线性伪装", "≈ fine_tune", "≈ fine_tune（0.363–0.365）"),
    ("base + 剪 4 层", "0.645（高于 SFT）", "0.53–0.56（高于 SFT ✓ 同向）"),
    ("独立衍生", "—", "0.311"),
    ("跨家族地板", "0.235–0.240（5 个负例）", "0.254（2 个负例）"),
    ("fine_tune − 负例 margin", "≈0.52", "≈0.11"),
    ("逐 prompt 胜负（N=20）", "32/32", "28/32（对跨家族负例）"),
]
for i, (a, b, c) in enumerate(rows, start=1):
    cell_fmt(tbl.cell(i, 0), a, size=12, align=PP_ALIGN.LEFT)
    cell_fmt(tbl.cell(i, 1), b, size=12)
    cell_fmt(tbl.cell(i, 2), c, size=12)
bullets(s, [
    "保持：相对谱系排序 / 伪装隐形 / 负例地板 / 随 N 收敛行为",
    "变化：绝对值整体压缩 —— 实用部署必须家族内 + 规模内相对标定（与 v2 ROC 结论互相印证）",
], top=5.5, size=16)

# ── 7 蒸馏线结果 ────────────────────────────────────────────────────────
s = slide_base(prs, "蒸馏线结果（dims=8）：负结果",
               "victim = Llama-3.2-1B；两个学生同基座同预算 3 epoch，唯一差异是训练文本（教师回答 vs Alpaca 原文）")
dorder = ["victim", "student_distill", "student_control", "neg_llama3_2_3b",
          "neg_sheared_1_3b", "neg_qwen2_5_7b", "neg_llama2_7b"]
dfill = {n: GREY for n in dorder}
dfill["student_distill"] = YEL; dfill["student_control"] = YEL
dfill["neg_llama3_2_3b"] = GRN; dfill["neg_sheared_1_3b"] = GRN
dfill["neg_qwen2_5_7b"] = GRN; dfill["neg_llama2_7b"] = GRN
result_table(s, "distill", dorder, dfill, size=12, height=3.4)
bullets(s, [
    "蒸馏信号不可区分：distill vs control 全部 N/d 差异 ≤0.003，在 seed set std（0.002–0.003）之内，方向不稳定",
    "教师本身（3B）得分 0.591 与无关负例无差别 —— 学生几何里没有可辨认的教师印记",
    "两学生高于负例的 ~0.03 是\"同基座被微调\"的共性效应（两臂共享），不是蒸馏贡献",
    "结论：\"被蒸馏过\"这一属性在 N≤128 预算下不可检测（比 v1 的\"认不出教师\"更进一步）",
], top=5.0, size=15)

# ── 8 internlm2 隔离 ────────────────────────────────────────────────────
s = slide_base(prs, "插曲：internlm2-7b 的数值发散与隔离")
bullets(s, [
    "现象：采集出的特征在深层全 NaN（第 21/31 层，覆盖全部 prompt/seed，含无扰动干净样本）",
    "排查链：",
    (1, "权重全部有限（逐张量校验 0 个非有限）→ 不是 checkpoint 损坏"),
    (1, "干净前向复现：bf16 从第 2 层、fp32 从第 4 层起全 NaN；NaN 起始层随 run 漂移（L2/L4/L12/L21）"),
    (1, "第 1 层激活幅值已达 ~4800（健康模型 O(1–10)）→ 前向处于发散边缘"),
    (1, "换 SDPA 注意力：单条前向有限，完整采集在第 2 个 prompt 仍失败"),
    "结论：模型 × 环境不兼容（自定义 modeling + transformers 5.20 CPU），非采集代码/协议问题",
    "处置：按 PLAN 风险表隔离该模型；跨家族面板保留 Qwen2.5-7B + Llama-3-8B（两者数值一致，地板结论不受影响）",
    "采集侧加固：collect.py 新增逐 prompt 有限性校验 —— 坏数据当场中止、不落盘",
], top=1.3, size=16)

# ── 9 局限 ──────────────────────────────────────────────────────────────
s = slide_base(prs, "局限（诚实记录）")
bullets(s, [
    "全程 CPU、单机：N=128 是预算上限；每模型只采一次（确定性验证过同机重跑逐位一致，但无跨机方差）",
    "蒸馏学生仅 1B（7B 学生 CPU 要 1–2 天）；教师数据 1600 条 Alpaca 指令 ×1 遍贪心",
    "distill−control 差 ~0.003 ≈ 1σ：更大 N 不排除出现微弱可分性，但论文设置（N=20）下是明确负结论",
    "跨家族负例剩 2 个（internlm2 隔离）；两版剪枝比例不同（16.7% vs 12.5%），\"剪枝 vs SFT 反转\"无法完全分离因素",
    "跨维度负例（4096 vs 2048）走各自独立 PCA，分数可比较但 PCA 基不同",
], top=1.35, size=17)

# ── 10 小结 ─────────────────────────────────────────────────────────────
s = slide_base(prs, "小结与后续工作")
bullets(s, [
    "【问题 1】协议可迁移到 7B：相对谱系排序可靠、三种白盒伪装全部隐形；但绝对阈值跨规模不可迁移 —— 判定必须家族内+规模内标定",
    "【问题 2】蒸馏不留指纹：序列级 KD 的几何差异淹没在\"同基座微调\"的共性信号里（负结果）",
    "【新维度】同底座剪枝副本是最强的\"同源\"信号（比 SFT 更近基座），鉴定\"是否源自基座 X\"时应重点利用",
    "",
    "后续（按性价比）：",
    (1, "补第三个跨家族负例（internlm2 需 GPU / transformers 4.x 环境）"),
    (1, "蒸馏线大 N 复核（N=256 或第 4 个 seed set，验证 ~1σ 差异是否真实）"),
    (1, "7B 蒸馏学生 + 跨机复跑（论文级声明需要）"),
    (1, "7B 面板 ROC 标定（v2 方法重做，给出家族内定量阈值）"),
], top=1.3, size=17)

out = os.path.join(HERE, "汇报_v4_7B规模验证.pptx")
prs.save(out)
print("saved:", out, f"({os.path.getsize(out)//1024} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
