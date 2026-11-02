# -*- coding: utf-8 -*-
"""生成 v4 组会汇报 PPT：数字从 scores/*.json 现算，与实验报告同口径。"""
import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
# V4_LAYER_SUBSET=mid2 -> 读 scores_mid2/（4 个存储层的中间 2 层），输出文件名加 _mid2
SUBSET = os.environ.get("V4_LAYER_SUBSET", "all")
SCORES = "scores" if SUBSET == "all" else f"scores_{SUBSET}"
# 显示名：neg_pruned_* 键名保留兼容，实为同底座结构变体（正例）
DISPLAY = {"neg_pruned_sparsegpt": "结构变体_sparsegpt",
           "neg_pruned_wanda": "结构变体_wanda",
           "neg_pruned_gblm": "结构变体_gblm"}
NS = (8, 20, 64, 128)          # PPT 表格展示的 N 列
DIMS = 8                        # 主展示维度

DARK = RGBColor(0x1F, 0x38, 0x64)   # 深蓝
HEAD = RGBColor(0xD9, 0xE2, 0xF3)   # 表头浅蓝
GREY = RGBColor(0xF2, 0xF2, 0xF2)   # victim 行
ORNG = RGBColor(0xFB, 0xE5, 0xD6)   # 同底座结构变体（正例）
GRN  = RGBColor(0xE2, 0xEF, 0xDA)   # 跨家族负例
YEL  = RGBColor(0xFF, 0xF2, 0xCC)   # SFT/伪装簇
RED  = RGBColor(0xC0, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def agg(line, name, dims, N):
    """跨 32 prompt 均值，3 个 seed set 的 mean/std；distance 取 3 seed 均值。"""
    rows = json.load(open(os.path.join(HERE, SCORES, f"{line}_{name}.json")))
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
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = WHITE
    if subtitle:
        tb = s.shapes.add_textbox(Inches(0.45), Inches(0.80), Inches(12.4), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        r = p.add_run(); r.text = subtitle
        r.font.size = Pt(15); r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    return s


def bullets(slide, items, left=0.5, top=1.25, width=12.3, height=5.8, size=18):
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
        cell_fmt(tbl.cell(i, 0), DISPLAY.get(name, name), size=size, bold=True, fill=fill, align=PP_ALIGN.LEFT)
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


from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE

import figures as _figs
FIGS = _figs.make_all()   # (fig1, fig2, fig3) 300 dpi 论文级图


def box(slide, x, y, w, h, text, fill, size=11, bold=False, color=None,
        dashed=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    """圆角矩形节点，支持多行文字（\\n 分行）。"""
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if dashed:
        sp.line.color.rgb = RGBColor(0x8A, 0x8A, 0x8A); sp.line.width = Pt(1.1)
        sp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else:
        sp.line.fill.background()
    tf = sp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Pt(3); tf.margin_top = tf.margin_bottom = Pt(1)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color if color else RGBColor(0x20, 0x20, 0x20)
    return sp


def arrow(slide, x, y, direction="right", w=0.32, h=0.26):
    """块状小箭头：direction=right/down，(x,y) 为左上角。"""
    shp = MSO_SHAPE.RIGHT_ARROW if direction == "right" else MSO_SHAPE.DOWN_ARROW
    sp = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = RGBColor(0xA6, 0xAE, 0xBE)
    sp.line.fill.background()


def link(slide, x1, y1, x2, y2, color=None, dashed=False, pt=1.2):
    """直线连接线（谱系树分支 / 流程汇入）。"""
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                   Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color or RGBColor(0x9A, 0xA2, 0xB2)
    c.line.width = Pt(pt)
    if dashed:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH


def note(slide, x, y, w, h, text, size=10.5, bold=False, color=None, align=PP_ALIGN.CENTER):
    """无底色小字标注。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color if color else RGBColor(0x60, 0x60, 0x60)
    return tb


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── 2 背景与两个问题 ────────────────────────────────────────────────────
s = slide_base(prs, "背景与要回答的两个问题")
bullets(s, [
    (0, "v3 已在 0.5B（Qwen2.5-0.5B）做出 0.52 margin、32/32 全可分 —— 但留下两个没回答的问题："),
    (1, "victim 太小、负例面板弱（更小模型），缺\"同底座结构变体\"维度"),
    (1, "v2 ROC 发现距离绝对尺度家族依赖 —— 协议能否真正迁移到 7B？"),
    (0, "问题 1：主协议在 7B 上对真实谱系（官方 SFT / 独立衍生 / 同底座剪枝 / 跨家族）的判别力？"),
    (0, "问题 2：序列级蒸馏（1B 学生 ← 3B 教师，同家族同词表）在 hidden 几何里留下的指纹能否被认出？"),
    (0, "方法：v3 协议原封不动 —— 配对种子 embedding 扰动采样 + 逐 prompt 指纹 + PCA→逐层 ALS（DP 关闭）"),
    (1, "主指标 distance_mean（逐层 ALS 平均 L2）；similarity = exp(−Σd/L)"),
], top=1.35, size=19)

# ── 3 方法原理 ①：配对种子扰动采样（论文级图） ─────────────────────────
def figure_slide(prs, title, img, subtitle=None):
    s = slide_base(prs, title, subtitle)
    s.shapes.add_picture(img, Inches(0.47), Inches(1.30), Inches(12.4), Inches(5.5))


figure_slide(prs, "方法原理 ①：配对种子扰动采样 —— 差异只来自模型", FIGS[0],
             "SCALE=2e-3 ｜ SEED_SETS=[123, 1123, 2123]，每组 128 样本 ｜ victim 与 suspect 同 seed 严格配对")

# ── 4 方法原理 ②：PCA + 逐层 ALS 对齐（论文级图） ──────────────────────
figure_slide(prs, "方法原理 ②：PCA 降维 + 逐层 ALS 对齐 —— 距离即指纹", FIGS[1],
             "diffip.preprocessing.project_sequences ＋ diffip.reversion.fit_step_alignment ｜ DP 关闭")

# ── 5 实验设计：两线谱系面板（论文级图） ────────────────────────────────
figure_slide(prs, "实验设计：两线谱系面板", FIGS[2],
             "主线 13 位（含隔离 1）＋ 蒸馏线 7 位（2 位复用主线 npz）｜ 协议与 v3 逐位一致")

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
result_table(s, "main", order, fillmap, size=13, height=5.5)
tb = s.shapes.add_textbox(Inches(0.45), Inches(6.92), Inches(12.4), Inches(0.5))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "黄=SFT/伪装簇（prune_dpoff=整层剪枝）　橙=同底座结构变体（正例，稀疏剪枝）　绿=跨家族负例｜N=128 时 SFT 簇 vs 跨家族 margin ≈ 0.11"
r.font.size = Pt(13); r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

# ── 5 主线解读 ──────────────────────────────────────────────────────────
s = slide_base(prs, "主线解读：五条结论")
bullets(s, [
    "【1 谱系排序在 7B 保持】 victim(1.0) ≫ 同底座结构变体(0.53–0.56) > SFT 簇(0.357–0.365) > 独立衍生(0.311) > 跨家族(0.254)",
    "【2 三种白盒伪装全部隐形】 weight_perm / rep_perm / scale 与 fine_tune 在噪声内一致（0.363–0.365）—— PCA+ALS 指纹对线性伪装不可见",
    "【3 结构变体是正例、且最像基座】 sparsegpt/wanda/gblm 为同底座稀疏权重剪枝（不删层），0.53 vs SFT 0.36，对跨家族负例 32/32 全胜；结论仅限稀疏剪枝（prune_dpoff 是整层删除，落在 SFT 簇）",
    "【4 绝对值随规模压缩】 fine_tune 0.76(0.5B)→0.36(7B)，margin 0.52→0.11，跨家族地板不动（0.25）—— 跨规模可迁移排序、不可迁移阈值",
    "【5 收敛行为一致】 N≥64 后 std ≤0.013；N=128 时全部 SFT 类 suspect 32/32 高于跨家族负例（N=20 为 27–29/32）；胜负只对跨家族负例计",
], top=1.35, size=19)

# ── 6 v3 对照 ───────────────────────────────────────────────────────────
s = slide_base(prs, "与 v3（0.5B）对照：什么保持、什么变化")
tbl = s.shapes.add_table(8, 3, Inches(0.7), Inches(1.3), Inches(11.9), Inches(4.1)).table
for j, h in enumerate(["指标（N=128, d=8）", "v3（Qwen2.5-0.5B）", "v4（Llama-2-7b）"]):
    cell_fmt(tbl.cell(0, j), h, size=15, bold=True, fill=HEAD)
rows = [
    ("fine_tune（官方 SFT）", "0.760", "0.363"),
    ("三种线性伪装", "≈ fine_tune", "≈ fine_tune（0.363–0.365）"),
    ("base + 剪枝（剪法不同）", "0.645 整层剪枝 4/24", "0.53–0.56 稀疏剪枝（高于 SFT，反转）"),
    ("独立衍生", "—", "0.311"),
    ("跨家族地板", "0.235–0.240（5 个负例）", "0.254（2 个负例）"),
    ("fine_tune − 负例 margin", "≈0.52", "≈0.11"),
    ("逐 prompt 胜负（N=20）", "32/32", "28/32（对跨家族负例）"),
]
for i, (a, b, c) in enumerate(rows, start=1):
    cell_fmt(tbl.cell(i, 0), a, size=13.5, align=PP_ALIGN.LEFT)
    cell_fmt(tbl.cell(i, 1), b, size=13.5)
    cell_fmt(tbl.cell(i, 2), c, size=13.5)
bullets(s, [
    "保持：相对谱系排序 / 伪装隐形 / 负例地板 / 随 N 收敛行为",
    "注意：v3 层剪枝 vs v4 稀疏剪枝，反转混了剪枝类型、比例、家族/规模三因素，不可分离",
    "变化：绝对值整体压缩 —— 实用部署必须家族内 + 规模内相对标定（与 v2 ROC 结论互相印证）",
], top=5.5, height=1.6, size=17)

# ── 7 蒸馏线结果 ────────────────────────────────────────────────────────
s = slide_base(prs, "蒸馏线结果（dims=8）：负结果",
               "victim = Llama-3.2-1B；两个学生同基座同预算 3 epoch，唯一差异是训练文本（教师回答 vs Alpaca 原文）")
dorder = ["victim", "student_distill", "student_control", "neg_llama3_2_3b",
          "neg_sheared_1_3b", "neg_qwen2_5_7b", "neg_llama2_7b"]
dfill = {n: GREY for n in dorder}
dfill["student_distill"] = YEL; dfill["student_control"] = YEL
dfill["neg_llama3_2_3b"] = GRN; dfill["neg_sheared_1_3b"] = GRN
dfill["neg_qwen2_5_7b"] = GRN; dfill["neg_llama2_7b"] = GRN
result_table(s, "distill", dorder, dfill, size=14, height=3.7)
bullets(s, [
    "蒸馏信号不可区分：distill vs control 全部 N/d 差异 ≤0.003，在 seed set std（0.002–0.003）之内，方向不稳定",
    "教师本身（3B）得分 0.591 与无关负例无差别 —— 学生几何里没有可辨认的教师印记",
    "两学生高于负例的 ~0.03 是\"同基座被微调\"的共性效应（两臂共享），不是蒸馏贡献",
    "结论：\"被蒸馏过\"这一属性在 N≤128 预算下不可检测（比 v1 的\"认不出教师\"更进一步）",
], top=5.1, height=2.2, size=16)

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
    "跨家族负例剩 2 个（internlm2 隔离）；剪法不同（v3 整层 4/24 vs v4 稀疏权重剪枝），\"剪枝 vs SFT 反转\"无法分离剪枝类型/比例/家族规模",
    "跨维度负例（4096 vs 2048）走各自独立 PCA，分数可比较但 PCA 基不同",
], top=1.35, size=19)

# ── 10 小结 ─────────────────────────────────────────────────────────────
s = slide_base(prs, "小结与后续工作")
bullets(s, [
    "【问题 1】协议可迁移到 7B：相对谱系排序可靠、三种白盒伪装全部隐形；但绝对阈值跨规模不可迁移 —— 判定必须家族内+规模内标定",
    "【问题 2】蒸馏不留指纹：序列级 KD 的几何差异淹没在\"同基座微调\"的共性信号里（负结果）",
    "【新维度】同底座结构变体（稀疏剪枝，正例）是最强的\"同源\"信号（比 SFT 更近基座）",
    "",
    "后续（按性价比）：",
    (1, "服务器跑中间 2 层重评分（V4_LAYER_SUBSET=mid2，免重采集；结果待出）"),
    (1, "补第三个跨家族负例（internlm2 需 GPU / transformers 4.x 环境）"),
    (1, "蒸馏线大 N 复核（N=256 或第 4 个 seed set，验证 ~1σ 差异是否真实）"),
    (1, "7B 蒸馏学生 + 跨机复跑（论文级声明需要）"),
    (1, "7B 面板 ROC 标定（v2 方法重做，给出家族内定量阈值）"),
], top=1.3, size=19)

# ── 每页演讲者备注（讲稿） ──────────────────────────────────────────────
NOTES = [
    # 2 背景与问题
    "先说动机。v3 在 0.5B 上做出 0.52 margin、32/32 全可分，但 victim 只有 0.5B，"
    "负例面板都是更小的模型，缺同底座结构变体这一维度。v2 的 ROC 标定又发现距离绝对尺度是家族依赖的，"
    "全局阈值跨家族不可迁移——所以协议到 7B 还灵不灵是未知数。"
    "所以 v4 回答两个问题：一，7B 上对真实谱系的判别力；二，蒸馏指纹能否被认出（v1 判负，这次用规范协议重验）。",
    # 3 方法原理 ①
    "先看方法核心。输入端是一条公共流水线：探针 prompt 过 embedding，再加 SCALE=2e-3 的高斯噪声。"
    "关键在配对种子：victim 和 suspect 用完全相同的种子序列，每组 128 样本、共 3 组——"
    "两边输入扰动逐位相同，之后算出的特征差异 100% 来自模型参数本身，这是整个指纹方法可比性的基础。"
    "另外两个白盒伪装 rep_perm / scale 就是在 suspect 特征上做线性变换派生的，免采集。",
    # 4 方法原理 ②
    "拿到两边特征怎么比？先 PCA 把 4096 维降到 8 或 20（满足 d≤4N−1），再逐层 ALS："
    "拟合线性映射 W 让 suspect 的特征云尽量对齐 victim，对齐残差就是逐层距离，"
    "平均得 distance_mean，similarity = exp(−距离)。"
    "核心假设：同源模型的层表示云只差一个线性映射——ALS 能对齐、距离小；"
    "异源模型云形状本质不同、对不齐、距离大。注意三种白盒伪装（置换、各向异性缩放）本身就是线性变换，"
    "天然落在'可对齐'集合里，所以必然隐形——这是后面结果里伪装全部无效的原因。"
    "右下标尺是主线实测相似度谱：0.254 跨家族 / 0.311 独立衍生 / 0.36 SFT / 0.54 同底座剪枝 / 1.0 自己。",
    # 5 谱系面板
    "实验设计两条线。主线 victim 是 Llama-2-7b-hf，面板覆盖：官方 chat SFT 及其三种白盒伪装（特征空间派生、免采集）、"
    "chat 剪 4 层（整层删除）、独立衍生中文版；另有三个第三方同底座结构变体（稀疏权重剪枝，不删层）——它们是真实同源的正例，"
    "这是本实验新增的维度；负例是 Qwen2.5-7B 和 Llama-3-8B 两个跨家族模型；internlm2-7b 因数值发散被隔离。"
    "蒸馏线 victim 是 Llama-3.2-1B：distill 学生在 3B 教师生成的回答上做序列级蒸馏，"
    "control 学生同指令同预算、文本换成 Alpaca 原始回答——双臂唯一差异是训练文本，"
    "distill 减 control 就是蒸馏的净效应。四个负例里两个直接复用主线 npz（协议逐位一致）。",
    # 6 主线结果表
    "主线结果表，dims=8，每格是 32 个 prompt 的 similarity 均值 ± 3 个种子组的 std。"
    "黄色是 SFT 和三种伪装，橙色是同底座结构变体（正例，稀疏剪枝），绿色是跨家族负例。"
    "排序很清晰：剪枝副本 0.53–0.56 高于 SFT 簇 0.36；独立衍生 0.31；跨家族两个模型都停在 0.254。"
    "N 越大 std 越小，N=128 时都在 0.01 以内，数值已收敛。",
    # 7 主线解读
    "五条结论。一，谱系排序在 7B 完整保持。二，三种白盒伪装全部隐形——与 fine_tune 在噪声内不可分，"
    "PCA+ALS 指纹对线性变换免疫，符合设计预期。三，最重要的新发现：同底座稀疏剪枝变体是正例，且比官方 SFT 更像基座，对跨家族负例全胜；"
    "注意它们是稀疏权重剪枝，不是删层，prune_dpoff 删层后仍落在 SFT 簇；与 v3（层剪枝）的反转无法分离剪法、比例和规模。"
    "四，绝对值随规模压缩：fine_tune 0.76→0.36，margin 0.52→0.11，负例地板不动——跨规模可迁移排序、不可迁移阈值。"
    "五，收敛行为与 v3 一致，N=128 时 SFT 类 suspect 全部 32/32 高于跨家族负例。",
    # 8 v3 对照
    "这张表把 v3 和 v4 直接对照。保持的：相对排序、伪装隐形、负例地板、随 N 收敛。"
    "变化的：绝对值整体压缩，margin 0.52→0.11。实用含义很直接：部署时必须在同家族、同规模内做相对标定，"
    "不能拿 0.5B 的阈值去判 7B——这与 v2 的 ROC 结论互相印证。",
    # 9 蒸馏线
    "蒸馏线是负结果。两个学生同一个 1B 基座、同样 3 epoch、同样 1600 条指令，唯一差异是训练文本："
    "distill 用 3B 教师回答，control 用 Alpaca 原始回答。结果两者 similarity 在所有 N 和维度下差异 ≤0.003，"
    "在种子组噪声内，方向还不稳定；教师本身 3B 得 0.591，与无关负例无差别。"
    "两学生高于负例的 ~0.03 是'同基座被微调'的共性效应，两臂共享，不是蒸馏贡献。"
    "结论：'被蒸馏过'这个属性在 N≤128 预算下不可检测——比 v1 的'认不出教师'更进一步。",
    # 10 internlm2
    "中间有个插曲。internlm2-7b 采出的特征深层全 NaN，连无扰动干净样本都是。排查链：权重全部有限，"
    "排除 checkpoint 损坏；干净前向复现，bf16 从第 2 层、fp32 从第 4 层发散，发散点随 run 漂移；"
    "第 1 层激活幅值 ~4800（健康模型 1–10）。换 SDPA 单条前向有限，完整采集第二个 prompt 仍失败。"
    "结论是模型 × 环境不兼容（自定义 modeling + transformers 5.20 CPU），非采集代码或协议问题。"
    "按 PLAN 风险表隔离；跨家族面板留 Qwen2.5-7B + Llama-3-8B，两者数值一致，地板结论不受影响。"
    "顺手给采集加了逐 prompt 有限性校验：坏数据当场中止、不落盘。",
    # 11 局限
    "局限如实说：全程 CPU，N=128 是预算上限；每模型只采一次，确定性只验证了同机重跑逐位一致，没有跨机方差；"
    "蒸馏学生仅 1B（7B 学生 CPU 要 1–2 天）；distill−control 差 ~0.003 ≈ 1σ，更大 N 不排除弱可分性，"
    "但论文设置（N=20）下是明确负结论；v3 是整层剪枝、v4 结构变体是稀疏剪枝，'剪枝 vs SFT 反转'无法分离剪枝类型、比例和家族规模。",
    # 12 小结
    "总结。问题一：协议可迁移到 7B，相对谱系排序可靠、三种白盒伪装全部隐形；绝对阈值跨规模不可迁移，"
    "实用判定必须家族内+规模内标定。问题二：蒸馏不留指纹。新维度：同底座结构变体是最强'同源'正信号。"
    "后续：先在服务器跑中间 2 层重评分，再补 internlm2 负例、蒸馏线大 N 复核、"
    "7B 蒸馏学生、7B 面板 ROC 标定。报告与交接文档都在仓库 llm_migration_v4/ 下。",
]
for slide, note in zip(prs.slides, NOTES):
    slide.notes_slide.notes_text_frame.text = note

out = os.path.join(HERE, "汇报_v4_7B规模验证.pptx" if SUBSET == "all" else f"汇报_v4_7B规模验证_{SUBSET}.pptx")
prs.save(out)
print("saved:", out, f"({os.path.getsize(out)//1024} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
