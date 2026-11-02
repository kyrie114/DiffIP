"""Build the advisor-report deck: llm_migration_v4 汇报.pptx (16:9).

Design: white cards on light background, navy/teal/orange palette matching
make_charts.py; Noto Sans CJK SC everywhere (large type sizes).
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
A = HERE / "ppt_assets"
OUT = HERE / "汇报PPT" / "v4_实验汇报.pptx"
OUT.parent.mkdir(exist_ok=True)

FONT = "Noto Sans CJK SC"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x15, 0x9A, 0x9C)
ORANGE = RGBColor(0xE8, 0x87, 0x1A)
RED = RGBColor(0xC7, 0x44, 0x3C)
GREEN = RGBColor(0x3D, 0x9A, 0x50)
PURPLE = RGBColor(0x7B, 0x5E, 0xA7)
INK = RGBColor(0x1E, 0x29, 0x3B)
MUT = RGBColor(0x64, 0x74, 0x8B)
PAPER = RGBColor(0xF4, 0xF7, 0xFB)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
LINEC = RGBColor(0xD7, 0xE0, 0xEC)
LIGHTNAV = RGBColor(0xEA, 0xF1, 0xF9)
LIGHTOR = RGBColor(0xFD, 0xF3, 0xE7)
LIGHTTE = RGBColor(0xE8, 0xF5, 0xF5)
LIGHTRE = RGBColor(0xFB, 0xEF, 0xEF)

SW, SH = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


# ----------------------------------------------------------------- helpers
def set_font(run, size, bold=False, color=INK, name=FONT, italic=False):
    f = run.font
    f.name = name
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    latin = rPr.find(qn("a:latin"))
    prev = latin
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            prev.addnext(el)
        el.set("typeface", name)
        prev = el


def tb(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, wrap=True):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    return tf


def para(tf, text, size, bold=False, color=INK, align=PP_ALIGN.LEFT,
         before=0, after=0, spacing=1.0, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    set_font(run, size, bold, color)
    return p


def rect(slide, x, y, w, h, fill, line=None, round_=False, line_w=0.75):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if round_:
        try:
            shp.adjustments[0] = 0.055
        except Exception:
            pass
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    return shp


def line_shape(slide, x, y, w, color, weight=2.25):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Pt(weight))
    ln.shadow.inherit = False
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    return ln


def circle(slide, x, y, d, fill, text, size, color=CARD, bold=True):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    c.shadow.inherit = False
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    tf = c.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    set_font(r, size, bold, color)
    return c


def pic(slide, name, x, y, w=None, h=None):
    kw = {}
    if w is not None:
        kw["width"] = w
    if h is not None:
        kw["height"] = h
    return slide.shapes.add_picture(str(A / name), x, y, **kw)


PAGE = [0]


def footer(slide):
    PAGE[0] += 1
    tf = tb(slide, Inches(0.55), Inches(7.08), Inches(12.2), Inches(0.3))
    para(tf, f"llm_migration_v4 · 导师汇报 · 2026-11-02          {PAGE[0]:02d}",
         11, False, MUT, first=True)


def header(slide, title, kicker=None):
    rect(slide, Inches(0.55), Inches(0.42), Inches(0.09), Inches(0.58), NAVY)
    tf = tb(slide, Inches(0.82), Inches(0.30), Inches(11.9), Inches(0.85),
            anchor=MSO_ANCHOR.MIDDLE)
    para(tf, title, 30, True, NAVY, first=True)
    if kicker:
        tf2 = tb(slide, Inches(0.82), Inches(1.02), Inches(11.9), Inches(0.35))
        para(tf2, kicker, 14.5, False, MUT, first=True)
    line_shape(slide, Inches(0.55), Inches(1.42), Inches(12.23), LINEC, weight=1.4)


def takeaway(slide, text, y=Inches(6.15), h=Inches(0.78), color=NAVY, bg=LIGHTNAV):
    rect(slide, Inches(0.55), y, Inches(12.23), h, bg, round_=True)
    tf = tb(slide, Inches(0.85), y + Inches(0.08), Inches(11.7), h - Inches(0.16),
            anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    r1 = p.add_run()
    r1.text = "▶  结论："
    set_font(r1, 17, True, color)
    r2 = p.add_run()
    r2.text = text
    set_font(r2, 18, True, color)


def bullet(tf, head, body, size_head=17, size_body=15.5, color=NAVY,
           mark="▪", mark_color=TEAL, first=False, after=9, spacing=1.12):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(after)
    p.line_spacing = spacing
    r = p.add_run()
    r.text = mark + "  "
    set_font(r, size_head, True, mark_color)
    r = p.add_run()
    r.text = head
    set_font(r, size_head, True, color)
    if body:
        r = p.add_run()
        r.text = "　" + body
        set_font(r, size_body, False, INK)


def new_slide():
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, SW, SH, PAPER)
    return s


# ================================================================= 2 agenda
s = new_slide()
header(s, "汇报提纲")
items = [
    ("01", "背景与问题", "v3（0.5B）站在哪，两个待回答的问题"),
    ("02", "实验设计", "主线谱系面板 + 蒸馏对照线，统一协议"),
    ("03", "主线结果", "谱系排序 / 伪装被看穿 / 跨规模规律 / 结构变体口径"),
    ("04", "蒸馏线与结论", "负结果复现 · 可迁移什么、不可迁移什么"),
]
x = Inches(0.72)
y = Inches(1.95)
for i, (num, t, d) in enumerate(items):
    px = x + (i % 2) * Inches(6.25)
    py = y + (i // 2) * Inches(2.25)
    w = Inches(5.85)
    h = Inches(1.85)
    rect(s, px, py, w, h, CARD, LINEC, round_=True)
    circle(s, px + Inches(0.42), py + Inches(0.42), Inches(0.62),
           [NAVY, TEAL, ORANGE, PURPLE][i], num, 17)
    tf = tb(s, px + Inches(1.30), py + Inches(0.28), w - Inches(1.6), h - Inches(0.5))
    para(tf, t, 21, True, NAVY, first=True)
    para(tf, d, 14.5, False, MUT, before=7, spacing=1.1)
footer(s)

# ================================================================= 3 background
s = new_slide()
header(s, "背景：v3 完成了什么，还差什么？", "v3 = 扰动采样 + 逐 prompt 指纹 + PCA/ALS（Qwen2.5-0.5B 面板）")
rect(s, Inches(0.55), Inches(1.75), Inches(5.95), Inches(3.55), CARD, LINEC, round_=True)
tf = tb(s, Inches(0.95), Inches(2.05), Inches(5.2), Inches(3.0))
para(tf, "v3 已得结论", 18, True, NAVY, first=True, after=10)
bullet(tf, "全场可分", "正例 margin 0.52，逐 prompt 32/32 全胜", 17, 15.5)
bullet(tf, "指纹有效", "线性伪装（白盒/置换）在特征空间不可见", 17, 15.5)
bullet(tf, "但面板太小", "victim 仅 0.5B，负例大多是更小模型，无“同底座结构变体”")
bullet(tf, "遗留问题", "ROC 发现距离绝对尺度家族依赖——全局阈值存疑")

for i, (bg, edge, tag, q) in enumerate([
        (LIGHTNAV, NAVY, "问题 ①", "主协议在 7B 上对真实谱系（官方 SFT / 独立衍生 /\n同底座剪枝变体 / 跨家族 7B-8B）判别力如何？"),
        (LIGHTOR, ORANGE, "问题 ②", "序列级蒸馏在 hidden 几何里留下的指纹，\n能否被认出？（v1 判负，协议升级后未重验）")]):
    py = Inches(1.75) + i * Inches(1.90)
    rect(s, Inches(6.85), py, Inches(5.93), Inches(1.68), bg, round_=True)
    rect(s, Inches(6.85), py, Inches(0.09), Inches(1.68), edge)
    tf = tb(s, Inches(7.20), py + Inches(0.17), Inches(5.35), Inches(1.4))
    para(tf, tag, 16.5, True, edge, first=True, after=5)
    para(tf, q, 15.5, False, INK, spacing=1.15)
takeaway(s, "用同一条 v3 协议（逐位不改）直接放大到 7B：真实谱系 + 强对照组，检验“可迁移性”。",
         y=Inches(5.85), h=Inches(0.75))
footer(s)

# ================================================================= 4 design
s = new_slide()
header(s, "实验设计：一个谱系面板 + 一条对照线", "victim 指纹 vs 各模型指纹的相似度；victim 自身 sim≡1 作天花板")
# main panel
rect(s, Inches(0.55), Inches(1.72), Inches(6.35), Inches(4.28), CARD, LINEC, round_=True)
tf = tb(s, Inches(0.90), Inches(1.95), Inches(5.75), Inches(3.9))
para(tf, "主线　victim = Llama-2-7b-hf（32 层）", 17.5, True, NAVY, first=True, after=8)
rows = [
    ("正例", "fine_tune：Llama-2-7b-chat（官方 SFT）", ORANGE),
    ("正例", "derivative_cn：Chinese-Llama-2-7b（独立衍生）", PURPLE),
    ("正例", "prune_dpoff：chat 剪 4 层（结构变换）", ORANGE),
    ("伪装", "chat + 3 种线性伪装（白盒置换/特征置换/缩放）", RED),
    ("正例", "同底座结构变体（正例，稀疏剪枝）×3（SparseGPT / Wanda / GBLM）", TEAL),
    ("易负例", "跨家族 ×2（Qwen2.5-7B / Llama-3-8B）", GREEN),
]
for role, name, c in rows:
    p = tf.add_paragraph()
    p.space_after = Pt(7.5)
    r = p.add_run()
    r.text = f"{role}｜"
    set_font(r, 14.5, True, c)
    r = p.add_run()
    r.text = f"　{name}"
    set_font(r, 15, False, INK)
para(tf, "（internlm2-7b 因本环境深层激活数值发散被隔离，非结论问题）",
     12.5, False, MUT, before=4)
# distill panel
rect(s, Inches(7.15), Inches(1.72), Inches(5.63), Inches(4.28), CARD, LINEC, round_=True)
tf = tb(s, Inches(7.50), Inches(1.95), Inches(5.0), Inches(3.9))
para(tf, "蒸馏线　victim = Llama-3.2-1B（16 层）", 17.5, True, NAVY, first=True, after=10)
bullet(tf, "student_distill", "同一 1B 学生在 3B 教师回答上蒸馏", 15, 14.5, mark_color=NAVY, after=7)
bullet(tf, "student_control", "同指令同预算、原始 Alpaca 回答", 15, 14.5, mark_color=ORANGE, after=7)
para(tf, "——关键对照：区分“学过教师”vs“只是被训练过”", 13, False, MUT, after=14)
para(tf, "负例 4 个：3B 教师本身 / Sheared-1.3B / Qwen2.5-7B / Llama-2-7B",
     14.5, False, INK, after=6)
para(tf, "同家族 + 同词表 → 蒸馏信号纯净；方案与 v3 逐位一致", 13.5, False, MUT)
takeaway(s, "32 prompts × 3 seed sets × N∈{8…128} × dims∈{8,20}；两线共 13 个模型特征、21888 个评分格。",
         y=Inches(6.22), h=Inches(0.70))
footer(s)

# ================================================================= 5 protocol
s = new_slide()
header(s, "统一协议（与 v3 逐位一致）+ 工程规模")
cards = [
    ("扰动采样", "配对种子输入 embedding 扰动 SCALE=2e-3，\n32 prompts 全谱系共享（配对比较）", NAVY, LIGHTNAV),
    ("逐层指纹", "4 个均匀层取 attention-mask 均值池化 →\nPCA(d) → 差异化投影逐层 ALS 对齐", TEAL, LIGHTTE),
    ("评分量", "similarity = exp(−Σd/L)；每格 1152 job，\n两线合计 21888 个评分格全部落盘", ORANGE, LIGHTOR),
]
x = Inches(0.55)
for t, d, c, bg in cards:
    rect(s, x, Inches(1.80), Inches(3.94), Inches(2.55), bg, round_=True)
    tf = tb(s, x + Inches(0.30), Inches(2.08), Inches(3.34), Inches(2.0))
    para(tf, t, 19, True, c, first=True, after=10)
    para(tf, d, 14.5, False, INK, spacing=1.25)
    x += Inches(4.145)
rect(s, Inches(0.55), Inches(4.70), Inches(12.23), Inches(1.30), CARD, LINEC, round_=True)
tf = tb(s, Inches(0.95), Inches(4.92), Inches(11.5), Inches(0.95),
        anchor=MSO_ANCHOR.MIDDLE)
para(tf, "执行规模：全程 CPU（GPU 被占）· Llama-2-7b 采集 38min/个 · 评分 13824+8064 行 · "
         "期间修复（rope 加载 / THP 卡机 / NaN 校验）并两次重跑验证数字逐位一致",
     15.5, False, INK, first=True, spacing=1.2)
takeaway(s, "协议零改动换规模——排序与结论的变化全部来自“规模与面板”，不是方法差异。",
         y=Inches(6.22), h=Inches(0.70))
footer(s)

# ================================================================= 6 c1 genealogy
s = new_slide()
header(s, "主线结果 ①：谱系排序在 7B 上保持，且多出新维度", "N=128，dims=8（dims=20 排序一致）· similarity 越高越像 victim")
pic(s, "c1_genealogy.png", Inches(1.05), Inches(1.30), h=Inches(4.65))
takeaway(s, "同底座结构变体（正例，稀疏剪枝）(0.54) > SFT 簇(0.36) > 独立衍生(0.31) > 跨家族地板(0.25)——"
            "三者是非结构化稀疏剪枝（只有 prune_dpoff 是整层删 4 层）：base 上稀疏剪枝比 SFT 更像基座（v3 面板没有这一层）。",
         y=Inches(6.10), h=Inches(0.80))
footer(s)

# ================================================================= 7 c2 convergence
s = new_slide()
header(s, "主线结果 ②：收敛行为与 v3 一致，地板不随规模变化", "similarity 随采样预算 N 的收敛（dims=8）")
pic(s, "c2_convergence.png", Inches(1.55), Inches(1.42), h=Inches(4.5))
takeaway(s, "N≥64 后各格 std ≤ 0.013、N=64→128 变化 ≤0.02——协议收敛；跨家族曲线贴住 v3 地板水平。",
         y=Inches(6.10), h=Inches(0.78))
footer(s)

# ================================================================= 8 c3 disguise
s = new_slide()
header(s, "主线结果 ③：三种伪装被看穿（分数≡fine_tune），prune_dpoff 归 SFT 簇", "指纹跟着“基座 + 是否 SFT”走，线性伪装对检测无效，谱系照样被认出")
pic(s, "c3_disguise.png", Inches(0.85), Inches(1.42), h=Inches(4.55))
takeaway(s, "weight_perm / rep_perm / scale 三种伪装与 official fine_tune 噪声级一致（0.363–0.365）；"
            "chat+剪4层(0.357)落 SFT 簇而非剪枝簇（0.54）。",
         y=Inches(6.10), h=Inches(0.78))
footer(s)

# ================================================================= 9 c4 scale
s = new_slide()
header(s, "主线结果 ④：跨规模对照——排序可迁移，绝对阈值不可以", "v3(0.5B) vs v4(7B) 同协议对比")
pic(s, "c4_scale.png", Inches(1.55), Inches(1.48), h=Inches(4.5))
takeaway(s, "正例绝对值 0.76→0.36、margin 0.52→0.11，但跨家族地板 ~0.24→0.254 几乎不动——"
            "跨规模可迁移的是相对排序，不是绝对阈值。",
         y=Inches(6.10), h=Inches(0.78))
footer(s)

# ================================================================= 10 c5
s = new_slide()
header(s, "主线结果 ⑤：结构变体（正例）分数高于 SFT 正例", "不是检测失败：胜负/AUC 主口径只对跨家族负例计")
pic(s, "c5_hardneg.png", Inches(1.15), Inches(1.42), h=Inches(4.55))
takeaway(s, "新口径下所有正例对跨家族负例逐 prompt 比较（见左图）；结构变体高于 SFT 只说明“正例 A 比正例 B 更像基座”。"
            "旧口径（若把结构变体当负例）下 SFT 只有 0–1/32，已弃用。",
         y=Inches(6.10), h=Inches(0.78))
footer(s)

# ================================================================= 11 c6 distill
s = new_slide()
header(s, "蒸馏线：负结果，且判得干净", "distill − control ≤0.003，方向随设置翻转（N=20 +0.004 / N=128 −0.003）")
pic(s, "c6_distill.png", Inches(1.00), Inches(1.38), h=Inches(4.5))
takeaway(s, "学生超负例的 margin(+0.03, 32/32) 两臂共享——是“同基座被训练过”的共性，"
            "不是教师蒸馏的指纹。教师本身与无关难例亦无差别。v1 结论在 v4 复现。",
         y=Inches(6.10), h=Inches(0.78), color=RED, bg=LIGHTRE)
footer(s)

# ================================================================= 12 conclusions
s = new_slide()
header(s, "结论")
items = [
    ("协议可迁移", "v3 结论在 7B 上方向性成立：同源高、跨家族低、线性伪装被看穿（对检测无效）；收敛行为一致。", TEAL),
    ("谱系有新秩序", "同底座结构变体（正例，稀疏剪枝）(0.54) > SFT(0.36) > 独立衍生(0.31)；"
                    "指纹 ≈ 基座 + 是否 SFT。", NAVY),
    ("只有排序可迁移", "绝对阈值随规模带漂移（0.76→0.36），地板不动（~0.25）——"
                      "部署需按家族重标定。", ORANGE),
    ("蒸馏指纹不可分", "强对照下 distill ≡ control（≤0.003）：“被蒸馏过”在 N≤128 预算下不可检测。", RED),
]
y = Inches(1.62)
for i, (t, d, c) in enumerate(items):
    h = Inches(1.14)
    rect(s, Inches(0.55), y, Inches(12.23), h, CARD, LINEC, round_=True)
    circle(s, Inches(0.92), y + Inches(0.27), Inches(0.60), c, str(i + 1), 18)
    tf = tb(s, Inches(1.85), y + Inches(0.12), Inches(10.7), h - Inches(0.22),
            anchor=MSO_ANCHOR.MIDDLE)
    para(tf, t, 19, True, c, first=True)
    para(tf, d, 15, False, INK, before=4, spacing=1.1)
    y += h + Inches(0.17)
footer(s)

# ================================================================= 13 limits
s = new_slide()
header(s, "局限与下一步")
rect(s, Inches(0.55), Inches(1.62), Inches(5.95), Inches(4.5), CARD, LINEC, round_=True)
tf = tb(s, Inches(0.95), Inches(1.88), Inches(5.2), Inches(4.0))
para(tf, "局限（复核过的）", 18, True, RED, first=True, after=10)
bullet(tf, "结构变体高于 SFT", "同底座结构变体（正例，稀疏剪枝）逐 prompt 高出 fine_tune 平均 +0.216——家族内排序需注明剪法", 15.5, 14)
bullet(tf, "混池 AUC 仅 0.63", "正例 vs 跨家族、跨 prompt 混池 576:192 行 AUC=0.634/0.638（次要指标）；逐 prompt 胜负是主口径", 15.5, 14)
bullet(tf, "internlm2 未判", "本环境深层 NaN 发散（生态问题）而非“指纹无关”", 15.5, 14)
bullet(tf, "面板仍小", "7B 级仅一个家族的全谱系；蒸馏线仅一组教师-学生", 15.5, 14)

rect(s, Inches(6.85), Inches(1.62), Inches(5.93), Inches(4.5), CARD, LINEC, round_=True)
tf = tb(s, Inches(7.25), Inches(1.88), Inches(5.2), Inches(4.0))
para(tf, "下一步", 18, True, TEAL, first=True, after=10)
bullet(tf, "家族标定", "绝对阈值改为逐家族 ROC 标定 + 相对分位数排序", 15.5, 14, mark_color=TEAL)
bullet(tf, "伪装扩展", "置换/缩放已被看穿；下一步测非线性等更强伪装", 15.5, 14, mark_color=TEAL)
bullet(tf, "蒸馏定量", "放开教师数据量/token 预算，找指纹消失的定量边界", 15.5, 14, mark_color=TEAL)
bullet(tf, "结构变体正用", "把同底座结构变体（稀疏剪枝）拿去做“变体归因”（哪个层被剪决定了相似度衰减）", 15.5, 14, mark_color=TEAL)
takeaway(s, "负例告诉我们：＋协议底线在哪，✗ 哪些“看似可分”的结论要在强对照下重说。",
         y=Inches(6.35), h=Inches(0.66))
footer(s)

# ================================================================= 14 end
s = new_slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(6.62), SW, Inches(0.10), ORANGE)
tf = tb(s, Inches(1.2), Inches(2.55), Inches(10.9), Inches(1.6))
para(tf, "谢谢，恳请指正", 42, True, CARD, first=True)
para(tf, "全部数据、图与协议脚本可复现：scores/*.json → analysis.py → make_charts.py",
     17, False, RGBColor(0xC9, 0xD8, 0xEC), before=18)
tf = tb(s, Inches(1.2), Inches(4.6), Inches(10.9), Inches(0.8))
para(tf, "备份页（备问）：逐格数值表 / internlm2 隔离记录 / THP 卡机修复 / 两次独立重跑一致性",
     14, False, RGBColor(0x8F, 0xA8, 0xC9), first=True)
footer(s)

prs.save(OUT)
print("saved ->", OUT, f"({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
