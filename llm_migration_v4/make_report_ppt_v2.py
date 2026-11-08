"""
构建面向导师的超高质量精编汇报 PPT (16 页完整版，完美对齐 PPT文案.md 规范)
输出: 汇报_DiffIP_大模块与消融实验_v2.pptx
"""
import os
import sys
from pathlib import Path
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = Path("/home/lushunxin/work/DiffIP/llm_migration_v4")
FIGS = HERE / "汇报PPT_v2" / "figs"
OUT_PPT = HERE / "汇报_DiffIP_大模块与消融实验_v2.pptx"

# 颜色规范
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x15, 0x9A, 0x9C)
ORANGE = RGBColor(0xE8, 0x87, 0x1A)
RED = RGBColor(0xC7, 0x44, 0x3C)
PURPLE = RGBColor(0x7B, 0x5E, 0xA7)
GREEN = RGBColor(0x3D, 0x9A, 0x50)
INK = RGBColor(0x1E, 0x29, 0x3B)
MUT = RGBColor(0x64, 0x74, 0x8B)
PAPER = RGBColor(0xF4, 0xF7, 0xFB)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
LINEC = RGBColor(0xD7, 0xE0, 0xEC)
LIGHTNAV = RGBColor(0xEA, 0xF1, 0xF9)
LIGHTTE = RGBColor(0xE8, 0xF5, 0xF5)
LIGHTOR = RGBColor(0xFD, 0xF3, 0xE7)
LIGHTRE = RGBColor(0xFB, 0xEF, 0xEF)

FONT = "Noto Sans CJK SC"

SW, SH = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]

def set_font(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)

def add_card(slide, x, y, w, h, fill=CARD, line=LINEC):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_header(slide, title, kicker, page_num):
    # 背景
    add_card(slide, 0, 0, 13.333, 7.5, fill=PAPER, line=None)
    # 左侧竖条
    add_card(slide, 0.55, 0.42, 0.09, 0.58, fill=NAVY, line=None)
    # 标题
    tb = slide.shapes.add_textbox(Inches(0.82), Inches(0.30), Inches(11.9), Inches(0.85))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title
    set_font(r, 26, bold=True, color=NAVY)
    
    # Kicker
    tb_k = slide.shapes.add_textbox(Inches(0.82), Inches(1.02), Inches(11.9), Inches(0.35))
    tf_k = tb_k.text_frame; tf_k.word_wrap = True
    p_k = tf_k.paragraphs[0]
    r_k = p_k.add_run(); r_k.text = kicker
    set_font(r_k, 13, bold=False, color=MUT)
    
    # 分隔线
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.42), Inches(12.23), Inches(0.015))
    line.fill.solid(); line.fill.fore_color.rgb = LINEC
    line.line.fill.background()
    
    # 页脚
    tb_f = slide.shapes.add_textbox(Inches(0.55), Inches(7.08), Inches(12.2), Inches(0.3))
    tf_f = tb_f.text_frame; tf_f.word_wrap = True
    p_f = tf_f.paragraphs[0]
    r_f = p_f.add_run(); r_f.text = f"llm_migration_v4 · 导师汇报 · 2026-11-08    {page_num:02d}/16"
    set_font(r_f, 10, bold=False, color=MUT)

def add_bottom_bar(slide, text, color=TEAL, fill=LIGHTTE):
    card = add_card(slide, 0.55, 6.15, 12.23, 0.78, fill=fill, line=color)
    tb = slide.shapes.add_textbox(Inches(0.75), Inches(6.20), Inches(11.8), Inches(0.68))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    set_font(r, 14, bold=True, color=color)

def add_two_column_slide(slide, title, kicker, page_num, img_path, card_title, bullets, takeaway, color=TEAL, fill=LIGHTTE, img_w=6.8):
    add_header(slide, title, kicker, page_num)
    
    # 放置图片 (保持宽高比)
    if img_path.exists():
        with Image.open(img_path) as im:
            iw, ih = im.size
        target_w = img_w
        target_h = target_w * (ih / iw)
        # 上限 4.2 in
        if target_h > 4.3:
            target_h = 4.3
            target_w = target_h * (iw / ih)
        slide.shapes.add_picture(str(img_path), Inches(0.55), Inches(1.65), Inches(target_w), Inches(target_h))
    
    # 右侧卡片
    rx = 0.55 + target_w + 0.35
    rw = 12.23 - target_w - 0.35
    add_card(slide, rx, 1.65, rw, 4.3, fill=CARD, line=LINEC)
    
    tb = slide.shapes.add_textbox(Inches(rx + 0.25), Inches(1.80), Inches(rw - 0.5), Inches(4.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = card_title
    set_font(r, 15, bold=True, color=NAVY)
    
    for b in bullets:
        p_b = tf.add_paragraph()
        p_b.space_before = Pt(8)
        r_b = p_b.add_run(); r_b.text = f"• {b}"
        set_font(r_b, 11.5, bold=False, color=INK)
        
    add_bottom_bar(slide, takeaway, color=color, fill=fill)

# ==============================================================================
# PAGE 1: 封面
# ==============================================================================
slide1 = prs.slides.add_slide(BLANK)
add_card(slide1, 0, 0, 13.333, 7.5, fill=PAPER, line=None)
add_card(slide1, 0, 0, 0.25, 7.5, fill=NAVY, line=None)

tb_t = slide1.shapes.add_textbox(Inches(0.95), Inches(1.55), Inches(11.0), Inches(1.3))
tf = tb_t.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "从相似度到谱系判定"
set_font(r, 38, bold=True, color=NAVY)

tb_sub = slide1.shapes.add_textbox(Inches(0.98), Inches(2.9), Inches(11.0), Inches(0.7))
tf_sub = tb_sub.text_frame; tf_sub.word_wrap = True
p = tf_sub.paragraphs[0]
r = p.add_run(); r.text = "v4 7B 面板: 绝对阈值、相对分数与无关模型压力测试 (大模块 + 3 个消融实验)"
set_font(r, 18, bold=False, color=MUT)

div = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.98), Inches(3.85), Inches(5.8), Inches(0.03))
div.fill.solid(); div.fill.fore_color.rgb = TEAL
div.line.fill.background()

tb_info = slide1.shapes.add_textbox(Inches(0.98), Inches(4.25), Inches(9.0), Inches(1.5))
tf_info = tb_info.text_frame; tf_info.word_wrap = True
p = tf_info.paragraphs[0]
r = p.add_run(); r.text = "汇报组织: 1 个大模块核心方法 + 3 个严谨消融实验 + 5 模型 20 对留出扩盘\n汇报体系: 导师汇报 · 2026-11-08 · 工作目录 llm_migration_v4"
set_font(r, 14, bold=False, color=INK)

# ==============================================================================
# PAGE 2: 汇报提要 (站住与没站住)
# ==============================================================================
slide2 = prs.slides.add_slide(BLANK)
add_header(slide2, "汇报提要: 站住的和没站住的", "本次汇报的总体结论与学术边界 (诚实汇报)", 2)

# 左卡片 (站住的)
add_card(slide2, 0.55, 1.65, 5.95, 4.3, fill=CARD, line=LINEC)
add_card(slide2, 0.55, 1.65, 5.95, 0.1, fill=TEAL, line=None)
tb2_l = slide2.shapes.add_textbox(Inches(0.85), Inches(1.9), Inches(5.35), Inches(3.8))
tf = tb2_l.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "✔ 站住的 (核心大模块与阳性结果)"
set_font(r, 16, bold=True, color=NAVY)

items_ok = [
    "7B 谱系相对排序完全保持: 结构变体(0.54) > SFT 簇(0.363) > 独立衍生(0.311) > 跨家族(0.254)。",
    "同 Prompt 相对差判成立: 减去参照负例后，AUC 0.9998，SFT 留出召回率高达 97%~100%。",
    "配对几何是区分基石: 打乱样本配对后分数暴跌至地板 (0.253，胜率 5/32)。",
    "特征空间攻击无法逃逸: 9 种非线性变换全部未能抹除谱系，混入 >75% 能量才触及地板。"
]
for item in items_ok:
    p = tf.add_paragraph()
    p.space_before = Pt(8)
    r = p.add_run(); r.text = f"• {item}"
    set_font(r, 11, color=INK)

# 右卡片 (没站住的)
add_card(slide2, 6.83, 1.65, 5.95, 4.3, fill=CARD, line=LINEC)
add_card(slide2, 6.83, 1.65, 5.95, 0.1, fill=RED, line=None)
tb2_r = slide2.shapes.add_textbox(Inches(7.13), Inches(1.9), Inches(5.35), Inches(3.8))
tf = tb2_r.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "✖ 没站住的 (严谨记录的阴性与边界)"
set_font(r, 16, bold=True, color=RED)

items_fail = [
    "绝对分数无法跨规模迁移: 0.5B 阈值搬到 7B，仅接 32% SFT，同时误接 23% 跨家族负例。",
    "序列级蒸馏线完全不可检: distill 与 control 差仅 -0.003，AUC 0.493 (干净阴性)。",
    "独立衍生无法与官方 SFT 共用阈值: Chinese-llama 间隔仅 0.019，召回率停在 50%。",
    "无关模型 20 对留出测试为 18/20: mpt-7b 尾部极值抖动导致 2 对微小误报 (需保护带)。"
]
for item in items_fail:
    p = tf.add_paragraph()
    p.space_before = Pt(8)
    r = p.add_run(); r.text = f"• {item}"
    set_font(r, 11, color=INK)

add_bottom_bar(slide2, "一句话底线: 相对排序与配对判定完全站住；绝对阈值不可跨规模；蒸馏线是干净阴性。", TEAL, LIGHTTE)

# ==============================================================================
# PAGE 3: 为什么扩大到 7B
# ==============================================================================
slide3 = prs.slides.add_slide(BLANK)
add_header(slide3, "为什么必须从 0.5B 放大到 7B？", "从玩具模型到真实工业级谱系的必然跨越", 3)

tb3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.3))
tf3 = tb3.text_frame; tf3.word_wrap = True

reasons = [
    ("v3 (0.5B) 的局限性与误读风险", 
     "0.5B 面板以 Qwen2.5-0.5B 为基座，负例选用 SmolLM2-135M 等极小规模模型，导致负例太容易分开 (Similarity ~0.24 vs 正例 ~0.76)。这容易给审稿人留下「指纹只在玩具规模生效」的质疑。"),
    ("同底座稀疏结构变体 (正例) 维度缺失", 
     "在真实开源生态中，模型被剪枝 (SparseGPT/Wanda/GBLM) 是极常见的衍生手段。0.5B 没有同底座真实稀疏剪枝模型，无法探究「剪枝扰动」与「SFT 漂移」在几何上的相对远近。"),
    ("绝对阈值的家族与规模依赖", 
     "v2/v3 的实验发现尺度依赖严重。如果不放大到 7B，就无法回答核心学术问题:「DiffIP 测量出的究竟是绝对的指纹高度，还是相对的几何结构？能否跨规模设置通用阈值？」"),
    ("最终选型: Llama-2-7b 全谱系面板", 
     "本地库中 Llama-2-7b 拥有完整的官方 chat、3 个同底座真实剪枝变体、独立二开衍生版 (Chinese-llama-2)，以及同量级跨家族 7B/8B (Qwen2.5, Llama-3)。")
]

for idx, (head, body) in enumerate(reasons):
    p = tf3.paragraphs[0] if idx == 0 else tf3.add_paragraph()
    if idx > 0: p.space_before = Pt(12)
    r = p.add_run(); r.text = f"{idx+1}. {head}\n"
    set_font(r, 14, bold=True, color=NAVY)
    r2 = p.add_run(); r2.text = body
    set_font(r2, 11.5, color=INK)

add_bottom_bar(slide3, "核心动机: 只有在 7B 真实谱系上检验，才能确立 DiffIP 面向现代 LLM 的真正科学价值。", NAVY, LIGHTNAV)

# ==============================================================================
# PAGE 4: 实验方法与流程
# ==============================================================================
slide4 = prs.slides.add_slide(BLANK)
add_header(slide4, "【大模块核心方法】DiffIP 几何指纹对齐流程", "严格无损放大自 DiffIP 原文的四阶段测量协议", 4)

method_cards = [
    ("Stage 1: 配对嵌入加噪", "同一 Prompt 在 Embedding 上施加高斯扰动 (Scale=2e-3)，两模型严格共享随机种子，构建 N=128 扰动样本。"),
    ("Stage 2: 逐层隐状态抽取", "前向推理抓取 Transformer 隐层 (默认抽取 4 均匀层: L0, L10, L21, L31)，做 Attention-Mask 加权均值池化。"),
    ("Stage 3: PCA 降维与流形对齐", "每层独立 PCA 降至 8 维；使用 ALS (交替最小二乘法) 拟合旋转 Q、各向异性缩放 S 与位移 h，消除仿射变换计算 L2 误差。"),
    ("Stage 4: 相对参照差值判定", "针对 Prompt 语义难度带来的绝对分数偏移，提出同一 Prompt 减去参照负例: Sim(Suspect) - Sim(Reference)。")
]

for idx, (title, text) in enumerate(method_cards):
    cx = 0.55 + idx * 3.12
    add_card(slide4, cx, 1.7, 2.95, 4.2, fill=CARD, line=LINEC)
    add_card(slide4, cx, 1.7, 2.95, 0.1, fill=TEAL if idx < 3 else ORANGE, line=None)
    
    tb = slide4.shapes.add_textbox(Inches(cx + 0.15), Inches(1.9), Inches(2.65), Inches(3.8))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = f"Step {idx+1}\n{title}\n"
    set_font(r, 13, bold=True, color=NAVY)
    
    p2 = tf.add_paragraph()
    p2.space_before = Pt(8)
    r2 = p2.add_run(); r2.text = text
    set_font(r2, 11, color=INK)

add_bottom_bar(slide4, "设计精髓: 通过配对扰动诱发流形几何，通过 ALS 回退消除表面伪装，通过相对差消除 Prompt 难度偏差。", TEAL, LIGHTTE)

# ==============================================================================
# PAGE 5: 7B 基线谱系排序 (图 1)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="大模块实测: 7B 谱系梯队排序严谨成立",
    kicker="基座 >> 稀疏结构变体 > SFT 簇 > 独立衍生 > 跨家族地板 (无翻转)",
    page_num=5,
    img_path=FIGS / "fig_baseline.png",
    card_title="核心数值与谱系梯队",
    bullets=[
        "稀疏结构变体 (正例): 0.528~0.559 (SparseGPT/Wanda/GBLM)，未微调而只剪枝，几何极其靠近基座。",
        "官方 SFT 簇: 0.357~0.365 (chat、通道置换、缩放、层剪枝)，微调引起明显但有限的几何漂移。",
        "独立衍生版: 0.311 (Chinese-llama-2-7b)，独立继续预训练带来的漂移大于官方微调。",
        "跨家族负例地板: 0.254 (Qwen2.5-7B, Llama-3-8B)，与 0.5B 地板 (0.24) 高度一致！",
        "逐 Prompt 胜负: 结构变体与 SFT 对负例取得 32/32 全胜；衍生版 25/32。"
    ],
    takeaway="结论: 谱系相对排序在 7B 规模完全保持！同底座稀疏剪枝变体比 SFT 更靠近基座。",
    color=TEAL, fill=LIGHTTE
)

# ==============================================================================
# PAGE 6: 蒸馏线干净阴性 (图 2)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="蒸馏线对照: 序列级蒸馏无法在几何中检出 (干净阴性)",
    kicker="Llama-3.2 1B 学生 / 3B 教师严格双盲微调实验",
    page_num=6,
    img_path=FIGS / "fig_distill.png",
    card_title="科学对照与实验判读",
    bullets=[
        "严格双盲对照设计: Distill (在教师生成答案上训) vs Control (在原始 Alpaca 上训)，同预算同轮次。",
        "得分高度重合: Distill 相似度 0.620，Control 0.623 (差 -0.003)，落在种子标准差 (0.005) 之内。",
        "AUC 完全随机: AUC 仅 0.493，96 对测试中只有 40 对 Distill 更高，统计上无法区分。",
        "审稿防御意义: 两学生均比负例高 0.03，证明那是「被微调过」的基座共性，绝非蒸馏指纹！",
        "学术态度: 不强行编造结论，承认序列级黑盒蒸馏不传递中间层几何指纹，避免假阳性。"
    ],
    takeaway="结论: 序列级 KD 不改变中间层流形几何。此阴性结果极具学术价值，避免了错误指控。",
    color=RED, fill=LIGHTRE
)

# ==============================================================================
# PAGE 7: 消融一: 层数选型 (图 3)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="消融实验 1: 抽层策略选型 (even4 vs mid2 vs all32)",
    kicker="为什么稀疏均匀 4 层已经是性能峰值？深层局部噪声的发现",
    page_num=7,
    img_path=FIGS / "fig_mid2.png",
    card_title="层数对比与消融发现",
    bullets=[
        "mid2 (中间两层): 排序虽未翻转，但 SFT 到地板的间隔从 0.109 收窄到 0.079，辨识度下降。",
        "全层扫描 (All 32): 间隔进一步收缩至 0.043，微调簇留出召回率跌至 71%，误报率恶化至 12.5%。",
        "末端 4 层 (Last 4): 间隔仅 0.033，召回率仅 66%~78%，表现最差。",
        "核心机理解释: 后半段网络强受训练目标与注意力局部细节主导，引入大量特异性噪声，掩盖了骨架指纹。",
        "工程价值: 均匀 4 层 (L0, L10, L21, L31) 节约 87.5% 存储与算力，且达到最高判别间隔！"
    ],
    takeaway="消融结论: 并不是层数越多越好！深层局部差异是高频噪声，均匀稀疏 4 层是最佳实践。",
    color=ORANGE, fill=LIGHTOR
)

# ==============================================================================
# PAGE 8: 判定机制: 绝对阈值失效 (图 4)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="判定机制突破: 为什么全局绝对阈值行不通？",
    kicker="Prompt 语义难度导致绝对分数方差极大，AUC 仅 0.702",
    page_num=8,
    img_path=FIGS / "fig_absolute.png",
    card_title="绝对阈值崩溃原因拆解",
    bullets=[
        "逐条 Prompt 标准差高达 0.25: fine_tune 与负例在单条 Prompt 上的均值相关性高达 0.956！",
        "语义难度主导高度: 数学类 Prompt 均值达 0.533，而问答类仅 0.077；负例在数学类得分远超正例在问答类。",
        "绝对阈值切分失败: 若将阈值设为低误报点 (0.786)，SFT 召回率暴跌至 6%；整体 AUC 仅 0.702。",
        "命题 T 失败: 将 0.5B 的阈值 (0.499) 原样搬到 7B，仅能接纳 32% SFT，同时误收 23% 跨家族负例。",
        "科学定性: 绝对数值不可跨规模迁移，必须寻找同 Prompt 下的相对标定方式。"
    ],
    takeaway="机制突破: 绝对阈值失效是由于 Prompt 难度引起的基底平移，必须引入同 Prompt 差分消除！",
    color=RED, fill=LIGHTRE
)

# ==============================================================================
# PAGE 9: 判定机制: 相对消除与 AUC 0.9998 (图 5)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="判定机制确立: 同 Prompt 相对消除与 AUC 0.9998",
    kicker="sim(Suspect) - sim(Ref): 优雅消除语义难度，召回达 97%~100%",
    page_num=9,
    img_path=FIGS / "fig_relative.png",
    card_title="相对判定机制与留出验证",
    bullets=[
        "优雅消除基底效应: 分数改为 Sim(模型) - Sim(无关参照)，留出另一个负例作为盲测检验。",
        "AUC 跃升至 0.9998: 部署正类 (剪枝 + SFT) 留出召回率高达 97%~100%，留出误报率仅 0%~3%！",
        "双向对称通过: Qwen 做参照 / Llama-3 做留出，及反向对称，均完全通过预注册线。",
        "严格因果检验 (实验 F): 若减去错开的 Prompt，AUC 跌回 0.644；若只减常数，AUC 跌回 0.702。",
        "独立衍生边界: Chinese-llama 间隔仅 0.019，召回率 50%，证实衍生模型不能与 SFT 混用同一阈值。"
    ],
    takeaway="判定基石: 必须减同一条 Prompt！相对参照差分机制彻底解决了 LLM 指纹的落地判定问题。",
    color=GREEN, fill=LIGHTTE
)

# ==============================================================================
# PAGE 10: 实验消融: 配对因果与混合攻击 (图 6)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="消融实验 2: 配对因果与无关模型混入极限",
    kicker="打乱配对分数立即坍塌到地板；拉齐能量后需混入 >75% 才失效",
    page_num=10,
    img_path=FIGS / "fig_break.png",
    card_title="配对因果与能量稀释分析",
    bullets=[
        "打乱配对几何坍塌 (Shuffle): 将微调模型样本顺序打乱后，相似度降至 0.253，胜率跌至 5/32！",
        "因果实锤: 证明指纹不是静态词表或输出统计，而是微小输入扰动诱发的「配对几何流形响应」。",
        "RMS 能量修正 (实验 G): 微调模型激活 RMS 是 Llama-3 的 2.54 倍，插值 75% 时能量仍占 42%。",
        "拉齐能量插值: 掺入 50% 无关模型后，得分 0.312，对 Qwen 仍胜 29/32；谱系依然稳定存在。",
        "极限阈值: 必须将无关模型掺入比例提升至 75% (微调能量仅剩 10%)，均值才贴近地板 (0.259)。"
    ],
    takeaway="消融发现: 配对关系是指纹的灵魂；在真实篡改中，只有掺入压倒性无关表征才能抹除谱系。",
    color=PURPLE, fill=PAPER
)

# ==============================================================================
# PAGE 11: 消融三: 非线性攻击韧性 (图 7)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="消融实验 3: 特征空间 9 种非线性伪装韧性",
    kicker="8/9 稳健且 32/32 全胜；最强平方变换仅降 0.019，高出地板 0.090",
    page_num=11,
    img_path=FIGS / "fig_nonlinear.png",
    card_title="非线性攻击防御明细",
    bullets=[
        "预注册攻击族: 包含符号保持平方根、置换复合、Tanh 饱和、各向异性缩放、符号平方、块旋转等。",
        "8 种攻击完全稳健: 得分与未伪装 fine_tune (0.363) 的偏差均在 0.018 带内，战胜负例率全部保持 32/32！",
        "最强攻击——符号保持平方: 分数微降至 0.344，但仍高出负例地板 0.090，胜负依然达到 31/32！",
        "逐维正态分位数化 (RankGauss): 将特征强行拉成标准正态后，得分 0.348，依然 32/32 全胜。",
        "防御结论: 特征空间单调变换与各向异性变形无法隐匿谱系几何，DiffIP 具备极强抗篡改韧性。"
    ],
    takeaway="消融结论: 非线性伪装是「削弱但不可抹除」！恶意攻击者无法靠表面特征变换逃脱检测。",
    color=TEAL, fill=LIGHTTE
)

# ==============================================================================
# PAGE 12: 维数扫描消融 (图 8)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="消融扩展: PCA 降维空间维数扫描 (8~40 维)",
    kicker="8 维至 40 维表现平稳，符合预注册门限，默认锁定 8 维",
    page_num=12,
    img_path=FIGS / "fig_dim.png",
    card_title="维数消融数据分析",
    bullets=[
        "全维度通过留出测试: 8, 10, 15, 20, 30, 40 维在两个参照方向上的强正例与 SFT 召回率均 ≥90%，误报 ≤10%。",
        "间隔增益微弱: 间隔中位数从 8 维的 0.092 仅微升至 40 维的 0.098 (+0.006)，标准差仅降 0.003。",
        "未达预注册切换线: 预注册规则规定「替换 8 维必须间隔提高 ≥0.02 且标准差不恶化」，因此保持 8 维。",
        "复现精度验证: 8 维与已存分数最大绝对差仅 2.0e-5，通过严格数值复现检验。",
        "工程意义: 8 维在保证判定效力的同时，极大地降低了 ALS 矩阵对齐的计算复杂度，利于大规模部署。"
    ],
    takeaway="消融结论: 谱系流形高度浓缩于低维空间！8 维兼具极佳的数学紧凑性与判别鲁棒性。",
    color=PURPLE, fill=PAPER
)

# ==============================================================================
# PAGE 13: 抽层重跑数据表 (图 9)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="消融扩展: 全层特征重采与抽层量化对齐",
    kicker="均匀 4 层以 0.092 间隔显著超越 8 层 (0.066) 与全部层 (0.043)",
    page_num=13,
    img_path=FIGS / "fig_layers.png",
    card_title="全层重采小盘量化验证",
    bullets=[
        "全 32 层重新采集验证: 覆盖受害模型、chat 微调、SparseGPT、Qwen2.5、Llama-3 全量隐状态。",
        "均匀 4 层复现检验: 重打分数与实验一最大绝对差仅 4.0e-6，复现门完全通过。",
        "量化间隔递减: 均匀 4 层 (0.092) > 均匀 8 层 (0.066) > 全部 32 层 (0.043) > 末端 4 层 (0.033)。",
        "召回率断崖: 全部 32 层的微调簇留出召回率下滑至 0.71 和 0.79，误报率超过 0.10，直接被否决！",
        "架构异构兼容: Qwen 为 28 层，与 32 层模型采用等比例均匀抽层，成功实现跨架构等价对齐。"
    ],
    takeaway="定量结论: 均匀 4 层抽样策略在精度与召回率上均处于绝对领先，正式锁定为最终协议标准。",
    color=NAVY, fill=LIGHTNAV
)

# ==============================================================================
# PAGE 14: 无关模型 20 对留出扩盘 (图 10)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="模型扩盘验证: 5 个无关模型在 20 对参照下的压力测试",
    kicker="Qwen, Llama-3, Llama-1, Baichuan, MPT: 18/20 对原生通过",
    page_num=14,
    img_path=FIGS / "fig_unrel_matrix.png",
    card_title="20 对交叉留出测试结果",
    bullets=[
        "扩充至 5 个异构无关模型: 涵盖中英文、不同词表、不同预训练架构，构建 20 对两两留出组合。",
        "18 / 20 对完美通过: 强正例与微调簇留出召回率达 100%，留出负例误报率低于 3%，AUC 达 0.9999！",
        "未完全通过的两对: 均出在以 MPT-7B 为留出负例的组合 (种子 2 误报 0.125 和 0.188，略超 0.10)。",
        "分布高度一致: MPT-7B 均值 0.2528，与其它 4 个无关模型 (0.2525~0.2537) 完全一致，并非其更像基座。",
        "边界成因: 阈值由种子 0 的 32 条最大值确定，MPT 在留出种子上的极值微弱抖动导致误报。"
    ],
    takeaway="扩盘结论: 相对判定在绝大多数异构模型上极度稳健；MPT-7B 暴露了单种子极值估计的边界点。",
    color=GREEN, fill=LIGHTTE
)

# ==============================================================================
# PAGE 15: MPT 边界诊断与保护带 (图 11)
# ==============================================================================
add_two_column_slide(
    slide=prs.slides.add_slide(BLANK),
    title="边界归因与诊断: MPT 尾部极值与保护带效应",
    kicker="排序与 AUC (0.9999) 完好，事后 +0.002 保护带全量 20/20 通过",
    page_num=15,
    img_path=FIGS / "fig_unrel_diag.png",
    card_title="尾部极值诊断与改进路径",
    bullets=[
        "阈值微调即可过线: Qwen→MPT 只需将阈值抬高 0.0004，Llama-3→MPT 只需抬高 0.0018 即可压入 0.10。",
        "事后保护带验证: 引入 +0.002 的极值安全保护带后，20 对测试全部通过 (20/20)，且召回率仍为 100%！",
        "预注册学术诚实度: 保护带属于事后诊断分析，我们严格在报告中登记为 18/20，不粉饰预注册结论。",
        "诊断结论: 失败并非几何对齐失效或模型语义相似，而是「仅凭 32 个负例估计极值尾部」的统计波动。",
        "下一轮规划: 将安全保护带 (Margin Buffer) 或更多 Prompt 样本正式写入预注册协议。"
    ],
    takeaway="学术态度: 彻底查清未达标原因——是样本量极值统计方差而非方法失效，为后续完善指明方向。",
    color=ORANGE, fill=LIGHTOR
)

# ==============================================================================
# PAGE 16: 总结与规划
# ==============================================================================
slide16 = prs.slides.add_slide(BLANK)
add_header(slide16, "工作总结与面向论文发表的后续规划", "研究闭环概览与下一步工作路线图", 16)

final_cards = [
    ("大模块落地完成 (主线确立)", "DiffIP 在 7B 上的有效性确立",
     "成功将 DiffIP 放大至 7B 真实模型，创立了同 Prompt 相对消除机制，在 7B 全谱系上达成 AUC 0.9998。已封装成开箱即用的 LineagePanel 生产级接口。"),
    ("三大消融体系完整闭环", "深入回答了关于层、维、攻击的本质",
     "消融 1 证明 4 层均匀采样兼具最优性与轻量化；消融 2 证实 8 维及配对流形是核心机理；消融 3 证实对 9 种非线性伪装具备压倒性抗性，需插值 >75% 才能抹除。"),
    ("后续工作路线 (论文冲刺)", "完善保护带与增加基座评测",
     "在预注册协议中加入 +0.002 保护带以达成 20/20 完美扩盘；整理消融图表与理论推导，直接对接顶会 (ICML/NeurIPS/ACL) 的正文与附录章节。")
]

for idx, (head, sub, body) in enumerate(final_cards):
    cx = 0.55 + idx * 4.15
    add_card(slide16, cx, 1.7, 3.9, 4.2, fill=CARD, line=LINEC)
    add_card(slide16, cx, 1.7, 3.9, 0.1, fill=NAVY if idx==0 else (TEAL if idx==1 else ORANGE), line=None)
    
    tb = slide16.shapes.add_textbox(Inches(cx + 0.2), Inches(1.9), Inches(3.5), Inches(3.8))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = f"{head}\n"
    set_font(r, 14, bold=True, color=NAVY)
    
    p_sub = tf.add_paragraph()
    p_sub.space_before = Pt(3)
    r_sub = p_sub.add_run(); r_sub.text = f"{sub}\n"
    set_font(r_sub, 11, bold=True, color=TEAL if idx==1 else (ORANGE if idx==2 else MUT))
    
    p_b = tf.add_paragraph()
    p_b.space_before = Pt(10)
    r_b = p_b.add_run(); r_b.text = body
    set_font(r_b, 10.5, color=INK)

add_bottom_bar(slide16, "总体判读: 1 个大模块 + 3 个核心消融 + 5 异构模型扩盘已完全具备顶会论文与项目汇报质量！", NAVY, LIGHTNAV)

# 保存
prs.save(str(OUT_PPT))
print(f"✅ 16页精编版 PPT 生成成功: {OUT_PPT}")
